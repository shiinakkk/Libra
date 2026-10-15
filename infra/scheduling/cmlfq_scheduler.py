"""Support code for Cmlfq scheduler."""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from RL_Framework.infra.scheduling.base import (
    BaseScheduler,
    InstanceHandle,
    LoadBalanceStrategy,
    MetricsFeedbackConfig,
    SchedulingResult,
)
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import (
    CausalPrefixTree,
    PrefixTreeNode,
    Trajectory,
)
from RL_Framework.infra.scheduling.cmlfq_shared_state import SharedCMLFQLoadState
from RL_Framework.infra.scheduling.cmlfq_tool_state import (
    ToolReturnState,
    DefaultToolStateExtractor,
    ToolStateExtractor,
    ToolStateExtractorRegistry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass
class CMLFQRequestState:
    """C m l f q request state implementation."""
    request_id: str
    prompt_id: str
    current_bucket: str
    current_instance_index: int = -1
    collected_return_states: list[ToolReturnState] = field(default_factory=list)
    generated_tokens: int = 0
    tool_return_token_positions: list[int] = field(default_factory=list)
    total_output_tokens: int = 0
    created_at: float = field(default_factory=time.time)
    has_migrated: bool = False
    input_tokens: int = 0


@dataclass
class CMLFQMigrationDecision:
    """C m l f q migration decision implementation."""
    should_migrate: bool
    reason: str
    current_bucket: str
    target_bucket: str = ""
    mean_remaining_length: float = 0.0
    p90_remaining_length: float = 0.0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class CMLFQScheduler(BaseScheduler):
    """C m l f q scheduler implementation."""

    DEFAULT_BUCKETS = {
        "short": {"tp_degrees": [1, 2], "max_tokens": 5000},
        "long": {"tp_degrees": [4, 8], "max_tokens": 50000},
    }

    def __init__(
        self,
        buckets: dict[str, dict] | None = None,
        bucket_thresholds: dict[str, int] | None = None,
        tool_state_extractor: ToolStateExtractor | None = None,
        prefix_tree: CausalPrefixTree | None = None,
        load_balance_strategy: str = "least_connections",
        max_queue_length: int = 100,
        enable_fallback: bool = True,
        rebuild_interval: int = 50,
        max_recent_trajectories: int = 5000,
        tree_path: str = "",
        tree_persist_interval: int = 1,
        shared_load_dir: str = "",
        shared_load_ttl_s: float = 60.0,
        shared_load_heartbeat_s: float = 10.0,
        load_metric: str = "tokens",
        kv_capacity_tokens_by_tp: dict[int, int] | None = None,
        feedback: MetricsFeedbackConfig | None = None,
        prefix_affinity: bool = False,
        prefix_affinity_max_entries: int = 8192,
    ):
        # In-bucket selection reuses the base ranking stack (capacity
        # normalization, cross-rank aggregate, /metrics bias, occupancy
        # admission, equal-load rotation). Only the BASE ESTIMATE differs:
        # ``active_tokens`` here is the sum of prefix-tree residual charges
        # (see ``_set_request_charge``), not the per-category output EMA
        # that ``_record_route`` would install. Hence "tokens" rather than
        # the base default "requests" -- ranking peers by request count
        # throws that estimate away.
        super().__init__(
            name="C-MLFQ",
            load_metric=load_metric,
            kv_capacity_tokens_by_tp=kv_capacity_tokens_by_tp,
            feedback=feedback,
            prefix_affinity=prefix_affinity,
            prefix_affinity_max_entries=prefix_affinity_max_entries,
        )


        self._buckets = buckets or dict(self.DEFAULT_BUCKETS)
        self._bucket_thresholds = bucket_thresholds or {
            name: cfg.get("max_tokens", 5000)
            for name, cfg in self._buckets.items()
        }

        self._sorted_bucket_names = sorted(
            self._bucket_thresholds.keys(),
            key=lambda b: self._bucket_thresholds[b],
        )


        self._bucket_rules: dict[str, Any] = {}
        for bname, bcfg in self._buckets.items():
            tp_degrees = bcfg.get("tp_degrees", [1, 2])
            all_other_tps = []
            for other_name, other_cfg in self._buckets.items():
                if other_name != bname:
                    all_other_tps.extend(other_cfg.get("tp_degrees", []))
            self._bucket_rules[bname] = {
                "preferred_tp_degrees": tp_degrees,
                "fallback_tp_degrees": all_other_tps,
            }


        if load_balance_strategy == "round_robin":
            self._strategy = LoadBalanceStrategy.ROUND_ROBIN
        else:
            self._strategy = LoadBalanceStrategy.LEAST_CONNECTIONS
        self._max_queue_length = max_queue_length
        self._enable_fallback = enable_fallback
        self._rebuild_interval = max(0, rebuild_interval)
        self._max_recent_trajectories = max(1, max_recent_trajectories)
        self._tree_path = tree_path
        self._tree_persist_interval = max(1, tree_persist_interval)
        self._rank = int(os.environ.get("RANK", "0"))
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._current_epoch = -1
        self._rr_index: dict[int, int] = {}
        self._shared_load = (
            SharedCMLFQLoadState(
                directory=shared_load_dir,
                ttl_s=shared_load_ttl_s,
                heartbeat_interval_s=shared_load_heartbeat_s,
            )
            if shared_load_dir
            else None
        )
        # One state object serves both ranking paths: our own
        # ``_active_count`` reads and ``load_of``'s aggregate layer.
        if self._shared_load is not None:
            self.attach_shared_state(self._shared_load)


        self.prefix_tree = prefix_tree or CausalPrefixTree()
        if tool_state_extractor:
            self._tool_registry = ToolStateExtractorRegistry()
            self._tool_registry.register(tool_state_extractor)
        else:
            self._tool_registry = ToolStateExtractorRegistry.create_default_registry()


        self._request_states: dict[str, CMLFQRequestState] = {}
        # request_id -> (instance_index, charged_tokens). Our book of
        # record for in-flight token load: each entry is context +
        # E[residual] from the prefix tree, revised on every tool return
        # and relocated on migration. Bucket peers share a TP degree, so
        # ``load_of`` ranks them on the sum of these charges rather than
        # on raw request counts. No learned EMA by design -- the tree's
        # conditional histogram is the estimator.
        self._request_charges: dict[str, tuple[int, int]] = {}
        self._recent_trajectories: list[Trajectory] = []


        self._migration_count = 0
        self._tool_return_count = 0
        self._tree_update_count = 0

        logger.info(
            f"[CMLFQScheduler] initialized: buckets={self._sorted_bucket_names}, "
            f"thresholds={self._bucket_thresholds}, "
            f"shared_load_dir={shared_load_dir or 'disabled'}, tree_path={tree_path or 'disabled'}"
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def schedule(
        self,
        input_tokens: int,
        prompt_id: str = "",
        n_samples: int = 1,
        epoch: int = -1,
        max_new_tokens: int = 0,
    ) -> SchedulingResult:
        """Schedule."""
        with self._lock:
            self._stats.total_requests += 1

        request_id = uuid.uuid4().hex


        shortest_bucket = self._sorted_bucket_names[0] if self._sorted_bucket_names else "short"


        result = self._route_to_bucket(
            bucket=shortest_bucket,
            input_tokens=input_tokens,
            prompt_id=prompt_id,
            reason="cmlfq_initial_placement",
        )


        if result.instance_index >= 0:
            with self._lock:
                self._request_states[request_id] = CMLFQRequestState(
                    request_id=request_id,
                    prompt_id=prompt_id,
                    current_bucket=result.category,
                    current_instance_index=result.instance_index if result.instance_index >= 0 else -1,
                    input_tokens=max(0, int(input_tokens)),
                )
                # Initial token charge: prompt + expected residual from the
                # tree root (no tool history yet). Falls back to the shortest
                # bucket's midpoint prior when the tree has no observations.
                prior = self._expected_residual(
                    prompt_id, [], shortest_bucket
                )
                self._set_request_charge(
                    request_id, result.instance_index,
                    int(max(0, int(input_tokens)) + prior),
                )
                handle = self.get_instance_handle(result.instance_index)
                if handle is not None:
                    self._affinity_record(prompt_id, handle)
            result.request_id = request_id

        return result

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def on_tool_return(
        self,
        request_id: str,
        tool_result: Any,
        generated_tokens: int = 0,
    ) -> CMLFQMigrationDecision:
        """On tool return."""
        self._tool_return_count += 1

        req_state = self._request_states.get(request_id)
        if req_state is None:
            return CMLFQMigrationDecision(
                should_migrate=False,
                reason="request_not_found",
                current_bucket="unknown",
            )


        req_state.generated_tokens = generated_tokens


        tool_state = self._tool_registry.extract(tool_result)
        req_state.collected_return_states.append(tool_state)
        req_state.tool_return_token_positions.append(generated_tokens)

        # Refresh this request's token reservation on its current instance:
        # the tool history just changed, so the residual expectation (and
        # therefore its weight for in-bucket ranking) changed with it.
        with self._lock:
            self._set_request_charge(
                request_id,
                req_state.current_instance_index,
                int(
                    req_state.input_tokens
                    + req_state.generated_tokens
                    + self._expected_residual(
                        req_state.prompt_id,
                        req_state.collected_return_states,
                        req_state.current_bucket,
                    )
                ),
            )

        node = self.prefix_tree.lookup_with_fallback(
            req_state.prompt_id,
            req_state.collected_return_states,
        )

        if node is None or node.visit_count == 0:
            return CMLFQMigrationDecision(
                should_migrate=False,
                reason="tree_node_not_found",
                current_bucket=req_state.current_bucket,
            )


        target_bucket = self.prefix_tree.get_bucket_for_node(
            node, self._bucket_thresholds
        )

        if target_bucket is None:
            return CMLFQMigrationDecision(
                should_migrate=False,
                reason="mean_p90_disagree",
                current_bucket=req_state.current_bucket,
                mean_remaining_length=node.mean_remaining_length,
                p90_remaining_length=node.p90_remaining_length,
            )


        if target_bucket == req_state.current_bucket:
            return CMLFQMigrationDecision(
                should_migrate=False,
                reason="already_in_target_bucket",
                current_bucket=req_state.current_bucket,
                target_bucket=target_bucket,
                mean_remaining_length=node.mean_remaining_length,
                p90_remaining_length=node.p90_remaining_length,
            )


        logger.info(
            f"[CMLFQ] migration decision: request={request_id}, "
            f"{req_state.current_bucket} -> {target_bucket}, "
            f"mean={node.mean_remaining_length:.0f}, p90={node.p90_remaining_length:.0f}, "
            f"tool={tool_state.to_key()}"
        )

        return CMLFQMigrationDecision(
            should_migrate=True,
            reason="causal_signal_agree",
            current_bucket=req_state.current_bucket,
            target_bucket=target_bucket,
            mean_remaining_length=node.mean_remaining_length,
            p90_remaining_length=node.p90_remaining_length,
        )

    def execute_migration(
        self,
        request_id: str,
        decision: CMLFQMigrationDecision,
    ) -> SchedulingResult:
        """Execute migration."""
        if not decision.should_migrate:
            raise ValueError("Cannot execute migration: decision.should_migrate is False")

        req_state = self._request_states.get(request_id)
        if req_state is None:
            raise ValueError(f"Request {request_id} not found")

        old_idx = req_state.current_instance_index


        result = self._route_to_bucket(
            bucket=decision.target_bucket,
            input_tokens=0,
            prompt_id=req_state.prompt_id,
            reason="cmlfq_migration",
            allow_affinity=False,
        )

        if result.instance_index >= 0:
            old_handle = self.get_instance_handle(old_idx)
            if old_handle:
                with self._lock:
                    # Count-only: the charge is relocated to the
                    # destination handle below, not released.
                    self._decrement_active(old_handle)
            req_state.current_bucket = result.category
            req_state.current_instance_index = result.instance_index
            moved = result.instance_index != old_idx
            req_state.has_migrated = req_state.has_migrated or moved
            # Re-charge on the destination with the freshest residual.
            with self._lock:
                self._set_request_charge(
                    request_id,
                    result.instance_index,
                    int(
                        req_state.input_tokens
                        + req_state.generated_tokens
                        + self._expected_residual(
                            req_state.prompt_id,
                            req_state.collected_return_states,
                            result.category,
                        )
                    ),
                )
            result.request_id = request_id
            self._migration_count += int(moved)
            with self._lock:
                self._stats.migrated_routes += int(moved)

        return result

    def has_request(self, request_id: str) -> bool:
        """Return whether a C-MLFQ request is still active."""
        return request_id in self._request_states

    def get_request_route(self, request_id: str) -> SchedulingResult | None:
        """Get request route."""
        req_state = self._request_states.get(request_id)
        if req_state is None:
            return None
        handle = self.get_instance_handle(req_state.current_instance_index)
        if handle is None:
            return None
        return SchedulingResult(
            instance_index=handle.index,
            tp_degree=handle.tp_degree,
            category=req_state.current_bucket,
            is_fallback=False,
            reason="cmlfq_resume",
            prompt_id=req_state.prompt_id,
            request_id=request_id,
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def on_request_done(
        self,
        instance_index: int,
        prompt_id: str = "",
        final_bucket: str = "",
        output_tokens: int = 0,
    ):
        """On request done."""
        if not prompt_id:
            handle = self.get_instance_handle(instance_index)
            if handle:
                with self._lock:
                    self._decrement_active(handle)
            return

        request_id = next(
            (
                state.request_id
                for state in self._request_states.values()
                if (
                    state.prompt_id == prompt_id
                    and state.current_instance_index == instance_index
                )
            ),
            "",
        )
        if request_id:
            self.finish_request(request_id, output_tokens)
            return

        handle = self.get_instance_handle(instance_index)
        if handle:
            with self._lock:
                self._decrement_active(handle)

    def finish_request(self, request_id: str, total_output_tokens: int):
        """Finish request."""
        req_state = self._request_states.pop(request_id, None)
        if req_state is None:
            return

        with self._lock:
            charge = self._request_charges.pop(request_id, None)
            handle = self.get_instance_handle(req_state.current_instance_index)
            if handle:
                if charge is not None and charge[0] == req_state.current_instance_index:
                    self._decrement_active(handle, charge[1])
                else:
                    # The charge sits on a different handle than the
                    # request's current instance (a migration landed
                    # between the two bookkeeping updates): release it
                    # where it actually is.
                    if charge is not None:
                        stale = self.get_instance_handle(charge[0])
                        if stale is not None:
                            self._discharge_tokens_only(stale, charge[1])
                    self._decrement_active(handle)

        req_state.total_output_tokens = max(0, total_output_tokens)
        if not req_state.collected_return_states:
            return

        remaining_lengths = [
            max(0, total_output_tokens - position)
            for position in req_state.tool_return_token_positions
        ]
        trajectory = Trajectory(
            prompt_id=req_state.prompt_id,
            return_states=list(req_state.collected_return_states),
            total_remaining_lengths=remaining_lengths,
            total_length=total_output_tokens,
        )
        self.prefix_tree.insert(trajectory)
        self._recent_trajectories.append(trajectory)
        if len(self._recent_trajectories) > self._max_recent_trajectories:
            self._recent_trajectories = self._recent_trajectories[
                -self._max_recent_trajectories:
            ]
        self._tree_update_count += 1
        if self._tree_update_count % self._tree_persist_interval == 0:
            self._persist_prefix_tree()

    def cancel_request(self, request_id: str):
        """Cancel request."""
        req_state = self._request_states.pop(request_id, None)
        if req_state is None:
            return
        handle = self.get_instance_handle(req_state.current_instance_index)
        with self._lock:
            charge = self._request_charges.pop(request_id, None)
            if handle is not None:
                if charge is not None and charge[0] == req_state.current_instance_index:
                    self._decrement_active(handle, charge[1])
                else:
                    if charge is not None:
                        stale = self.get_instance_handle(charge[0])
                        if stale is not None:
                            self._discharge_tokens_only(stale, charge[1])
                    self._decrement_active(handle)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def update_tree_from_trajectories(self, trajectories: list[Trajectory]):
        """Update tree from trajectories."""
        for traj in trajectories:
            self.prefix_tree.insert(traj)
        self._tree_update_count += len(trajectories)
        logger.info(
            f"[CMLFQ] updated the prefix tree in a batch: {len(trajectories)} trajectories, "
            f"tree statistics={self.prefix_tree.get_stats()}"
        )

    def rebuild_tree(self, trajectories: list[Trajectory]):
        """Rebuild tree."""
        self.prefix_tree.rebuild(trajectories)
        logger.info(
            f"[CMLFQ] prefix-tree rebuild complete: tree statistics={self.prefix_tree.get_stats()}"
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _route_to_bucket(
        self,
        bucket: str,
        input_tokens: int,
        prompt_id: str = "",
        reason: str = "",
        allow_affinity: bool = True,
    ) -> SchedulingResult:
        """Route to bucket."""
        rule = self._bucket_rules.get(bucket)
        if rule is None:

            bucket = self._sorted_bucket_names[0] if self._sorted_bucket_names else "short"
            rule = self._bucket_rules.get(bucket)

        global_counts = self._get_global_active_counts()
        with self._lock:
            self._stats.category_counts[bucket] += 1

            # Prefix affinity, inside the chosen bucket only: RL rollouts
            # replay the same prompt (n_samples, shared system prompts,
            # retried trajectories), so a warm prefix cache on the sticky
            # peer beats a marginally emptier cold one. Never applied to a
            # migration -- that route exists precisely to leave the
            # current instance, and stickiness would cancel it.
            if allow_affinity and self._prefix_affinity_enabled:
                affinity_candidates = self._selectable(
                    [
                        h for h in self._instances
                        if h.is_ready
                        and h.tp_degree in rule["preferred_tp_degrees"]
                        and (
                            self._max_queue_length <= 0
                            or self._active_count(h, global_counts)
                            < self._max_queue_length
                        )
                    ]
                )
                selected = self._affinity_pick(prompt_id, affinity_candidates)
                if selected is not None:
                    self._stats.preferred_routes += 1
                    self._stats.category_tp_counts[bucket][selected.tp_degree] += 1
                    self._increment_active(selected)
                    return SchedulingResult(
                        instance_index=selected.index,
                        tp_degree=selected.tp_degree,
                        category=bucket,
                        is_fallback=False,
                        reason="prefix_affinity",
                        prompt_id=prompt_id,
                        request_id="",
                    )

            selected = self._try_select(
                rule["preferred_tp_degrees"], global_counts
            )
            if selected is not None:
                self._stats.preferred_routes += 1
                self._stats.category_tp_counts[bucket][selected.tp_degree] += 1
                self._increment_active(selected)
                return SchedulingResult(
                    instance_index=selected.index,
                    tp_degree=selected.tp_degree,
                    category=bucket,
                    is_fallback=False,
                    reason=reason,
                    prompt_id=prompt_id,
                    request_id="",
                )


            if self._enable_fallback and rule["fallback_tp_degrees"]:
                selected = self._try_select(
                    rule["fallback_tp_degrees"], global_counts
                )
                if selected is not None:
                    self._stats.fallback_routes += 1
                    self._stats.category_tp_counts[bucket][selected.tp_degree] += 1
                    self._increment_active(selected)
                    return SchedulingResult(
                        instance_index=selected.index,
                        tp_degree=selected.tp_degree,
                        category=self.bucket_for_tp(selected.tp_degree, bucket),
                        is_fallback=True,
                        reason=f"{reason}_fallback",
                        prompt_id=prompt_id,
                        request_id="",
                    )


            all_ready = self._selectable(
                [h for h in self._instances if h.is_ready]
            )
            if all_ready:
                selected = self._min_load_rotating(all_ready)
                self._stats.fallback_routes += 1
                self._stats.category_tp_counts[bucket][selected.tp_degree] += 1
                self._increment_active(selected)
                return SchedulingResult(
                    instance_index=selected.index,
                    tp_degree=selected.tp_degree,
                    category=self.bucket_for_tp(selected.tp_degree, bucket),
                    is_fallback=True,
                    reason="global_fallback",
                    prompt_id=prompt_id,
                    request_id="",
                )


            self._stats.failed_routes += 1
            return SchedulingResult(
                instance_index=-1,
                tp_degree=0,
                category=bucket,
                is_fallback=False,
                reason="no_available_instance",
                prompt_id=prompt_id,
                request_id="",
            )

    def bucket_for_tp(self, tp_degree: int, preferred: str = "") -> str:
        """Report the bucket of the actual instance, including on fallback."""
        matches = [
            name for name in self._sorted_bucket_names
            if tp_degree in self._buckets[name].get("tp_degrees", [])
        ]
        if preferred in matches:
            return preferred
        return matches[0] if matches else "unmapped"

    def _try_select(
        self,
        tp_preferences: list[int],
        global_counts: dict[str, dict[str, int]] | None = None,
    ) -> Optional[InstanceHandle]:
        """Pick an instance inside a bucket.

        Bucket peers share a TP degree, so this is where load actually
        decides placement. The filter order matters: comparability under
        the metric first (``_selectable`` also applies occupancy admission
        with hysteresis and the preemption penalty), then the configured
        concurrency cap, then the ranking.
        """
        for tp_degree in tp_preferences:
            candidates = [
                h for h in self._instances_by_tp.get(tp_degree, [])
                if h.is_ready
            ]
            if not candidates:
                continue
            candidates = self._selectable(candidates)
            if self._max_queue_length > 0:
                candidates = [
                    h for h in candidates
                    if self._active_count(h, global_counts) < self._max_queue_length
                ]
            if not candidates:
                continue
            if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
                idx = self._rr_index.get(tp_degree, 0) % len(candidates)
                self._rr_index[tp_degree] = idx + 1
                return candidates[idx]
            # ``load_of`` ranks on the tree-residual charge, corrected by
            # the cross-rank aggregate and the /metrics bias; rotation
            # keeps equal-load peers from starving when requests complete
            # between routing calls (low-concurrency multi-turn rollouts).
            return self._min_load_rotating(candidates)
        return None

    def _get_global_active_counts(self) -> dict[str, dict[str, int]] | None:
        if self._shared_load is None:
            return None
        return self._shared_load.aggregate()

    @staticmethod
    def _active_count(
        handle: InstanceHandle,
        global_counts: dict[str, dict[str, int]] | None,
    ) -> int:
        if global_counts is None:
            return handle.active_requests
        entry = global_counts.get(handle.instance_id)
        if entry is None:
            return handle.active_requests
        if isinstance(entry, dict):
            return int(entry.get("requests", 0))
        return int(entry)  # legacy integer snapshot

    def _increment_active(self, handle: InstanceHandle, delta_tokens: int = 0) -> None:
        """Count a request, optionally with its token charge.

        Deliberately not ``inc_active``: that installs a fixed FIFO
        reservation, while our charges are revised in place as the
        residual estimate changes. ``_request_charges`` is our book of
        record; the handle only carries the running total.
        """
        handle.active_requests += 1
        handle.charge_tokens(delta_tokens)
        if self._shared_load is not None:
            self._shared_load.adjust(
                handle.instance_id, 1, max(0, int(delta_tokens))
            )

    def _decrement_active(self, handle: InstanceHandle, delta_tokens: int = 0) -> None:
        """Retire a request, and with it ``delta_tokens`` of its charge.

        Defaults to retiring the request count only, for the paths where
        the charge is relocated rather than released (a migration moves
        it to the destination handle).
        """
        handle.active_requests = max(0, handle.active_requests - 1)
        retired = handle.discharge_tokens(delta_tokens)
        if retired != max(0, int(delta_tokens)):
            logger.warning(
                "[CMLFQ] token charge mismatch on %s: tried to retire %d, "
                "retired %d (handle drained by a reconfigure?)",
                handle.instance_id, int(delta_tokens), retired,
            )
        if self._shared_load is not None:
            self._shared_load.adjust(handle.instance_id, -1, -retired)

    def _discharge_tokens_only(self, handle: InstanceHandle, tokens: int) -> None:
        """Retire tokens without touching the request count."""
        retired = handle.discharge_tokens(int(tokens))
        if self._shared_load is not None and retired:
            self._shared_load.adjust(handle.instance_id, 0, -retired)

    def _expected_residual(
        self,
        prompt_id: str,
        return_states: list[ToolReturnState],
        fallback_bucket: str,
    ) -> float:
        """Expected remaining length from the tree; bucket-midpoint prior.

        No learned EMA by design: within a bucket lengths are bounded and
        migrations cap the tail, so the tree's conditional histogram (or a
        cheap midpoint prior) is all the routing signal we need.
        """
        try:
            distribution = self.prefix_tree.residual_distribution(
                prompt_id, return_states, self._bucket_thresholds
            )
        except Exception:
            distribution = []
        if distribution:
            return sum(
                probability * remaining
                for _, probability, remaining in distribution
            )
        return float(self._bucket_thresholds.get(fallback_bucket, 5000)) / 2.0

    def _set_request_charge(self, request_id: str, instance_index: int, tokens: int) -> None:
        """Move a request's token reservation to (instance_index, tokens).

        Callers must hold ``self._lock``.
        """
        tokens = max(0, int(tokens))
        previous = self._request_charges.pop(request_id, None)
        if previous is not None and previous[1]:
            old_handle = self.get_instance_handle(previous[0])
            if old_handle is not None:
                self._discharge_tokens_only(old_handle, previous[1])
        handle = self.get_instance_handle(instance_index)
        if handle is not None and tokens:
            handle.charge_tokens(tokens)
            if self._shared_load is not None:
                self._shared_load.adjust(handle.instance_id, 0, tokens)
        if tokens:
            self._request_charges[request_id] = (instance_index, tokens)

    def _rank_tree_path(self, epoch: int | None = None) -> str:
        path = Path(self._tree_path)
        suffix = path.suffix or ".json"
        stem = path.name[:-len(path.suffix)] if path.suffix else path.name
        epoch_suffix = f".step_{epoch}" if epoch is not None else ""
        return str(path.with_name(f"{stem}.rank_{self._rank}{epoch_suffix}{suffix}"))

    def _persist_prefix_tree(self, epoch: int | None = None) -> None:
        if not self._tree_path:
            return
        self.prefix_tree.save(self._rank_tree_path())
        if epoch is not None:
            self.prefix_tree.save(self._rank_tree_path(epoch))

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def on_epoch_start(self, epoch: int):
        """On epoch start."""
        self._current_epoch = epoch
        logger.info(
            f"[CMLFQ] Epoch {epoch} started, "
            f"active requests={len(self._request_states)}, "
            f"tree statistics={self.prefix_tree.get_stats()}"
        )

    def on_epoch_end(self, epoch: int):
        """On epoch end."""
        if (
            self._rebuild_interval > 0
            and epoch > 0
            and epoch % self._rebuild_interval == 0
            and self._recent_trajectories
        ):
            self.prefix_tree.rebuild(self._recent_trajectories)
        self._persist_prefix_tree(epoch=epoch)
        logger.info(
            f"[CMLFQ] Epoch {epoch} ended, "
            f"tool_returns={self._tool_return_count}, "
            f"migrations={self._migration_count}, "
            f"tree_updates={self._tree_update_count}"
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def print_stats(self):
        """Print stats."""
        super().print_stats()
        print(f"\n  [C-MLFQ extended statistics]")
        print(f"  tool returns: {self._tool_return_count}")
        print(f"  migrations: {self._migration_count}")
        print(f"  tree updates: {self._tree_update_count}")
        print(f"  active requests: {len(self._request_states)}")
        tree_stats = self.prefix_tree.get_stats()
        print(f"  prefix tree: prompts={tree_stats['n_prompts']}, "
              f"nodes={tree_stats['total_nodes']}, "
              f"max_depth={tree_stats['max_depth']}")

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, hetero_config: Any) -> "CMLFQScheduler":
        """From config."""
        sched = hetero_config.scheduling

        buckets = getattr(sched, "cmlfq_buckets", None) or cls.DEFAULT_BUCKETS
        bucket_thresholds = {
            name: cfg.get("max_tokens", 5000)
            for name, cfg in buckets.items()
        }

        prefix_tree = CausalPrefixTree()
        tree_path = getattr(sched, "cmlfq_tree_path", "")
        if tree_path:
            path = Path(tree_path)
            suffix = path.suffix or ".json"
            stem = path.name[:-len(path.suffix)] if path.suffix else path.name
            rank = int(os.environ.get("RANK", "0"))
            rank_tree_path = path.with_name(f"{stem}.rank_{rank}{suffix}")
            try:
                prefix_tree.load(str(rank_tree_path))
            except FileNotFoundError:
                try:
                    prefix_tree.load(tree_path)
                except FileNotFoundError:
                    logger.warning(
                        "[CMLFQ] prefix-tree files do not exist: %s, %s",
                        rank_tree_path,
                        tree_path,
                    )

        tool_state_extractor = DefaultToolStateExtractor(
            small_threshold=getattr(
                sched, "cmlfq_payload_small_threshold", 500
            ),
            large_threshold=getattr(
                sched, "cmlfq_payload_large_threshold", 5000
            ),
        )

        return cls(
            buckets=buckets,
            bucket_thresholds=bucket_thresholds,
            tool_state_extractor=tool_state_extractor,
            prefix_tree=prefix_tree,
            load_balance_strategy=sched.load_balance_strategy,
            max_queue_length=sched.max_queue_length,
            enable_fallback=sched.enable_fallback,
            rebuild_interval=getattr(
                sched, "cmlfq_rebuild_interval", 50
            ),
            tree_path=tree_path,
            tree_persist_interval=getattr(
                sched, "cmlfq_tree_persist_interval", 1
            ),
            shared_load_dir=getattr(sched, "cmlfq_shared_load_dir", ""),
            shared_load_ttl_s=getattr(
                sched, "cmlfq_shared_load_ttl_s", 60.0
            ),
            shared_load_heartbeat_s=getattr(
                sched, "cmlfq_shared_load_heartbeat_s", 10.0
            ),
            # In-bucket ranking: "tokens" keeps the prefix-tree residual
            # charge as the load figure; "kv_tokens" additionally
            # normalizes by each bucket's KV capacity (needs a capacity
            # table, auto-estimated at config load).
            load_metric=getattr(sched, "cmlfq_load_metric", "tokens"),
            kv_capacity_tokens_by_tp=getattr(
                sched, "kv_capacity_tokens_by_tp", None
            ),
            feedback=MetricsFeedbackConfig.from_scheduling(sched),
            prefix_affinity=bool(getattr(sched, "prefix_affinity", False)),
            prefix_affinity_max_entries=int(
                getattr(sched, "prefix_affinity_max_entries", 8192)
            ),
        )
