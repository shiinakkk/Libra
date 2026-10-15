"""Token accounting for C-MLFQ in-bucket selection (tree-residual billing).

Bucket peers share a TP degree, so selection inside a bucket now ranks by
estimated in-flight tokens: each request is charged ``context + E[residual]``
from the prefix tree (bucket-midpoint prior when the tree is empty — no
learned EMA), refreshed on tool returns and migrations, and retired by exact
value. The cross-rank protocol carries {requests, tokens}.
"""

import json
import tempfile
import time
import unittest

from RL_Framework.infra.scheduling.base import MetricsFeedbackConfig
from RL_Framework.infra.scheduling.cmlfq_cost import CMLFQRoutingCosts
from RL_Framework.infra.scheduling.cmlfq_cost_scheduler import CMLFQCostScheduler
from RL_Framework.infra.scheduling.cmlfq_prefix_tree import Trajectory
from RL_Framework.infra.scheduling.cmlfq_scheduler import (
    CMLFQMigrationDecision,
    CMLFQScheduler,
)
from RL_Framework.infra.scheduling.cmlfq_shared_state import SharedCMLFQLoadState
from RL_Framework.infra.scheduling.cmlfq_tool_state import DefaultToolStateExtractor
from RL_Framework.infra.scheduling.metrics_feed import InstanceMetrics

STATE = DefaultToolStateExtractor().extract(
    {"tool_type": "search", "output": "ok", "payload_tokens": 2}
)
EVENT = {"tool_type": "search", "output": "ok", "payload_tokens": 2}

BUCKETS = {
    "short": {"tp_degrees": [1], "max_tokens": 5000},
    "long": {"tp_degrees": [4], "max_tokens": 50000},
}
TOPOLOGY = [
    {"instance_id": "s", "host": "host-a", "tp_degree": 1},
    {"instance_id": "l", "host": "host-b", "tp_degree": 4},
]


def two_peer_scheduler(**kwargs):
    scheduler = CMLFQScheduler(
        buckets={"short": {"tp_degrees": [1], "max_tokens": 5000},
                 "long": {"tp_degrees": [4], "max_tokens": 50000}},
        **kwargs,
    )
    scheduler.register_instance(0, "short_a", 1)
    scheduler.register_instance(1, "short_b", 1)
    scheduler.register_instance(2, "long_a", 4)
    return scheduler


def cost_scheduler(lengths=(10000,)):
    costs = CMLFQRoutingCosts()
    costs.decode_ms = {"1": {"short": 2000, "long": 20000},
                       "4": {"short": 4000, "long": 5000}}
    costs.migrations = [{
        "source_tp": 1, "target_tp": 4, "source_host": "host-a",
        "target_host": "host-b", "seq_len": 1000,
        "transfer_ms": 100, "recompute_ms": 1000,
    }]
    scheduler = CMLFQCostScheduler(buckets=BUCKETS, routing_costs=costs)
    scheduler.register_instance(0, "s", 1)
    scheduler.register_instance(1, "l", 4)
    scheduler.configure_runtime(TOPOLOGY)
    for length in lengths:
        scheduler.prefix_tree.insert(Trajectory("p", [STATE], [length], length + 1000))
    return scheduler


class TestCharging(unittest.TestCase):
    def test_initial_charge_uses_tree_root_and_midpoint_fallback(self):
        scheduler = two_peer_scheduler()
        # Empty tree: midpoint prior of the shortest bucket (5000/2).
        scheduler.schedule(100, prompt_id="cold")
        self.assertEqual(scheduler.get_instance_handle(0).active_tokens, 100 + 2500)

        # A bootstrapped tree: root observations are the total lengths.
        warm = two_peer_scheduler()
        warm.prefix_tree.insert(Trajectory("p", [STATE], [1000], 3000))
        warm.schedule(100, prompt_id="p")
        self.assertEqual(warm.get_instance_handle(0).active_tokens, 100 + 3000)

    def test_bucket_peers_rank_by_tokens_not_request_count(self):
        scheduler = two_peer_scheduler()
        heavy = scheduler.schedule(4000, prompt_id="heavy")
        self.assertEqual(heavy.instance_index, 0)

        # Three light arrivals must pile onto the other peer (2510 each,
        # far below the heavy peer's 6500). Legacy request-count ranking
        # would have ping-ponged the third one back to instance 0.
        lights = [scheduler.schedule(10, prompt_id=f"light{i}") for i in range(3)]
        self.assertEqual([r.instance_index for r in lights], [1, 1, 1])

        scheduler.finish_request(heavy.request_id, 0)
        for light in lights:
            scheduler.finish_request(light.request_id, 0)
        for handle in scheduler._instances:
            self.assertEqual((handle.active_requests, handle.active_tokens), (0, 0))
        self.assertEqual(scheduler._request_charges, {})

    def test_tool_return_refreshes_charge_in_place(self):
        scheduler = two_peer_scheduler()
        scheduler.prefix_tree.insert(Trajectory("p", [STATE], [1000], 3000))
        request = scheduler.schedule(100, prompt_id="p")
        handle = scheduler.get_instance_handle(request.instance_index)
        before = handle.active_tokens
        self.assertEqual(before, 100 + 3000)

        scheduler.on_tool_return(request.request_id, EVENT, generated_tokens=900)
        # 100 input + 900 generated + E[residual | state] (=1000).
        self.assertEqual(handle.active_tokens, 100 + 900 + 1000)
        self.assertEqual(handle.active_requests, 1)

    def test_legacy_execute_migration_moves_charge(self):
        scheduler = two_peer_scheduler()
        # Node residual (10000) lands in the long bucket -> migration.
        scheduler.prefix_tree.insert(Trajectory("p", [STATE], [10000], 11000))
        request = scheduler.schedule(100, prompt_id="p")
        scheduler.on_tool_return(request.request_id, EVENT, generated_tokens=900)

        decision = scheduler.on_tool_return(request.request_id, EVENT, generated_tokens=2000)
        self.assertTrue(decision.should_migrate)
        result = scheduler.execute_migration(request.request_id, decision)
        self.assertEqual(result.instance_index, 2)  # long_a
        source = scheduler.get_instance_handle(request.instance_index)
        target = scheduler.get_instance_handle(result.instance_index)
        self.assertEqual(source.active_tokens, 0)
        self.assertEqual(source.active_requests, 0)
        # 100 input + 2000 generated + 10000 residual on the target.
        self.assertEqual(target.active_tokens, 100 + 2000 + 10000)
        self.assertEqual(scheduler._request_charges[request.request_id],
                         (result.instance_index, 100 + 2000 + 10000))

        scheduler.finish_request(request.request_id, 0)
        for handle in scheduler._instances:
            self.assertEqual((handle.active_requests, handle.active_tokens), (0, 0))


class TestCostReservation(unittest.TestCase):
    def test_commit_rollback_and_finish_conserve_tokens(self):
        scheduler = cost_scheduler()
        request = scheduler.schedule(100, "p").request_id
        self.assertTrue(scheduler.on_tool_return(request, EVENT, 900).should_migrate)

        decision = scheduler.reserve_generation(request)
        # Provisional target charge mirrors the request-count double-count.
        self.assertEqual([h.active_requests for h in scheduler._instances], [1, 1])
        self.assertEqual(scheduler.get_instance_handle(1).active_tokens, 11000)
        # Committed charge on the source: 100 input + E[root]=11000.
        self.assertEqual(scheduler._request_charges[request], (0, 11100))

        scheduler.rollback_generation(request)
        self.assertEqual([h.active_requests for h in scheduler._instances], [1, 0])
        self.assertEqual(scheduler.get_instance_handle(1).active_tokens, 0)
        self.assertEqual(scheduler.get_instance_handle(0).active_tokens, 11100)

        decision = scheduler.reserve_generation(request)
        scheduler.commit_generation(request, decision)
        self.assertEqual([h.active_requests for h in scheduler._instances], [0, 1])
        self.assertEqual(scheduler.get_instance_handle(0).active_tokens, 0)
        self.assertEqual(scheduler._request_charges[request], (1, 11000))

        scheduler.finish_request(request, 0)
        self.assertEqual(scheduler.get_instance_handle(1).active_tokens, 0)
        self.assertEqual(scheduler._request_charges, {})

    def test_resume_refresh_updates_charge_and_rollback_restores(self):
        # Short-bin residual keeps the trajectory on its TP1 instance, so
        # reserve goes through the stay-refresh path instead of migrating.
        scheduler = cost_scheduler(lengths=(1000,))
        request = scheduler.schedule(100, "p").request_id
        initial = scheduler.get_instance_handle(0).active_tokens  # 2100
        self.assertFalse(scheduler.on_tool_return(request, EVENT, 900).should_migrate)

        decision = scheduler.reserve_generation(request)
        self.assertFalse(decision.should_migrate)
        self.assertEqual(decision.previous_charge, initial)
        # Refreshed: context 1000 + E[residual | state] = 1000.
        self.assertEqual(scheduler.get_instance_handle(0).active_tokens, 2000)
        self.assertEqual(scheduler._request_charges[request], (0, 2000))

        scheduler.rollback_generation(request)
        self.assertEqual(scheduler.get_instance_handle(0).active_tokens, initial)
        self.assertEqual(scheduler._request_charges[request], (0, initial))

    def test_cost_tie_break_prefers_lighter_token_peer(self):
        # Short-bin dominant: the two TP1 peers beat TP4 on decode cost,
        # leaving an exact cost tie between them for tokens to break.
        scheduler = cost_scheduler(lengths=(1000,))
        # Two extra TP1 peers on the same host as the source.
        scheduler.register_instance(2, "s2", 1)
        scheduler.register_instance(3, "s3", 1)
        topology = list(TOPOLOGY) + [
            {"instance_id": "s2", "host": "host-a", "tp_degree": 1},
            {"instance_id": "s3", "host": "host-a", "tp_degree": 1},
        ]
        scheduler.configure_runtime(topology)
        # Same-TP peer moves are only routable when profiled.
        scheduler.routing_costs.migrations.append({
            "source_tp": 1, "target_tp": 1, "source_host": "host-a",
            "target_host": "host-a", "seq_len": 1000,
            "transfer_ms": 100, "recompute_ms": 1000,
        })
        # s2 carries another rank's weight; s3 is clean.
        scheduler.get_instance_handle(2).charge_tokens(500)

        request = scheduler.schedule(100, "p").request_id
        scheduler.on_tool_return(request, EVENT, 900)
        # Take the source (stay) option off the table so the choice is
        # purely between the two cost-identical TP1 peers.
        scheduler.get_instance_handle(0).is_ready = False
        decision = scheduler._choose(request)
        self.assertEqual(decision.should_migrate, True)
        self.assertEqual(decision.target_instance_index, 3)


class TestInBucketSelection(unittest.TestCase):
    """Bucket peers share a TP degree, so load decides placement here."""

    def test_capacity_normalization_can_outrank_raw_tokens(self):
        scheduler = two_peer_scheduler(
            load_metric="kv_tokens",
            kv_capacity_tokens_by_tp={1: 10000, 4: 40000},
            feedback=MetricsFeedbackConfig(enabled=True),
        )
        a, b = (scheduler.get_instance_handle(i) for i in (0, 1))
        # short_a carries twice the token charge of short_b...
        a.charge_tokens(5200)
        b.charge_tokens(2600)

        # ...but vLLM reports a far larger KV cache on it, so the
        # occupancy ratio is lower and it is the better target.
        class Feed:
            def get(self, iid):
                return InstanceMetrics(
                    iid, time.time(),
                    kv_capacity_tokens=1_000_000 if iid == "short_a" else 10_000,
                )

        scheduler.attach_metrics_feed(Feed())
        self.assertLess(scheduler.load_of(a), scheduler.load_of(b))
        self.assertEqual(scheduler.schedule(100, prompt_id="c").instance_index, 0)

    def test_equal_load_peers_rotate_instead_of_starving(self):
        """A plain min() always re-picked the first registered peer."""
        scheduler = two_peer_scheduler()
        chosen = []
        for _ in range(4):
            result = scheduler.schedule(100, prompt_id="p")
            chosen.append(result.instance_index)
            scheduler.finish_request(result.request_id, 0)
        self.assertEqual({0, 1}, set(chosen), f"starved a peer: {chosen}")

    def test_admission_skips_over_occupied_peer(self):
        scheduler = two_peer_scheduler(
            feedback=MetricsFeedbackConfig(enabled=True),
        )
        # short_b holds charge, so on load alone short_a would win.
        scheduler.get_instance_handle(1).charge_tokens(5000)

        class Feed:
            def get(self, iid):
                return InstanceMetrics(
                    iid, time.time(),
                    gpu_cache_usage=0.99 if iid == "short_a" else 0.1,
                )

        scheduler.attach_metrics_feed(Feed())
        self.assertEqual(scheduler.schedule(100, prompt_id="p").instance_index, 1)

    def test_recent_preemption_steers_away_from_peer(self):
        scheduler = two_peer_scheduler(
            feedback=MetricsFeedbackConfig(enabled=True),
        )
        scheduler.get_instance_handle(1).charge_tokens(5000)

        class Feed:
            def get(self, iid):
                return InstanceMetrics(
                    iid, time.time(), gpu_cache_usage=0.1,
                    last_preemption_at=(
                        time.time() if iid == "short_a" else 0.0
                    ),
                )

        scheduler.attach_metrics_feed(Feed())
        self.assertEqual(scheduler.schedule(100, prompt_id="p").instance_index, 1)

    def test_prefix_affinity_keeps_a_prompt_on_its_peer(self):
        scheduler = two_peer_scheduler(prefix_affinity=True)
        first = scheduler.schedule(100, prompt_id="shared")
        # Same prompt replayed (n_samples / retry): a warm prefix cache
        # beats the now-emptier cold peer.
        second = scheduler.schedule(100, prompt_id="shared")
        self.assertEqual(second.instance_index, first.instance_index)
        self.assertEqual(second.reason, "prefix_affinity")
        # A different prompt is free to balance away.
        other = scheduler.schedule(100, prompt_id="elsewhere")
        self.assertNotEqual(other.instance_index, first.instance_index)

    def test_affinity_is_off_without_the_flag(self):
        scheduler = two_peer_scheduler()
        first = scheduler.schedule(100, prompt_id="shared")
        second = scheduler.schedule(100, prompt_id="shared")
        self.assertNotEqual(second.instance_index, first.instance_index)

    def test_migration_is_not_pulled_back_by_affinity(self):
        """Same-bucket migration must leave the sticky peer.

        The affinity map points at the source instance, so without the
        ``allow_affinity=False`` gate the migration would re-select it
        and silently become a no-op.
        """
        scheduler = two_peer_scheduler(prefix_affinity=True)
        result = scheduler.schedule(100, prompt_id="shared")
        state = scheduler._request_states[result.request_id]
        source = scheduler.get_instance_handle(result.instance_index)
        self.assertEqual(scheduler._affinity["shared"], source.instance_id)
        decision = CMLFQMigrationDecision(
            should_migrate=True, reason="test",
            current_bucket=state.current_bucket, target_bucket="short",
        )
        moved = scheduler.execute_migration(result.request_id, decision)
        self.assertNotEqual(moved.instance_index, result.instance_index)
        self.assertNotEqual(moved.reason, "prefix_affinity")


class TestAccountingSubstrate(unittest.TestCase):
    """C-MLFQ owns its charge book; the FIFO reservation queue is base's."""

    def test_cmlfq_never_touches_the_fifo_reservation_book(self):
        scheduler = two_peer_scheduler()
        result = scheduler.schedule(100, prompt_id="p")
        scheduler.on_tool_return(result.request_id, EVENT, generated_tokens=900)
        decision = CMLFQMigrationDecision(
            should_migrate=True, reason="test",
            current_bucket="short", target_bucket="long",
        )
        scheduler.execute_migration(result.request_id, decision)
        for index in range(3):
            handle = scheduler.get_instance_handle(index)
            self.assertEqual(
                len(handle._token_estimates), 0,
                f"instance {index} leaked a FIFO reservation",
            )
        scheduler.finish_request(result.request_id, 0)
        for index in range(3):
            handle = scheduler.get_instance_handle(index)
            self.assertEqual(
                (handle.active_requests, handle.active_tokens), (0, 0)
            )

    def test_discharge_clamps_instead_of_going_negative(self):
        scheduler = two_peer_scheduler()
        handle = scheduler.get_instance_handle(0)
        handle.charge_tokens(500)
        self.assertEqual(handle.discharge_tokens(900), 500)
        self.assertEqual(handle.active_tokens, 0)
        self.assertEqual(handle.discharge_tokens(100), 0)
        self.assertEqual(handle.active_tokens, 0)

    def test_charges_are_revisable_in_place(self):
        """A tool return changes the residual, not the request identity."""
        scheduler = two_peer_scheduler()
        result = scheduler.schedule(100, prompt_id="p")
        handle = scheduler.get_instance_handle(result.instance_index)
        before = handle.active_tokens
        requests_before = handle.active_requests
        scheduler.on_tool_return(result.request_id, EVENT, generated_tokens=900)
        self.assertNotEqual(handle.active_tokens, before)
        self.assertEqual(handle.active_requests, requests_before)


class TestSharedProtocol(unittest.TestCase):
    def test_publishes_tokens_and_reads_legacy_int_files(self):
        import time
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            mine = SharedCMLFQLoadState(directory, writer_id="rank_0")
            # cache_ttl_s=0 so a peer file written mid-test is visible
            # immediately; the production default caches for ~1s because
            # load_of() reads this per candidate under the scheduler lock.
            remote = SharedCMLFQLoadState(
                directory, writer_id="rank_1", cache_ttl_s=0.0
            )
            try:
                mine.adjust("s", delta_requests=2, delta_tokens=6500)
                totals = remote.aggregate()
                self.assertEqual(totals["s"], {"requests": 2, "tokens": 6500})

                # A legacy integer snapshot carries no token data, so the
                # cluster token sum would be an undercount. Request counts
                # still aggregate, but the "tokens" key is dropped
                # entirely: readers must fall back to their local token
                # accounting rather than rank on a known-low number.
                legacy = {
                    "writer_id": "rank_9", "updated_at": time.time(),
                    "counts": {"s": 3},
                }
                (Path(directory) / "rank_9.json").write_text(
                    json.dumps(legacy), encoding="utf-8"
                )
                totals = remote.aggregate()
                self.assertEqual(totals["s"], {"requests": 5})
            finally:
                mine.close()
                remote.close()

    def test_aggregate_caches_peer_reads(self):
        """The default TTL keeps load_of() off the filesystem per candidate."""
        import time
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            reader = SharedCMLFQLoadState(directory, writer_id="rank_0")
            try:
                self.assertEqual(reader.aggregate(), {})
                (Path(directory) / "rank_7.json").write_text(
                    json.dumps({
                        "writer_id": "rank_7", "updated_at": time.time(),
                        "counts": {"s": {"requests": 1, "tokens": 900}},
                    }),
                    encoding="utf-8",
                )
                # Still the cached empty snapshot.
                self.assertEqual(reader.aggregate(), {})
                # Our own publish invalidates it, so the peer appears.
                reader.adjust("s", delta_requests=1, delta_tokens=100)
                self.assertEqual(
                    reader.aggregate()["s"], {"requests": 2, "tokens": 1000}
                )
            finally:
                reader.close()

    def test_scheduler_publishes_token_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            scheduler = two_peer_scheduler(shared_load_dir=directory)
            try:
                request = scheduler.schedule(100, prompt_id="p")
                entry = scheduler._shared_load.aggregate()["short_a"]
                self.assertEqual(entry["tokens"], 100 + 2500)
                scheduler.finish_request(request.request_id, 0)
                entry = scheduler._shared_load.aggregate()["short_a"]
                self.assertEqual(entry, {"requests": 0, "tokens": 0})
            finally:
                scheduler._shared_load.close()


if __name__ == "__main__":
    unittest.main()
