"""Cross-process load snapshots for C-MLFQ schedulers."""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path


class SharedCMLFQLoadState:
    """Publish per-rank counters and aggregate them from a shared filesystem.

    Each process is the only writer of its own snapshot. Atomic rename keeps
    readers from observing partial JSON, while heartbeats let readers discard
    counters left behind by a failed training process.

    The published counts carry both in-flight request counts and estimated
    in-flight tokens; readers fall back to request-only data (and their local
    token accounting) when a peer file predates the token extension.
    """

    def __init__(
        self,
        directory: str,
        ttl_s: float = 60.0,
        heartbeat_interval_s: float = 10.0,
        writer_id: str = "",
        cache_ttl_s: float = 1.0,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        self.writer_id = writer_id or f"rank_{rank}"
        self.ttl_s = max(heartbeat_interval_s * 2.0, ttl_s)
        self.heartbeat_interval_s = max(1.0, heartbeat_interval_s)
        self.cache_ttl_s = max(0.0, float(cache_ttl_s))
        self._path = self.directory / f"{self.writer_id}.json"
        self._counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cache: dict[str, dict[str, int]] | None = None
        self._cache_at: float = 0.0
        self._publish()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"cmlfq-heartbeat-{self.writer_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        atexit.register(self.close)

    def adjust(
        self,
        instance_id: str,
        delta_requests: int = 0,
        delta_tokens: int = 0,
    ) -> None:
        """Apply deltas to this rank's published accounting."""
        with self._lock:
            entry = self._counts.setdefault(
                instance_id, {"requests": 0, "tokens": 0}
            )
            entry["requests"] = max(0, entry["requests"] + int(delta_requests))
            entry["tokens"] = max(0, entry["tokens"] + int(delta_tokens))
            self._publish_locked()
            self._cache = None

    def increment(self, instance_id: str) -> None:
        self.adjust(instance_id, delta_requests=1)

    def decrement(self, instance_id: str) -> None:
        self.adjust(instance_id, delta_requests=-1)

    # ``BaseScheduler.attach_shared_state`` speaks add()/totals(); keep the
    # C-MLFQ names as the primary API and alias the base protocol onto them
    # so one state object can serve both ranking paths.
    def add(
        self,
        instance_id: str,
        delta_requests: int = 0,
        delta_tokens: int = 0,
    ) -> None:
        """Alias of :meth:`adjust` for the base-scheduler protocol."""
        self.adjust(instance_id, delta_requests, delta_tokens)

    def totals(self) -> dict[str, dict[str, int]]:
        """Alias of :meth:`aggregate` for the base-scheduler protocol."""
        return self.aggregate()

    def aggregate(self) -> dict[str, dict[str, int]]:
        """Cluster-wide sums over live rank files (cached).

        Returns ``{instance_id: {"requests": int, ["tokens": int]}}``. The
        ``tokens`` key is present only when EVERY live contributor for that
        instance published token data: a legacy peer (integer counts, or a
        dict without "tokens") makes the cluster token sum an undercount,
        so the key is omitted entirely and readers fall back to their own
        local token accounting instead of ranking on a too-small number.
        """
        now = time.monotonic()
        with self._lock:
            if (
                self._cache is not None
                and self.cache_ttl_s > 0.0
                and now - self._cache_at < min(self.cache_ttl_s, self.ttl_s)
            ):
                return {iid: dict(e) for iid, e in self._cache.items()}

        wall = time.time()
        totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "tokens": 0}
        )
        # instance_id -> every contributor so far published a token count.
        token_complete: dict[str, bool] = {}
        for path in self.directory.glob("rank_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                updated_at = float(payload.get("updated_at", 0.0))
                if wall - updated_at > self.ttl_s:
                    continue
                for instance_id, entry in payload.get("counts", {}).items():
                    iid = str(instance_id)
                    total = totals[iid]
                    token_complete.setdefault(iid, True)
                    if isinstance(entry, dict):
                        total["requests"] += max(0, int(entry.get("requests", 0)))
                        if "tokens" in entry:
                            total["tokens"] += max(0, int(entry["tokens"]))
                        else:
                            token_complete[iid] = False
                    else:
                        total["requests"] += max(0, int(entry))
                        token_complete[iid] = False
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

        result: dict[str, dict[str, int]] = {}
        for iid, entry in totals.items():
            published = {"requests": entry["requests"]}
            if token_complete.get(iid, False):
                published["tokens"] = entry["tokens"]
            result[iid] = published

        with self._lock:
            self._cache = result
            self._cache_at = time.monotonic()
        return {iid: dict(e) for iid, e in result.items()}

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        with self._lock:
            self._counts.clear()
            self._publish_locked()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_s):
            self._publish()

    def _publish(self) -> None:
        with self._lock:
            self._publish_locked()

    def _publish_locked(self) -> None:
        payload = {
            "writer_id": self.writer_id,
            "rank": int(os.environ.get("RANK", "0")),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "updated_at": time.time(),
            "counts": {
                iid: dict(entry) for iid, entry in self._counts.items()
            },
        }
        tmp_path = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, self._path)
