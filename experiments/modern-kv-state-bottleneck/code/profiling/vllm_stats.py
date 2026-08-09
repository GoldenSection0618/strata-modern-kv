"""vLLM internal statistics collection.

Captures SchedulerStats from the EngineCore output stream.  The V1 engine
emits SchedulerStats on every step when log_stats is enabled, but does not
expose the KV cache manager internals to the LLM client process (they live
inside the EngineCore subprocess).  We therefore patch
``engine_core.get_output()`` to record the latest stats without changing
engine behaviour.

Prefix cache stats and eviction events are per-step deltas (drained by the
engine), so we accumulate them in the client.  ``kv_cache_usage`` is a
snapshot value and is not accumulated.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KVStats:
    """Snapshot of vLLM internal cache statistics."""

    # KV cache usage (0.0 – 1.0), snapshot at last step
    usage: float
    # Prefix cache stats (cumulative since collector start / last reset)
    prefix_requests: int
    prefix_queries: int
    prefix_hits: int
    prefix_preempted_requests: int
    prefix_preempted_queries: int
    prefix_preempted_hits: int
    # Scheduler state (latest step)
    num_running_reqs: int
    num_waiting_reqs: int
    # Cumulative number of KV cache eviction events (offload tier)
    kv_eviction_events: int
    # Timestamp
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


class VLLMStatsCollector:
    """Collects SchedulerStats from the EngineCore output stream."""

    def __init__(self, llm):
        self.llm = llm
        self._scheduler_stats: Any = None  # latest SchedulerStats
        self._patched = False
        self._accum = {
            "prefix_requests": 0,
            "prefix_queries": 0,
            "prefix_hits": 0,
            "prefix_preempted_requests": 0,
            "prefix_preempted_queries": 0,
            "prefix_preempted_hits": 0,
            "kv_eviction_events": 0,
        }
        self._patch_get_output()

    def _patch_get_output(self):
        """Patch engine_core.get_output() to capture SchedulerStats."""
        engine = getattr(self.llm, "llm_engine", None)
        if engine is None:
            logger.warning("Cannot find llm_engine on LLM instance")
            return
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is None:
            logger.warning("Cannot find engine_core on llm_engine")
            return

        orig_get_output = engine_core.get_output
        collector = self

        def patched_get_output(*args, **kwargs):
            outputs = orig_get_output(*args, **kwargs)
            ss = getattr(outputs, "scheduler_stats", None)
            if ss is not None:
                collector._scheduler_stats = ss
                pcs = getattr(ss, "prefix_cache_stats", None)
                if pcs is not None:
                    acc = collector._accum
                    acc["prefix_requests"] += int(getattr(pcs, "requests", 0) or 0)
                    acc["prefix_queries"] += int(getattr(pcs, "queries", 0) or 0)
                    acc["prefix_hits"] += int(getattr(pcs, "hits", 0) or 0)
                    acc["prefix_preempted_requests"] += int(
                        getattr(pcs, "preempted_requests", 0) or 0
                    )
                    acc["prefix_preempted_queries"] += int(
                        getattr(pcs, "preempted_queries", 0) or 0
                    )
                    acc["prefix_preempted_hits"] += int(
                        getattr(pcs, "preempted_hits", 0) or 0
                    )
                events = getattr(ss, "kv_cache_eviction_events", None)
                if events:
                    collector._accum["kv_eviction_events"] += len(events)
            return outputs

        engine_core.get_output = patched_get_output
        self._patched = True
        logger.info("Patched engine_core.get_output to capture SchedulerStats")

    def collect(self) -> KVStats:
        """Collect a snapshot of current cache statistics."""
        stats = KVStats(
            usage=0.0,
            prefix_requests=0,
            prefix_queries=0,
            prefix_hits=0,
            prefix_preempted_requests=0,
            prefix_preempted_queries=0,
            prefix_preempted_hits=0,
            num_running_reqs=0,
            num_waiting_reqs=0,
            kv_eviction_events=0,
            timestamp=time.perf_counter(),
        )

        if not self._patched:
            logger.warning("Stats collector not patched; stats unavailable")
            return stats

        ss = self._scheduler_stats
        if ss is None:
            logger.warning(
                "No SchedulerStats captured yet. "
                "Ensure disable_log_stats=False."
            )
            return stats

        try:
            stats.usage = float(ss.kv_cache_usage)
            stats.num_running_reqs = int(ss.num_running_reqs)
            stats.num_waiting_reqs = int(ss.num_waiting_reqs)
        except Exception as e:
            logger.debug("Failed to parse scheduler fields: %s", e)

        stats.prefix_requests = self._accum["prefix_requests"]
        stats.prefix_queries = self._accum["prefix_queries"]
        stats.prefix_hits = self._accum["prefix_hits"]
        stats.prefix_preempted_requests = self._accum["prefix_preempted_requests"]
        stats.prefix_preempted_queries = self._accum["prefix_preempted_queries"]
        stats.prefix_preempted_hits = self._accum["prefix_preempted_hits"]
        stats.kv_eviction_events = self._accum["kv_eviction_events"]

        return stats

    def reset_prefix_cache(self):
        """Clear the prefix cache via the public LLM API (RPC)."""
        if self.llm is not None:
            try:
                self.llm.reset_prefix_cache()
                logger.info("Prefix cache reset via LLM API")
            except Exception as e:
                logger.warning("Failed to reset prefix cache via LLM: %s", e)

    def is_available(self) -> bool:
        """Return True if scheduler stats are being captured."""
        return self._patched and self._scheduler_stats is not None
