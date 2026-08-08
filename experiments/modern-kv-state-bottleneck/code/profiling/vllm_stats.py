"""vLLM internal statistics collection.

Wraps KVCacheManager and Scheduler APIs to collect cache-related stats
before and after each measured request.
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

    # KV cache usage (0.0 – 1.0)
    usage: float
    # Prefix cache stats
    prefix_requests: int
    prefix_queries: int
    prefix_hits: int
    prefix_preempted_requests: int
    prefix_preempted_queries: int
    prefix_preempted_hits: int
    # Timestamp
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


class VLLMStatsCollector:
    """Collects internal vLLM statistics from the engine.

    Accesses the KVCacheManager and Scheduler through the engine's
    internal attributes.  The exact attribute path may vary across
    vLLM versions; this implementation targets vLLM 0.26.0 V1 engine.
    """

    def __init__(self, llm):
        self.llm = llm
        self._kv_manager = None
        self._scheduler = None
        self._log_stats = False
        self._find_internals()

    def _find_internals(self):
        """Locate KVCacheManager and Scheduler in the engine."""
        engine = getattr(self.llm, "llm_engine", None)
        if engine is None:
            logger.warning("Cannot find llm_engine on LLM instance")
            return

        # V1 engine stores scheduler in engine.scheduler or engine.engine_core
        for attr in ("scheduler", "engine_core"):
            obj = getattr(engine, attr, None)
            if obj is not None:
                # Try to find kv_cache_manager on the scheduler
                kv_mgr = getattr(obj, "kv_cache_manager", None)
                if kv_mgr is not None:
                    self._kv_manager = kv_mgr
                    self._scheduler = obj
                    self._log_stats = getattr(obj, "log_stats", False) or True
                    logger.info(
                        "Found KVCacheManager on %s and Scheduler", attr
                    )
                    return

        # Fallback: search deeper
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None:
            scheduler = getattr(engine_core, "scheduler", None)
            if scheduler is not None:
                kv_mgr = getattr(scheduler, "kv_cache_manager", None)
                if kv_mgr is not None:
                    self._kv_manager = kv_mgr
                    self._scheduler = scheduler
                    self._log_stats = True
                    logger.info("Found KVCacheManager via engine_core.scheduler")
                    return

        logger.warning(
            "Could not locate KVCacheManager/Scheduler. "
            "Internal stats will be empty."
        )

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
            timestamp=time.perf_counter(),
        )

        if self._kv_manager is not None:
            try:
                stats.usage = float(self._kv_manager.usage)
            except Exception as e:
                logger.debug("Failed to get usage: %s", e)

        if self._kv_manager is not None:
            try:
                pcs = self._kv_manager.make_prefix_cache_stats()
                if pcs is not None:
                    stats.prefix_requests = pcs.requests
                    stats.prefix_queries = pcs.queries
                    stats.prefix_hits = pcs.hits
                    stats.prefix_preempted_requests = pcs.preempted_requests
                    stats.prefix_preempted_queries = pcs.preempted_queries
                    stats.prefix_preempted_hits = pcs.preempted_hits
            except Exception as e:
                logger.debug("Failed to get prefix cache stats: %s", e)

        return stats

    def reset_prefix_cache(self):
        """Clear the prefix cache."""
        if self._kv_manager is not None:
            try:
                self._kv_manager.reset_prefix_cache()
                logger.info("Prefix cache reset via KVCacheManager")
            except Exception as e:
                logger.warning("Failed to reset prefix cache: %s", e)
        elif self.llm is not None:
            try:
                self.llm.reset_prefix_cache()
                logger.info("Prefix cache reset via LLM API")
            except Exception as e:
                logger.warning("Failed to reset prefix cache via LLM: %s", e)

    def is_available(self) -> bool:
        """Return True if internal stats collection is available."""
        return self._kv_manager is not None
