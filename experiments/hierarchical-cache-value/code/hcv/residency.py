"""Cold- and warm-cache preparation with observed cache state.

Cold: flush the cache (L1 GPU + L2 host) and verify emptiness from the
gauges (``kv_used_tokens`` ~ 0 and ``hicache_host_used_tokens`` ~ 0
when observable).  An unobservable gauge never counts as zero.

Warm: run a fixed cache-population trace (a prefix of the formal trace,
run serially so the shared prefixes are installed deterministically),
then record the *observed* occupancy.  Executing warm-up is not enough:
the occupancy evidence is recorded and validated before the formal phase
starts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from hcv.http_client import SGLangHTTPClient
from hcv.metrics import CacheStats, snapshot_from_scrape
from hcv.workload import Trace

logger = logging.getLogger(__name__)


@dataclass
class ObservedCacheState:
    """Observed cache state after a preparation step."""

    kv_used_tokens: Optional[float] = None
    kv_available_tokens: Optional[float] = None
    kv_evictable_tokens: Optional[float] = None
    hicache_host_used_tokens: Optional[float] = None
    hicache_host_total_tokens: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    num_requests_total: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "kv_used_tokens": self.kv_used_tokens,
            "kv_available_tokens": self.kv_available_tokens,
            "kv_evictable_tokens": self.kv_evictable_tokens,
            "hicache_host_used_tokens": self.hicache_host_used_tokens,
            "hicache_host_total_tokens": self.hicache_host_total_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "num_requests_total": self.num_requests_total,
        }

    @property
    def empty_verified(self) -> bool:
        """True when occupancy gauges are observable and empty.

        Unobservable gauges (None) never count as verified empty.
        """
        if self.kv_used_tokens is None:
            return False
        if self.kv_used_tokens > 1.0:
            return False
        if self.hicache_host_used_tokens is None:
            # host gauge missing: L2 emptiness cannot be verified
            return False
        return self.hicache_host_used_tokens <= 1.0


def snapshot_cache_state(client: SGLangHTTPClient) -> ObservedCacheState:
    """Snapshot the observable cache gauges."""
    stats: CacheStats = snapshot_from_scrape(client.scrape_metrics())
    return ObservedCacheState(
        kv_used_tokens=stats.kv_used_tokens,
        kv_available_tokens=stats.kv_available_tokens,
        kv_evictable_tokens=stats.kv_evictable_tokens,
        hicache_host_used_tokens=stats.hicache_host_used_tokens,
        hicache_host_total_tokens=stats.hicache_host_total_tokens,
        cache_hit_rate=stats.cache_hit_rate,
        num_requests_total=stats.num_requests_total,
    )


def prepare_cold(client: SGLangHTTPClient) -> ObservedCacheState:
    """Flush the cache and verify emptiness; raise on verification failure."""
    ok = client.flush_cache()
    if not ok:
        raise RuntimeError("/flush_cache returned non-200")
    state = snapshot_cache_state(client)
    if not state.empty_verified:
        logger.warning(
            "cold-cache emptiness not fully verified (kv_used=%s host_used=%s)",
            state.kv_used_tokens, state.hicache_host_used_tokens,
        )
    return state


def prepare_warm(
    client: SGLangHTTPClient,
    trace: Trace,
    warmup_requests: int,
    request_id_offset: int = 0,
) -> ObservedCacheState:
    """Install the warm state by running a fixed population prefix.

    The population prefix is the first ``warmup_requests`` records of
    the formal trace (deterministic, identical for GPU-only and
    hierarchical pairs).  Returns the observed occupancy afterwards.
    """
    records = trace.records[:warmup_requests]
    if not records:
        raise ValueError("warmup_requests must be > 0")
    for i, rec in enumerate(records):
        gen = client.generate(
            request_id=request_id_offset + i,
            input_ids=rec.input_ids,
            max_new_tokens=rec.output_tokens,
        )
        if not gen.ok:
            raise RuntimeError(f"warmup request {i} failed: {gen.error}")
    state = snapshot_cache_state(client)
    logger.info(
        "warm cache: kv_used=%s host_used=%s (populated %d requests)",
        state.kv_used_tokens, state.hicache_host_used_tokens, len(records),
    )
    return state


def warm_state_grew(state: ObservedCacheState, baseline: ObservedCacheState) -> bool:
    """True when warm occupancy observably grew relative to the baseline.

    Conservative: any of the occupancy gauges that are observable on
    both sides must show growth for the check to pass.
    """
    checks = []
    if state.kv_used_tokens is not None and baseline.kv_used_tokens is not None:
        checks.append(state.kv_used_tokens > baseline.kv_used_tokens)
    if state.hicache_host_used_tokens is not None and baseline.hicache_host_used_tokens is not None:
        checks.append(state.hicache_host_used_tokens > baseline.hicache_host_used_tokens)
    return bool(checks) and all(checks)
