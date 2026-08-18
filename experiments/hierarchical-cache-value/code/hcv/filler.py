"""Deterministic L1 (GPU) pressure filling and budgeting.

The filler's job is to deterministically evict previously-resident
reusable state from the GPU cache so that a later revisit must come from
L2 (host) or recompute.

Learned constraint: under ``write_through`` the host occupancy grows
*before* L1 eviction (every write-through backup is immediate), so host
occupancy must never be used as the filler early-stop condition.  The
filler stops only on L1 evidence:

* target pressure reached, AND
* L1 saturation evidence (``kv_evictable_tokens`` > 0 or free KV slots
  exhausted), or a best-effort cap reached.

Observed L1 capacity is preferred (``max_total_num_tokens``, else the
sum of used+available KV tokens).  When no observed capacity exists the
fallback pressure is ``max(6*context_length, 262144) +
protected_prefix_tokens + 4096``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from hcv.config import (
    FALLBACK_L1_FILLER_FLOOR,
    FALLBACK_L1_FILLER_MARGIN,
    FALLBACK_L1_FILLER_MULTIPLIER,
)
from hcv.http_client import SGLangHTTPClient
from hcv.metrics import CacheStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure budgeting functions
# ---------------------------------------------------------------------------


def observed_l1_capacity(stats: CacheStats) -> Optional[float]:
    """Observed L1 (GPU KV) capacity in tokens, or None when unobservable."""
    if stats.max_total_num_tokens is not None:
        return float(stats.max_total_num_tokens)
    if stats.kv_used_tokens is not None and stats.kv_available_tokens is not None:
        return float(stats.kv_used_tokens + stats.kv_available_tokens)
    return None


def filler_pressure_tokens(
    context_length: int,
    protected_prefix_tokens: int,
    observed_capacity: Optional[float] = None,
) -> float:
    """Target L1 filler pressure (unique tokens to push through L1).

    Uses the observed L1 capacity when available; otherwise the fallback
    formula ``max(6*context_length, 262144) + protected_prefix_tokens +
    4096``.
    """
    if observed_capacity is not None and observed_capacity > 0:
        return float(observed_capacity)
    fallback = max(
        FALLBACK_L1_FILLER_MULTIPLIER * int(context_length),
        FALLBACK_L1_FILLER_FLOOR,
    ) + int(protected_prefix_tokens) + FALLBACK_L1_FILLER_MARGIN
    return float(fallback)


def l1_saturation_evidence(stats: CacheStats) -> dict:
    """L1 saturation signals derived from a snapshot (never host gauges)."""
    evictable = stats.kv_evictable_tokens
    available = stats.kv_available_tokens
    used = stats.kv_used_tokens
    return {
        "kv_evictable_tokens": evictable,
        "kv_available_tokens": available,
        "kv_used_tokens": used,
        "l1_saturated": (
            (evictable is not None and evictable > 0)
            or (available is not None and available <= 1.0)
        )
        if (evictable is not None or available is not None)
        else None,
    }


def should_stop_filling(
    cumulative_unique_tokens: float,
    target_pressure: float,
    saturation: dict,
    best_effort_cap: float,
) -> tuple[bool, str]:
    """Early-stop decision.  **Never** consults host occupancy gauges.

    Returns (stop, reason).  Stops when the target is reached AND L1
    saturation is observed; otherwise keeps going until ``best_effort_cap``
    (best-effort evidence recorded by the caller).
    """
    if cumulative_unique_tokens >= target_pressure:
        if saturation.get("l1_saturated") is True:
            return True, "target_reached_and_l1_saturated"
        # target reached without observable saturation: keep filling a
        # bounded amount so eviction is still forced (host occupancy is
        # deliberately not part of this decision).
        if cumulative_unique_tokens >= best_effort_cap:
            return True, "best_effort_cap_reached_without_observed_saturation"
        return False, "target_reached_waiting_for_l1_saturation"
    if cumulative_unique_tokens >= best_effort_cap:
        return True, "best_effort_cap_reached"
    return False, "filling"


# ---------------------------------------------------------------------------
# Filler driver
# ---------------------------------------------------------------------------


@dataclass
class FillerPlan:
    """Filler plan produced before execution (recorded in metadata)."""

    target_pressure_tokens: float
    observed_capacity: Optional[float]
    capacity_source: str                 # "observed_max_total_num_tokens" | "observed_kv_sum" | "fallback_formula"
    protected_prefix_tokens: int
    filler_prefix_length: int
    filler_unique_count: int             # number of unique filler prefixes planned
    best_effort_cap: float
    context_length: int

    def to_dict(self) -> dict:
        return {
            "target_pressure_tokens": self.target_pressure_tokens,
            "observed_capacity": self.observed_capacity,
            "capacity_source": self.capacity_source,
            "protected_prefix_tokens": self.protected_prefix_tokens,
            "filler_prefix_length": self.filler_prefix_length,
            "filler_unique_count": self.filler_unique_count,
            "best_effort_cap": self.best_effort_cap,
            "context_length": self.context_length,
        }


@dataclass
class FillerResult:
    """Outcome of a filler run."""

    requests_sent: int = 0
    unique_tokens_pushed: int = 0
    stop_reason: str = ""
    target_pressure_tokens: float = 0.0
    best_effort_cap: float = 0.0
    l1_saturated: Optional[bool] = None
    final_snapshot: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.requests_sent > 0 and not self.errors

    def to_dict(self) -> dict:
        return {
            "requests_sent": self.requests_sent,
            "unique_tokens_pushed": self.unique_tokens_pushed,
            "stop_reason": self.stop_reason,
            "target_pressure_tokens": self.target_pressure_tokens,
            "best_effort_cap": self.best_effort_cap,
            "l1_saturated": self.l1_saturated,
            "final_snapshot": self.final_snapshot,
            "errors": self.errors,
            "ok": self.ok,
        }


def build_filler_plan(
    context_length: int,
    protected_prefix_tokens: int,
    observed_capacity: Optional[float] = None,
    capacity_source: str = "fallback_formula",
    filler_prefix_length: int = 512,
    best_effort_multiplier: float = 1.5,
) -> FillerPlan:
    """Plan the filler from observed capacity or the fallback formula.

    ``capacity_source`` records where ``observed_capacity`` came from
    (e.g. ``observed_max_total_num_tokens`` or ``observed_kv_sum``); it
    is metadata, not a runtime decision input.
    """
    target = filler_pressure_tokens(
        context_length, protected_prefix_tokens, observed_capacity
    )
    if observed_capacity is not None:
        source = capacity_source or "observed"
    else:
        source = "fallback_formula"
    unique_count = max(1, int(target) // max(1, filler_prefix_length) + 1)
    return FillerPlan(
        target_pressure_tokens=target,
        observed_capacity=observed_capacity,
        capacity_source=source,
        protected_prefix_tokens=protected_prefix_tokens,
        filler_prefix_length=filler_prefix_length,
        filler_unique_count=unique_count,
        best_effort_cap=target * best_effort_multiplier,
        context_length=context_length,
    )


def build_fixed_filler_plan(
    target_tokens: int,
    filler_prefix_length: int = 512,
) -> FillerPlan:
    """Build a cross-budget fixed-work plan for pressure experiments.

    Unlike the capability-gate plan, this plan deliberately does not
    adapt to each server's capacity. Paired pressure points therefore
    receive identical unique-token work and differ only in GPU budget.
    """
    target = float(target_tokens)
    return FillerPlan(
        target_pressure_tokens=target,
        observed_capacity=None,
        capacity_source="fixed_cross_budget_working_set",
        protected_prefix_tokens=0,
        filler_prefix_length=int(filler_prefix_length),
        filler_unique_count=max(
            1,
            (int(target_tokens) + filler_prefix_length - 1) // filler_prefix_length,
        ),
        best_effort_cap=target,
        context_length=int(filler_prefix_length),
    )


def run_filler(
    client: SGLangHTTPClient,
    plan: FillerPlan,
    seed: int,
    max_requests: int = 20000,
) -> FillerResult:
    """Push deterministic unique filler prefixes through L1.

    Each filler request uses a unique prefix (never revisited), so any
    L1 growth is entirely unique-token pressure.  The loop stops on L1
    evidence only (see :func:`should_stop_filling`); host occupancy
    gauges are recorded but never consulted.
    """
    from hcv.workload import _ids

    result = FillerResult(
        target_pressure_tokens=plan.target_pressure_tokens,
        best_effort_cap=plan.best_effort_cap,
    )
    rng_seed = seed * 104729 + 99991
    cumulative = 0
    for i in range(max_requests):
        prefix_ids = _ids(rng_seed + i * 31, plan.filler_prefix_length)
        suffix_ids = _ids(rng_seed + i * 31 + 1, 8)
        input_ids = prefix_ids + suffix_ids
        gen = client.generate(request_id=-1 - i, input_ids=input_ids, max_new_tokens=1)
        if not gen.ok:
            result.errors.append({"filler_req": i, "error": gen.error, "status": gen.status})
            break
        cumulative += len(prefix_ids)
        result.requests_sent += 1
        result.unique_tokens_pushed += len(prefix_ids)

        stats = _stats_from_client(client)
        sat = l1_saturation_evidence(stats)
        result.l1_saturated = sat.get("l1_saturated")
        stop, reason = should_stop_filling(
            float(cumulative), plan.target_pressure_tokens, sat, plan.best_effort_cap
        )
        if stop:
            result.stop_reason = reason
            break
        if i % 50 == 0 and i > 0:
            logger.info("filler progress: %d unique tokens (target %.0f)", cumulative, plan.target_pressure_tokens)
    else:
        result.stop_reason = "max_requests_reached"

    result.final_snapshot = _stats_to_dict(client)
    return result


def _stats_from_client(client: SGLangHTTPClient) -> CacheStats:
    """Scrape and build a CacheStats snapshot."""
    from hcv.metrics import snapshot_from_scrape

    return snapshot_from_scrape(client.scrape_metrics())


def _stats_to_dict(client: SGLangHTTPClient) -> dict:
    stats = _stats_from_client(client)
    return {
        "kv_used_tokens": stats.kv_used_tokens,
        "kv_available_tokens": stats.kv_available_tokens,
        "kv_evictable_tokens": stats.kv_evictable_tokens,
        "max_total_num_tokens": stats.max_total_num_tokens,
        "hicache_host_used_tokens": stats.hicache_host_used_tokens,
        "hicache_host_total_tokens": stats.hicache_host_total_tokens,
    }
