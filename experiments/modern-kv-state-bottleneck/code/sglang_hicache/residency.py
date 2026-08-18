"""SGLang cache-residency preparation and evidence evaluation.

Three residency modes with these exact meanings:

* ``recompute`` — server started with ``--disable-radix-cache``; the
  preparation is a cache flush and validation asserts the repeated
  prefix is NOT served from cache (``cached_tokens == 0``).
* ``gpu_hit`` — radix cache enabled, HiCache disabled; the prefix is
  warmed into GPU L1 and validation asserts the measured request matches
  it from the device tier using either per-request metadata or the public
  ``prefill_effective_tokens_total{mode="device_hit"}`` counter in an
  isolated before/after request window.
* ``cpu_hit`` — HiCache L1 GPU + L2 host enabled; the prefix is warmed,
  then evicted from GPU by filling L1 with distinct filler KV, and
  validation asserts the next request restores it from host — the per-
  request is attributed to host L2 by either per-request tier metadata or
  an exact ``prefill_effective_tokens_total{mode="host_hit"}`` delta with
  no device-hit delta in the same isolated window. ``load_back_tokens_total``
  is recorded as corroboration when this model exposes it; missing metrics
  are unsupported, not zero.

The evidence evaluation functions are pure and unit-tested; they never
silently treat a missing metric as zero.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Optional

from sglang_hicache.metrics import CacheStats, CacheStatsDelta

logger = logging.getLogger(__name__)

#: Default filler prompt length and batch size (tokens per filler request,
#: number of concurrent filler requests per batch).
DEFAULT_FILLER_LEN = 4096
DEFAULT_FILLER_BATCH = 8
FILLER_VOCAB = 32000
# The pinned Qwen3-30B-A3B A100 server reported 227,712 L1 KV tokens while
# the public metrics scrape omitted that capacity. Keep the fallback above
# that observed pool and add one filler-page margin before validation.
FALLBACK_L1_CAPACITY_TOKENS = 262144
FALLBACK_MARGIN_TOKENS = 4096


# ---------------------------------------------------------------------------
# Pure evidence evaluation (unit-tested, no network)
# ---------------------------------------------------------------------------


def _details(meta_info: dict) -> Optional[dict]:
    details = meta_info.get("cached_tokens_details")
    if isinstance(details, dict):
        return details
    return None


def _host_tokens(details: Optional[dict]) -> int:
    return int((details or {}).get("host", 0) or 0)


def _device_tokens(details: Optional[dict]) -> int:
    return int((details or {}).get("device", 0) or 0)


def evaluate_recompute_evidence(meta_info: dict, prefix_len: int) -> tuple[bool, str]:
    """recompute: the repeated prefix must NOT be served from cache."""
    cached = int(meta_info.get("cached_tokens", 0) or 0)
    if cached == 0:
        return True, f"no cached tokens (cached_tokens=0) as expected for recompute"
    return (
        False,
        f"recompute unexpectedly served {cached}/{prefix_len} tokens from cache "
        f"(radix cache should be disabled)",
    )


def evaluate_gpu_hit_evidence(
    meta_info: dict,
    prefix_len: int,
    delta: CacheStatsDelta | None = None,
) -> tuple[bool, str]:
    """gpu_hit: the measured request must match the prefix from device L1."""
    cached = int(meta_info.get("cached_tokens", 0) or 0)
    details = _details(meta_info)
    host = _host_tokens(details)
    device = _device_tokens(details)

    if cached == prefix_len and host == 0 and device == prefix_len:
        return (
            True,
            f"prefix fully matched from device tier (cached={cached}, "
            f"device={device}, host={host})",
        )

    if delta is not None:
        device_delta = delta.get("prefill_device_hit_tokens")
        host_delta = delta.get("prefill_host_hit_tokens")
        if (
            device_delta is not None
            and device_delta >= prefix_len
            and host_delta is not None
            and host_delta == 0
        ):
            return (
                True,
                f"isolated public device-tier metric proves the hit "
                f"(device_hit_delta={device_delta:.0f}, "
                f"host_hit_delta={host_delta if host_delta is not None else 'unsupported'}; "
                f"native metadata cached={cached}, details={details!r})",
            )

    reason_parts = [f"cached={cached}/{prefix_len}"]
    if details is None:
        reason_parts.append("cached_tokens_details missing (no tier breakdown)")
    else:
        reason_parts.append(f"device={device}, host={host}")
    if delta is not None:
        dd = delta.get("prefill_device_hit_tokens")
        hd = delta.get("prefill_host_hit_tokens")
        reason_parts.append(f"device_hit_delta={dd if dd is not None else 'unsupported'}")
        reason_parts.append(f"host_hit_delta={hd if hd is not None else 'unsupported'}")
    return False, "GPU-resident hit evidence insufficient: " + ", ".join(reason_parts)


def evaluate_cpu_hit_evidence(
    meta_info: dict,
    prefix_len: int,
    delta: CacheStatsDelta | None = None,
) -> tuple[bool, str]:
    """cpu_hit: the measured request must restore the prefix from host L2.

    A PASS requires public host-tier attribution in the same isolated
    before/after request window. Per-request metadata is corroboration:

    1. per-request host-tier evidence:
       ``meta_info.cached_tokens_details == {"device": 0, "host": prefix_len}``
       (the pinned implementation's per-request tier breakdown — see
       ``output_streamer.get_cached_tokens_details`` and
       ``schedule_batch.split_cached_prefix_by_tier``);
    2. public tier evidence from the same window:
       ``prefill_effective_tokens_total{mode="host_hit"}`` delta >= prefix_len
       and no positive device-hit delta. ``load_back_tokens_total`` is
       retained as corroboration when exposed.

    Missing metrics are unsupported, never zero: a delta of ``None``
    (metric absent on either side) cannot contribute to a pass. The public
    counter is request-specific here because callers take its
    before/after snapshots around one synchronous request only. This is the
    authoritative fallback for pinned Qwen3.5 hybrid runs, whose native
    response leaves cached-token metadata at zero in this SGLang commit.
    """
    cached = int(meta_info.get("cached_tokens", 0) or 0)
    details = _details(meta_info)
    host = _host_tokens(details)
    device = _device_tokens(details)

    # -- class (1): per-request host-tier evidence -----------------------
    per_request_ok = (
        details is not None
        and cached == prefix_len
        and host == prefix_len
        and device == 0
    )

    # -- class (2): public tier evidence from the same window ------------
    if delta is None:
        load_back = None
        host_hit = None
        device_hit = None
    else:
        load_back = delta.get("load_back_tokens_total")
        host_hit = delta.get("prefill_host_hit_tokens")
        device_hit = delta.get("prefill_device_hit_tokens")
    public_ok = (
        host_hit is not None
        and host_hit >= prefix_len
        and device_hit is not None
        and device_hit == 0
    )

    if public_ok:
        load_back_text = (
            f"{load_back:.0f}" if load_back is not None else "unsupported"
        )
        return (
            True,
            f"prefix attributed to host tier (cached={cached}, host={host}, "
            f"device={device}); host_hit_delta="
            f"{host_hit if host_hit is not None else 'unsupported'}, "
            f"device_hit_delta={device_hit if device_hit is not None else 'unsupported'}, "
            f"load_back_delta={load_back_text}",
        )

    # -- failure / unsupported reason ------------------------------------
    parts = []
    unsupported = False
    if details is None:
        unsupported = True
        parts.append(
            "cached_tokens_details missing (no per-request tier breakdown; "
            f"got cached={cached}/{prefix_len})"
        )
    elif not per_request_ok:
        if device == prefix_len and host == 0:
            parts.append(
                "prefix is still GPU-resident "
                f"(device={device}, host={host}); L1 eviction did not happen "
                "(filler pressure too low or eviction policy kept the node on device)"
            )
        else:
            parts.append(
                f"per-request breakdown cached={cached}, host={host}, device={device}"
            )
    if host_hit is None:
        unsupported = True
        parts.append("prefill_host_hit_tokens missing/unsupported in this window")
    if device_hit is None:
        unsupported = True
        parts.append("prefill_device_hit_tokens missing/unsupported in this window")
    if host_hit is not None and device_hit is not None and not public_ok:
        parts.append(
            f"public tier deltas do not prove an isolated host hit "
            f"(host_hit={host_hit}, device_hit={device_hit}, load_back={load_back})"
        )
    parts.append(
        "isolated prefill_host_hit tier evidence required"
    )
    prefix = (
        "CPU-resident hit unsupported"
        if unsupported
        else "CPU-resident hit evidence insufficient"
    )
    return False, f"{prefix}: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Filler budgeting (pure)
# ---------------------------------------------------------------------------


def compute_filler_budget(
    snapshot: CacheStats,
    prefix_len: int,
    margin_tokens: int = 4096,
) -> Optional[int]:
    """Tokens of distinct filler KV needed to evict the warmed prefix.

    Fills the remaining GPU KV pool plus a margin so LRU eviction must
    evict the oldest node (the warmed prefix) and preserve it to host.

    Returns ``None`` when pool capacity cannot be observed (metrics
    missing) — callers must then use the documented fallback heuristic.
    """
    total = snapshot.max_total_num_tokens
    used = snapshot.kv_used_tokens
    if total is None or used is None:
        return None
    free = total - used
    if free < 0:
        free = 0.0
    return int(free + prefix_len + margin_tokens)


def fallback_filler_budget(
    context_length: int, prefix_len: int, factor: int = 6
) -> int:
    """Documented fallback when pool metrics are unavailable.

    Uses the larger of ``factor`` full-contexts and the documented
    262,144-token A100 capacity ceiling, then adds the protected prefix
    and one filler-page margin. Validation still gates the result.
    """
    pressure = max(factor * context_length, FALLBACK_L1_CAPACITY_TOKENS)
    return pressure + prefix_len + FALLBACK_MARGIN_TOKENS


# ---------------------------------------------------------------------------
# Filler prompt construction (deterministic)
# ---------------------------------------------------------------------------


def build_filler_prompts(
    n_prompts: int,
    filler_len: int,
    seed: int,
    vocab: int = FILLER_VOCAB,
) -> list[list[int]]:
    """Build distinct filler token sequences (never equal to the prefix).

    Uses a dedicated RNG stream so fillers are independent of the
    workload prefix construction.
    """
    rng = random.Random(seed)
    prompts = []
    for _ in range(n_prompts):
        prompts.append([rng.randrange(vocab) for _ in range(filler_len)])
    return prompts


# ---------------------------------------------------------------------------
# Preparation procedures (network-bound)
# ---------------------------------------------------------------------------


@dataclass
class CPUHitPrepEvidence:
    """What happened during cpu_hit preparation (warm + evict)."""

    warmup_ok: bool = False
    warmup_error: str = ""
    filler_batches: int = 0
    filler_tokens_sent: int = 0
    filler_budget: Optional[int] = None
    host_used_before: Optional[float] = None
    host_used_after: Optional[float] = None
    host_used_delta: Optional[float] = None
    budget_source: str = "metrics"

    def to_dict(self) -> dict:
        return {
            "warmup_ok": self.warmup_ok,
            "warmup_error": self.warmup_error,
            "filler_batches": self.filler_batches,
            "filler_tokens_sent": self.filler_tokens_sent,
            "filler_budget": self.filler_budget,
            "host_used_before": self.host_used_before,
            "host_used_after": self.host_used_after,
            "host_used_delta": self.host_used_delta,
            "budget_source": self.budget_source,
        }


def prepare_recompute(client) -> None:
    """recompute preparation: nothing cached, just flush to a clean slate."""
    client.flush_cache()


def prepare_gpu_hit(client, prefix_ids: list[int], request_id: int = 0) -> GenerateResultLike:
    """gpu_hit preparation: flush, then warm the prefix into GPU L1."""
    client.flush_cache()
    return warm_prefix(client, prefix_ids, request_id=request_id)


def warm_prefix(client, prefix_ids: list[int], request_id: int = 0):
    """Send a prefix-only warmup request (max_new_tokens=1)."""
    result = client.generate(prefix_ids, max_new_tokens=1, request_id=request_id)
    if not result.ok:
        raise RuntimeError(f"prefix warmup failed: {result.error}")
    return result


def prepare_cpu_hit(
    client,
    prefix_ids: list[int],
    snapshot_fn: Callable[[], CacheStats],
    *,
    context_length: int,
    seed: int = 42,
    filler_len: int = DEFAULT_FILLER_LEN,
    filler_batch: int = DEFAULT_FILLER_BATCH,
    max_batches: int = 64,
    margin_tokens: int = 4096,
    request_id_base: int = 1000,
) -> CPUHitPrepEvidence:
    """cpu_hit preparation: flush, warm prefix, evict it from GPU to host.

    Steps:
      1. flush_cache();
      2. warm the shared prefix (prefix-only request);
      3. read pool metrics -> filler budget (free pool + prefix + margin);
      4. send distinct filler batches until filler tokens >= budget or
         ``max_batches`` is reached;
      5. return prep evidence for the validation gate.

    With ``write_through``, host occupancy grows as soon as filler KV is
    written and therefore does not prove that the warmed prefix left L1.
    It is recorded as evidence but must not terminate pressure early.
    The validation gate still performs the definitive check on the next
    measured request (host-tier per-request evidence), so an insufficient
    eviction is caught there, not silently accepted.
    """
    client.flush_cache()
    warm_prefix(client, prefix_ids, request_id=request_id_base)
    before = snapshot_fn()
    host_used_before = before.hicache_host_used_tokens

    budget = compute_filler_budget(before, len(prefix_ids), margin_tokens=margin_tokens)
    if budget is None:
        budget = fallback_filler_budget(context_length, len(prefix_ids))
        budget_source = "fallback-heuristic"
        logger.warning(
            "Pool capacity metrics unavailable; using fallback filler budget=%d",
            budget,
        )
    else:
        budget_source = "metrics"

    sent = 0
    batches = 0
    for batch_idx in range(max_batches):
        if sent >= budget:
            break
        remaining = budget - sent
        n_prompts = min(filler_batch, max(1, remaining // max(filler_len, 1)))
        if n_prompts == 0:
            break
        prompts = build_filler_prompts(n_prompts, filler_len, seed + batch_idx)
        for i, pids in enumerate(prompts):
            # Globally unique request ids across batches (SGLang rids must
            # not be reused while a request may still be tracked).
            rid = request_id_base + 1 + batch_idx * filler_batch + i
            r = client.generate(pids, max_new_tokens=1, request_id=rid)
            if not r.ok:
                logger.warning("filler request failed: %s", r.error)
        sent += n_prompts * filler_len
        batches += 1

    after = snapshot_fn()
    delta = (
        after.hicache_host_used_tokens - host_used_before
        if after.hicache_host_used_tokens is not None and host_used_before is not None
        else None
    )
    evidence = CPUHitPrepEvidence(
        warmup_ok=True,
        filler_batches=batches,
        filler_tokens_sent=sent,
        filler_budget=budget,
        host_used_before=host_used_before,
        host_used_after=after.hicache_host_used_tokens,
        host_used_delta=delta,
        budget_source=budget_source,
    )
    logger.info(
        "cpu_hit prep: batches=%d filler_tokens=%d budget=%d host_delta=%s",
        batches, sent, budget,
        f"{delta:.0f}" if delta is not None else "unsupported",
    )
    return evidence


# ---------------------------------------------------------------------------
# Prefix-pool residency preparation (Exp3)
# ---------------------------------------------------------------------------


@dataclass
class PrefixPoolPrepEvidence:
    """What happened during pool-based residency preparation.

    ``fits_l1`` is ``None`` when pool capacity metrics are unavailable;
    ``False`` means the warmed prefix pool provably exceeds the observed
    GPU KV pool — a ``gpu_hit`` load point must then be treated as
    unsupported (the load-point dominance gate is the final arbiter).
    """

    mode: str = ""
    warmup_ok: bool = False
    warmup_error: str = ""
    prefixes_warmed: int = 0
    pool_tokens: Optional[int] = None
    fits_l1: Optional[bool] = None
    l1_capacity_tokens: Optional[float] = None
    filler_batches: int = 0
    filler_tokens_sent: int = 0
    filler_budget: Optional[int] = None
    budget_source: str = "metrics"
    host_used_before: Optional[float] = None
    host_used_after: Optional[float] = None
    host_used_delta: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "warmup_ok": self.warmup_ok,
            "warmup_error": self.warmup_error,
            "prefixes_warmed": self.prefixes_warmed,
            "pool_tokens": self.pool_tokens,
            "fits_l1": self.fits_l1,
            "l1_capacity_tokens": self.l1_capacity_tokens,
            "filler_batches": self.filler_batches,
            "filler_tokens_sent": self.filler_tokens_sent,
            "filler_budget": self.filler_budget,
            "budget_source": self.budget_source,
            "host_used_before": self.host_used_before,
            "host_used_after": self.host_used_after,
            "host_used_delta": self.host_used_delta,
        }


def _pool_prefixes(pool) -> list[list[int]]:
    """Extract the per-family prefix token lists from a prefix pool."""
    return [fam.prefix_ids for fam in pool.families]


def prepare_gpu_hit_pool(
    client,
    pool,
    snapshot_fn: Callable[[], CacheStats],
    *,
    request_id_base: int = 6000,
) -> PrefixPoolPrepEvidence:
    """gpu_hit preparation for an Exp3 prefix pool: warm every family prefix.

    Flushes, then warms ALL family prefixes into GPU L1 (the bounded pool
    must fit the observed L1 capacity — ``fits_l1`` records whether the
    pool provably fits; the load-point dominance gate enforces it).
    """
    client.flush_cache()
    prefixes = _pool_prefixes(pool)
    pool_tokens = sum(len(p) for p in prefixes)

    snapshot = snapshot_fn()
    capacity = snapshot.max_total_num_tokens
    used = snapshot.kv_used_tokens
    fits_l1: Optional[bool] = None
    free: Optional[float] = None
    if capacity is not None and used is not None:
        free = capacity - used
        fits_l1 = pool_tokens <= free
        if not fits_l1:
            logger.warning(
                "gpu_hit pool (%d tokens) does not fit observed L1 free capacity "
                "(%.0f tokens); device dominance will fail and the load point "
                "must be labelled unsupported",
                pool_tokens, free,
            )

    for idx, pids in enumerate(prefixes):
        warm_prefix(client, pids, request_id=request_id_base + idx)

    return PrefixPoolPrepEvidence(
        mode="gpu_hit",
        warmup_ok=True,
        prefixes_warmed=len(prefixes),
        pool_tokens=pool_tokens,
        fits_l1=fits_l1,
        l1_capacity_tokens=free,
    )


def prepare_cpu_hit_pool(
    client,
    pool,
    snapshot_fn: Callable[[], CacheStats],
    *,
    context_length: int,
    seed: int = 42,
    filler_len: int = DEFAULT_FILLER_LEN,
    filler_batch: int = DEFAULT_FILLER_BATCH,
    max_batches: int = 128,
    margin_tokens: int = 4096,
    request_id_base: int = 7000,
) -> PrefixPoolPrepEvidence:
    """cpu_hit preparation for an Exp3 prefix pool.

    Steps:
      1. flush_cache();
      2. warm ALL family prefixes (distinct prefixes, distinct rids);
      3. read pool metrics -> filler budget (free pool + pool tokens + margin);
      4. apply L1 pressure with distinct filler KV AFTER warming so the
         formal load window contains many independently restorable host
         prefixes (LRU eviction moves the warmed prefixes to host L2);
      5. return prep evidence for the load-point dominance gate.

    Host occupancy is observational only: under ``write_through`` it
    increases before L1 eviction and cannot be used as an early-stop signal.
    """
    client.flush_cache()
    prefixes = _pool_prefixes(pool)
    pool_tokens = sum(len(p) for p in prefixes)
    for idx, pids in enumerate(prefixes):
        warm_prefix(client, pids, request_id=request_id_base + idx)

    before = snapshot_fn()
    host_used_before = before.hicache_host_used_tokens

    budget = compute_filler_budget(before, pool_tokens, margin_tokens=margin_tokens)
    if budget is None:
        budget = fallback_filler_budget(context_length, pool_tokens)
        budget_source = "fallback-heuristic"
        logger.warning(
            "Pool capacity metrics unavailable; using fallback filler budget=%d",
            budget,
        )
    else:
        budget_source = "metrics"

    sent = 0
    batches = 0
    for batch_idx in range(max_batches):
        if sent >= budget:
            break
        remaining = budget - sent
        n_prompts = min(filler_batch, max(1, remaining // max(filler_len, 1)))
        if n_prompts == 0:
            break
        prompts = build_filler_prompts(n_prompts, filler_len, seed + 1000 + batch_idx)
        for i, pids in enumerate(prompts):
            rid = request_id_base + 100 + batch_idx * filler_batch + i
            r = client.generate(pids, max_new_tokens=1, request_id=rid)
            if not r.ok:
                logger.warning("filler request failed: %s", r.error)
        sent += n_prompts * filler_len
        batches += 1

    after = snapshot_fn()
    delta = (
        after.hicache_host_used_tokens - host_used_before
        if after.hicache_host_used_tokens is not None and host_used_before is not None
        else None
    )
    evidence = PrefixPoolPrepEvidence(
        mode="cpu_hit",
        warmup_ok=True,
        prefixes_warmed=len(prefixes),
        pool_tokens=pool_tokens,
        filler_batches=batches,
        filler_tokens_sent=sent,
        filler_budget=budget,
        budget_source=budget_source,
        host_used_before=host_used_before,
        host_used_after=after.hicache_host_used_tokens,
        host_used_delta=delta,
    )
    logger.info(
        "cpu_hit pool prep: prefixes_warmed=%d pool_tokens=%d batches=%d "
        "filler_tokens=%d budget=%d host_delta=%s",
        len(prefixes), pool_tokens, batches, sent, budget,
        f"{delta:.0f}" if delta is not None else "unsupported",
    )
    return evidence


# Re-exported for convenience (avoids an extra import in runners).
from sglang_hicache.http_client import GenerateResult as GenerateResultLike  # noqa: E402
