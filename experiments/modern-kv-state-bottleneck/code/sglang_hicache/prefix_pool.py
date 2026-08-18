"""Deterministic prefix-pool construction and tier-dominance decisions (Exp3).

Exp3 drives *concurrent* requests, so a single warmed shared prefix is
wrong for the cache-residency question: the first request would restore it
from host and every later concurrent request would be a GPU hit.  Instead
Exp3 builds a **prefix pool**:

* ``build_prefix_families`` — deterministic, pairwise-distinct prefix
  families (each with its own suffix pool) built from the same corpus
  rules as ``workload.SGLangWorkload``;
* ``PrefixPool`` — deterministic scheduling of prompt families through a
  load window (round-robin cycle), so concurrent requests touch distinct
  prefixes;
* ``classify_tier`` / ``aggregate_tier_hits`` — per-request device/host
  tier breakdown aggregated over a load window;
* ``dominance_decision`` — a load point may only be reported under the
  requested residency when that tier dominates by a documented threshold
  (``HIT_DOMINANCE_THRESHOLD``); a mostly-GPU load point is never
  silently called ``cpu_hit`` — it is labelled unsupported instead.

All functions are pure and unit-tested (no CUDA, SGLang, network, or
model weights).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from sglang_hicache.workload import _FALLBACK_TEXT

logger = logging.getLogger(__name__)

#: Default dominance threshold: the requested residency tier must account
#: for at least this fraction of the requests with cache-hit evidence in a
#: load window.  Pinned in metadata and sbatch (``HIT_DOMINANCE_THRESHOLD``).
HIT_DOMINANCE_THRESHOLD = 0.8

#: HiCache is pinned to 64-token pages in the Exp1-3 sbatch files. Prefix
#: families must differ within this first cache page: merely differing later
#: in a 16K prefix is insufficient because one host restore would make every
#: family sharing the leading radix path a subsequent device hit.
DISTINCT_PREFIX_HEAD_TOKENS = 64


@dataclass
class PrefixFamily:
    """One distinct shared-prefix family: prefix + request-specific suffixes."""

    family_id: int
    prefix_ids: list[int]
    suffix_pool: list[list[int]] = field(default_factory=list)

    @property
    def prefix_len(self) -> int:
        return len(self.prefix_ids)


def _tile_corpus(all_tokens: list[int], need: int) -> list[int]:
    """Tile the corpus (deterministically) until it holds ``need`` tokens."""
    if len(all_tokens) >= need:
        return all_tokens
    factor = need // len(all_tokens) + 2
    return all_tokens * factor


def build_prefix_families(
    tokenize_fn: Callable[[str], list[int]],
    prefix_len: int,
    suffix_len: int,
    n_families: int,
    n_suffixes: int = 1,
    base_text: str | None = None,
    seed: int = 42,
) -> list[PrefixFamily]:
    """Build ``n_families`` deterministic, pairwise-distinct prefix families.

    Each family owns a distinct ``[offset, offset+prefix_len)`` span of the
    tokenized corpus and a suffix pool drawn from its own suffix region, so
    no two families share a prefix and no suffix overlaps another family's
    prefix.  Family placement is seeded (``seed``) and therefore
    deterministic across runs.
    """
    if n_families <= 0:
        raise ValueError(f"n_families must be > 0, got {n_families}")
    if prefix_len <= 0 or suffix_len <= 0:
        raise ValueError(f"prefix_len/suffix_len must be > 0, got {prefix_len}/{suffix_len}")
    total = prefix_len + suffix_len
    family_span = prefix_len + n_suffixes * suffix_len

    text = base_text or _FALLBACK_TEXT
    all_tokens = _tile_corpus(tokenize_fn(text), family_span * n_families)
    corpus_len = len(all_tokens)
    required = family_span * n_families
    if corpus_len < required:
        raise ValueError(
            f"cannot place {n_families} distinct families (span={family_span} "
            f"tokens each) in a {corpus_len}-token corpus"
        )

    # Rejection sampling becomes unreliable when the requested families fill
    # most of the corpus (for example 12 x 32K in 417,921 tokens). Allocate
    # guaranteed non-overlapping slots and distribute the available slack as
    # deterministic random gaps. Uneven gaps matter for repetitive corpora:
    # equally spaced physical spans can otherwise contain identical prefixes.
    rng = random.Random(seed)
    slack = corpus_len - required
    offsets = []
    for _attempt in range(max(1, n_families * 20)):
        if slack:
            # Stars-and-bars: split `slack` unused tokens into n+1 gaps
            # without a token-by-token loop.
            bars = sorted(rng.sample(range(slack + n_families), n_families))
            gaps = [bars[0]]
            gaps.extend(bars[i] - bars[i - 1] - 1 for i in range(1, len(bars)))
            gaps.append(slack + n_families - 1 - bars[-1])
        else:
            gaps = [0] * (n_families + 1)

        candidate = []
        cursor = gaps[0]
        for i in range(n_families):
            candidate.append(cursor)
            cursor += family_span + gaps[i + 1]

        head_len = min(prefix_len, DISTINCT_PREFIX_HEAD_TOKENS)
        prefix_heads = {
            tuple(all_tokens[off : off + head_len]) for off in candidate
        }
        if len(prefix_heads) == n_families:
            offsets = candidate
            rng.shuffle(offsets)
            break
    if not offsets:
        raise ValueError(
            f"cannot construct {n_families} cache-page-distinct, non-overlapping "
            f"prefix families from the {corpus_len}-token corpus "
            f"(distinct head tokens={min(prefix_len, DISTINCT_PREFIX_HEAD_TOKENS)})"
        )

    families: list[PrefixFamily] = []
    for i, off in enumerate(offsets):
        prefix = all_tokens[off : off + prefix_len]
        suffix_pool = [
            all_tokens[off + prefix_len + s * suffix_len : off + total + s * suffix_len]
            for s in range(n_suffixes)
        ]
        families.append(PrefixFamily(family_id=i, prefix_ids=prefix, suffix_pool=suffix_pool))
    logger.info(
        "Built %d prefix families (prefix=%d, suffix=%d, suffixes/family=%d)",
        n_families, prefix_len, suffix_len, n_suffixes,
    )
    return families


def schedule_families(n_requests: int, n_families: int) -> list[int]:
    """Deterministic family schedule for a load window.

    Round-robin cycle ``0..n_families-1``: request ``i`` uses family
    ``i % n_families``, so consecutive arrivals touch distinct prefixes and
    each family recurs exactly ``n_requests // n_families`` times (plus the
    remainder).  Deterministic by construction (no RNG).
    """
    if n_families <= 0:
        raise ValueError(f"n_families must be > 0, got {n_families}")
    return [i % n_families for i in range(n_requests)]


class PrefixPool:
    """Deterministic prompt pool over a list of prefix families."""

    def __init__(self, families: list[PrefixFamily], seed: int = 42):
        if not families:
            raise ValueError("PrefixPool requires at least one family")
        self.families = list(families)
        self.seed = seed

    @property
    def n_families(self) -> int:
        return len(self.families)

    @property
    def prefix_ids(self) -> list[list[int]]:
        return [f.prefix_ids for f in self.families]

    def prompt_for(self, request_index: int) -> list[int]:
        """Prompt for request ``request_index`` (deterministic schedule)."""
        fam = self.families[request_index % self.n_families]
        suffix = fam.suffix_pool[(request_index // self.n_families) % len(fam.suffix_pool)]
        return fam.prefix_ids + suffix

    def prompts(self, n_requests: int) -> list[list[int]]:
        return [self.prompt_for(i) for i in range(n_requests)]

    def to_metadata(self) -> dict:
        return {
            "prefix_pool_size": self.n_families,
            "prefix_tokens_per_family": self.families[0].prefix_len,
            "suffix_tokens_per_family": len(self.families[0].suffix_pool[0])
            if self.families[0].suffix_pool
            else 0,
        }


# ---------------------------------------------------------------------------
# Per-request tier classification and load-window aggregation (pure)
# ---------------------------------------------------------------------------


def classify_tier(details: Optional[dict]) -> str:
    """Classify one request's ``cached_tokens_details``.

    Returns ``"device"`` (only device tokens), ``"host"`` (only host
    tokens), ``"mixed"`` (both tiers), or ``"none"`` (no cache match or
    breakdown absent).
    """
    if not isinstance(details, dict):
        return "none"
    device = int(details.get("device", 0) or 0)
    host = int(details.get("host", 0) or 0)
    if device > 0 and host > 0:
        return "mixed"
    if device > 0:
        return "device"
    if host > 0:
        return "host"
    return "none"


def aggregate_tier_hits(records: list[dict]) -> dict:
    """Aggregate per-request tier breakdowns over a load window.

    ``records`` are dicts carrying ``cached_tokens_details`` (e.g.
    ``HttpRequestRecord.to_dict()``).  Returns counts plus device/host
    ratios over requests with any hit evidence (``None`` when no request
    reported a hit — missing evidence is unsupported, not zero).
    """
    counts = {"device": 0, "host": 0, "mixed": 0, "none": 0}
    for r in records:
        counts[classify_tier(r.get("cached_tokens_details"))] += 1
    total = counts["device"] + counts["host"] + counts["mixed"]
    return {
        **counts,
        "evidence_source": "per_request_metadata",
        "total_hits": total,
        "device_ratio": round(counts["device"] / total, 4) if total else None,
        "host_ratio": round(counts["host"] / total, 4) if total else None,
    }


def aggregate_tier_metric_delta(delta) -> dict:
    """Aggregate public per-tier token counters for one isolated load window.

    This is the authoritative Exp3 fallback for runtimes such as pinned
    Qwen3.5 hybrid whose native ``/generate`` responses omit per-request
    tier metadata. Both peer-tier counters must be present; missing is never
    treated as zero.
    """
    device = delta.get("prefill_device_hit_tokens")
    host = delta.get("prefill_host_hit_tokens")
    load_back = delta.get("load_back_tokens_total")
    supported = device is not None and host is not None
    total = device + host if supported else None

    # With concurrent HiCache requests, SGLang may start H->D restoration
    # while requests wait for a prefill slot. By the time a later request is
    # admitted, its restored prefix is reported as `device_hit` even though it
    # originated in host L2 at window start. The public load-back counter is
    # the authoritative residency-transition evidence for this case. Cap it
    # at total cache-hit tokens because hybrid Qwen also transfers a handful
    # of auxiliary Mamba state slots.
    restored = min(max(load_back or 0, 0), total) if total is not None else None
    host_origin = max(host, restored) if supported else None
    device_origin = max(0, total - host_origin) if supported else None
    return {
        "evidence_source": "window_metric_delta_with_load_back",
        "device_tokens_reported_at_admission": device,
        "host_tokens_reported_at_admission": host,
        "load_back_tokens": load_back,
        "device_tokens": device_origin,
        "host_tokens": host_origin,
        "total_hit_tokens": total,
        "total_hits": total,
        "device_ratio": round(device_origin / total, 4) if total else None,
        "host_ratio": round(host_origin / total, 4) if total else None,
        "metrics_supported": supported,
    }


def dominance_decision(
    agg: dict,
    requested_residency: str,
    threshold: float = HIT_DOMINANCE_THRESHOLD,
) -> tuple[bool, str, str]:
    """Decide whether a load point may be reported under the requested tier.

    Returns ``(ok, label, reason)``.  ``label`` is ``"unsupported"`` when
    the requested residency does not dominate by ``threshold`` — a
    mostly-GPU load point is never silently called ``cpu_hit``.  ``ok`` is
    always ``True`` for ``recompute`` (no cache-hit dominance requirement).
    """
    total_value = agg.get("total_hit_tokens", agg.get("total_hits", 0))
    total = float(total_value or 0)
    device_ratio = agg.get("device_ratio")
    host_ratio = agg.get("host_ratio")

    if requested_residency == "recompute":
        # Recompute has no cache-hit expectation: no dominance requirement,
        # and a load window with zero hits is exactly the expected result.
        return (True, "recompute", "recompute: no cache-hit dominance requirement")

    if not agg.get("metrics_supported", True):
        return (
            False,
            "unsupported",
            "window tier counters missing (both device_hit and host_hit are "
            "required); missing evidence is unsupported, not zero",
        )

    if total == 0:
        return (
            False,
            "unsupported",
            "no tier-hit evidence in the load window; missing evidence is "
            "unsupported, not zero",
        )

    if requested_residency == "cpu_hit":
        if host_ratio is not None and host_ratio >= threshold:
            return (
                True,
                "cpu_hit",
                f"host hits dominate (host_ratio={host_ratio:.2%} >= "
                f"{threshold:.0%}, device_ratio={device_ratio:.2%})",
            )
        return (
            False,
            "unsupported",
            f"requested cpu_hit but host hits do not dominate "
            f"(host_ratio={host_ratio:.2%} < {threshold:.0%}, "
            f"device_ratio={device_ratio:.2%}); a mostly-GPU load point "
            f"must not be labelled cpu_hit",
        )

    if requested_residency == "gpu_hit":
        if device_ratio is not None and device_ratio >= threshold:
            return (
                True,
                "gpu_hit",
                f"device hits dominate (device_ratio={device_ratio:.2%} >= "
                f"{threshold:.0%}, host_ratio={host_ratio:.2%})",
            )
        return (
            False,
            "unsupported",
            f"requested gpu_hit but device hits do not dominate "
            f"(device_ratio={device_ratio:.2%} < {threshold:.0%}, "
            f"host_ratio={host_ratio:.2%})",
        )

    return (
        False,
        "unsupported",
        f"unknown residency mode {requested_residency!r}",
    )
