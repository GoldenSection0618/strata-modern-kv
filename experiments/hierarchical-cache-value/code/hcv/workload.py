"""Deterministic text-only trace construction and reuse summaries.

Design
------
* Every trace is fully reproducible from a configuration subset (seed +
  structural parameters) and carries a readable stable ``trace_id``.
* Requests are grouped into deterministic prefix families
  (``fam0001``...).  Within a family, the first request installs the
  shared prefix; later eligible slots either revisit that prefix or use
  a *matched unique prefix* with the same length at the same position.
* Eligible revisit slots are FIXED across reuse levels: every position
  after a family's first request is eligible.  A deterministic per-slot
  priority decides whether a slot revisits at a given fraction, so
  lowering the revisit fraction replaces revisits with matched unique
  prefixes without reordering requests or changing hotspot structure.
* Requests are sent to the server as exact ``input_ids`` (deterministic
  pseudo-token IDs from a local encoder, text-only workload).  Prefix
  length is therefore exact by construction and never depends on a
  tokenizer.
* Reuse distance is defined as the number of intervening requests
  between a revisit slot and its family's first occurrence.

Reuse summaries record request-weighted and token/state-volume-weighted
reuse, unique-prefix counts, and reuse-distance statistics so that
trace validity can be checked independently of runtime behavior.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Optional

#: Pseudo-token ID range (safe below typical hybrid-model vocab sizes).
_TOKEN_ID_MIN = 100
_TOKEN_ID_MAX = 999

_CANONICAL_FIELDS = (
    "seed", "num_prefix_families", "family_size", "prefix_length",
    "suffix_length_min", "suffix_length_max", "output_length",
    "revisit_fraction", "request_count",
)


@dataclass
class TraceRecord:
    """One request in a trace."""

    idx: int
    family_id: str
    is_revisit: bool
    revisit_of_idx: Optional[int]  # first occurrence idx of the family
    reuse_distance: Optional[int]  # intervening requests
    prefix_tokens: int
    suffix_tokens: int
    output_tokens: int
    input_ids: list[int]
    prefix_key: str

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "family_id": self.family_id,
            "is_revisit": self.is_revisit,
            "revisit_of_idx": self.revisit_of_idx,
            "reuse_distance": self.reuse_distance,
            "prefix_tokens": self.prefix_tokens,
            "suffix_tokens": self.suffix_tokens,
            "output_tokens": self.output_tokens,
            "input_ids": self.input_ids,
            "prefix_key": self.prefix_key,
        }


@dataclass
class Trace:
    """A complete request trace."""

    trace_id: str
    config: dict
    records: list[TraceRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def record(self, idx: int) -> TraceRecord:
        return self.records[idx]

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "config": self.config,
            "records": [r.to_dict() for r in self.records],
        }


# ---------------------------------------------------------------------------
# Deterministic pseudo-token encoder
# ---------------------------------------------------------------------------


def _ids(seed: int, length: int) -> list[int]:
    """Deterministic pseudo-token ID list of ``length`` tokens."""
    rng = random.Random(seed)
    lo, hi = _TOKEN_ID_MIN, _TOKEN_ID_MAX
    return [rng.randint(lo, hi) for _ in range(length)]


# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------


def canonical_config(seed: int, num_families: int, family_size: int,
                     prefix_length: int, suffix_min: int, suffix_max: int,
                     output_length: int, revisit_fraction: float,
                     request_count: int) -> dict:
    """Canonical config dict (the input to trace_id)."""
    return {
        "seed": int(seed),
        "num_prefix_families": int(num_families),
        "family_size": int(family_size),
        "prefix_length": int(prefix_length),
        "suffix_length_min": int(suffix_min),
        "suffix_length_max": int(suffix_max),
        "output_length": int(output_length),
        "revisit_fraction": float(revisit_fraction),
        "request_count": int(request_count),
    }


def compute_trace_id(seed: int, num_prefix_families: int, family_size: int,
                     prefix_length: int, suffix_length_min: int, suffix_length_max: int,
                     output_length: int, revisit_fraction: float,
                     request_count: int) -> str:
    """Readable stable trace identifier derived from the canonical config."""
    cfg = canonical_config(seed, num_prefix_families, family_size, prefix_length,
                           suffix_length_min, suffix_length_max, output_length,
                           revisit_fraction, request_count)
    reuse = str(cfg["revisit_fraction"]).replace(".", "p")
    return (
        f"s{cfg['seed']}-f{cfg['num_prefix_families']}x{cfg['family_size']}"
        f"-p{cfg['prefix_length']}-s{cfg['suffix_length_min']}_{cfg['suffix_length_max']}"
        f"-o{cfg['output_length']}-r{reuse}-n{cfg['request_count']}"
    )


def _family_prefix(family_id: str, prefix_length: int, seed: int) -> list[int]:
    """Deterministic shared prefix for one family."""
    fam_seed = int(family_id[3:]) + seed * 1000003
    return _ids(fam_seed, prefix_length)


def _unique_prefix(slot_idx: int, prefix_length: int, seed: int) -> list[int]:
    """Deterministic matched unique prefix (never revisited by design)."""
    return _ids(seed * 7919 + slot_idx * 104729 + 13, prefix_length)


def _suffix(slot_idx: int, family_id: str, length: int, seed: int) -> list[int]:
    rng = random.Random(seed * 31 + slot_idx * 65537 + int(family_id[3:]))
    lo, hi = _TOKEN_ID_MIN, _TOKEN_ID_MAX
    return [rng.randint(lo, hi) for _ in range(length)]


def build_trace(
    seed: int,
    num_prefix_families: int,
    family_size: int,
    prefix_length: int,
    suffix_length_min: int,
    suffix_length_max: int,
    output_length: int,
    revisit_fraction: float,
    request_count: int = 0,
) -> Trace:
    """Build a deterministic trace.

    ``request_count`` truncates the generated request list (0 = full).
    Every slot after a family's first request is an eligible revisit
    slot; a slot revisits iff its deterministic priority <
    ``revisit_fraction``.  Replaced slots use a matched unique prefix.
    """
    cfg = canonical_config(seed, num_prefix_families, family_size, prefix_length,
                           suffix_length_min, suffix_length_max, output_length,
                           revisit_fraction, request_count)
    trace_id = compute_trace_id(**cfg)

    # Deterministic per-slot revisit priority (fixed across reuse levels).
    prio_rng = random.Random(seed * 7 + 12345)

    family_prefixes: dict[str, list[int]] = {}
    first_idx: dict[str, int] = {}
    records: list[TraceRecord] = []
    idx = 0

    for r in range(family_size):
        for fam in range(num_prefix_families):
            if request_count and idx >= request_count:
                break
            family_id = f"fam{fam + 1:04d}"
            if family_id not in family_prefixes:
                family_prefixes[family_id] = _family_prefix(family_id, prefix_length, seed)
                first_idx[family_id] = idx

            is_eligible = r > 0
            is_revisit = False
            revisit_of = None
            distance = None
            if is_eligible:
                priority = prio_rng.random()
                is_revisit = priority < revisit_fraction
                if is_revisit:
                    revisit_of = first_idx[family_id]
                    distance = idx - revisit_of - 1

            if is_revisit:
                prefix_ids = family_prefixes[family_id]
            elif r == 0:
                prefix_ids = family_prefixes[family_id]
            else:
                prefix_ids = _unique_prefix(idx, prefix_length, seed)

            suffix_len = suffix_length_min
            if suffix_length_max > suffix_length_min:
                suffix_len = suffix_length_min + prio_rng.randint(
                    0, suffix_length_max - suffix_length_min
                )
            suffix_ids = _suffix(idx, family_id, suffix_len, seed)
            input_ids = list(prefix_ids) + suffix_ids
            prefix_key = family_id if is_revisit or r == 0 else f"unique:{idx}"

            records.append(TraceRecord(
                idx=idx,
                family_id=family_id,
                is_revisit=is_revisit,
                revisit_of_idx=revisit_of,
                reuse_distance=distance,
                prefix_tokens=len(prefix_ids),
                suffix_tokens=len(suffix_ids),
                output_tokens=int(output_length),
                input_ids=input_ids,
                prefix_key=prefix_key,
            ))
            idx += 1
        if request_count and idx >= request_count:
            break

    trace = Trace(trace_id=trace_id, config=cfg, records=records)
    validate_trace(trace)
    return trace


def build_trace_from_config(cfg: dict) -> Trace:
    """Build a trace from a canonical config dict (round-trip safe)."""
    return build_trace(
        seed=cfg["seed"],
        num_prefix_families=cfg["num_prefix_families"],
        family_size=cfg["family_size"],
        prefix_length=cfg["prefix_length"],
        suffix_length_min=cfg["suffix_length_min"],
        suffix_length_max=cfg["suffix_length_max"],
        output_length=cfg["output_length"],
        revisit_fraction=cfg["revisit_fraction"],
        request_count=cfg["request_count"],
    )


# ---------------------------------------------------------------------------
# Invariants and summaries
# ---------------------------------------------------------------------------


def validate_trace(trace: Trace) -> None:
    """Enforce trace invariants (raises AssertionError on violation)."""
    assert trace.records, "trace must not be empty"
    seen_prefixes: dict[str, int] = {}
    first_idx: dict[str, int] = {}
    prev_idx = -1
    for rec in trace.records:
        assert rec.idx == prev_idx + 1, "idx must be contiguous"
        prev_idx = rec.idx
        assert rec.prefix_tokens > 0 and rec.suffix_tokens >= 0
        assert len(rec.input_ids) == rec.prefix_tokens + rec.suffix_tokens

        if not rec.is_revisit:
            # unique prefixes must never repeat anywhere in the trace
            key = rec.prefix_key
            if key in seen_prefixes:
                raise AssertionError(
                    f"prefix {key[:8]} reused at idx {rec.idx} and {seen_prefixes[key]}"
                )
            seen_prefixes[key] = rec.idx
        else:
            assert rec.revisit_of_idx is not None
            target = first_idx.get(rec.family_id)
            assert target == rec.revisit_of_idx, "revisit must target family first occurrence"
            assert rec.reuse_distance is not None and rec.reuse_distance >= 0
        first_idx.setdefault(rec.family_id, rec.idx)


def reuse_summary(trace: Trace) -> dict:
    """Request-weighted and token-weighted reuse + distance statistics."""
    total = len(trace.records)
    revisits = [r for r in trace.records if r.is_revisit]
    n_rev = len(revisits)

    prefix_tokens_all = sum(r.prefix_tokens for r in trace.records)
    prefix_tokens_rev = sum(r.prefix_tokens for r in revisits)
    suffix_tokens_all = sum(r.suffix_tokens for r in trace.records)
    unique_prefixes = len({r.prefix_key for r in trace.records})

    distances = sorted(r.reuse_distance for r in revisits if r.reuse_distance is not None)
    dist = {}
    if distances:
        dist = {
            "min": distances[0],
            "max": distances[-1],
            "mean": round(sum(distances) / len(distances), 3),
            "p50": distances[len(distances) // 2],
        }

    return {
        "request_count": total,
        "revisit_requests": n_rev,
        "revisit_fraction_request_weighted": round(n_rev / total, 4) if total else 0.0,
        "revisit_fraction_token_weighted": (
            round(prefix_tokens_rev / prefix_tokens_all, 4) if prefix_tokens_all else 0.0
        ),
        "unique_prefix_count": unique_prefixes,
        "total_prefix_tokens": prefix_tokens_all,
        "total_suffix_tokens": suffix_tokens_all,
        "reuse_distance": dist,
        "families": sorted({r.family_id for r in trace.records}),
    }


# ---------------------------------------------------------------------------
# Locality-unchanged validation (Exp3)
# ---------------------------------------------------------------------------


def validate_locality_unchanged(hi_trace: Trace, lo_trace: Trace) -> None:
    """Assert two reuse-level traces share ordering/hotspot structure.

    Request count, per-slot family assignment, suffix token counts and
    the reuse distances of slots that remain revisits must be identical.
    Only the revisit decision (and therefore the prefix identity) may
    differ.
    """
    assert len(hi_trace.records) == len(lo_trace.records), "request count differs"
    hi_rev = {r.idx for r in hi_trace.records if r.is_revisit}
    lo_rev = {r.idx for r in lo_trace.records if r.is_revisit}
    assert lo_rev <= hi_rev, "lower-reuse trace revisits slots the higher one does not"
    common = hi_rev & lo_rev
    for idx in sorted(common):
        a = hi_trace.record(idx)
        b = lo_trace.record(idx)
        assert a.family_id == b.family_id, f"family changed at slot {idx}"
        assert a.reuse_distance == b.reuse_distance, f"distance changed at slot {idx}"
        assert a.suffix_tokens == b.suffix_tokens, f"suffix changed at slot {idx}"
    # ordering/hotspot structure: family assignment identical at every slot
    for a, b in zip(hi_trace.records, lo_trace.records):
        assert a.family_id == b.family_id, "family assignment changed"
        assert a.suffix_tokens == b.suffix_tokens, "suffix structure changed"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_trace(path: str, trace: Trace) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(trace.to_dict(), fh, sort_keys=True)


def load_trace(path: str) -> Trace:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    records = [TraceRecord(**r) for r in data["records"]]
    trace = Trace(trace_id=data["trace_id"], config=data["config"], records=records)
    validate_trace(trace)
    return trace
