"""Tests for hcv.workload: determinism, invariants, reuse summaries, locality."""

from __future__ import annotations

import json
import tempfile

from hcv.workload import (
    TraceRecord,
    build_trace,
    build_trace_from_config,
    compute_trace_id,
    load_trace,
    reuse_summary,
    save_trace,
    validate_locality_unchanged,
    validate_trace,
)

KW = dict(seed=42, num_prefix_families=8, family_size=20, prefix_length=512,
          suffix_length_min=16, suffix_length_max=64, output_length=1,
          revisit_fraction=0.5, request_count=0)


def test_determinism_same_config_same_trace():
    a = build_trace(**KW)
    b = build_trace(**KW)
    assert a.trace_id == b.trace_id
    assert [r.input_ids for r in a.records] == [r.input_ids for r in b.records]
    assert [r.is_revisit for r in a.records] == [r.is_revisit for r in b.records]


def test_trace_id_stable():
    cfg = {k: v for k, v in KW.items() if k != "request_count"}
    cfg["request_count"] = 0
    assert compute_trace_id(**cfg) == build_trace(**KW).trace_id


def test_invariants_hold():
    trace = build_trace(**KW)
    validate_trace(trace)  # raises on violation
    # no forward references
    for rec in trace.records:
        if rec.is_revisit:
            assert rec.revisit_of_idx is not None and rec.revisit_of_idx < rec.idx
            assert rec.reuse_distance is not None and rec.reuse_distance >= 0


def test_unique_prefixes_never_repeat():
    trace = build_trace(**KW)
    seen = {}
    for rec in trace.records:
        if not rec.is_revisit:
            assert rec.prefix_key not in seen, "unique prefix repeated"
            seen[rec.prefix_key] = rec.idx


def test_fixed_eligible_slots_across_reuse_levels():
    hi = build_trace(**{**KW, "revisit_fraction": 0.75})
    lo = build_trace(**{**KW, "revisit_fraction": 0.25})
    assert len(hi.records) == len(lo.records)
    hi_rev = {r.idx for r in hi.records if r.is_revisit}
    lo_rev = {r.idx for r in lo.records if r.is_revisit}
    assert lo_rev <= hi_rev, "lower-reuse trace revisits slots the higher one does not"
    # family assignment + suffix structure identical at every position
    for a, b in zip(hi.records, lo.records):
        assert a.family_id == b.family_id
        assert a.suffix_tokens == b.suffix_tokens


def test_locality_unchanged():
    hi = build_trace(**{**KW, "revisit_fraction": 0.75})
    lo = build_trace(**{**KW, "revisit_fraction": 0.25})
    validate_locality_unchanged(hi, lo)  # raises on drift
    # common revisit slots keep identical distance
    hi_rev = {r.idx for r in hi.records if r.is_revisit}
    lo_rev = {r.idx for r in lo.records if r.is_revisit}
    for idx in sorted(hi_rev & lo_rev):
        assert hi.record(idx).reuse_distance == lo.record(idx).reuse_distance


def test_reuse_summary():
    trace = build_trace(**KW)
    s = reuse_summary(trace)
    assert s["request_count"] == len(trace.records)
    assert 0.0 <= s["revisit_fraction_request_weighted"] <= 1.0
    assert s["revisit_fraction_token_weighted"] <= 1.0
    assert s["unique_prefix_count"] >= 1
    dist = s["reuse_distance"]
    assert dist["min"] >= 0 and dist["max"] >= dist["min"]
    # request-weighted fraction approximately matches the configured 0.5
    assert abs(s["revisit_fraction_request_weighted"] - 0.5) < 0.2


def test_zero_reuse_negative_control():
    trace = build_trace(**{**KW, "revisit_fraction": 0.0})
    assert all(not r.is_revisit for r in trace.records)
    s = reuse_summary(trace)
    assert s["revisit_fraction_request_weighted"] == 0.0
    # every slot got a matched unique prefix with the same prefix length
    assert all(r.prefix_tokens == KW["prefix_length"] for r in trace.records)


def test_request_count_truncation():
    trace = build_trace(**{**KW, "request_count": 100})
    assert len(trace.records) == 100
    assert [r.idx for r in trace.records] == list(range(100))


def test_roundtrip_save_load():
    trace = build_trace(**KW)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/trace.json"
        save_trace(path, trace)
        loaded = load_trace(path)
        assert loaded.trace_id == trace.trace_id
        assert [r.input_ids for r in loaded.records] == [r.input_ids for r in trace.records]


def test_build_from_config_roundtrip():
    trace = build_trace(**KW)
    again = build_trace_from_config(trace.config)
    assert again.trace_id == trace.trace_id
    assert len(again.records) == len(trace.records)


def test_prefix_length_exact():
    trace = build_trace(**KW)
    for rec in trace.records:
        assert rec.prefix_tokens == KW["prefix_length"]
        assert len(rec.input_ids) == rec.prefix_tokens + rec.suffix_tokens
