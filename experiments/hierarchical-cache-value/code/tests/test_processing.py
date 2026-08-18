"""Tests for hcv.analysis: derived metrics, aggregation, filtering, locality."""

from __future__ import annotations

from types import SimpleNamespace


def test_pressure_eviction_uses_direct_probe_not_write_through_or_gauges():
    from hcv.run_exp2 import eviction_evidence
    from hcv.hierarchy import PROBE_GPU_HIT, PROBE_GPU_ONLY_EVICTION, ProbeResult

    write_through_only = SimpleNamespace(
        total_delta={"deltas": {"hicache_backup_tokens_total": 1000.0}},
        end_snapshot={"kv_evictable_tokens": 1000.0, "kv_available_tokens": 5000.0},
    )
    gpu_hit = ProbeResult(PROBE_GPU_HIT, 512, True, "ok")
    evidence = eviction_evidence(write_through_only, "gpu_only", gpu_hit)
    assert evidence["eviction_observed"] is False
    assert evidence["gpu_residency_preserved"] is True

    saturated = SimpleNamespace(
        total_delta={"deltas": {"hicache_backup_tokens_total": None}},
        end_snapshot={"kv_evictable_tokens": 1000.0, "kv_available_tokens": 0.0},
    )
    recompute = ProbeResult(PROBE_GPU_ONLY_EVICTION, 512, True, "ok")
    evidence = eviction_evidence(saturated, "gpu_only", recompute)
    assert evidence["eviction_observed"] is True
    assert evidence["gpu_residency_preserved"] is False

import json
import os
import tempfile

from hcv.analysis import (
    aggregate_over_reps,
    derive_pair,
    gate_status_of,
    process_exp1,
    process_exp2,
    process_exp3,
    run_validity,
    write_csv,
    write_processed,
)
from hcv.config import ARCH_GPU_ONLY, ARCH_HIERARCHICAL, GATE_FULL


def test_derive_pair_recomputation_reduction():
    gpu = {"recomputed_tokens": 1000.0, "gpu_hit_tokens": 500.0,
           "cpu_hit_tokens": 0.0, "ttft_p50_ms": 120.0,
           "throughput_req_per_s": 8.0}
    hier = {"recomputed_tokens": 300.0, "gpu_hit_tokens": 400.0,
            "cpu_hit_tokens": 200.0, "ttft_p50_ms": 90.0,
            "throughput_req_per_s": 10.0}
    d = derive_pair(gpu, hier)
    assert d["recomputation_reduction"] == 700.0
    assert d["relative_recomputation_reduction"] == 0.7
    assert abs(d["relative_ttft_improvement"] - 0.25) < 1e-9
    assert abs(d["throughput_gain"] - 0.25) < 1e-9
    assert abs(d["cpu_tier_contribution"] - 200.0 / 600.0) < 1e-9
    assert d["gpu_hit_tokens_gpu_only"] == 500.0
    assert d["gpu_hit_tokens_hierarchical"] == 400.0


def test_derive_pair_division_by_zero_safe():
    d = derive_pair({"recomputed_tokens": 0.0, "ttft_p50_ms": 0.0,
                     "throughput_req_per_s": 0.0},
                    {"recomputed_tokens": 0.0})
    assert d["relative_recomputation_reduction"] is None
    assert "throughput_gain" not in d  # not computed when gpu throughput 0


def test_aggregate_over_reps():
    records = [{"x": 1.0}, {"x": 3.0}, {"x": 5.0}]
    agg = aggregate_over_reps(records, "x")
    assert agg["mean"] == 3.0
    assert agg["n"] == 3
    assert agg["std"] >= 0.0
    # None values are excluded, empty -> None
    assert aggregate_over_reps([{"x": None}, {"x": None}], "x") == {"mean": None, "std": None, "n": 0}


def test_run_validity_requires_full_gate():
    run = {"metadata": {"validity_status": "valid"}, "validation": {"status": GATE_FULL}}
    assert run_validity(run) == (True, "")
    run2 = {"metadata": {"validity_status": "valid"}, "validation": {"status": "partial"}}
    ok, reason = run_validity(run2)
    assert not ok and "gate" in reason
    run3 = {"metadata": {"validity_status": "invalid"}, "validation": {"status": GATE_FULL}}
    ok, reason = run_validity(run3)
    assert not ok


def test_gate_status_of_missing():
    assert gate_status_of({"validation": {}}) == "missing"


def _make_run(results_root: str, experiment: str, tag: str, md: dict,
              validation: dict, measurements: list[dict]) -> None:
    base = os.path.join(results_root, experiment, tag)
    os.makedirs(os.path.join(base, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base, "results"), exist_ok=True)
    with open(os.path.join(base, "metadata.json"), "w") as fh:
        json.dump(md, fh)
    with open(os.path.join(base, "validation.json"), "w") as fh:
        json.dump(validation, fh)
    with open(os.path.join(base, "raw", "measurements.jsonl"), "w") as fh:
        for rec in measurements:
            fh.write(json.dumps(rec) + "\n")


def test_process_exp1_filters_invalid_and_builds_cells():
    with tempfile.TemporaryDirectory() as tmp:
        valid_md = {"architecture": ARCH_GPU_ONLY, "cache_initial_state": "cold",
                    "validity_status": "valid"}
        valid_val = {"status": GATE_FULL}
        meas = [{"kind": "repetition", "validity": {"valid": True},
                 "summary": {"gpu_hit_tokens": 10.0, "recomputed_tokens": 5.0,
                             "ttft_p50_ms": 1.0, "throughput_req_per_s": 2.0}}]
        _make_run(tmp, "exp1", "run-a", valid_md, valid_val, meas)
        _make_run(tmp, "exp1", "run-b", {**valid_md, "validity_status": "invalid"},
                  valid_val, meas)
        out = process_exp1(tmp)
        assert "gpu_only/cold" in out["cells"]
        cell = out["cells"]["gpu_only/cold"]
        assert cell["runs"][0]["run_tag"] == "run-a"
        assert cell["runs"][0]["aggregate"]["gpu_hit_tokens"]["mean"] == 10.0
        assert any(e["run_tag"] == "run-b" for e in out["excluded"])


def test_process_exp2_builds_curves():
    with tempfile.TemporaryDirectory() as tmp:
        for arch in (ARCH_GPU_ONLY, ARCH_HIERARCHICAL):
            md = {"architecture": arch, "pressure_label": "Medium",
                  "validity_status": "valid", "gpu_cache_budget": 0.65}
            val = {"status": GATE_FULL}
            meas = [{"kind": "pressure_point", "validity": {"valid": True},
                     "eviction_evidence": {"eviction_observed": True},
                     "summary": {"gpu_hit_tokens": 100.0, "cpu_hit_tokens": 50.0,
                                 "recomputed_tokens": 200.0, "ttft_p50_ms": 80.0,
                                 "throughput_req_per_s": 5.0}}]
            _make_run(tmp, "exp2", f"run-{arch}", md, val, meas)
        out = process_exp2(tmp)
        assert "Medium" in out["curves"]
        curve = out["curves"]["Medium"]
        assert curve["recomputed_tokens_gpu_only"] == 200.0
        assert curve["recomputed_tokens_hierarchical"] == 200.0
        assert curve["recomputation_reduction"] == 0.0
        # cpu tier contribution only when hierarchical has cpu hits
        assert curve["cpu_tier_contribution"] is not None


def test_process_exp3_locality_checked():
    from hcv.workload import build_trace, save_trace

    with tempfile.TemporaryDirectory() as tmp:
        hi = build_trace(seed=1, num_prefix_families=4, family_size=8,
                         prefix_length=64, suffix_length_min=8,
                         suffix_length_max=16, output_length=1,
                         revisit_fraction=0.75, request_count=0)
        lo = build_trace(seed=1, num_prefix_families=4, family_size=8,
                         prefix_length=64, suffix_length_min=8,
                         suffix_length_max=16, output_length=1,
                         revisit_fraction=0.25, request_count=0)
        for arch in (ARCH_GPU_ONLY, ARCH_HIERARCHICAL):
            for frac, trace in ((0.75, hi), (0.25, lo)):
                tag = f"run-{arch}-{frac}"
                base = os.path.join(tmp, "exp3", tag)
                os.makedirs(os.path.join(base, "raw", "traces"), exist_ok=True)
                os.makedirs(os.path.join(base, "results"), exist_ok=True)
                save_trace(os.path.join(base, "raw", "traces", f"trace-{trace.trace_id}.json"), trace)
                md = {"architecture": arch, "configured_revisit_fraction": frac,
                      "validity_status": "valid", "gpu_cache_budget": 0.6,
                      "trace_id": trace.trace_id}
                val = {"status": GATE_FULL}
                meas = [{"kind": "reuse_point", "validity": {"valid": True},
                         "actual_reuse_request_weighted": frac,
                         "reuse_distance": {"min": 0, "max": 10},
                         "summary": {"gpu_hit_tokens": 10.0, "cpu_hit_tokens": 1.0,
                                     "recomputed_tokens": 20.0,
                                     "throughput_req_per_s": 3.0}}]
                _make_run(tmp, "exp3", tag, md, val, meas)
        out = process_exp3(tmp)
        assert out["locality"]["checked"] is True
        assert all(p["ok"] for p in out["locality"]["pairs"])
        assert "0.25" in out["curves"] and "0.75" in out["curves"]


def test_writers_deterministic_and_separate():
    with tempfile.TemporaryDirectory() as tmp:
        p = write_processed(tmp, "x", {"a": [1, 2], "b": {"c": 3}})
        with open(p) as fh:
            assert json.load(fh) == {"a": [1, 2], "b": {"c": 3}}
        rows = [{"run_tag": "r1", "v": 1.5}, {"run_tag": "r2", "v": 2.5}]
        csv_path = write_csv(tmp, "t", rows)
        with open(csv_path) as fh:
            text = fh.read()
        assert text.splitlines()[0] == "run_tag,v"
        assert "r2" in text
