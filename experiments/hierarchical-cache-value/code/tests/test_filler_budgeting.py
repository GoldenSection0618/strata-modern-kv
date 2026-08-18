"""Tests for hcv.filler: capacity source, fallback formula, early-stop rule."""

from __future__ import annotations

import os

from hcv.filler import (
    build_fixed_filler_plan,
    build_filler_plan,
    filler_pressure_tokens,
    l1_saturation_evidence,
    observed_l1_capacity,
    should_stop_filling,
)
from hcv.metrics import CacheStats


def test_shared_calibration_path_matches_sweep_consumer():
    from hcv.calibrate import shared_calibration_path

    root = os.path.join("tmp", "results")
    assert shared_calibration_path(root) == os.path.join(
        root, "exp2", "calibration", "calibration.json"
    )


def test_pressure_budget_places_medium_near_calibrated_floor():
    from hcv.calibrate import pressure_budget

    calib = {"min_valid_fraction": 0.55}
    assert pressure_budget(calib, "Low") == 0.85
    assert pressure_budget(calib, "Medium") == 0.65
    assert pressure_budget(calib, "High") == 0.60


def test_fixed_pressure_plan_is_capacity_independent_and_exact():
    plan = build_fixed_filler_plan(81920, filler_prefix_length=512)
    assert plan.target_pressure_tokens == 81920
    assert plan.best_effort_cap == 81920
    assert plan.filler_unique_count == 160
    assert plan.observed_capacity is None
    assert plan.capacity_source == "fixed_cross_budget_working_set"


def test_fallback_formula():
    ctx = 4096
    protected = 512
    target = filler_pressure_tokens(ctx, protected, observed_capacity=None)
    expected = max(6 * ctx, 262144) + protected + 4096
    assert target == expected
    # large context drives the 6x term above the 262144 floor
    target_big = filler_pressure_tokens(100000, 0, None)
    assert target_big == 6 * 100000 + 4096


def test_observed_capacity_preferred():
    stats = CacheStats(max_total_num_tokens=50000.0)
    cap = observed_l1_capacity(stats)
    assert cap == 50000.0
    target = filler_pressure_tokens(4096, 0, observed_capacity=cap)
    assert target == 50000.0


def test_observed_capacity_kv_sum_fallback():
    stats = CacheStats(kv_used_tokens=1000.0, kv_available_tokens=9000.0)
    assert observed_l1_capacity(stats) == 10000.0


def test_no_observed_capacity_is_none():
    stats = CacheStats()
    assert observed_l1_capacity(stats) is None


def test_l1_saturation_evidence_never_uses_host():
    # host gauges may be huge while L1 is unsaturated: saturation must
    # depend only on kv_evictable/kv_available/kv_used
    stats = CacheStats(kv_evictable_tokens=0.0, kv_available_tokens=5000.0,
                       hicache_host_used_tokens=999999.0)
    ev = l1_saturation_evidence(stats)
    assert ev["l1_saturated"] is False  # host occupancy must not matter
    stats2 = CacheStats(kv_evictable_tokens=10.0, kv_available_tokens=0.0)
    assert l1_saturation_evidence(stats2)["l1_saturated"] is True


def test_early_stop_requires_l1_saturation():
    sat_ok = {"l1_saturated": True}
    sat_no = {"l1_saturated": False}
    stop, reason = should_stop_filling(10000.0, 10000.0, sat_no, 20000.0)
    assert not stop
    assert "waiting_for_l1_saturation" in reason
    stop, reason = should_stop_filling(10000.0, 10000.0, sat_ok, 20000.0)
    assert stop and reason == "target_reached_and_l1_saturated"


def test_early_stop_best_effort_cap():
    sat_no = {"l1_saturated": False}
    stop, reason = should_stop_filling(21000.0, 10000.0, sat_no, 20000.0)
    assert stop and "best_effort_cap" in reason


def test_early_stop_never_consults_host_occupancy():
    # even with the host pool full, the stop decision must be identical
    # when L1 signals are the same (host gauges are not inputs)
    import inspect

    src = inspect.getsource(should_stop_filling)
    assert "hicache_host" not in src
    assert "host_used" not in src


def test_build_filler_plan_records_source():
    plan = build_filler_plan(4096, 512, observed_capacity=None)
    assert plan.capacity_source == "fallback_formula"
    assert plan.target_pressure_tokens == filler_pressure_tokens(4096, 512, None)
    plan2 = build_filler_plan(4096, 512, observed_capacity=70000.0,
                              capacity_source="observed_max_total_num_tokens")
    assert plan2.capacity_source == "observed_max_total_num_tokens"
    assert plan2.target_pressure_tokens == 70000.0
    assert plan2.filler_unique_count >= 70000 // 512
