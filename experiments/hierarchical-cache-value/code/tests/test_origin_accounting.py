"""Tests for hcv.load_driver: concurrent window origin accounting."""

from __future__ import annotations

from hcv.load_driver import (
    ORIGIN_DEVICE,
    ORIGIN_HOST,
    ORIGIN_MIXED,
    ORIGIN_UNKNOWN,
    classify_window_origin,
)


def test_run_load_records_final_partial_window():
    import time
    import hcv.load_driver as driver
    from hcv.metrics import CacheStats
    from hcv.workload import build_trace

    class FakeGenerate:
        ok = True
        error = ""
        status = 200

        def __init__(self, request_id):
            self.request_id = request_id
            self.t_send = time.time()

        def to_dict(self):
            return {"ok": True, "request_id": self.request_id, "ttft_ms": 1.0}

    class FakeClient:
        def __init__(self):
            self.requests = 0

        def generate(self, request_id, input_ids, max_new_tokens):
            self.requests += 1
            return FakeGenerate(request_id)

        def scrape_metrics(self):
            return CacheStats(
                num_requests_total=float(self.requests),
                prefill_input_tokens=float(self.requests * 64),
                prefill_device_hit_tokens=0.0,
                prefill_host_hit_tokens=0.0,
                prefill_storage_hit_tokens=0.0,
                load_back_tokens_total=0.0,
                num_queue_reqs=0.0,
                num_running_reqs=0.0,
            )

    trace = build_trace(
        seed=1, num_prefix_families=1, family_size=2,
        prefix_length=64, suffix_length_min=8, suffix_length_max=8,
        output_length=1, revisit_fraction=1.0, request_count=2,
    )
    original_snapshot = driver.snapshot_from_scrape
    driver.snapshot_from_scrape = lambda value: value
    try:
        result = driver.run_load(
            FakeClient(), trace, trace.records, concurrency=1,
            window_requests=32,
        )
    finally:
        driver.snapshot_from_scrape = original_snapshot

    assert len(result.windows) == 1
    assert result.windows[0].requests_completed == 2


def test_device_resident_window():
    origin, capped = classify_window_origin(
        device_hit_delta=1000.0, host_hit_delta=0.0, storage_hit_delta=0.0,
        load_back_delta=0.0, tier_hits_total=1000.0)
    assert origin == ORIGIN_DEVICE
    assert capped == 0.0


def test_host_restored_window():
    origin, capped = classify_window_origin(
        device_hit_delta=100.0, host_hit_delta=900.0, storage_hit_delta=0.0,
        load_back_delta=950.0, tier_hits_total=1000.0)
    assert origin == ORIGIN_HOST
    # load_back capped at tier-hit volume (no double counting)
    assert capped == 950.0


def test_restored_while_queued_is_host_not_device():
    # admission-time device hits can represent prefixes restored while
    # queued: large load_back + modest device hits -> host origin
    origin, capped = classify_window_origin(
        device_hit_delta=400.0, host_hit_delta=100.0, storage_hit_delta=0.0,
        load_back_delta=500.0, tier_hits_total=500.0)
    assert origin == ORIGIN_HOST
    assert capped == 500.0
    # naive device-first reading would have said device_resident (400>100)


def test_mixed_window():
    origin, capped = classify_window_origin(
        device_hit_delta=700.0, host_hit_delta=200.0, storage_hit_delta=0.0,
        load_back_delta=100.0, tier_hits_total=900.0)
    assert origin == ORIGIN_MIXED
    assert capped == 100.0


def test_unknown_when_no_tier_hits():
    origin, capped = classify_window_origin(
        device_hit_delta=0.0, host_hit_delta=0.0, storage_hit_delta=0.0,
        load_back_delta=0.0, tier_hits_total=0.0)
    assert origin == ORIGIN_UNKNOWN


def test_cap_prevents_double_counting():
    # load_back exceeds total tier hits: capped at the total
    origin, capped = classify_window_origin(
        device_hit_delta=100.0, host_hit_delta=200.0, storage_hit_delta=0.0,
        load_back_delta=2000.0, tier_hits_total=300.0)
    assert capped == 300.0
    # 200 host + 300 capped = 500 > 300 total: share clamps to host-origin
    assert origin == ORIGIN_HOST


def test_missing_metrics_never_treated_as_zero_hits():
    origin, capped = classify_window_origin(
        device_hit_delta=None, host_hit_delta=None, storage_hit_delta=None,
        load_back_delta=None, tier_hits_total=None)
    assert origin == ORIGIN_UNKNOWN
    assert capped is None


def test_missing_load_back_is_unknown_not_zero():
    origin, capped = classify_window_origin(
        device_hit_delta=1000.0, host_hit_delta=0.0, storage_hit_delta=0.0,
        load_back_delta=None, tier_hits_total=1000.0)
    assert origin == ORIGIN_UNKNOWN
    assert capped is None
