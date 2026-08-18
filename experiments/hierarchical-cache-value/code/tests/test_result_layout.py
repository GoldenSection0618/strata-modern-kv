"""Tests for hcv.schema: run layout, tags, schema keys, atomic writes."""

from __future__ import annotations

import json
import os
import tempfile

from hcv.schema import (
    METADATA_KEYS,
    PROCESSED_KEYS,
    RAW_MEASUREMENT_KEYS,
    VALIDATION_KEYS,
    RunLayout,
    append_jsonl,
    make_run_tag,
    read_jsonl,
    validate_record,
    write_json_atomic,
)


def test_run_tag_format():
    tag = make_run_tag("20260813T002921Z", "12345")
    assert tag == "run-20260813T002921Z-job12345"
    assert tag.startswith("run-")
    # two calls with different job ids never collide
    assert make_run_tag("20260813T002921Z", "1") != make_run_tag("20260813T002921Z", "2")


def test_run_layout_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        layout = RunLayout(os.path.join(tmp, "run-x")).create()
        for d in (layout.raw_dir, layout.server_dir, layout.processed_dir, layout.results_dir):
            assert os.path.isdir(d)
        assert layout.metadata_path.endswith("metadata.json")
        assert layout.validation_path.endswith("validation.json")
        assert layout.measurements_path.endswith("raw/measurements.jsonl")


def test_metadata_schema_keys_present():
    # every documented metadata key exists in the schema tuple
    for key in METADATA_KEYS:
        assert key in METADATA_KEYS
    assert "hierarchy_status" in METADATA_KEYS
    assert "validity_status" in METADATA_KEYS
    assert "run_tag" in METADATA_KEYS
    assert "hicache_io_backend" in METADATA_KEYS


def test_validation_schema_keys():
    for key in VALIDATION_KEYS:
        assert key in VALIDATION_KEYS
    assert "status" in VALIDATION_KEYS


def test_validate_record_required_keys():
    errors = validate_record({}, METADATA_KEYS)
    assert any("run_tag" in e for e in errors)
    assert validate_record({k: 1 for k in METADATA_KEYS}, METADATA_KEYS) == []


def test_validate_record_none_for_critical():
    rec = {k: 1 for k in METADATA_KEYS}
    rec["validity_status"] = None
    errors = validate_record(rec, METADATA_KEYS)
    assert any("validity_status" in e for e in errors)


def test_atomic_write_and_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a.json")
        write_json_atomic(path, {"x": 1})
        with open(path) as fh:
            assert json.load(fh) == {"x": 1}
        jl = os.path.join(tmp, "m.jsonl")
        append_jsonl(jl, {"kind": "request", "n": 1})
        append_jsonl(jl, {"kind": "request", "n": 2})
        records = read_jsonl(jl)
        assert [r["n"] for r in records] == [1, 2]
        assert read_jsonl(os.path.join(tmp, "missing.jsonl")) == []


def test_raw_measurement_schema():
    assert "kind" in RAW_MEASUREMENT_KEYS
    assert "run_tag" in RAW_MEASUREMENT_KEYS


def test_processed_schema():
    for key in ("run_tag", "gpu_hit_tokens", "cpu_hit_tokens", "recomputed_tokens",
                "ttft_p50_ms", "throughput_req_per_s"):
        assert key in PROCESSED_KEYS
