"""Raw / processed / result schemas and run-directory layout.

Raw data must never be overwritten: every run writes into its own
``run-<UTC>-job<id>`` directory (unique tag), and processing scripts
only read raw directories.

Layout inside one run directory::

    <results_root>/<experiment>/run-<UTC>-job<id>/
    ├── metadata.json          # run metadata (schema METADATA_KEYS)
    ├── validation.json        # hierarchy gate outcome (controls reportability)
    ├── calibration.json       # Exp2 calibration outcome (when applicable)
    ├── raw/
    │   ├── measurements.jsonl # per-request + window records
    │   └── snapshots.jsonl    # metric snapshots
    ├── server/                # server stdout/stderr (raw evidence)
    ├── processed/             # deterministic processing output
    └── results/               # summary tables/figures data
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Required metadata keys (results/README.md contract)
# ---------------------------------------------------------------------------

METADATA_KEYS = (
    "experiment",                 # validation | exp1 | exp2 | exp3 | exp4
    "model_role",                 # primary | secondary
    "model_id",                   # exact model identifier
    "model_revision",             # exact revision when known
    "hardware",                   # GPU form factor
    "driver",                     # NVIDIA driver
    "cuda_build",                 # PyTorch CUDA build
    "runtime_version",            # sglang version
    "runtime_commit",             # sglang commit
    "precision",                  # bfloat16
    "cache_dtype",                # cache dtype when known
    "architecture",               # gpu_only | hierarchical
    "hierarchy_status",           # full | partial | unsupported | invalid_infrastructure
    "validated_state_groups",     # list
    "cache_initial_state",        # cold | warm
    "gpu_cache_budget",           # mem-fraction-static
    "cpu_tier_budget",            # hicache size/ratio
    "observed_gpu_occupancy",     # kv_used_tokens snapshot
    "trace_id",                   # workload trace identifier
    "prefix_length",              # intended prefix tokens
    "context_distribution",       # intended suffix distribution
    "configured_revisit_fraction",
    "actual_reuse_request_weighted",
    "actual_reuse_token_weighted",
    "reuse_distance_summary",
    "output_length",
    "offered_load_condition",     # concurrency + arrival rate
    "achieved_request_rate",
    "effective_concurrency",
    "preemption_count",
    "calibration_id",             # Exp2 calibration identifier
    "run_timestamp",
    "repetition_index",
    "validity_status",            # valid | invalid | partial | unsupported
    "invalid_reason",
    "sglang_env_dir",
    "hicache_io_backend",
    "hicache_mem_layout",
    "hicache_write_policy",
    "page_size",
    "smoke",
    "run_tag",
    "config_sources",
    "static_checks",
    "runtime_provenance",
    "pressure_label",
)

#: Validation outcome keys (validation.json; controls reportability).
VALIDATION_KEYS = (
    "status",              # full | partial | unsupported | invalid_infrastructure
    "model",
    "architecture",
    "probe_results",
    "state_group_evidence",
    "concurrent_windows",
    "reasons",
    "infra_checks",
)

#: Raw measurement record keys (measurements.jsonl).
RAW_MEASUREMENT_KEYS = (
    "kind",                # request | window | snapshot | filler | calibration
    "run_tag",
)

#: Processed record keys (processed/aggregate.json).
PROCESSED_KEYS = (
    "run_tag",
    "experiment",
    "architecture",
    "hierarchy_status",
    "validity_status",
    "gpu_hit_tokens",
    "cpu_hit_tokens",
    "eviction_tokens",
    "recomputed_tokens",
    "restore_tokens",
    "restore_bytes",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "throughput_req_per_s",
    "achieved_request_rate",
    "preemption_count",
    "effective_concurrency",
)


def validate_record(record: dict, required_keys: tuple) -> list[str]:
    """Return a list of missing-required-key errors ([] when valid)."""
    errors = []
    for key in required_keys:
        if key not in record:
            errors.append(f"missing required key {key!r}")
        elif record[key] is None and key in ("validity_status", "hierarchy_status"):
            errors.append(f"required key {key!r} must not be None")
    return errors


@dataclass
class RunLayout:
    """Filesystem layout for one run directory."""

    root: str

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.root, "metadata.json")

    @property
    def validation_path(self) -> str:
        return os.path.join(self.root, "validation.json")

    @property
    def calibration_path(self) -> str:
        return os.path.join(self.root, "calibration.json")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def measurements_path(self) -> str:
        return os.path.join(self.raw_dir, "measurements.jsonl")

    @property
    def snapshots_path(self) -> str:
        return os.path.join(self.raw_dir, "snapshots.jsonl")

    @property
    def server_dir(self) -> str:
        return os.path.join(self.root, "server")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.root, "results")

    def create(self) -> "RunLayout":
        for d in (self.raw_dir, self.server_dir, self.processed_dir, self.results_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        return self


def make_run_tag(utc: str, job_id: str) -> str:
    """Unique run tag ``run-<UTC>-job<id>`` (never reused/overwritten)."""
    return f"run-{utc}-job{job_id}"


def write_json_atomic(path: str, obj: dict) -> None:
    """Write JSON via temp-file + rename so readers never see partial data."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_jsonl(path: str, obj: dict) -> None:
    """Append one JSON record to a JSONL file (raw evidence stream)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read all JSON records from a JSONL file ([] when missing)."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
