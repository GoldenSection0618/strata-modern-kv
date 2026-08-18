"""Frozen runtime configuration, precedence, pairing checks, server args.

This group's hierarchy-only baseline is frozen:

* I/O backend: ``direct`` (standard CUDA-copy; GPU-assisted ``kernel``
  I/O belongs to the Page Granularity / GPU-Assisted I/O group)
* host layout: ``page_first_direct``
* write policy: ``write_through``
* page size: 64
* public metrics: ``--enable-metrics``
* environment: ``sglang-hicache-cu129-torch211`` @ pinned commit

Config precedence (later layers win, each layer's wins are recorded):

1. code defaults
2. JSON config file (``configs/<exp>.json``)
3. environment variables (``HCV_*``)
4. CLI arguments

``SMOKE=1`` never silently overwrites a user-provided value: it only
scales derived run-size knobs whose final source is a default or config
file, and the exact smoke adjustments are recorded in metadata.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from hcv import PINNED_SGLANG_COMMIT, PINNED_SGLANG_VERSION

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

#: Canonical environment prefix name (must not fall back to qwen/gemma4/sglang).
SGLANG_ENV_NAME = "sglang-hicache-cu129-torch211"

#: Environment marker files that must exist inside the prefix.
ENV_MARKER_FILES = (
    "sglang_commit.txt",
    "provenance.json",
    "pip_freeze.txt",
    "conda_list.txt",
    "cu129_complete.txt",
    "jit_toolchain_complete.txt",
)

#: Prefix-local JIT toolchain executables required by every sbatch.
JIT_TOOLCHAIN_BINS = (
    "bin/x86_64-conda-linux-gnu-gcc",
    "bin/x86_64-conda-linux-gnu-g++",
    "bin/nvcc",
)

#: Frozen hierarchy-only baseline (this group; kernel I/O belongs to group 3).
FROZEN_HICACHE_IO_BACKEND = "direct"
FROZEN_HICACHE_MEM_LAYOUT = "page_first_direct"
FROZEN_HICACHE_WRITE_POLICY = "write_through"
FROZEN_PAGE_SIZE = 64

VALID_IO_BACKENDS = ("direct", "kernel")
VALID_MEM_LAYOUTS = ("layer_first", "page_first", "page_first_direct")
VALID_WRITE_POLICIES = ("write_back", "write_through", "write_through_selective")
VALID_MODELS = ("qwen", "gemma")

#: Cache architecture values.
ARCH_GPU_ONLY = "gpu_only"
ARCH_HIERARCHICAL = "hierarchical"
VALID_ARCHITECTURES = (ARCH_GPU_ONLY, ARCH_HIERARCHICAL)

#: Cache initial state values.
STATE_COLD = "cold"
STATE_WARM = "warm"
VALID_INITIAL_STATES = (STATE_COLD, STATE_WARM)

#: Hierarchy gate outcomes (validation.json status values).
GATE_FULL = "full"
GATE_PARTIAL = "partial"
GATE_UNSUPPORTED = "unsupported"
GATE_INVALID_INFRA = "invalid_infrastructure"
VALID_GATE_STATUSES = (GATE_FULL, GATE_PARTIAL, GATE_UNSUPPORTED, GATE_INVALID_INFRA)

#: Model registry (paths resolved against DL_ROOT at runtime).
MODEL_SPECS = {
    "qwen": {
        "id": "Qwen/Qwen3.5-9B",
        "cache_dir": "cache/huggingface/models--Qwen--Qwen3.5-9B",
        "role": "primary",
        "state_groups": ["attention_kv", "gated_delta_recurrent"],
        "default_context": 262144,
    },
    "gemma": {
        "id": "google/gemma-4-12B-it",
        "cache_dir": "cache/huggingface/models--google--gemma-4-12B-it",
        "role": "secondary",
        "state_groups": ["local_sliding_window", "global_attention"],
        "default_context": 262144,
    },
}

#: Default GPU memory fraction (start of Exp2 calibration ladder).
DEFAULT_MEM_FRACTION = 0.85

#: Fallback L1 filler pressure (tokens) when no observed capacity exists:
#: max(6*context_length, 262144) + protected_prefix_tokens + 4096.
FALLBACK_L1_FILLER_MULTIPLIER = 6
FALLBACK_L1_FILLER_FLOOR = 262144
FALLBACK_L1_FILLER_MARGIN = 4096


def dl_root() -> str:
    """Resolve the project DL_ROOT (env override allowed for tests)."""
    return os.environ.get("DL_ROOT", os.path.join(os.path.expanduser("~"), "yanglihan", "dl-stack"))


def model_path_for(model: str, dlroot: str | None = None) -> str:
    """Absolute model cache path for a model key."""
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown model {model!r}; valid: {VALID_MODELS}")
    root = dlroot or dl_root()
    return os.path.join(root, MODEL_SPECS[model]["cache_dir"])


def env_dir_for(dlroot: str | None = None) -> str:
    """Absolute canonical SGLang environment prefix path."""
    root = dlroot or dl_root()
    return os.path.join(root, "envs", SGLANG_ENV_NAME)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """One runnable experiment configuration (all knobs resolved once)."""

    # experiment identity
    experiment: str = "validation"          # validation | exp1 | exp2 | exp3 | exp4
    model: str = "qwen"
    architecture: str = ARCH_GPU_ONLY
    initial_state: str = STATE_WARM
    trace_id: str = ""                       # resolved from workload config

    # workload
    seed: int = 20260813
    num_prefix_families: int = 16
    family_size: int = 24                    # requests per family template
    prefix_length: int = 512                 # intended prefix tokens
    suffix_length_min: int = 16
    suffix_length_max: int = 64
    output_length: int = 1                   # max_new_tokens (TTFT == latency)
    revisit_fraction: float = 0.5            # Exp3 variable; Exp1/2 fixed
    request_count: int = 0                   # 0 -> derived from families

    # serving load
    concurrency: int = 8
    arrival_rate: float = 0.0                # 0 -> saturation (concurrency-bounded)
    offered_requests: int = 0                # 0 -> all trace requests

    # GPU cache budget (Exp2 sweep variable)
    mem_fraction_static: float = DEFAULT_MEM_FRACTION
    max_total_num_tokens: int = 0            # 0 -> let mem-fraction govern
    page_size: int = FROZEN_PAGE_SIZE

    # hierarchical CPU tier (hierarchical runs; recorded even when unused)
    hicache_size_tokens: int = 0             # 0 -> runtime default
    hicache_ratio: float = 0.0               # 0 -> unused (size wins when >0)
    hicache_io_backend: str = FROZEN_HICACHE_IO_BACKEND
    hicache_mem_layout: str = FROZEN_HICACHE_MEM_LAYOUT
    hicache_write_policy: str = FROZEN_HICACHE_WRITE_POLICY

    # calibration / sweep controls (Exp2)
    pressure_label: str = "Low"              # Low|Medium|High|VeryHigh
    calib_ladder: tuple = (0.85, 0.75, 0.65, 0.55, 0.45, 0.35)
    calib_probe_requests: int = 120

    # repetitions / execution
    n_warmup: int = 1
    n_repeats: int = 3
    smoke: bool = False
    ready_timeout_s: int = 1800

    # output
    results_root: str = ""                   # absolute; resolved by runner
    run_tag: str = ""                        # run-<UTC>-job<id>

    # provenance
    sglang_env_dir: str = ""
    sglang_commit: str = PINNED_SGLANG_COMMIT
    sglang_version: str = PINNED_SGLANG_VERSION

    # records which layer set each field (defaults/file/env/cli/smoke)
    sources: dict = field(default_factory=dict)

    # -- derived helpers ----------------------------------------------------

    @property
    def hierarchy_enabled(self) -> bool:
        return self.architecture == ARCH_HIERARCHICAL

    @property
    def model_id(self) -> str:
        return MODEL_SPECS[self.model]["id"]

    @property
    def model_state_groups(self) -> list[str]:
        return list(MODEL_SPECS[self.model]["state_groups"])

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def build_server_argv(self, port: int, host: str = "127.0.0.1") -> list[str]:
        """Authoritative ``launch_server`` argument list for this config."""
        args = [
            "--model-path", model_path_for(self.model),
            "--host", host,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--tp", "1",
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--page-size", str(self.page_size),
            "--enable-metrics",
        ]
        if self.max_total_num_tokens > 0:
            args += ["--max-total-tokens", str(self.max_total_num_tokens)]
        if self.hierarchy_enabled:
            args.append("--enable-hierarchical-cache")
            if self.hicache_size_tokens > 0:
                args += ["--hicache-size", str(self.hicache_size_tokens)]
            elif self.hicache_ratio > 0:
                args += ["--hicache-ratio", str(self.hicache_ratio)]
            args += [
                "--hicache-io-backend", self.hicache_io_backend,
                "--hicache-mem-layout", self.hicache_mem_layout,
                "--hicache-write-policy", self.hicache_write_policy,
            ]
        return args


# ---------------------------------------------------------------------------
# Precedence-aware loading
# ---------------------------------------------------------------------------

#: Fields that SMOKE may reduce when their final source is default/file.
_SMOKE_REDUCIBLE = ("n_repeats", "n_warmup", "request_count", "offered_requests",
                    "calib_probe_requests", "family_size", "num_prefix_families",
                    "max_total_num_tokens")

#: Scalar field names with known JSON-compatible types (for env parsing).
_INT_FIELDS = {
    "seed", "num_prefix_families", "family_size", "prefix_length",
    "suffix_length_min", "suffix_length_max", "output_length", "request_count",
    "concurrency", "offered_requests", "max_total_num_tokens", "page_size",
    "hicache_size_tokens", "calib_probe_requests", "n_warmup", "n_repeats",
    "ready_timeout_s",
}
_FLOAT_FIELDS = {
    "arrival_rate", "mem_fraction_static", "hicache_ratio", "revisit_fraction",
}
_BOOL_FIELDS = {"smoke"}
_STR_FIELDS = {
    "experiment", "model", "architecture", "initial_state", "trace_id",
    "pressure_label", "run_tag", "results_root", "sglang_env_dir",
    "sglang_commit", "sglang_version", "hicache_io_backend",
    "hicache_mem_layout", "hicache_write_policy",
}
_LIST_FIELDS = {"calib_ladder"}
_ALL_FIELDS = _INT_FIELDS | _FLOAT_FIELDS | _BOOL_FIELDS | _STR_FIELDS | _LIST_FIELDS


def _coerce(field_name: str, raw: Any) -> Any:
    """Coerce a raw value to the field's declared type."""
    if field_name in _INT_FIELDS:
        return int(raw)
    if field_name in _FLOAT_FIELDS:
        return float(raw)
    if field_name in _BOOL_FIELDS:
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if field_name in _LIST_FIELDS:
        if isinstance(raw, str):
            return tuple(float(x) for x in raw.split(",") if x.strip())
        return tuple(float(x) for x in raw)
    return str(raw)


def _apply_layer(cfg: ExperimentConfig, layer: dict, source: str,
                 explicit: set[str]) -> set[str]:
    """Apply one config layer; return the set of fields it set."""
    touched: set[str] = set()
    for key, value in layer.items():
        if key not in _ALL_FIELDS:
            raise ValueError(f"unknown config field {key!r} (source={source})")
        if key in ("sources",):
            continue
        setattr(cfg, key, _coerce(key, value))
        cfg.sources[key] = source
        touched.add(key)
    return touched


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_config(
    config_path: str | None = None,
    cli: dict | None = None,
    env: dict | None = None,
    smoke: bool = False,
    explicit_fields: set[str] | None = None,
) -> ExperimentConfig:
    """Resolve an :class:`ExperimentConfig` with recorded precedence.

    Layer order: code defaults < JSON file < ``HCV_*`` environment
    variables < CLI overrides.  ``explicit_fields`` names the fields the
    user set through env/CLI so that SMOKE never overwrites them.
    """
    env = dict(os.environ if env is None else env)
    cli = dict(cli or {})
    explicit = set(explicit_fields or {})

    cfg = ExperimentConfig()
    for f in _ALL_FIELDS:
        cfg.sources[f] = "default"

    touched: set[str] = set()
    if config_path:
        touched |= _apply_layer(cfg, _load_json(config_path), "file", explicit)

    env_layer = {}
    for key, value in env.items():
        if key.startswith("HCV_") and value != "":
            field_name = key[4:].lower()
            if field_name in _ALL_FIELDS:
                env_layer[field_name] = value
    touched |= _apply_layer(cfg, env_layer, "env", explicit)
    explicit |= set(env_layer)

    touched |= _apply_layer(cfg, cli, "cli", explicit)
    explicit |= set(cli)

    # Validate enums after all layers.
    _validate_enums(cfg)

    # SMOKE only touches reducible fields that the user did not set.
    if smoke:
        smoke_adjust: dict[str, Any] = {}
        if "n_repeats" not in explicit:
            smoke_adjust["n_repeats"] = max(1, min(cfg.n_repeats, 1))
        # Exp2/3 must preserve their formal working set and preparation;
        # shortening either makes pressure/reuse evidence disappear.
        short_workload_smoke = cfg.experiment in ("validation", "exp1", "exp4")
        if "n_warmup" not in explicit and short_workload_smoke:
            smoke_adjust["n_warmup"] = max(0, min(cfg.n_warmup, 1))
        if "request_count" not in explicit and cfg.request_count > 0 and short_workload_smoke:
            smoke_adjust["request_count"] = max(8, cfg.request_count // 8)
        if "offered_requests" not in explicit and cfg.offered_requests > 0:
            smoke_adjust["offered_requests"] = max(8, cfg.offered_requests // 8)
        if "family_size" not in explicit and short_workload_smoke:
            # At concurrency 4, very short traces are dominated by worker
            # ramp-up/drain-down and cannot validate effective concurrency.
            # Keep at least 4 x 16 = 64 requests when request_count is
            # derived from the family grid, still 1/6 of Exp1 formal.
            smoke_adjust["family_size"] = max(16, cfg.family_size // 4)
        if "num_prefix_families" not in explicit and short_workload_smoke:
            smoke_adjust["num_prefix_families"] = max(2, cfg.num_prefix_families // 4)
        if "calib_probe_requests" not in explicit and short_workload_smoke:
            smoke_adjust["calib_probe_requests"] = max(16, cfg.calib_probe_requests // 4)
        # Keep the hierarchy gate evidentiary in smoke mode: shrink the
        # actual L1 pool instead of truncating the eviction filler before it
        # can demonstrate a GPU eviction and CPU load-back.  Formal runs
        # retain the mem-fraction-governed pool (0), and an explicit env/CLI
        # value remains authoritative.
        if "max_total_num_tokens" not in explicit and cfg.experiment == "validation":
            smoke_adjust["max_total_num_tokens"] = (
                min(cfg.max_total_num_tokens, 32768)
                if cfg.max_total_num_tokens > 0 else 32768
            )
        for key, value in smoke_adjust.items():
            setattr(cfg, key, _coerce(key, value))
            cfg.sources[key] = "smoke"
        cfg.smoke = True

    # Derived fields that must always reflect the resolved inputs.
    cfg.sglang_env_dir = env_dir_for()
    return cfg


def _validate_enums(cfg: ExperimentConfig) -> None:
    if cfg.model not in VALID_MODELS:
        raise ValueError(f"invalid model {cfg.model!r}; valid: {VALID_MODELS}")
    if cfg.architecture not in VALID_ARCHITECTURES:
        raise ValueError(f"invalid architecture {cfg.architecture!r}")
    if cfg.initial_state not in VALID_INITIAL_STATES:
        raise ValueError(f"invalid initial_state {cfg.initial_state!r}")
    if cfg.hicache_io_backend not in VALID_IO_BACKENDS:
        raise ValueError(f"invalid hicache_io_backend {cfg.hicache_io_backend!r}")
    if cfg.hicache_mem_layout not in VALID_MEM_LAYOUTS:
        raise ValueError(f"invalid hicache_mem_layout {cfg.hicache_mem_layout!r}")
    if cfg.hicache_write_policy not in VALID_WRITE_POLICIES:
        raise ValueError(f"invalid hicache_write_policy {cfg.hicache_write_policy!r}")
    if cfg.sglang_commit != PINNED_SGLANG_COMMIT:
        raise ValueError(
            f"sglang_commit {cfg.sglang_commit!r} != pinned {PINNED_SGLANG_COMMIT!r}"
        )
    if cfg.experiment not in ("validation", "exp1", "exp2", "exp3", "exp4"):
        raise ValueError(f"invalid experiment {cfg.experiment!r}")


# ---------------------------------------------------------------------------
# Paired-configuration checks
# ---------------------------------------------------------------------------

#: Fields allowed to differ between a GPU-only / hierarchical pair.
_PAIR_ALLOWED_DIFF = {
    "architecture", "hierarchy_enabled",
    "hicache_size_tokens", "hicache_ratio", "hicache_io_backend",
    "hicache_mem_layout", "hicache_write_policy", "sources",
}


def assert_paired(gpu_only: ExperimentConfig, hierarchical: ExperimentConfig) -> None:
    """Assert the GPU-only / hierarchical pair differs ONLY in the
    intended architecture difference (hierarchy enablement + CPU tier).

    Every other comparison-relevant field (model, workload, GPU budget,
    page size, scheduler, load, trace) must be identical.
    """
    a = gpu_only.to_dict()
    b = hierarchical.to_dict()
    diffs = [k for k in a if a[k] != b.get(k)]
    unexpected = [k for k in diffs if k not in _PAIR_ALLOWED_DIFF]
    if unexpected:
        raise ValueError(
            "paired configs differ in non-architecture fields: "
            f"{sorted(unexpected)}"
        )
    if gpu_only.architecture != ARCH_GPU_ONLY:
        raise ValueError("gpu_only config must have architecture=gpu_only")
    if hierarchical.architecture != ARCH_HIERARCHICAL:
        raise ValueError("hierarchical config must have architecture=hierarchical")


def pair_configs(base: ExperimentConfig) -> tuple[ExperimentConfig, ExperimentConfig]:
    """Derive the paired (GPU-only, hierarchical) configs from a base.

    The base's architecture field is ignored; both cells keep every
    other resolved field identical.
    """
    import copy

    gpu = copy.deepcopy(base)
    gpu.architecture = ARCH_GPU_ONLY
    hier = copy.deepcopy(base)
    hier.architecture = ARCH_HIERARCHICAL
    assert_paired(gpu, hier)
    return gpu, hier
