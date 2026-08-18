"""Shared runner scaffolding: server launch, metadata, snapshots, traces.

Every runner follows the same discipline:

1. resolve the config once (precedence recorded in ``sources``);
2. run static provenance checks (markers, toolchain, model path, log path);
3. create the unique ``run-<UTC>-job<id>`` directory;
4. launch the SGLang server with the resolved argv (bounded lifecycle);
5. execute the experiment, appending raw evidence to ``raw/``;
6. write ``metadata.json`` and ``validation.json`` (reportability gate);
7. shut the server down gracefully.

Servers are launched only by the sbatch files on compute nodes; this
module never launches anything on the login node.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from hcv.config import ExperimentConfig, dl_root, model_path_for
from hcv.http_client import SGLangHTTPClient
from hcv.provenance import (
    check_jit_toolchain,
    check_log_path,
    check_markers,
    check_model_path,
    collect_runtime_provenance,
)
from hcv.schema import RunLayout, append_jsonl, make_run_tag, write_json_atomic
from hcv.server_lifecycle import SGLangServerProcess, pick_free_port
from hcv.workload import build_trace_from_config, load_trace

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def utc_now() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def job_id() -> str:
    return os.environ.get("SLURM_JOB_ID", "local")


def parse_common_args(argv: Optional[list] = None) -> dict:
    """Parse the shared runner CLI (config, results root, run tag, smoke)."""
    import argparse

    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--config", default="", help="JSON config file path")
    p.add_argument("--results-root", default="", help="results root dir")
    p.add_argument("--run-tag", default="", help="run tag override")
    p.add_argument("--smoke", action="store_true", help="smoke mode")
    p.add_argument("--model", default="", help="model key override (qwen|gemma)")
    p.add_argument("--experiment", default="", help="experiment id override")
    p.add_argument("--log-dir", default="", help="slurm log dir")
    p.add_argument("--repetition", type=int, default=0, help="repetition index")
    p.add_argument("--ready-timeout", type=int, default=0,
                   help="server ready timeout seconds override")
    p.add_argument("--n-repeats", type=int, default=0, help="repetitions override")
    p.add_argument("--architecture", default="",
                   help="architecture override (gpu_only|hierarchical)")
    p.add_argument("--revisit-fraction", type=float, default=-1.0,
                   help="revisit fraction override (exp3)")
    p.add_argument("--pressure-label", default="",
                   help="pressure label override (exp2/exp3)")
    p.add_argument("--initial-state", default="",
                   help="initial cache state override (cold|warm)")
    ns = p.parse_args(argv)
    return vars(ns)


def resolve_config(
    args: dict,
    experiment: str,
    defaults: dict | None = None,
) -> ExperimentConfig:
    """Resolve the experiment config from CLI + config file + env.

    The config file supplies experiment defaults; CLI values win over
    them; ``HCV_*`` environment variables participate through
    :func:`hcv.config.load_config`.  User-provided values (config file /
    env / CLI) are protected from SMOKE reductions.
    """
    from hcv.config import load_config

    cli: dict = {}
    if args.get("experiment"):
        cli["experiment"] = args["experiment"]
    if args.get("model"):
        cli["model"] = args["model"]
    if args.get("run_tag"):
        cli["run_tag"] = args["run_tag"]
    if args.get("results_root"):
        cli["results_root"] = args["results_root"]
    if args.get("smoke"):
        cli["smoke"] = True
    if args.get("ready_timeout"):
        cli["ready_timeout_s"] = args["ready_timeout"]
    if args.get("n_repeats"):
        cli["n_repeats"] = args["n_repeats"]
    if args.get("architecture"):
        cli["architecture"] = args["architecture"]
    if args.get("revisit_fraction") is not None and args.get("revisit_fraction") >= 0:
        cli["revisit_fraction"] = args["revisit_fraction"]
    if args.get("pressure_label"):
        cli["pressure_label"] = args["pressure_label"]
    if args.get("initial_state"):
        cli["initial_state"] = args["initial_state"]
    cli.update(defaults or {})
    return load_config(
        config_path=args.get("config") or None,
        cli=cli,
        smoke=bool(args.get("smoke")),
        explicit_fields=set(cli),
    )


def resolve_run_tag(cfg: ExperimentConfig) -> str:
    """Run tag: user RUN_TAG override wins; else unique run-<UTC>-job<id>."""
    if cfg.run_tag:
        return cfg.run_tag
    return make_run_tag(utc_now(), job_id())


def results_root_default() -> str:
    """Results root inside the current worktree
    (``experiments/hierarchical-cache-value/results``).

    Derived from this module's location so it never points into the
    source repository or another worktree.
    """
    here = Path(__file__).resolve().parent  # .../hierarchical-cache-value/code/hcv
    return str(here.parent.parent / "results")


def run_dir_for(cfg: ExperimentConfig, tag: str) -> str:
    root = cfg.results_root or results_root_default()
    return os.path.join(root, cfg.experiment, tag)


def log_dir_default() -> str:
    """Default sbatch log dir (the cluster-mandated ~/logs location)."""
    return os.path.join(os.path.expanduser("~"), "logs")


# ---------------------------------------------------------------------------
# Static checks (usable in dry-run and before launch)
# ---------------------------------------------------------------------------


def static_checks(cfg: ExperimentConfig, log_dir: str) -> dict:
    """Run all static provenance checks; returns a check summary dict.

    Raises RuntimeError with the first hard failure.  Each check's ok/errors
    are recorded so dry-run and real runs share the same gate.
    """
    env_ok, env_errors = check_markers(cfg.sglang_env_dir, cfg.sglang_commit)
    jit_ok, jit_errors = check_jit_toolchain(cfg.sglang_env_dir)
    model_ok, model_errors = check_model_path(model_path_for(cfg.model))
    log_ok, log_errors = check_log_path(log_dir)
    checks = {
        "env_markers_ok": env_ok,
        "env_markers_errors": env_errors,
        "jit_toolchain_ok": jit_ok,
        "jit_toolchain_errors": jit_errors,
        "model_path_ok": model_ok,
        "model_path_errors": model_errors,
        "log_path_ok": log_ok,
        "log_path_errors": log_errors,
        "sglang_env_dir": cfg.sglang_env_dir,
        "sglang_commit": cfg.sglang_commit,
        "model_path": model_path_for(cfg.model),
        "model_id": cfg.model_id,
    }
    if not env_ok:
        raise RuntimeError("environment marker check failed: " + "; ".join(env_errors))
    if not jit_ok:
        raise RuntimeError("JIT toolchain check failed: " + "; ".join(jit_errors))
    if not model_ok:
        raise RuntimeError("model path check failed: " + "; ".join(model_errors))
    if not log_ok:
        raise RuntimeError("log path check failed: " + "; ".join(log_errors))
    return checks


# ---------------------------------------------------------------------------
# Server launch helper
# ---------------------------------------------------------------------------


def launch_server(
    cfg: ExperimentConfig,
    log_dir: str,
    run_layout: RunLayout,
) -> tuple[SGLangServerProcess, SGLangHTTPClient, int]:
    """Launch the server with the resolved argv and wait for readiness.

    Returns (process, client, port).  Raises on launch/readiness failure.
    """
    port = pick_free_port()
    argv = cfg.build_server_argv(port=port)
    stdout_path = os.path.join(run_layout.server_dir, "server.stdout")
    stderr_path = os.path.join(run_layout.server_dir, "server.stderr")
    proc = SGLangServerProcess(
        argv=argv,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        env={},
        ready_timeout_s=float(cfg.ready_timeout_s),
    )
    proc.start()
    client = SGLangHTTPClient(base_url=f"http://127.0.0.1:{port}")
    try:
        proc.wait_ready(client.health)
    except Exception:
        proc.stop()
        raise
    return proc, client, port


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def load_or_build_trace(cfg: ExperimentConfig, cache_dir: str) -> object:
    """Load the trace from ``cache_dir`` if present, else build and cache.

    The trace is deterministic from the config; caching just avoids
    rebuilding it for every repetition.  Returns a ``hcv.workload.Trace``.
    """
    trace_cfg = {
        "seed": cfg.seed,
        "num_prefix_families": cfg.num_prefix_families,
        "family_size": cfg.family_size,
        "prefix_length": cfg.prefix_length,
        "suffix_length_min": cfg.suffix_length_min,
        "suffix_length_max": cfg.suffix_length_max,
        "output_length": cfg.output_length,
        "revisit_fraction": cfg.revisit_fraction,
        "request_count": cfg.request_count,
    }
    from hcv.workload import compute_trace_id

    trace_id = compute_trace_id(**trace_cfg)
    path = os.path.join(cache_dir, f"trace-{trace_id}.json")
    if os.path.exists(path):
        return load_trace(path)
    trace = build_trace_from_config(trace_cfg)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    from hcv.workload import save_trace

    save_trace(path, trace)
    return trace


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def base_metadata(cfg: ExperimentConfig, tag: str, checks: dict,
                  runtime_provenance: Optional[dict] = None) -> dict:
    """Build the base run metadata dict (schema: hcv.schema.METADATA_KEYS)."""
    return {
        "experiment": cfg.experiment,
        "model_role": "primary" if cfg.model == "qwen" else "secondary",
        "model_id": cfg.model_id,
        "model_revision": "",
        "hardware": "",
        "driver": "",
        "cuda_build": "",
        "runtime_version": cfg.sglang_version,
        "runtime_commit": cfg.sglang_commit,
        "precision": "bfloat16",
        "cache_dtype": "",
        "architecture": cfg.architecture,
        "hierarchy_status": "unsupported",
        "validated_state_groups": cfg.model_state_groups,
        "cache_initial_state": cfg.initial_state,
        "gpu_cache_budget": cfg.mem_fraction_static,
        "cpu_tier_budget": {
            "hicache_size_tokens": cfg.hicache_size_tokens,
            "hicache_ratio": cfg.hicache_ratio,
        },
        "observed_gpu_occupancy": None,
        "trace_id": cfg.trace_id or "",
        "prefix_length": cfg.prefix_length,
        "context_distribution": {
            "suffix_min": cfg.suffix_length_min,
            "suffix_max": cfg.suffix_length_max,
        },
        "configured_revisit_fraction": cfg.revisit_fraction,
        "actual_reuse_request_weighted": None,
        "actual_reuse_token_weighted": None,
        "reuse_distance_summary": None,
        "output_length": cfg.output_length,
        "offered_load_condition": {
            "concurrency": cfg.concurrency,
            "arrival_rate": cfg.arrival_rate,
        },
        "achieved_request_rate": None,
        "effective_concurrency": None,
        "preemption_count": None,
        "calibration_id": "",
        "run_timestamp": utc_now(),
        "repetition_index": 0,
        "validity_status": "pending",
        "invalid_reason": "",
        "sglang_env_dir": cfg.sglang_env_dir,
        "hicache_io_backend": cfg.hicache_io_backend,
        "hicache_mem_layout": cfg.hicache_mem_layout,
        "hicache_write_policy": cfg.hicache_write_policy,
        "page_size": cfg.page_size,
        "smoke": cfg.smoke,
        "pressure_label": cfg.pressure_label,
        "run_tag": tag,
        "config_sources": dict(cfg.sources),
        "static_checks": checks,
        "runtime_provenance": runtime_provenance or {},
    }
