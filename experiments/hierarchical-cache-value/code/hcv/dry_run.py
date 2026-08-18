"""DRY_RUN=1 validation: static checks without launching any server.

Runs inside a minimal dry-run sbatch.  Validates, without
starting a server or importing torch:

* exact environment markers (files + pinned commit);
* prefix-local compiler executables;
* model path;
* log path writability;
* config precedence resolution (recorded per-field sources);
* resolved smoke vs formal values (never silently overwritten);
* server argv construction for both architectures;
* GPU-only / hierarchical pairing invariants;
* Slurm job-name prefix (``ylh-hcv-*``) when inside a job.

Environment/dry-run success is NOT full-hierarchy proof (see
``hcv.hierarchy`` for the mechanism gate).
"""

from __future__ import annotations

import json
import logging
import os
import sys

from hcv.config import (
    ARCH_GPU_ONLY,
    ARCH_HIERARCHICAL,
    ExperimentConfig,
    pair_configs,
)
from hcv.provenance import (
    check_jit_toolchain,
    check_log_path,
    check_markers,
    check_model_path,
)
from hcv.run_common import (
    log_dir_default,
    parse_common_args,
    resolve_config,
)
from hcv.schema import RunLayout, write_json_atomic

logger = logging.getLogger(__name__)


def validate(cfg: ExperimentConfig, log_dir: str) -> dict:
    """Run all static checks; returns a report dict (never launches)."""
    report: dict = {
        "dry_run": True,
        "model": cfg.model,
        "model_id": cfg.model_id,
        "model_path": cfg.model_path if hasattr(cfg, "model_path") else _model_path(cfg),
        "sglang_env_dir": cfg.sglang_env_dir,
        "sglang_commit": cfg.sglang_commit,
        "sglang_version": cfg.sglang_version,
        "experiment": cfg.experiment,
        "architecture": cfg.architecture,
        "initial_state": cfg.initial_state,
        "smoke": cfg.smoke,
        "resolved_values": _resolved_values(cfg),
        "config_sources": dict(cfg.sources),
        "server_argv_gpu_only": None,
        "server_argv_hierarchical": None,
        "paired_ok": None,
        "checks": {},
    }

    # environment markers + toolchain + model + log
    env_ok, env_errors = check_markers(cfg.sglang_env_dir, cfg.sglang_commit)
    jit_ok, jit_errors = check_jit_toolchain(cfg.sglang_env_dir)
    model_ok, model_errors = check_model_path(_model_path(cfg))
    log_ok, log_errors = check_log_path(log_dir)
    report["checks"] = {
        "env_markers_ok": env_ok,
        "env_markers_errors": env_errors,
        "jit_toolchain_ok": jit_ok,
        "jit_toolchain_errors": jit_errors,
        "model_path_ok": model_ok,
        "model_path_errors": model_errors,
        "log_path_ok": log_ok,
        "log_path_errors": log_errors,
    }

    # server argv for both architectures + pairing invariant
    try:
        gpu_cfg, hier_cfg = pair_configs(cfg)
        report["server_argv_gpu_only"] = gpu_cfg.build_server_argv(port=0)
        report["server_argv_hierarchical"] = hier_cfg.build_server_argv(port=0)
        report["paired_ok"] = True
    except ValueError as e:
        report["paired_ok"] = False
        report["paired_error"] = str(e)

    # job-name prefix check (only meaningful inside a Slurm job)
    job_name = os.environ.get("SLURM_JOB_NAME", "")
    if job_name:
        report["job_name"] = job_name
        report["job_name_prefix_ok"] = job_name.startswith("ylh-hcv-")

    report["ok"] = (
        env_ok and jit_ok and model_ok and log_ok and report.get("paired_ok", False)
        and (not job_name or report.get("job_name_prefix_ok", True))
    )
    return report


def _model_path(cfg: ExperimentConfig) -> str:
    from hcv.config import model_path_for

    return model_path_for(cfg.model)


def _resolved_values(cfg: ExperimentConfig) -> dict:
    """Resolved smoke/formal values (what a real run would use)."""
    return {
        "n_repeats": cfg.n_repeats,
        "n_warmup": cfg.n_warmup,
        "request_count": cfg.request_count,
        "num_prefix_families": cfg.num_prefix_families,
        "family_size": cfg.family_size,
        "prefix_length": cfg.prefix_length,
        "revisit_fraction": cfg.revisit_fraction,
        "concurrency": cfg.concurrency,
        "mem_fraction_static": cfg.mem_fraction_static,
        "page_size": cfg.page_size,
        "hicache_io_backend": cfg.hicache_io_backend,
        "hicache_mem_layout": cfg.hicache_mem_layout,
        "hicache_write_policy": cfg.hicache_write_policy,
        "hicache_size_tokens": cfg.hicache_size_tokens,
        "hicache_ratio": cfg.hicache_ratio,
        "pressure_label": cfg.pressure_label,
        "calib_ladder": list(cfg.calib_ladder),
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                        stream=sys.stdout)
    args = parse_common_args(argv)
    cfg = resolve_config(args, experiment=args.get("experiment") or "validation")
    log_dir = args.get("log_dir") or log_dir_default()
    report = validate(cfg, log_dir)

    # write the report next to the (planned) run directory for auditability
    from hcv.run_common import resolve_run_tag, run_dir_for

    tag = resolve_run_tag(cfg)
    layout = RunLayout(run_dir_for(cfg, tag)).create()
    write_json_atomic(os.path.join(layout.results_dir, "dry_run.json"), report)

    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        print("DRY_RUN FAILED: see checks above", file=sys.stderr)
        return 1
    print("DRY_RUN OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
