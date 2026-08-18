"""Experiment 2 calibration: minimum GPU budget without preemption.

Before the pressure sweep, a calibration job determines the smallest
GPU memory fraction at which the fixed active workload runs without
active-request preemption (runtime retraction), OOM, or request failures.
The sweep then compresses only the reusable-cache headroom above that
floor.

Calibration output (shared ``calibration.json`` under the exp2 results
root) records, per ladder attempt:

* mem fraction;
* preemption windows, queue depth, request failures;
* whether the attempt is valid;
* the resolved ``min_valid_fraction`` (floor) and ``calibration_id``.

Each pressure point job reads this file; a missing calibration file is a
hard error (the sweep must never silently guess the floor).
"""

from __future__ import annotations

import json
import logging
import os
import time
import sys

from hcv.config import ExperimentConfig
from hcv.load_driver import run_load
from hcv.residency import prepare_cold, snapshot_cache_state
from hcv.run_common import (
    base_metadata,
    job_id,
    launch_server,
    load_or_build_trace,
    log_dir_default,
    parse_common_args,
    resolve_config,
    resolve_run_tag,
    results_root_default,
    run_dir_for,
    setup_logging,
    static_checks,
    utc_now,
)
from hcv.schema import RunLayout, append_jsonl, write_json_atomic
from hcv.workload import reuse_summary

logger = logging.getLogger(__name__)


def shared_calibration_path(results_root: str) -> str:
    """Canonical calibration path consumed by Exp2 and Exp3 jobs."""
    return os.path.join(results_root, "exp2", "calibration", "calibration.json")


def calibration_id(ladder: tuple, trace_id: str, concurrency: int, probe_requests: int) -> str:
    ladder_id = "-".join(str(x).replace(".", "p") for x in ladder)
    return f"{trace_id}-c{concurrency}-n{probe_requests}-l{ladder_id}"


def attempt_valid(run, concurrency: int, tolerance: float = 0.5) -> dict:
    """Decide validity of one calibration attempt."""
    reasons = []
    if run.preemption_windows > 0:
        reasons.append("active-request preemption (retracted-request counter increased)")
    if run.gen_errors:
        reasons.append(f"{len(run.gen_errors)} request failures")
    ec = None
    if run.windows and all(w.effective_concurrency is not None for w in run.windows):
        ecs = [w.effective_concurrency for w in run.windows if w.effective_concurrency is not None]
        if ecs:
            ec = sum(ecs) / len(ecs)
            if abs(ec - concurrency) > tolerance:
                reasons.append(f"effective concurrency drift ({ec:.2f} vs {concurrency})")
    return {"valid": not reasons, "reasons": reasons, "effective_concurrency": ec}


def run_calibration(cfg: ExperimentConfig, log_dir: str, layout: RunLayout) -> dict:
    """Run the calibration ladder and write the shared calibration file."""
    trace = load_or_build_trace(cfg, os.path.join(layout.raw_dir, "traces"))
    cid = calibration_id(tuple(cfg.calib_ladder), trace.trace_id,
                         cfg.concurrency, cfg.calib_probe_requests)
    attempts: list[dict] = []
    min_valid: float | None = None

    for frac in cfg.calib_ladder:
        attempt_cfg = cfg  # reuse; server argv built per launch below
        attempt_cfg.mem_fraction_static = float(frac)
        logger.info("calibration attempt: mem-fraction=%.2f", frac)
        try:
            proc, client, _port = launch_server(attempt_cfg, log_dir, layout)
        except Exception as e:  # noqa: BLE001
            logger.warning("calibration attempt %.2f launch failed: %s", frac, e)
            attempts.append({"mem_fraction": frac, "launch_error": str(e), "valid": False,
                             "reasons": ["server launch failed"]})
            continue
        try:
            prepare_cold(client)
            probe_records = trace.records[: cfg.calib_probe_requests]
            run = run_load(client, trace, probe_records, concurrency=cfg.concurrency,
                           request_id_offset=0, window_requests=32, max_windows=50)
            verdict = attempt_valid(run, cfg.concurrency)
            attempts.append({
                "mem_fraction": frac,
                "valid": verdict["valid"],
                "reasons": verdict["reasons"],
                "effective_concurrency": verdict["effective_concurrency"],
                "preemption_windows": run.preemption_windows,
                "max_queue_reqs": run.max_queue_reqs,
                "requests_failed": len(run.gen_errors),
            })
            if verdict["valid"]:
                min_valid = float(frac)
        finally:
            proc.stop()

    calib = {
        "calibration_id": cid,
        "trace_id": trace.trace_id,
        "concurrency": cfg.concurrency,
        "probe_requests": cfg.calib_probe_requests,
        "ladder": list(cfg.calib_ladder),
        "min_valid_fraction": min_valid,
        "attempts": attempts,
        "utc": utc_now(),
        "job": job_id(),
    }
    write_json_atomic(layout.calibration_path, calib)
    return calib


def pressure_budget(calib: dict, label: str, default_fraction: float = 0.85) -> float:
    """Map a pressure label to a mem fraction above the calibration floor.

    * Low: default fraction (maximum headroom)
    * Medium: floor + 0.10 (onset candidate above High)
    * High: floor + 0.05 (small headroom above the floor)
    * VeryHigh: floor (only meaningful when floor itself is valid)
    """
    floor = calib.get("min_valid_fraction")
    if floor is None:
        raise ValueError("calibration has no valid floor; sweep cannot proceed")
    floor = float(floor)
    label = str(label).lower()
    if label == "low":
        return float(default_fraction)
    if label == "medium":
        return round(min(default_fraction, floor + 0.10), 2)
    if label == "high":
        return round(min(default_fraction, floor + 0.05), 2)
    if label == "veryhigh":
        return floor
    raise ValueError(f"unknown pressure label {label!r}")


def main(argv=None) -> int:
    setup_logging()
    args = parse_common_args(argv)
    cfg = resolve_config(args, experiment="exp2")
    tag = resolve_run_tag(cfg)
    log_dir = args.get("log_dir") or log_dir_default()
    checks = static_checks(cfg, log_dir)
    layout = RunLayout(run_dir_for(cfg, tag)).create()

    calib = run_calibration(cfg, log_dir, layout)
    # Mirror the calibration to the shared exp2 root for sweep jobs.
    shared_path = shared_calibration_path(cfg.results_root or results_root_default())
    os.makedirs(os.path.dirname(shared_path), exist_ok=True)
    write_json_atomic(shared_path, calib)

    md = base_metadata(cfg, tag, checks)
    md.update({
        "experiment": "exp2_calibration",
        "calibration_id": calib["calibration_id"],
        "validity_status": "valid" if calib["min_valid_fraction"] is not None else "invalid",
        "invalid_reason": "" if calib["min_valid_fraction"] is not None
                          else "no valid floor found on ladder",
    })
    write_json_atomic(layout.metadata_path, md)
    logger.info("calibration complete: floor=%.2f id=%s",
                calib["min_valid_fraction"], calib["calibration_id"])
    return 0


def _exp2_root() -> str:
    from hcv.run_common import results_root_default

    return os.path.join(results_root_default(), "exp2")


if __name__ == "__main__":
    sys.exit(main())
