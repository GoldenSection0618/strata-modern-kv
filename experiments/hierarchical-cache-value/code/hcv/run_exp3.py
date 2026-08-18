"""Experiment 3: Prefix Reuse Scaling — one reuse level per job.

Each runner invocation executes one (architecture, reuse-level) point at
the fixed pressure budget selected from Experiment 2 (same calibration
floor derivation as ``hcv.run_exp2``).  Only the revisit fraction
changes: the trace keeps identical eligible revisit slots, request
ordering, prefix-length distribution, suffix structure and hotspot
concentration; lower-reuse traces replace revisits with matched unique
prefixes at the same positions (``hcv.workload.build_trace``).

Validity:
* actual request-weighted reuse within tolerance of the configured
  fraction;
* preemption == 0, no request failures, concurrency within tolerance;
* CPU-tier capacity eviction absent (hierarchical runs);
* locality-unchanged cross-level checks are performed by
  ``hcv.analysis`` on the saved traces.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time

from hcv.calibrate import pressure_budget
from hcv.config import ARCH_GPU_ONLY, ARCH_HIERARCHICAL
from hcv.filler import build_fixed_filler_plan, run_filler
from hcv.load_driver import run_load
from hcv.residency import prepare_warm, snapshot_cache_state, warm_state_grew
from hcv.run_common import (
    base_metadata,
    job_id,
    launch_server,
    load_or_build_trace,
    log_dir_default,
    parse_common_args,
    resolve_config,
    resolve_run_tag,
    run_dir_for,
    setup_logging,
    static_checks,
    utc_now,
)
from hcv.schema import RunLayout, append_jsonl, write_json_atomic
from hcv.workload import reuse_summary
from hcv.run_exp1 import find_latest_gate_status
from hcv.run_exp2 import (
    FIXED_PRESSURE_FILLER_TOKENS,
    cpu_tier_eviction_evidence,
    load_calibration,
)

logger = logging.getLogger(__name__)

REUSE_LEVELS = (0.0, 0.25, 0.5, 0.75)
REUSE_TOLERANCE_FLOOR = 0.05


def reuse_validity(
    configured: float, actual: float, eligible_slots: int, total_slots: int
) -> dict:
    # The configured level is a deterministic Bernoulli threshold, not an
    # exact quota.  Accept ordinary finite-trace variation (3 sigma), while
    # always preserving/reporting the actual realized fraction.
    expected = configured * eligible_slots / max(1, total_slots)
    sigma = math.sqrt(eligible_slots * configured * (1.0 - configured)) / max(1, total_slots)
    tolerance = max(REUSE_TOLERANCE_FLOOR, 3.0 * sigma)
    reasons = []
    if abs(actual - expected) > tolerance:
        reasons.append(
            f"actual reuse {actual:.3f} outside expected {expected:.3f} "
            f"for configured threshold {configured:.3f}; "
            f"tolerance {tolerance:.3f}"
        )
    return {
        "valid": not reasons,
        "reasons": reasons,
        "expected_reported_fraction": expected,
        "tolerance": tolerance,
    }


def main(argv=None) -> int:
    setup_logging()
    args = parse_common_args(argv)
    cfg = resolve_config(args, experiment="exp3")
    tag = resolve_run_tag(cfg)
    log_dir = args.get("log_dir") or log_dir_default()
    checks = static_checks(cfg, log_dir)

    architecture = cfg.architecture
    level = cfg.revisit_fraction
    if architecture not in (ARCH_GPU_ONLY, ARCH_HIERARCHICAL):
        logger.error("exp3 requires architecture; got %s", architecture)
        return 2
    if level not in REUSE_LEVELS:
        logger.warning("reuse level %.2f not in standard levels %s; continuing",
                       level, REUSE_LEVELS)

    results_root = cfg.results_root or _exp3_base()
    calib = load_calibration(results_root)
    budget = pressure_budget(calib, cfg.pressure_label)
    cfg.mem_fraction_static = budget
    cfg.sources["mem_fraction_static"] = "exp3_fixed_pressure"
    logger.info("exp3 point: arch=%s reuse=%.2f budget=%.2f calib_id=%s",
                architecture, level, budget, calib["calibration_id"])

    layout = RunLayout(run_dir_for(cfg, tag)).create()
    trace = load_or_build_trace(cfg, os.path.join(layout.raw_dir, "traces"))
    reuse = reuse_summary(trace)
    gate_status = find_latest_gate_status(results_root, cfg.model)

    start_mono = time.monotonic()
    proc, client, _port = launch_server(cfg, log_dir, layout)
    repetitions = []
    try:
        for rep in range(max(1, cfg.n_repeats)):
            rep_start = time.monotonic()
            base_state = snapshot_cache_state(client)
            warm_state = prepare_warm(client, trace, min(cfg.n_warmup, len(trace.records)))
            warm_grew = warm_state_grew(warm_state, base_state)
            filler_plan = build_fixed_filler_plan(
                FIXED_PRESSURE_FILLER_TOKENS,
                filler_prefix_length=cfg.prefix_length,
            )
            filler = run_filler(client, filler_plan, seed=cfg.seed + rep * 1009)
            append_jsonl(layout.measurements_path, {
                "kind": "pressure_preparation",
                "run_tag": tag,
                "repetition_index": rep,
                "architecture": architecture,
                "configured_revisit_fraction": level,
                "plan": filler_plan.to_dict(),
                "result": filler.to_dict(),
            })
            if not filler.ok:
                raise RuntimeError(f"fixed pressure preparation failed: {filler.errors}")
            formal_records = trace.records[cfg.n_warmup:]
            if not formal_records:
                raise RuntimeError("formal phase empty after warm-up")
            run = run_load(client, trace, formal_records, concurrency=cfg.concurrency,
                           request_id_offset=rep * 100000, window_requests=64, max_windows=400)
            duration_s = time.monotonic() - rep_start
            cpu_ev = cpu_tier_eviction_evidence(run)
            eligible_slots = len(trace.records) - cfg.num_prefix_families
            rvalid = reuse_validity(
                level,
                reuse["revisit_fraction_request_weighted"],
                eligible_slots,
                len(trace.records),
            )
            reasons = list(rvalid["reasons"])
            if run.preemption_windows > 0:
                reasons.append("active-request preemption")
            if run.gen_errors:
                reasons.append(f"{len(run.gen_errors)} request failures")
            if cpu_ev.get("near_capacity"):
                reasons.append("uncontrolled CPU-tier capacity eviction")
            ecs = [w.effective_concurrency for w in run.windows if w.effective_concurrency is not None]
            if ecs:
                mean_ec = sum(ecs) / len(ecs)
                if abs(mean_ec - cfg.concurrency) > 0.5:
                    reasons.append(f"effective concurrency drift ({mean_ec:.2f})")
            validity = {"valid": not reasons, "reasons": reasons}
            summary = {
                "requests_completed": sum(1 for r in run.requests if r.get("ok")),
                "requests_failed": len(run.gen_errors),
                "gpu_hit_tokens": run.total_delta.get("deltas", {}).get("prefill_device_hit_tokens"),
                "cpu_hit_tokens": run.total_delta.get("deltas", {}).get("prefill_host_hit_tokens"),
                "recomputed_tokens": run.total_delta.get("deltas", {}).get("prefill_input_tokens"),
                "restore_tokens": run.total_delta.get("deltas", {}).get("load_back_tokens_total"),
                "restore_bytes": run.total_delta.get("deltas", {}).get("load_back_bytes_total"),
                "throughput_req_per_s": round(sum(1 for r in run.requests if r.get("ok")) / max(duration_s, 1e-6), 3),
                "preemption_windows": run.preemption_windows,
                "max_queue_reqs": run.max_queue_reqs,
                "duration_s": round(duration_s, 3),
            }
            rec = {
                "kind": "reuse_point",
                "run_tag": tag,
                "repetition_index": rep,
                "architecture": architecture,
                "configured_revisit_fraction": level,
                "actual_reuse_request_weighted": reuse["revisit_fraction_request_weighted"],
                "actual_reuse_token_weighted": reuse["revisit_fraction_token_weighted"],
                "reuse_distance": reuse["reuse_distance"],
                "unique_prefix_count": reuse["unique_prefix_count"],
                "mem_fraction_static": budget,
                "calibration_id": calib["calibration_id"],
                "cpu_tier_evidence": cpu_ev,
                "validity": validity,
                "summary": summary,
                "warm_grew": warm_grew,
            }
            append_jsonl(layout.measurements_path, rec)
            repetitions.append(rec)
            logger.info("rep %d done: valid=%s", rep, validity["valid"])
    finally:
        proc.stop()

    total_s = time.monotonic() - start_mono
    valid_reps = [r for r in repetitions if r["validity"]["valid"]]
    cell = {
        "architecture": architecture,
        "configured_revisit_fraction": level,
        "mem_fraction_static": budget,
        "calibration_id": calib["calibration_id"],
        "n_repetitions": len(repetitions),
        "n_valid_repetitions": len(valid_reps),
        "reportable": gate_status["status"] == "full" and len(valid_reps) > 0,
        "gate_status": gate_status,
        "total_wall_s": round(total_s, 3),
    }
    write_json_atomic(os.path.join(layout.results_dir, "cell_summary.json"), cell)
    md = base_metadata(cfg, tag, checks)
    md.update({
        "hierarchy_status": gate_status["status"],
        "validity_status": "valid" if len(valid_reps) == len(repetitions) else "invalid",
        "invalid_reason": "" if len(valid_reps) == len(repetitions) else
                          "some repetitions invalid (see raw)",
        "calibration_id": calib["calibration_id"],
        "trace_id": trace.trace_id,
        "configured_revisit_fraction": level,
        "actual_reuse_request_weighted": reuse["revisit_fraction_request_weighted"],
        "actual_reuse_token_weighted": reuse["revisit_fraction_token_weighted"],
        "reuse_distance_summary": reuse["reuse_distance"],
        "gpu_cache_budget": budget,
    })
    write_json_atomic(layout.metadata_path, md)
    logger.info("point complete: arch=%s reuse=%.2f reportable=%s",
                architecture, level, cell["reportable"])
    return 0


def _exp3_base() -> str:
    from hcv.run_common import results_root_default

    return results_root_default()


if __name__ == "__main__":
    sys.exit(main())
