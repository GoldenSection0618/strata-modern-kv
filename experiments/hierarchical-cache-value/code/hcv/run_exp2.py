"""Experiment 2: GPU Cache Pressure Scaling — one pressure point per job.

Each runner invocation executes one (architecture, pressure-label) point
on the fixed warm-cache steady-state workload.  The GPU budget is
derived from the shared ``calibration.json`` floor (see ``hcv.calibrate``);
only the reusable-cache headroom above the floor is compressed.

Pressure validity (recorded per run; invalid points stay visible in raw
but are excluded from the main capacity curve by ``hcv.analysis``):

* OOM / server failure / request failures;
* active-request preemption (runtime retracted-request counter);
* effective-concurrency drift;
* uncontrolled CPU-tier capacity eviction (hierarchical runs where host
  occupancy reaches the host pool capacity);
* pressure evidence mismatch: High/Medium points must show observed L1
  eviction; Low must show little eviction.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from hcv.calibrate import pressure_budget
from hcv.config import ARCH_GPU_ONLY, ARCH_HIERARCHICAL
from hcv.filler import build_fixed_filler_plan, run_filler
from hcv.hierarchy import PROBE_CPU_HIT, PROBE_GPU_HIT, PROBE_GPU_ONLY_EVICTION
from hcv.load_driver import run_load
from hcv.probes import ProbeSpec, run_serial_probe
from hcv.residency import prepare_cold, prepare_warm, snapshot_cache_state, warm_state_grew
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
from hcv.run_exp1 import _percentile, find_latest_gate_status

logger = logging.getLogger(__name__)

FIXED_PRESSURE_FILLER_TOKENS = 81920


def load_calibration(results_root: str) -> dict:
    """Read the shared exp2 calibration file (hard error when missing)."""
    path = os.path.join(results_root, "exp2", "calibration", "calibration.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"exp2 calibration file missing at {path}; run the calibration "
            "sbatch before any pressure point"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def cpu_tier_eviction_evidence(run) -> dict:
    """Host-pool capacity pressure check for hierarchical runs."""
    host_used = run.end_snapshot.get("hicache_host_used_tokens")
    host_total = run.end_snapshot.get("hicache_host_total_tokens")
    if host_used is None or host_total is None or host_total <= 0:
        return {"observable": False, "near_capacity": False}
    ratio = host_used / host_total
    return {"observable": True, "near_capacity": ratio >= 0.95, "ratio": round(ratio, 3)}


def eviction_evidence(run, architecture: str, pressure_probe) -> dict:
    """Pressure evidence from the formal window plus a serial diagnostic.

    Hierarchical eviction is established by host-hit plus H->D load-back in
    the measured workload.  A single serial prefix is diagnostic only: the
    runtime may consume/retire that specific host copy even while other
    workload entries restore normally.
    """
    total = run.total_delta.get("deltas", run.total_delta)
    backup = total.get("hicache_backup_tokens_total")
    evictable = run.end_snapshot.get("kv_evictable_tokens")
    available = run.end_snapshot.get("kv_available_tokens")
    host_hit = total.get("prefill_host_hit_tokens")
    load_back = total.get("load_back_tokens_total")
    if load_back is None:
        pools = run.total_delta.get("pool_deltas", {}).get("load_back_tokens_total", {})
        load_back = sum(pools.values()) if pools else None
    formal_restore = (
        architecture == ARCH_HIERARCHICAL
        and host_hit is not None and host_hit > 0
        and load_back is not None and load_back > 0
    )
    return {
        "eviction_tokens": backup,
        "kv_evictable_tokens": evictable,
        "kv_available_tokens": available,
        "direct_probe": pressure_probe.to_dict(),
        "formal_host_hit_tokens": host_hit,
        "formal_load_back_tokens": load_back,
        "eviction_observed": (
            formal_restore
            if architecture == ARCH_HIERARCHICAL
            else pressure_probe.name != PROBE_GPU_HIT and pressure_probe.ok
        ),
        "gpu_residency_preserved": pressure_probe.name == PROBE_GPU_HIT and pressure_probe.ok,
    }


def pressure_validity(label: str, ev: dict, cpu_ev: dict, run, concurrency: int) -> dict:
    """Point validity for the main capacity curve."""
    reasons = []
    if run.gen_errors:
        reasons.append(f"{len(run.gen_errors)} request failures")
    if run.preemption_windows > 0:
        reasons.append("active-request preemption")
    ecs = [w.effective_concurrency for w in run.windows if w.effective_concurrency is not None]
    if ecs:
        mean_ec = sum(ecs) / len(ecs)
        if abs(mean_ec - concurrency) > 0.5:
            reasons.append(f"effective concurrency drift ({mean_ec:.2f} vs {concurrency})")
    if cpu_ev.get("near_capacity"):
        reasons.append("uncontrolled CPU-tier capacity eviction")
    label = label.lower()
    probe = ev.get("direct_probe", {})
    if label == "low" and not probe.get("ok"):
        reasons.append(f"direct Low-residency probe failed: {probe.get('reason', 'missing result')}")
    if label in ("medium", "high", "veryhigh") and not ev.get("eviction_observed"):
        reasons.append("pressure evidence missing (protected prefix was not evicted)")
    if label == "low" and not ev.get("gpu_residency_preserved"):
        reasons.append("Low point did not preserve the protected prefix in L1")
    return {"valid": not reasons, "reasons": reasons}


def main(argv=None) -> int:
    setup_logging()
    args = parse_common_args(argv)
    cfg = resolve_config(args, experiment="exp2")
    tag = resolve_run_tag(cfg)
    log_dir = args.get("log_dir") or log_dir_default()
    checks = static_checks(cfg, log_dir)

    architecture = cfg.architecture
    label = cfg.pressure_label
    if architecture not in (ARCH_GPU_ONLY, ARCH_HIERARCHICAL):
        logger.error("exp2 requires architecture; got %s", architecture)
        return 2

    results_root = cfg.results_root or _exp2_base()
    calib = load_calibration(results_root)
    budget = pressure_budget(calib, label)
    cfg.mem_fraction_static = budget
    cfg.sources["mem_fraction_static"] = "exp2_pressure_budget"
    logger.info("exp2 point: arch=%s label=%s budget=%.2f calib_id=%s",
                architecture, label, budget, calib["calibration_id"])

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
            # Pick the last actual family revisit in the warm phase.  It has
            # write-through history from the family's first occurrence and
            # is recently resident in L1, avoiding both stale-entry and
            # fresh-prefix admission artifacts.
            warm_records = trace.records[:min(cfg.n_warmup, len(trace.records))]
            protected_record = next(r for r in reversed(warm_records) if r.is_revisit)
            protected = protected_record.input_ids
            pre_probe = run_serial_probe(client, ProbeSpec(
                name=PROBE_GPU_HIT,
                input_ids=protected,
                prefix_length=cfg.prefix_length,
                request_id=900000000 + rep * 10,
            ))
            if not pre_probe.ok:
                raise RuntimeError(f"protected prefix was not resident before pressure: {pre_probe.reason}")
            if label.lower() == "low":
                # Low is the unforced warm steady state.  Injecting a long
                # one-pass filler would itself change locality/LRU age and
                # manufacture pressure even when ample token capacity remains.
                post_probe = pre_probe
            else:
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
                    "pressure_label": label,
                    "plan": filler_plan.to_dict(),
                    "result": filler.to_dict(),
                })
                if not filler.ok:
                    raise RuntimeError(f"fixed pressure preparation failed: {filler.errors}")
                post_probe_name = (
                    PROBE_CPU_HIT
                    if architecture == ARCH_HIERARCHICAL
                    else PROBE_GPU_ONLY_EVICTION
                )
                post_probe = run_serial_probe(client, ProbeSpec(
                    name=post_probe_name,
                    input_ids=protected,
                    prefix_length=cfg.prefix_length,
                    request_id=900000001 + rep * 10,
                ))
            append_jsonl(layout.measurements_path, {
                "kind": "pressure_probe",
                "run_tag": tag,
                "repetition_index": rep,
                "architecture": architecture,
                "pressure_label": label,
                "filler_applied": label.lower() != "low",
                "protected_trace_index": protected_record.idx,
                "before_filler": pre_probe.to_dict(),
                "after_filler": post_probe.to_dict(),
            })
            formal_records = trace.records[cfg.n_warmup:]
            if not formal_records:
                raise RuntimeError("formal phase empty after warm-up")
            run = run_load(client, trace, formal_records, concurrency=cfg.concurrency,
                           request_id_offset=rep * 100000, window_requests=64, max_windows=400)
            duration_s = time.monotonic() - rep_start
            ev = eviction_evidence(run, architecture, post_probe)
            cpu_ev = cpu_tier_eviction_evidence(run)
            validity = pressure_validity(label, ev, cpu_ev, run, cfg.concurrency)
            summary = {
                "requests_completed": sum(1 for r in run.requests if r.get("ok")),
                "requests_failed": len(run.gen_errors),
                "gpu_hit_tokens": run.total_delta.get("deltas", {}).get("prefill_device_hit_tokens"),
                "cpu_hit_tokens": run.total_delta.get("deltas", {}).get("prefill_host_hit_tokens"),
                "recomputed_tokens": run.total_delta.get("deltas", {}).get("prefill_input_tokens"),
                "restore_tokens": run.total_delta.get("deltas", {}).get("load_back_tokens_total"),
                "restore_bytes": run.total_delta.get("deltas", {}).get("load_back_bytes_total"),
                "ttft_p50_ms": _percentile([r.get("ttft_ms", 0.0) for r in run.requests if r.get("ok")], 50),
                "ttft_p90_ms": _percentile([r.get("ttft_ms", 0.0) for r in run.requests if r.get("ok")], 90),
                "throughput_req_per_s": round(sum(1 for r in run.requests if r.get("ok")) / max(duration_s, 1e-6), 3),
                "preemption_windows": run.preemption_windows,
                "max_queue_reqs": run.max_queue_reqs,
                "duration_s": round(duration_s, 3),
            }
            rec = {
                "kind": "pressure_point",
                "run_tag": tag,
                "repetition_index": rep,
                "architecture": architecture,
                "pressure_label": label,
                "mem_fraction_static": budget,
                "calibration_id": calib["calibration_id"],
                "eviction_evidence": ev,
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
        "pressure_label": label,
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
        "actual_reuse_request_weighted": reuse["revisit_fraction_request_weighted"],
        "actual_reuse_token_weighted": reuse["revisit_fraction_token_weighted"],
        "reuse_distance_summary": reuse["reuse_distance"],
        "gpu_cache_budget": budget,
    })
    write_json_atomic(layout.metadata_path, md)
    logger.info("point complete: arch=%s label=%s budget=%.2f reportable=%s",
                architecture, label, budget, cell["reportable"])
    return 0


def _exp2_base() -> str:
    from hcv.run_common import results_root_default

    return results_root_default()


if __name__ == "__main__":
    sys.exit(main())
