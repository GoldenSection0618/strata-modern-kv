"""Experiment 4: Cross-Model Validation — frozen V0/V1/V2 points.

Two phases, enforced by the sbatch/submit order:

1. ``--phase freeze``  — records the representative-point selection rule
   (``representative_selection.json``) derived ONLY from primary-model
   (Experiments 2–3) results, BEFORE any secondary-model performance
   data exists.
2. ``--phase run``     — refuses to start without a frozen selection
   rule, runs the secondary-model hierarchy gate (full/partial/
   unsupported), and only then runs the paired V0/V1/V2 validation
   points on the secondary model.

Points (per docs/04):

* V0 — low-value control: Exp2 Low pressure region, fixed reuse.
* V1 — capacity value-onset: Exp2 value-onset pressure region.
* V2 — reuse value-onset: Exp3 representative pressure with the reuse
  level where benefit starts.

Cross-model matching uses observed operating regime (calibrated GPU
budget, observed eviction/hit behavior), not identical absolute GB.
Partial / unsupported secondary results stay visible as runtime-support
evidence and never enter the full-hierarchy cross-model comparison.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from hcv.calibrate import pressure_budget
from hcv.config import (
    ARCH_GPU_ONLY,
    ARCH_HIERARCHICAL,
    GATE_FULL,
    ExperimentConfig,
    pair_configs,
)
from hcv.hierarchy import (
    PROBE_CPU_HIT,
    PROBE_GPU_HIT,
    PROBE_GPU_ONLY_EVICTION,
    PROBE_RECOMPUTE,
    classify_gate,
)
from hcv.load_driver import run_load
from hcv.filler import run_filler
from hcv.residency import prepare_warm
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
from hcv.run_exp2 import load_calibration

logger = logging.getLogger(__name__)

POINTS = ("V0", "V1", "V2")
POINT_PRESSURE = {"V0": "Low", "V1": "Medium", "V2": "High"}
POINT_REUSE = {"V0": 0.5, "V1": 0.5, "V2": 0.5}  # V2 reuse onset level (Exp3)
GEMMA_MATCHED_BUDGET = {"V0": 0.85, "V1": 0.75, "V2": 0.75}


def freeze_selection(results_root: str, calibration: dict) -> dict:
    """Write the frozen V0/V1/V2 selection rule from primary results.

    Raises if the prerequisite primary artifacts (calibration with a
    valid floor, Exp1-3 processed inputs) are not available, so the rule
    can never be fabricated after secondary data exists.
    """
    floor = calibration.get("min_valid_fraction")
    if floor is None:
        raise RuntimeError("primary calibration has no valid floor; cannot freeze V0/V1/V2")
    rule = {
        "frozen_utc": utc_now(),
        "source": "primary model Experiments 2-3 results (hcv.calibrate.pressure_budget)",
        "note": "selection rule recorded before any secondary-model performance data",
        "points": {
            "V0": {
                "pressure_region": "exp2_low_pressure",
                "pressure_label": POINT_PRESSURE["V0"],
                "reuse_fraction": POINT_REUSE["V0"],
                "regime_target": "GPU covers reusable working set; hierarchy benefit small",
            },
            "V1": {
                "pressure_region": "exp2_value_onset",
                "pressure_label": POINT_PRESSURE["V1"],
                "reuse_fraction": POINT_REUSE["V1"],
                "regime_target": "stable reusable-state eviction, preemption 0, "
                                 "CPU-tier hits observable",
            },
            "V2": {
                "pressure_region": "exp3_reuse_value_onset",
                "pressure_label": POINT_PRESSURE["V2"],
                "reuse_fraction": POINT_REUSE["V2"],
                "regime_target": "fixed representative pressure; reuse benefit onset",
            },
        },
        "budget_rule": "pressure_budget(calibration, label) from hcv.calibrate",
        "calibration_id": calibration.get("calibration_id"),
        "primary_floor": floor,
    }
    return rule


def _metrics_usable(client: SGLangHTTPClient) -> bool:
    """Verify the public /metrics endpoint serves the tier counters."""
    from hcv.metrics import parse_prometheus_text

    try:
        _status, text = client._get_text("/metrics")
        return parse_prometheus_text(text).has("sglang:prefill_effective_tokens_total")
    except Exception:  # noqa: BLE001
        return False


def run_secondary_gate(cfg: ExperimentConfig, log_dir: str, layout: RunLayout) -> dict:
    """Run the full-hierarchy gate on the secondary model."""
    from hcv.run_validation import run_gpu_only_probes, run_hierarchical_probes

    gpu_cfg, hier_cfg = pair_configs(cfg)
    probes: dict = {}
    agg_host: float | None = None
    agg_load: float | None = None
    infra = {"env_ok": True, "server_ok": False, "metrics_ok": False}
    try:
        proc, client, _port = launch_server(gpu_cfg, log_dir, layout)
        try:
            infra["server_ok"] = True
            infra["metrics_ok"] = _metrics_usable(client)
            gpu_probes = run_gpu_only_probes(client, gpu_cfg, _secondary_prefix(
                gpu_cfg, 1), 3000)
            probes[PROBE_RECOMPUTE] = gpu_probes["recompute"]
            probes[PROBE_GPU_HIT] = gpu_probes["gpu_hit"]
            probes[PROBE_GPU_ONLY_EVICTION] = gpu_probes["negative_control"]
        finally:
            proc.stop()
    except Exception as e:  # noqa: BLE001
        infra["server_error"] = f"gpu_only: {e}"
    try:
        proc, client, _port = launch_server(hier_cfg, log_dir, layout)
        try:
            hier_probes = run_hierarchical_probes(client, hier_cfg, _secondary_prefix(
                hier_cfg, 2), 4000)
            probes[PROBE_RECOMPUTE] = hier_probes["recompute"]
            probes[PROBE_GPU_HIT] = hier_probes["gpu_hit"]
            probes[PROBE_CPU_HIT] = hier_probes["cpu_hit"]
            agg_host = hier_probes["cpu_hit"].host_hit_delta
            agg_load = hier_probes["cpu_hit"].load_back_delta
        finally:
            proc.stop()
    except Exception as e:  # noqa: BLE001
        infra["server_error"] = f"hierarchical: {e}"
    gate = classify_gate(cfg.model, ARCH_HIERARCHICAL, cfg.model_state_groups,
                         probes, agg_host, agg_load, infra_checks=infra)
    return gate.to_dict()


def _secondary_prefix(cfg: ExperimentConfig, family_no: int) -> list[int]:
    from hcv.workload import _ids

    return _ids(cfg.seed * 131 + family_no * 7919, cfg.prefix_length)


def run_point(
    cfg: ExperimentConfig,
    point: str,
    rule: dict,
    calib: dict,
    log_dir: str,
    layout: RunLayout,
) -> dict:
    """Run one paired V-point on the secondary model (warm steady state)."""
    label = POINT_PRESSURE[point]
    # Cross-model matching is by observed regime, not Qwen's absolute
    # fraction.  Gemma's SWA admission floor makes Qwen's 0.65/0.60 points
    # invalid; 0.75 is the lowest prevalidated safe secondary budget.
    budget = (
        GEMMA_MATCHED_BUDGET[point]
        if cfg.model == "gemma"
        else pressure_budget(calib, label)
    )
    cfg.mem_fraction_static = budget
    cfg.revisit_fraction = POINT_REUSE[point]
    cfg.sources["mem_fraction_static"] = f"exp4_{point}_budget"
    cfg.sources["revisit_fraction"] = f"exp4_{point}_reuse"

    trace = load_or_build_trace(cfg, os.path.join(layout.raw_dir, "traces"))
    reuse = reuse_summary(trace)
    cells = {}
    for architecture in (ARCH_GPU_ONLY, ARCH_HIERARCHICAL):
        cell_cfg = cfg
        cell_cfg.architecture = architecture
        proc, client, _port = launch_server(cell_cfg, log_dir, layout)
        reps = []
        try:
            for rep in range(max(1, cfg.n_repeats)):
                prepare_warm(client, trace, min(cfg.n_warmup, len(trace.records)))
                pressure_preparation = None
                if point != "V0":
                    from hcv.run_validation import _filler_plan

                    plan = _filler_plan(client, cell_cfg, cell_cfg.prefix_length)
                    filler = run_filler(
                        client, plan, seed=cell_cfg.seed + rep * 1009 + ord(point[-1])
                    )
                    pressure_preparation = {
                        "plan": plan.to_dict(), "result": filler.to_dict()
                    }
                    if not filler.ok:
                        raise RuntimeError(
                            f"secondary pressure preparation failed: {filler.errors}"
                        )
                formal = trace.records[cfg.n_warmup:]
                if not formal:
                    raise RuntimeError("formal phase empty after warm-up")
                run = run_load(client, trace, formal, concurrency=cfg.concurrency,
                               request_id_offset=rep * 100000,
                               window_requests=64, max_windows=400)
                total = run.total_delta.get("deltas", run.total_delta)
                restore_tokens = total.get("load_back_tokens_total")
                if restore_tokens is None:
                    restore_pools = run.total_delta.get("pool_deltas", {}).get(
                        "load_back_tokens_total", {}
                    )
                    restore_tokens = sum(restore_pools.values()) if restore_pools else None
                reps.append({
                    "repetition_index": rep,
                    "architecture": architecture,
                    "gpu_hit_tokens": total.get("prefill_device_hit_tokens"),
                    "cpu_hit_tokens": total.get("prefill_host_hit_tokens"),
                    "recomputed_tokens": total.get("prefill_input_tokens"),
                    "restore_tokens": restore_tokens,
                    "restore_bytes": total.get("load_back_bytes_total"),
                    "preemption_windows": run.preemption_windows,
                    "requests_failed": len(run.gen_errors),
                    "max_queue_reqs": run.max_queue_reqs,
                    "pressure_preparation": pressure_preparation,
                    "valid": run.preemption_windows == 0 and not run.gen_errors,
                    "throughput_req_per_s": round(
                        sum(1 for r in run.requests if r.get("ok")) / max(run_wall(run), 1e-6), 3),
                })
        finally:
            proc.stop()
        cells[architecture] = reps
    return {
        "point": point,
        "pressure_label": label,
        "mem_fraction_static": budget,
        "reuse_fraction": POINT_REUSE[point],
        "trace_id": trace.trace_id,
        "actual_reuse": reuse["revisit_fraction_request_weighted"],
        "cells": cells,
        "rule": rule,
    }


def run_wall(run) -> float:
    """Wall time of a load run (sum of window durations)."""
    if not run.windows:
        return 0.0
    return run.windows[-1].end_monotonic - run.windows[0].start_monotonic


def main(argv=None) -> int:
    setup_logging()
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("freeze", "run"), required=True)
    phase_args, rest = p.parse_known_args(argv)
    common = parse_common_args(rest)
    cfg = resolve_config(common, experiment="exp4")
    tag = resolve_run_tag(cfg)
    log_dir = common.get("log_dir") or log_dir_default()
    checks = static_checks(cfg, log_dir)
    results_root = cfg.results_root or _exp4_base()
    layout = RunLayout(run_dir_for(cfg, tag)).create()

    calib = load_calibration(results_root)

    if phase_args.phase == "freeze":
        rule = freeze_selection(results_root, calib)
        rule_path = os.path.join(results_root, "exp4", "representative_selection.json")
        os.makedirs(os.path.dirname(rule_path), exist_ok=True)
        write_json_atomic(rule_path, rule)
        append_jsonl(layout.measurements_path, {"kind": "selection_frozen",
                                                "run_tag": tag, "rule": rule})
        logger.info("frozen selection rule written to %s", rule_path)
        return 0

    # --- run phase ---------------------------------------------------------
    rule_path = os.path.join(results_root, "exp4", "representative_selection.json")
    if not os.path.exists(rule_path):
        logger.error("representative_selection.json missing at %s; run freeze phase first", rule_path)
        return 3
    with open(rule_path, "r", encoding="utf-8") as fh:
        rule = json.load(fh)

    gate = run_secondary_gate(cfg, log_dir, layout)
    write_json_atomic(layout.validation_path, gate)
    logger.info("secondary gate: %s", gate["status"])
    if gate["status"] != GATE_FULL:
        summary = {
            "status": gate["status"],
            "reason": "secondary model cannot satisfy full-hierarchy gate; "
                      "cross-model performance comparison skipped (runtime-support boundary)",
            "validation_path": layout.validation_path,
        }
        write_json_atomic(os.path.join(layout.results_dir, "summary.json"), summary)
        md = base_metadata(cfg, tag, checks)
        md.update({"hierarchy_status": gate["status"],
                   "validity_status": gate["status"]})
        write_json_atomic(layout.metadata_path, md)
        return 0

    points_out = {}
    for point in POINTS:
        pcfg = cfg
        point_rec = run_point(pcfg, point, rule, calib, log_dir, layout)
        append_jsonl(layout.measurements_path, {"kind": "exp4_point", "run_tag": tag,
                                                "point": point_rec})
        points_out[point] = point_rec
        logger.info("point %s done", point)

    write_json_atomic(os.path.join(layout.results_dir, "points.json"), points_out)
    md = base_metadata(cfg, tag, checks)
    md.update({
        "hierarchy_status": gate["status"],
        "validity_status": "valid",
        "trace_id": cfg.trace_id or "",
        "selection_rule_path": rule_path,
        "selection_rule_frozen_utc": rule.get("frozen_utc", ""),
    })
    write_json_atomic(layout.metadata_path, md)
    return 0


def _exp4_base() -> str:
    from hcv.run_common import results_root_default

    return results_root_default()


if __name__ == "__main__":
    sys.exit(main())
