"""Experiment 1: Baseline Benefit — GPU-only vs hierarchical x cold/warm.

Each runner invocation executes one (architecture, initial-state) cell
with ``n_repeats`` independent repetitions on the fixed paired trace and
GPU budget.  The submit script alternates the architecture order so
machine drift does not systematically favor one side.

Per repetition the runner records:

* total tier deltas (GPU hit / CPU hit / eviction / recompute / restore);
* per-window phased aggregates (cold-cache progression evidence);
* TTFT distribution, throughput, achieved rate, preemption, effective
  concurrency;
* validity (preemption == 0, no gen errors, concurrency within
  tolerance).

``hierarchy_status`` is copied from the newest reference
``validation.json`` for the model: reportable results require the gate
to be ``full`` (see ``hcv.analysis`` filtering).
"""

from __future__ import annotations

import logging
import os
import statistics
import sys
import time

from hcv.config import (
    ARCH_GPU_ONLY,
    ARCH_HIERARCHICAL,
    STATE_COLD,
    STATE_WARM,
)
from hcv.load_driver import run_load
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

logger = logging.getLogger(__name__)


def find_latest_gate_status(results_root: str, model: str) -> dict:
    """Latest ``validation.json`` status for a model (reportability gate)."""
    import json as _json
    from pathlib import Path

    base = Path(results_root) / "validation"
    if not base.is_dir():
        return {"status": "missing", "path": ""}
    candidates = []
    for run_dir in sorted(base.glob("run-*"), reverse=True):
        vpath = run_dir / "validation.json"
        if not vpath.is_file():
            continue
        try:
            data = _json.loads(vpath.read_text())
        except ValueError:
            continue
        if data.get("model") == model:
            candidates.append((run_dir.name, data.get("status", "unknown")))
    if not candidates:
        return {"status": "missing", "path": ""}
    name, status = candidates[0]
    return {"status": status, "path": str(base / name / "validation.json")}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]


def summarize_repetition(run, duration_s: float) -> dict:
    """Derive the per-repetition metric summary from a LoadRunResult."""
    reqs = run.requests
    ok_reqs = [r for r in reqs if r.get("ok")]
    ttfts = [float(r.get("ttft_ms", 0.0)) for r in ok_reqs]
    total = run.total_delta.get("deltas", run.total_delta)

    def d(key):
        return total.get(key)

    n_windows = len(run.windows)
    phases: dict[str, dict] = {}
    if n_windows >= 4:
        q = n_windows // 4
        labels = ["q1", "q2", "q3", "q4"]
        for i, label in enumerate(labels):
            ws = run.windows[i * q:(i + 1) * q]
            dev = sum(w.device_hit_delta for w in ws if w.device_hit_delta is not None)
            host = sum(w.host_hit_delta for w in ws if w.host_hit_delta is not None)
            inp = sum(w.tier_hits_total for w in ws if w.tier_hits_total is not None) or 0
            phases[label] = {
                "windows": len(ws),
                "device_hit_tokens": round(dev, 1),
                "host_hit_tokens": round(host, 1),
                "tier_hits_total": round(inp, 1),
            }
    return {
        "requests_completed": len(ok_reqs),
        "requests_failed": len(reqs) - len(ok_reqs),
        "gpu_hit_tokens": d("prefill_device_hit_tokens"),
        "cpu_hit_tokens": d("prefill_host_hit_tokens"),
        "recomputed_tokens": d("prefill_input_tokens"),
        "eviction_tokens": d("hicache_backup_tokens_total"),
        "restore_tokens": d("load_back_tokens_total"),
        "restore_bytes": d("load_back_bytes_total"),
        "ttft_p50_ms": round(_percentile(ttfts, 50), 3) if ttfts else None,
        "ttft_p90_ms": round(_percentile(ttfts, 90), 3) if ttfts else None,
        "ttft_p99_ms": round(_percentile(ttfts, 99), 3) if ttfts else None,
        "ttft_mean_ms": round(statistics.fmean(ttfts), 3) if ttfts else None,
        "throughput_req_per_s": round(len(ok_reqs) / duration_s, 3) if duration_s > 0 else None,
        "achieved_request_rate": round(len(ok_reqs) / duration_s, 3) if duration_s > 0 else None,
        "preemption_windows": run.preemption_windows,
        "concurrency_drift_windows": run.concurrency_drift_windows,
        "max_queue_reqs": run.max_queue_reqs,
        "effective_concurrency": _effective_concurrency(ok_reqs, duration_s),
        "phases": phases,
        "windows": n_windows,
    }


def _effective_concurrency(ok_reqs: list[dict], duration_s: float) -> float | None:
    if not ok_reqs or duration_s <= 0:
        return None
    total_ttft_s = sum(float(r.get("ttft_ms", 0.0)) for r in ok_reqs) / 1000.0
    return round(total_ttft_s / duration_s, 3)


def _validity(summary: dict, concurrency: int, tolerance: float = 0.5) -> dict:
    reasons = []
    if summary["preemption_windows"] > 0:
        reasons.append("active-request preemption observed")
    if summary["requests_failed"] > 0:
        reasons.append(f"{summary['requests_failed']} requests failed")
    ec = summary["effective_concurrency"]
    if ec is not None and abs(ec - concurrency) > tolerance:
        reasons.append(f"effective concurrency drift ({ec} vs {concurrency})")
    return {
        "valid": not reasons,
        "reasons": reasons,
    }


def main(argv=None) -> int:
    setup_logging()
    args = parse_common_args(argv)
    cfg = resolve_config(args, experiment="exp1")
    tag = resolve_run_tag(cfg)
    log_dir = args.get("log_dir") or log_dir_default()
    checks = static_checks(cfg, log_dir)

    architecture = cfg.architecture
    initial_state = cfg.initial_state
    if architecture not in (ARCH_GPU_ONLY, ARCH_HIERARCHICAL):
        logger.error("exp1 requires architecture gpu_only|hierarchical, got %s", architecture)
        return 2
    if initial_state not in (STATE_COLD, STATE_WARM):
        logger.error("exp1 requires initial_state cold|warm, got %s", initial_state)
        return 2

    layout = RunLayout(run_dir_for(cfg, tag)).create()
    trace_cache = os.path.join(layout.raw_dir, "traces")
    trace = load_or_build_trace(cfg, trace_cache)
    reuse = reuse_summary(trace)
    logger.info("exp1 cell: arch=%s state=%s trace=%s requests=%d",
                architecture, initial_state, trace.trace_id, len(trace))

    gate_status = find_latest_gate_status(cfg.results_root or _results_root(), cfg.model)

    start_mono = time.monotonic()
    proc, client, _port = launch_server(cfg, log_dir, layout)
    repetitions = []
    try:
        for rep in range(max(1, cfg.n_repeats)):
            rep_start = time.monotonic()
            warm_state = None
            if initial_state == STATE_WARM:
                base_state = snapshot_cache_state(client)
                warm_state = prepare_warm(client, trace, min(cfg.n_warmup, len(trace.records)))
                warm_grew = warm_state_grew(warm_state, base_state)
                formal_records = trace.records[cfg.n_warmup:]
                logger.info("rep %d warm populated=%d grew=%s", rep, cfg.n_warmup, warm_grew)
            else:
                prepare_cold(client)
                formal_records = trace.records
                warm_grew = None

            if not formal_records:
                raise RuntimeError("formal phase empty after warm-up/initialization")

            run = run_load(
                client, trace, formal_records,
                concurrency=cfg.concurrency,
                request_id_offset=rep * 100000,
                window_requests=64,
                max_windows=400,
            )
            duration_s = time.monotonic() - rep_start
            summary = summarize_repetition(run, duration_s)
            validity = _validity(summary, cfg.concurrency)
            rep_record = {
                "kind": "repetition",
                "run_tag": tag,
                "repetition_index": rep,
                "architecture": architecture,
                "initial_state": initial_state,
                "summary": summary,
                "validity": validity,
                "warm_grew": warm_grew,
                "warm_state": warm_state.to_dict() if warm_state else None,
                "duration_s": round(duration_s, 3),
            }
            append_jsonl(layout.measurements_path, rep_record)
            repetitions.append(rep_record)
            logger.info("rep %d done: valid=%s ttft_p50=%.1fms throughput=%.2f/s",
                        rep, validity["valid"],
                        summary["ttft_p50_ms"] or float("nan"),
                        summary["throughput_req_per_s"] or float("nan"))
    finally:
        proc.stop()

    total_s = time.monotonic() - start_mono
    valid_reps = [r for r in repetitions if r["validity"]["valid"]]
    summary_all = {
        "architecture": architecture,
        "initial_state": initial_state,
        "trace_id": trace.trace_id,
        "trace_reuse": reuse,
        "gate_status": gate_status,
        "n_repetitions": len(repetitions),
        "n_valid_repetitions": len(valid_reps),
        "reportable": gate_status["status"] == "full" and len(valid_reps) > 0,
        "total_wall_s": round(total_s, 3),
    }
    write_json_atomic(os.path.join(layout.results_dir, "cell_summary.json"), summary_all)

    md = base_metadata(cfg, tag, checks)
    md.update({
        "hierarchy_status": gate_status["status"],
        "validity_status": "valid" if len(valid_reps) == len(repetitions) else "invalid",
        "invalid_reason": "" if len(valid_reps) == len(repetitions) else
                          "some repetitions invalid (see raw)",
        "trace_id": trace.trace_id,
        "actual_reuse_request_weighted": reuse["revisit_fraction_request_weighted"],
        "actual_reuse_token_weighted": reuse["revisit_fraction_token_weighted"],
        "reuse_distance_summary": reuse["reuse_distance"],
        "repetition_index": 0,
        "achieved_request_rate": summary_all["trace_reuse"].get("request_count", 0) / max(total_s, 1e-6),
    })
    write_json_atomic(layout.metadata_path, md)
    logger.info("cell complete: arch=%s state=%s reportable=%s",
                architecture, initial_state, summary_all["reportable"])
    return 0


def _results_root() -> str:
    from hcv.run_common import results_root_default

    return results_root_default()


if __name__ == "__main__":
    sys.exit(main())
