"""Experiment 3 (SGLang path): Request-Rate Scaling / Concurrency Pressure.

Drives the *public SGLang server* with concurrent HTTP ``/generate``
requests (never a synchronous in-process engine).  Two phases:

  1. Capacity calibration — probe offered rates, estimate sustainable
     capacity (achieved tracks offered within a fixed ratio);
  2. Formal sweep — 7 normalized-load points (0.25x..1.30x capacity),
     each repeated, with per-request queueing/service/TTFT separation.

Cache residency uses an explicit **prefix pool** (``prefix_pool.py``):
deterministic distinct prefix families scheduled round-robin through the
load window, warmed (and for ``cpu_hit`` evicted to host L2) *before* the
window, so the formal load window contains many independently restorable
host prefixes instead of one shared prefix that becomes a GPU hit after
the first request.  Every load point records the per-request device/host
tier breakdown and only keeps the requested residency label when that
tier dominates by ``--hit-dominance-threshold``; otherwise the load point
is labelled unsupported (a mostly-GPU load point is never silently called
``cpu_hit``).

Results land in ``results/sglang/exp3/<model>/<ctx>k-<pct>pct-<mode>/run-<tag>/``
with the same summary schema as the vLLM path so
``analysis/exp4_synthesis.py`` can consume them.

Usage:
    python3 sglang_hicache/run_exp3.py \\
        --model qwen --model-path /path/to/model --residency cpu_hit \\
        --output-dir results/sglang/exp3/qwen/32k-50pct-cpu_hit
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sglang_hicache.config import SGLangServerConfig
from sglang_hicache.io import (
    run_output_dir,
    save_metadata,
    save_summary,
    save_unsupported,
    save_validation,
    utc_run_tag,
    utcnow,
    write_json,
)
from sglang_hicache.load_driver import HttpLoadDriver, LoadDriverConfig
from sglang_hicache.metrics import diff_snapshots
from sglang_hicache.prefix_pool import (
    HIT_DOMINANCE_THRESHOLD,
    PrefixPool,
    aggregate_tier_hits,
    aggregate_tier_metric_delta,
    build_prefix_families,
    dominance_decision,
)
from sglang_hicache.residency import (
    prepare_cpu_hit_pool,
    prepare_gpu_hit_pool,
    prepare_recompute,
)
from sglang_hicache.session import server_metadata, start_server_session
from sglang_hicache.summary import (
    aggregate_load_summaries,
    calibrate_sustainable_capacity,
    summarize_records,
    sweep_rates_from_capacity,
)
from sglang_hicache.validation import run_validation_gate
from sglang_hicache.workload import checkpoint_tokenize_fallback

logger = logging.getLogger("sglang-exp3")

TRACKING_RATIO = 0.85


def setup_logging(log_dir: str, model_label: str, mode: str) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = log_dir / f"sglang-exp3-{model_label}-{mode}-{ts}.log"
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger.info("Logging to %s", log_file)
    return log_file


def _prepare_residency(mode, client, pool, snapshot_fn, seed, context_length):
    """Pool-aware residency preparation for one measured load window."""
    if mode == "recompute":
        prepare_recompute(client)
        return None
    if mode == "gpu_hit":
        return prepare_gpu_hit_pool(client, pool, snapshot_fn)
    if mode == "cpu_hit":
        return prepare_cpu_hit_pool(
            client,
            pool,
            snapshot_fn,
            context_length=context_length,
            seed=seed,
        )
    raise ValueError(f"unknown residency mode: {mode}")


def _run_load_point(
    client,
    pool,
    rate,
    ceiling,
    n_requests,
    seed,
    requested_residency,
    threshold,
    snapshot_fn,
):
    """Run one load window over the deterministic prompt schedule.

    Returns ``(records, tier_agg, dom_ok, dom_label, dom_reason)`` where
    ``tier_agg`` is the per-request device/host tier breakdown and the
    dominance triple is the residency-dominance decision for this window.
    """
    config = LoadDriverConfig(
        offered_rate=rate,
        n_requests=n_requests,
        concurrency_ceiling=ceiling,
        arrival_pattern="poisson",
        seed=seed,
    )
    driver = HttpLoadDriver(client, pool.prompts(n_requests), config)
    kv_before = snapshot_fn()
    records = [r.to_dict() for r in driver.run()]
    kv_after = snapshot_fn()
    kv_delta = diff_snapshots(kv_after, kv_before)
    metadata_agg = aggregate_tier_hits(records)
    tier_agg = aggregate_tier_metric_delta(kv_delta)
    tier_agg["per_request_metadata"] = metadata_agg
    dom_ok, dom_label, dom_reason = dominance_decision(
        tier_agg, requested_residency, threshold
    )
    return (
        records, tier_agg, dom_ok, dom_label, dom_reason,
        kv_before, kv_after, kv_delta,
    )


def run_single(args) -> int:
    model_label = args.model_id.split("/")[-1].lower().replace(".", "")
    setup_logging(args.log_dir, model_label, args.residency)

    server_config = SGLangServerConfig(
        model_path=args.model_path,
        model_id=args.model_id,
        residency_mode=args.residency,
        port=args.port,
        mem_fraction_static=args.mem_fraction_static,
        page_size=args.page_size,
        tp_size=args.tp_size,
        max_running_requests=args.max_running_requests,
        hicache_ratio=args.hicache_ratio,
        hicache_size_gb=args.hicache_size,
        hicache_io_backend=args.hicache_io_backend,
        hicache_mem_layout=args.hicache_mem_layout,
        hicache_write_policy=args.hicache_write_policy,
        sglang_commit=args.sglang_commit,
    )

    logger.info("=" * 60)
    logger.info("Experiment 3 (SGLang): Request-Rate Scaling")
    logger.info("Model:     %s (%s)", args.model_id, args.model_path)
    logger.info("Context:   %d tokens (fixed)", args.context_length)
    logger.info("Prefix:    %.1f%%", args.prefix_ratio * 100)
    logger.info("Residency: %s", args.residency)
    logger.info("Prefix pool: %d families", args.prefix_pool_size)
    logger.info("Dominance threshold: %.0f%%", args.hit_dominance_threshold * 100)
    logger.info("Control:   %s", "yes" if args.control_mode else "no")
    logger.info("Output:    %s", args.output_dir)
    logger.info("=" * 60)

    server, client, snapshot_fn = start_server_session(
        server_config, args.log_dir, ready_timeout_s=args.ready_timeout
    )
    try:
        _run_with_server(args, server_config, client, snapshot_fn)
    finally:
        server.stop()
    return 0


def _run_with_server(args, server_config, client, snapshot_fn) -> None:
    base_text = None
    if args.base_text_path:
        base_text = Path(args.base_text_path).read_text(encoding="utf-8")

    prefix_len = int(args.context_length * args.prefix_ratio)
    suffix_len = args.context_length - prefix_len
    pool = PrefixPool(
        build_prefix_families(
            checkpoint_tokenize_fallback(client.tokenize, args.model_path),
            prefix_len,
            suffix_len,
            n_families=args.prefix_pool_size,
            n_suffixes=1,
            base_text=base_text,
            seed=args.seed,
        ),
        seed=args.seed,
    )
    logger.info(
        "Prefix pool: %d families x %d prefix tokens (suffix %d)",
        pool.n_families, prefix_len, suffix_len,
    )

    metadata = server_metadata(server_config, args.log_dir)
    metadata.update({
        "experiment": "exp3-request-rate-scaling",
        "model": args.model_id,
        "model_path": args.model_path,
        "context_length": args.context_length,
        "prefix_ratio": args.prefix_ratio,
        "prefix_tokens": prefix_len,
        "suffix_tokens": suffix_len,
        "residency_mode": args.residency,
        "run_tag": args.run_tag,
        "prefix_pool_size": args.prefix_pool_size,
        "hit_dominance_threshold": args.hit_dominance_threshold,
        "prefix_pool": pool.to_metadata(),
        "n_warmup": args.n_warmup,
        "n_measured": args.n_measured,
        "n_repeats": args.n_repeats,
        "concurrency_ceiling": args.concurrency_ceiling,
        "seed": args.seed,
        "control_mode": args.control_mode,
        "timestamp": utcnow(),
    })
    save_metadata(metadata, args.output_dir)

    # --- validation gate ---
    logger.info("--- Validation Gate ---")
    validation = run_validation_gate(
        client,
        server_config,
        input_ids=pool.prompt_for(0),
        prefix_ids=pool.families[0].prefix_ids,
        snapshot_fn=snapshot_fn,
        context_length=args.context_length,
        seed=args.seed,
    )
    save_validation(validation, args.output_dir)
    if not validation["all_passed"]:
        logger.error("Validation gate FAILED. Aborting.")
        return
    if args.residency == "cpu_hit" and validation.get("cpu_hit_supported") is False:
        logger.warning("CPU-resident hit not supported; labeling as unsupported.")
        save_unsupported(
            {
                "status": "unsupported",
                "runtime": "sglang",
                "reason": "cpu_hit validation evidence insufficient",
                "validation": validation,
                "model": args.model_id,
                "context_length": args.context_length,
                "residency_mode": args.residency,
            },
            args.output_dir,
        )
        return

    # --- warmup ---
    logger.info("--- Warmup (%d reps) ---", args.n_warmup)
    for i in range(args.n_warmup):
        if args.prefix_ratio > 0.0 and args.residency != "recompute":
            _prepare_residency(
                args.residency, client, pool, snapshot_fn,
                args.seed, args.context_length,
            )
        else:
            prepare_recompute(client)
        r = client.generate(pool.prompt_for(i), max_new_tokens=1, request_id=100 + i)
        if not r.ok:
            logger.warning("warmup request %d failed: %s", i, r.error[:200])
        logger.info("Warmup %d/%d done", i + 1, args.n_warmup)

    # --- Phase 1: capacity calibration ---
    if args.frozen_rates:
        sweep_rates = args.frozen_rates
        normalized_loads = [1.0] * len(sweep_rates)
        cal_result = None
        logger.info("Using frozen sweep rates: %s", sweep_rates)
    elif args.skip_calibration:
        sweep_rates = args.frozen_rates or [1.0, 2.0, 4.0]
        normalized_loads = [1.0] * len(sweep_rates)
        cal_result = None
    else:
        cal_rates = args.calibration_rates or [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        cal_result = _run_calibration(client, pool, cal_rates, args, snapshot_fn)
        cap = cal_result["sustainable_capacity"]
        sweep_rates = cal_result["sweep_rates"]
        normalized_loads = cal_result["normalized_loads"]

    # --- control mode: representative points only ---
    if args.control_mode:
        if args.frozen_rates:
            n_rates = len(sweep_rates)
            idxs = sorted({0, n_rates // 2, n_rates - 1}) if n_rates >= 3 else list(range(n_rates))
            sweep_rates = [sweep_rates[i] for i in idxs]
            normalized_loads = [normalized_loads[i] for i in idxs]
        else:
            target_loads = [0.25, 1.00, 1.30]
            seen: set[int] = set()
            selected = []
            for tl in target_loads:
                best = min(
                    range(len(normalized_loads)),
                    key=lambda i: abs(normalized_loads[i] - tl),
                )
                if best not in seen:
                    seen.add(best)
                    selected.append(best)
            sweep_rates = [sweep_rates[i] for i in selected]
            normalized_loads = [normalized_loads[i] for i in selected]
        logger.info("Control mode: rates=%s loads=%s", sweep_rates, normalized_loads)

    # --- Phase 2: formal sweep ---
    logger.info("--- Formal Sweep ---")
    logger.info("Load points: %s req/s", sweep_rates)
    logger.info("Reps per point: %d, Requests per rep: %d",
                args.n_repeats, args.n_measured)

    all_summaries: list[dict] = []
    rng = random.Random(args.seed)
    order = list(range(len(sweep_rates)))
    rng.shuffle(order)

    for idx in order:
        rate = sweep_rates[idx]
        nload = normalized_loads[idx] if idx < len(normalized_loads) else 1.0
        logger.info("Load point: rate=%.2f req/s (%.0f%% capacity) — %d reps x %d requests",
                    rate, nload * 100, args.n_repeats, args.n_measured)

        for rep in range(args.n_repeats):
            prep = _prepare_residency(
                args.residency, client, pool, snapshot_fn,
                args.seed, args.context_length,
            )
            (
                records, tier_agg, dom_ok, dom_label, dom_reason,
                kv_before, kv_after, kv_delta,
            ) = _run_load_point(
                client,
                pool,
                rate,
                args.concurrency_ceiling,
                args.n_measured,
                seed=args.seed + rep,
                requested_residency=args.residency,
                threshold=args.hit_dominance_threshold,
                snapshot_fn=snapshot_fn,
            )

            summary = summarize_records(records)
            summary["offered_rate"] = rate
            summary["normalized_load"] = round(nload, 3)
            summary["rep"] = rep
            summary["kv_stats_before"] = kv_before.to_dict()
            summary["kv_stats_after"] = kv_after.to_dict()
            summary["kv_stats_delta"] = kv_delta.to_dict()
            summary["cpu_hit_prep"] = prep.to_dict() if prep is not None else None
            summary["tier_breakdown"] = tier_agg
            summary["residency_dominance_ok"] = dom_ok
            summary["residency_dominance_label"] = dom_label
            summary["residency_dominance_reason"] = dom_reason
            if not dom_ok:
                summary["unsupported"] = True

            raw_dir = Path(args.output_dir) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                {
                    "offered_rate": rate,
                    "normalized_load": round(nload, 3),
                    "rep": rep,
                    "runtime": "sglang",
                    "records": records,
                    "tier_breakdown": tier_agg,
                    "residency_dominance_ok": dom_ok,
                    "residency_dominance_label": dom_label,
                    "residency_dominance_reason": dom_reason,
                    "summary": summary,
                },
                raw_dir / f"load_{rate:.2f}_rep_{rep:02d}.json",
            )
            all_summaries.append(summary)
            logger.info(
                "    achieved=%.1f, TTFT p50=%.1f p90=%.1f p99=%.1f, "
                "queue p50=%.1f, conc=%d",
                summary.get("achieved_throughput", 0),
                summary.get("ttft_p50_ms", 0),
                summary.get("ttft_p90_ms", 0),
                summary.get("ttft_p99_ms", 0),
                summary.get("queueing_p50_ms", 0),
                summary.get("active_concurrency_max", 0),
            )
            if not dom_ok:
                logger.warning(
                    "    RESIDENCY DOMINANCE FAILED (%s): %s",
                    args.residency, dom_reason,
                )
            if summary.get("queueing_p90_ms", 0) > summary.get("ttft_p90_ms", 0):
                logger.warning("    Queueing > 90th-percentile TTFT — possible unstable overload")

    aggregated = aggregate_load_summaries(all_summaries)
    save_summary(aggregated, args.output_dir)
    logger.info("--- Aggregated Summary ---")
    for a in aggregated:
        logger.info(
            "  rate=%.2f (%.0f%%): throughput=%.1f, TTFT p50=%.1f p90=%.1f p99=%.1f, "
            "dom_ok=%s",
            a["offered_rate"], a["normalized_load"] * 100,
            a["achieved_throughput_mean"] or 0,
            a["ttft_p50_mean"] or 0, a["ttft_p90_mean"] or 0, a["ttft_p99_mean"] or 0,
            a.get("residency_dominance_ok"),
        )


def _run_calibration(client, pool, cal_rates, args, snapshot_fn) -> dict:
    logger.info("=== Capacity Calibration ===")
    logger.info("Probe rates: %s", cal_rates)
    probes = []
    for rate in cal_rates:
        prep = _prepare_residency(
            args.residency, client, pool, snapshot_fn,
            args.seed, args.context_length,
        )
        (
            records, tier_agg, dom_ok, dom_label, dom_reason,
            kv_before, kv_after, kv_delta,
        ) = _run_load_point(
            client,
            pool,
            rate,
            args.concurrency_ceiling,
            n_requests=args.calibration_n,
            seed=args.seed,
            requested_residency=args.residency,
            threshold=args.hit_dominance_threshold,
            snapshot_fn=snapshot_fn,
        )
        summary = summarize_records(records)
        achieved = summary["achieved_throughput"]
        ratio = achieved / rate if rate > 0 else 1.0
        is_tracking = ratio > TRACKING_RATIO
        probe = {
            "offered_rate": rate,
            "achieved_throughput": achieved,
            "tracking_ratio": round(ratio, 3),
            "is_tracking": is_tracking,
            "ttft_p50_ms": summary["ttft_p50_ms"],
            "ttft_p90_ms": summary["ttft_p90_ms"],
            "queueing_p50_ms": summary["queueing_p50_ms"],
            "active_concurrency_max": summary["active_concurrency_max"],
            "tier_breakdown": tier_agg,
            "kv_stats_before": kv_before.to_dict(),
            "kv_stats_after": kv_after.to_dict(),
            "kv_stats_delta": kv_delta.to_dict(),
            "cpu_hit_prep": prep.to_dict() if prep is not None else None,
            "residency_dominance_ok": dom_ok,
            "residency_dominance_label": dom_label,
            "residency_dominance_reason": dom_reason,
        }
        probes.append(probe)
        logger.info(
            "  rate=%.1f -> achieved=%.1f (ratio=%.2f, tracking=%s, dom_ok=%s)",
            rate, achieved, ratio, is_tracking, dom_ok,
        )
        if not dom_ok:
            logger.warning(
                "  calibration probe residency dominance failed (%s): %s",
                args.residency, dom_reason,
            )

    capacity = calibrate_sustainable_capacity(probes, tracking_ratio=TRACKING_RATIO)
    sweep_rates = sweep_rates_from_capacity(capacity)
    normalized_loads = [0.25, 0.50, 0.70, 0.85, 1.00, 1.15, 1.30]
    result = {
        "model_id": args.model_id,
        "residency_mode": args.residency,
        "context_length": args.context_length,
        "probes": probes,
        "sustainable_capacity": round(capacity, 3),
        "sweep_rates": sweep_rates,
        "normalized_loads": normalized_loads,
        "tracking_ratio": TRACKING_RATIO,
    }
    write_json(result, Path(args.output_dir) / "calibration.json")
    logger.info("Sustainable capacity: %.2f req/s", capacity)
    logger.info("Sweep rates: %s", sweep_rates)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 (SGLang): Request-Rate Scaling"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--residency", required=True,
                        help="recompute | gpu_hit | cpu_hit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--prefix-ratio", type=float, default=0.5)
    parser.add_argument("--prefix-pool-size", type=int, default=16,
                        help="number of distinct prefix families in the Exp3 "
                             "prefix pool (pinned in metadata/sbatch)")
    parser.add_argument("--hit-dominance-threshold", type=float,
                        default=HIT_DOMINANCE_THRESHOLD,
                        help="minimum device/host hit fraction required to "
                             "label a load point under the requested residency")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-measured", type=int, default=30)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--concurrency-ceiling", type=int, default=64)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--hicache-size", type=int, default=0)
    parser.add_argument("--hicache-io-backend", default="kernel")
    parser.add_argument("--hicache-mem-layout", default="page_first")
    parser.add_argument("--hicache-write-policy", default="write_through")
    parser.add_argument("--base-text-path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default=None,
                        help="Unique run tag for the output directory "
                             "(default: UTC timestamp + SLURM job id)")
    parser.add_argument("--sglang-commit", default=None)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--calibration-n", type=int, default=10,
                        help="requests per calibration probe")
    parser.add_argument("--calibration-rates", default=None,
                        help="comma-separated probe rates")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--frozen-rates", default=None,
                        help="comma-separated offered rates (skip calibration)")
    parser.add_argument("--control-mode", action="store_true",
                        help="only representative load points (low/sat/overload)")
    parser.add_argument("--smoke", action="store_true",
                        help="Small smoke settings (1 rep, 5 requests, ceiling 4)")
    args = parser.parse_args()

    model_map = {
        "qwen": "Qwen/Qwen3.5-9B",
        "qwen3.5-9b": "Qwen/Qwen3.5-9B",
        "gemma": "google/gemma-4-12B",
        "gemma4-12b": "google/gemma-4-12B",
    }
    args.model_id = model_map.get(args.model.lower(), args.model)

    if args.smoke:
        args.n_warmup = 1
        args.n_measured = 5
        args.n_repeats = 1
        args.concurrency_ceiling = 4
        args.calibration_n = 5

    if args.frozen_rates:
        args.frozen_rates = [float(r) for r in args.frozen_rates.split(",")]
    if args.calibration_rates:
        args.calibration_rates = [float(r) for r in args.calibration_rates.split(",")]

    run_tag = utc_run_tag(args.run_tag)
    args.run_tag = run_tag
    args.output_dir = str(run_output_dir(args.output_dir, run_tag))

    sys.exit(run_single(args))


if __name__ == "__main__":
    main()
