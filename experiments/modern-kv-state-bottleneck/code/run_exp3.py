"""Experiment 3 main runner.

Request-rate scaling: measures TTFT, throughput, queueing, and active
concurrency under increasing offered load.

Two phases:
  1. Capacity calibration — estimate sustainable capacity
  2. Formal sweep — 7 load points (0.25× to 1.30× capacity)

Each load point sends n_measured requests at the target Poisson rate
and records per-request timing.

Usage:
    python3 run_exp3.py \
        --model qwen \
        --model-path /path/to/model \
        --residency cpu_hit \
        --output-dir results/exp3/qwen/32k-50pct-cpu_hit/

For recompute/gpu_hit control conditions, use --control-mode to run
only representative load points (low, near-saturation, overload).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure package-relative imports work when run as a script
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from configs.exp3_config import Exp3Config
from workload.token_workload import TokenWorkload
from runners.vllm_runner import VLLMRunner
from profiling.gpu_monitor import GPUMonitor
from profiling.vllm_stats import VLLMStatsCollector
from profiling.load_driver import (
    AsyncLoadDriver,
    LoadDriverConfig,
    RequestRecord,
    summarize_records,
)
from profiling.calibration import run_calibration, CalibrationResult
from validate import run_validation_gate

logger = logging.getLogger("exp3")


def setup_logging(log_dir: str, model_label: str, mode: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = log_dir / f"exp3-{model_label}-{mode}-{ts}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    for name in ("vllm", "urllib3", "filelock", "torch"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Logging to %s", log_file)
    return log_file


def save_metadata(runner: VLLMRunner, config: Exp3Config, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = runner.get_metadata()
    metadata["config"] = json.loads(config.to_json())
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved metadata: %s", path)


def save_validation(result: dict, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved validation: %s", path)


def save_load_point(
    records: list[RequestRecord],
    summary: dict,
    offered_rate: float,
    normalized_load: float,
    output_dir: str,
    rep: int,
):
    """Save raw and summary data for a single load point."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Save per-request records
    raw_path = raw_dir / f"load_{offered_rate:.2f}_rep_{rep:02d}.json"
    raw_data = {
        "offered_rate": offered_rate,
        "normalized_load": round(normalized_load, 3),
        "rep": rep,
        "records": [r.to_dict() for r in records],
        "summary": summary,
    }
    raw_path.write_text(
        json.dumps(raw_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved raw: %s", raw_path)


def save_summary(all_summaries: list[dict], output_dir: str):
    """Save aggregated summary across all load points and reps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by offered_rate
    by_rate: dict[float, list[dict]] = {}
    for s in all_summaries:
        rate = s["offered_rate"]
        by_rate.setdefault(rate, []).append(s)

    # Aggregate
    aggregated = []
    for rate in sorted(by_rate.keys()):
        reps = by_rate[rate]
        n = len(reps)

        def mean(key):
            return round(sum(r.get(key, 0) for r in reps) / n, 3)

        def max_val(key):
            return round(max(r.get(key, 0) for r in reps), 3)

        aggregated.append({
            "offered_rate": rate,
            "normalized_load": reps[0].get("normalized_load", 0),
            "n_reps": n,
            "achieved_throughput_mean": mean("achieved_throughput"),
            "ttft_p50_mean": mean("ttft_p50_ms"),
            "ttft_p90_mean": mean("ttft_p90_ms"),
            "ttft_p99_mean": mean("ttft_p99_ms"),
            "ttft_p50_max": max_val("ttft_p50_ms"),
            "ttft_p90_max": max_val("ttft_p90_ms"),
            "ttft_p99_max": max_val("ttft_p99_ms"),
            "queueing_p50_mean": mean("queueing_p50_ms"),
            "queueing_p90_mean": mean("queueing_p90_ms"),
            "service_p50_mean": mean("service_p50_ms"),
            "service_p90_mean": mean("service_p90_ms"),
            "active_concurrency_max": max_val("active_concurrency_max"),
        })

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("--- Aggregated Summary ---")
    for a in aggregated:
        logger.info(
            "  rate=%.2f (%.0f%%): throughput=%.1f, TTFT p50=%.1f p90=%.1f p99=%.1f, "
            "queue p50=%.1f",
            a["offered_rate"], a["normalized_load"] * 100,
            a["achieved_throughput_mean"],
            a["ttft_p50_mean"], a["ttft_p90_mean"], a["ttft_p99_mean"],
            a["queueing_p50_mean"],
        )
    logger.info("Saved: %s", summary_path)


def run_single(
    model_id: str,
    model_path: str,
    residency_mode: str,
    output_dir: str,
    log_dir: str,
    context_length: int = 32768,
    prefix_ratio: float = 0.5,
    n_warmup: int = 5,
    n_measured: int = 30,
    n_repeats: int = 3,
    concurrency_ceiling: int = 64,
    calibration_rates: list[float] | None = None,
    skip_calibration: bool = False,
    frozen_sweep_rates: list[float] | None = None,
    control_mode: bool = False,
    kv_offloading_size: int | None = None,
    cpu_offload_gb: float = 0.0,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 65536,
    block_size: int = 16,
    dtype: str = "bfloat16",
    gpu_id: int = 0,
    gpu_monitor_interval_ms: int = 10,
    base_text_path: str = "",
):
    """Run Experiment 3 for a single model × residency mode."""
    model_label = model_id.split("/")[-1].lower().replace(".", "")

    setup_logging(log_dir, model_label, residency_mode)

    logger.info("=" * 60)
    logger.info("Experiment 3: Request-Rate Scaling")
    logger.info("Model:     %s (%s)", model_id, model_path)
    logger.info("Context:   %d tokens (fixed)", context_length)
    logger.info("Prefix:    %.1f%%", prefix_ratio * 100)
    logger.info("Residency: %s", residency_mode)
    logger.info("Control:   %s", "yes (representative points only)" if control_mode else "no (full sweep)")
    logger.info("Output:    %s", output_dir)
    logger.info("=" * 60)

    # --- Initialize runner ---
    runner = VLLMRunner(
        model_path=model_path,
        model_id=model_id,
        residency_mode=residency_mode,
        context_length=context_length,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        block_size=block_size,
        kv_cache_metrics=True,
        dtype=dtype,
        kv_offloading_size=kv_offloading_size,
        cpu_offload_gb=cpu_offload_gb,
        gpu_id=gpu_id,
        gpu_monitor_interval_ms=gpu_monitor_interval_ms,
    )

    # --- Build workload ---
    base_text = None
    if base_text_path:
        base_text = Path(base_text_path).read_text(encoding="utf-8")

    tokenizer = runner.tokenizer
    # Use more reps than n_measured to have enough unique suffixes
    workload_n_reps = max(n_measured * n_repeats, 50)
    workload = TokenWorkload(
        tokenizer=tokenizer,
        total_tokens=context_length,
        prefix_ratio=prefix_ratio,
        n_reps=workload_n_reps,
        base_text=base_text,
    )

    # --- Save metadata ---
    config = Exp3Config(
        model_id=model_id,
        model_path=model_path,
        context_length=context_length,
        prefix_ratio=prefix_ratio,
        residency_mode=residency_mode,
        n_warmup=n_warmup,
        n_measured=n_measured,
        concurrency_ceiling=concurrency_ceiling,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        block_size=block_size,
        dtype=dtype,
        kv_offloading_size=kv_offloading_size,
        cpu_offload_gb=cpu_offload_gb,
        output_dir=output_dir,
        gpu_id=gpu_id,
    )
    save_metadata(runner, config, output_dir)

    # --- Validation gate ---
    logger.info("--- Validation Gate ---")
    skip_cpu = (residency_mode != "cpu_hit")
    validation = run_validation_gate(
        runner, workload, skip_cpu_hit=skip_cpu, residency_mode=residency_mode
    )
    save_validation(validation, output_dir)

    if not validation["all_passed"]:
        logger.error("Validation gate FAILED. Aborting.")
        runner.cleanup()
        return

    if residency_mode == "cpu_hit" and validation.get("cpu_hit_supported") is False:
        logger.warning("CPU-resident hit not supported. Labeling as unsupported.")
        result = {
            "status": "unsupported",
            "validation": validation,
            "model": model_id,
            "context_length": context_length,
            "residency_mode": residency_mode,
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "unsupported.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        runner.cleanup()
        return

    # --- Warmup ---
    logger.info("--- Warmup (%d reps) ---", n_warmup)
    sp = runner.get_sampling_params(max_tokens=1)
    for i in range(n_warmup):
        if prefix_ratio > 0.0 and residency_mode != "recompute":
            runner.reset_prefix_cache()
            runner.warmup_prefix(workload.get_prefix_ids())
        segment = workload.get_segment(i)
        runner.llm.generate([segment.to_tokens_prompt()], sp)
        logger.info("Warmup %d/%d done", i + 1, n_warmup)

    # --- Prepare prompts for load driver ---
    # Build a pool of unique prompts (different suffixes)
    prompt_pool = []
    for i in range(workload_n_reps):
        segment = workload.get_segment(i)
        prompt_pool.append(segment.to_tokens_prompt())

    # --- Phase 1: Capacity calibration ---
    if frozen_sweep_rates:
        # Use frozen rates from a previous calibration
        sweep_rates = frozen_sweep_rates
        normalized_loads = [1.0] * len(sweep_rates)  # placeholder
        logger.info("Using frozen sweep rates: %s", sweep_rates)
        cal_result = None
    elif skip_calibration:
        # Skip calibration entirely (e.g., for control modes with explicit rates)
        sweep_rates = frozen_sweep_rates or [1.0, 2.0, 4.0]
        normalized_loads = [1.0] * len(sweep_rates)
        cal_result = None
    else:
        cal_rates = calibration_rates or [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        cal_result = run_calibration(
            llm=runner.llm,
            sampling_params=sp,
            prompts=prompt_pool,
            model_id=model_id,
            residency_mode=residency_mode,
            context_length=context_length,
            probe_rates=cal_rates,
            n_requests_per_probe=10,
            concurrency_ceiling=concurrency_ceiling,
            output_dir=output_dir,
        )
        sweep_rates = cal_result.sweep_rates
        normalized_loads = cal_result.normalized_loads

    # --- Control mode: only representative points ---
    if control_mode:
        if frozen_sweep_rates:
            # Frozen rates from the primary (cpu_hit) calibration.
            # normalized_loads are placeholders here, so select by position:
            # low / near-saturation / overload.
            n_rates = len(sweep_rates)
            if n_rates >= 3:
                idxs = sorted({0, n_rates // 2, n_rates - 1})
            else:
                idxs = list(range(n_rates))
            sweep_rates = [sweep_rates[i] for i in idxs]
            normalized_loads = [normalized_loads[i] for i in idxs]
        else:
            # Select: low (0.25), near-saturation (1.00), overload (1.30)
            # Map normalized loads to indices
            target_loads = [0.25, 1.00, 1.30]
            filtered = []
            for tl in target_loads:
                # Find closest normalized load
                best_idx = min(
                    range(len(normalized_loads)),
                    key=lambda i: abs(normalized_loads[i] - tl),
                )
                filtered.append(best_idx)
            # Deduplicate while preserving order
            seen = set()
            filtered_unique = []
            for idx in filtered:
                if idx not in seen:
                    seen.add(idx)
                    filtered_unique.append(idx)
            sweep_rates = [sweep_rates[i] for i in filtered_unique]
            normalized_loads = [normalized_loads[i] for i in filtered_unique]
        logger.info("Control mode: selected rates %s (loads %s)", sweep_rates, normalized_loads)

    # --- Phase 2: Formal sweep ---
    logger.info("--- Formal Sweep ---")
    logger.info("Load points: %s req/s", sweep_rates)
    logger.info("Reps per point: %d, Requests per rep: %d", n_repeats, n_measured)

    all_summaries = []

    # Randomize load-point order to reduce systematic bias (doc §10)
    import random
    rng = random.Random(42)
    order = list(range(len(sweep_rates)))
    rng.shuffle(order)

    for idx in order:
        rate = sweep_rates[idx]
        nload = normalized_loads[idx] if idx < len(normalized_loads) else 1.0

        logger.info(
            "Load point: rate=%.2f req/s (%.0f%% capacity) — %d reps × %d requests",
            rate, nload * 100, n_repeats, n_measured,
        )

        for rep in range(n_repeats):
            logger.info("  Rep %d/%d", rep + 1, n_repeats)

            # Reset residency state before each rep
            if residency_mode == "recompute":
                runner.reset_prefix_cache()
            elif residency_mode == "gpu_hit":
                runner.reset_prefix_cache()
                if prefix_ratio > 0.0:
                    runner.warmup_prefix(workload.get_prefix_ids())
            elif residency_mode == "cpu_hit":
                runner.reset_prefix_cache()
                if prefix_ratio > 0.0:
                    runner.warmup_prefix(workload.get_prefix_ids())
                    runner.evict_prefix_to_cpu(workload.get_prefix_ids())

            # Collect KV stats before
            kv_before = runner.stats_collector.collect().to_dict()

            # Run load driver
            driver_config = LoadDriverConfig(
                offered_rate=rate,
                n_requests=n_measured,
                concurrency_ceiling=concurrency_ceiling,
                arrival_pattern="poisson",
            )
            driver = AsyncLoadDriver(
                runner.llm, sp, prompt_pool, driver_config
            )
            records = asyncio.run(driver.run())

            # Collect KV stats after
            kv_after = runner.stats_collector.collect().to_dict()

            summary = summarize_records(records)
            summary["offered_rate"] = rate
            summary["normalized_load"] = round(nload, 3)
            summary["rep"] = rep
            summary["kv_stats_before"] = kv_before
            summary["kv_stats_after"] = kv_after

            all_summaries.append(summary)
            save_load_point(records, summary, rate, nload, output_dir, rep)

            logger.info(
                "    achieved=%.1f, TTFT p50=%.1f p90=%.1f p99=%.1f, "
                "queue p50=%.1f, conc=%d",
                summary["achieved_throughput"],
                summary["ttft_p50_ms"], summary["ttft_p90_ms"],
                summary["ttft_p99_ms"],
                summary["queueing_p50_ms"],
                summary["active_concurrency_max"],
            )

            # Check for unstable overload
            if summary["queueing_p90_ms"] > summary["ttft_p90_ms"]:
                logger.warning(
                    "    Queueing > 50%% of TTFT — possible unstable overload"
                )

    # --- Aggregated summary ---
    save_summary(all_summaries, output_dir)

    runner.cleanup()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Request-Rate Scaling"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--residency", required=True,
                        help="recompute, gpu_hit, or cpu_hit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--prefix-ratio", type=float, default=0.5)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-measured", type=int, default=30)
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Repetitions per load point")
    parser.add_argument("--concurrency-ceiling", type=int, default=64)
    parser.add_argument("--kv-offloading-size", type=int, default=None)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-monitor-interval-ms", type=int, default=10)
    parser.add_argument("--base-text-path", default="")
    # Calibration control
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip calibration, use --frozen-rates")
    parser.add_argument("--frozen-rates", default=None,
                        help="Comma-separated offered rates (skip calibration)")
    parser.add_argument("--calibration-rates", default=None,
                        help="Comma-separated probe rates for calibration")
    # Control mode
    parser.add_argument("--control-mode", action="store_true",
                        help="Run only representative load points (low/sat/overload)")
    args = parser.parse_args()

    # Model ID mapping
    model_map = {
        "qwen3.5-9b": "Qwen/Qwen3.5-9B",
        "qwen": "Qwen/Qwen3.5-9B",
        "gemma4-12b": "google/gemma-4-12B",
        "gemma": "google/gemma-4-12B",
    }
    model_id = model_map.get(args.model.lower(), args.model)

    if args.log_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        log_dir = str(repo_root / "logs")
    else:
        log_dir = args.log_dir

    frozen_rates = None
    if args.frozen_rates:
        frozen_rates = [float(r) for r in args.frozen_rates.split(",")]

    cal_rates = None
    if args.calibration_rates:
        cal_rates = [float(r) for r in args.calibration_rates.split(",")]

    run_single(
        model_id=model_id,
        model_path=args.model_path,
        residency_mode=args.residency,
        output_dir=args.output_dir,
        log_dir=log_dir,
        context_length=args.context_length,
        prefix_ratio=args.prefix_ratio,
        n_warmup=args.n_warmup,
        n_measured=args.n_measured,
        n_repeats=args.n_repeats,
        concurrency_ceiling=args.concurrency_ceiling,
        calibration_rates=cal_rates,
        skip_calibration=args.skip_calibration,
        frozen_sweep_rates=frozen_rates,
        control_mode=args.control_mode,
        kv_offloading_size=args.kv_offloading_size,
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        dtype=args.dtype,
        gpu_id=args.gpu_id,
        gpu_monitor_interval_ms=args.gpu_monitor_interval_ms,
        base_text_path=args.base_text_path,
    )


if __name__ == "__main__":
    main()
