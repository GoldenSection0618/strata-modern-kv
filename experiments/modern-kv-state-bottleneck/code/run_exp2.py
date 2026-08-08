"""Experiment 2 main runner.

Fixed context length, sweep prefix ratio × residency mode.

Usage:
    python3 run_exp2.py \
        --model qwen \
        --model-path /path/to/model \
        --context-length 32768 \
        --prefix-ratio 0.5 \
        --residency cpu_hit \
        --output-dir results/exp2/qwen/32k-50pct-cpu_hit/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from configs.exp2_config import Exp2Config
from workload.token_workload import TokenWorkload
from runners.vllm_runner import VLLMRunner
from profiling.timing import measure_ttft
from validate import run_validation_gate

logger = logging.getLogger("exp2")


def setup_logging(log_dir: str, model_label: str, ctx: int, ratio: float, mode: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pct = int(ratio * 100)
    log_file = log_dir / f"exp2-{model_label}-{ctx}-{pct}pct-{mode}-{ts}.log"

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


def save_raw_result(result: dict, output_dir: str, rep: int):
    output_dir = Path(output_dir) / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rep_{rep:02d}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved raw result: %s", path)


def save_metadata(runner: VLLMRunner, config: Exp2Config, output_dir: str):
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


def run_single(
    model_id: str,
    model_path: str,
    context_length: int,
    prefix_ratio: float,
    residency_mode: str,
    output_dir: str,
    log_dir: str,
    n_warmup: int = 3,
    n_repeats: int = 10,
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
    model_label = model_id.split("/")[-1].lower().replace(".", "")

    setup_logging(log_dir, model_label, context_length, prefix_ratio, residency_mode)

    logger.info("=" * 60)
    logger.info("Experiment 2: Shared-Prefix Scaling")
    logger.info("Model:   %s (%s)", model_id, model_path)
    logger.info("Context: %d tokens (fixed)", context_length)
    logger.info("Prefix:  %.1f%% (%d shared / %d unique)",
                prefix_ratio * 100,
                int(context_length * prefix_ratio),
                context_length - int(context_length * prefix_ratio))
    logger.info("Residency: %s", residency_mode)
    logger.info("Output:  %s", output_dir)
    logger.info("=" * 60)

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

    base_text = None
    if base_text_path:
        base_text = Path(base_text_path).read_text(encoding="utf-8")

    tokenizer = runner.tokenizer
    workload = TokenWorkload(
        tokenizer=tokenizer,
        total_tokens=context_length,
        prefix_ratio=prefix_ratio,
        n_reps=n_repeats,
        base_text=base_text,
    )

    config = Exp2Config(
        model_id=model_id,
        model_path=model_path,
        context_length=context_length,
        prefix_ratios=[prefix_ratio],
        residency_modes=[residency_mode],
        n_warmup=n_warmup,
        n_repeats=n_repeats,
        kv_offloading_size=kv_offloading_size,
        cpu_offload_gb=cpu_offload_gb,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        block_size=block_size,
        dtype=dtype,
        output_dir=output_dir,
        gpu_id=gpu_id,
    )
    save_metadata(runner, config, output_dir)

    # Validation gate — skip for 0% ratio (no prefix to validate)
    if prefix_ratio == 0.0:
        logger.info("Skipping validation gate for 0% prefix ratio (no shared prefix)")
        validation = {"checks": [], "all_passed": True, "cpu_hit_supported": None,
                       "note": "0% prefix ratio — no cache hit to validate"}
        save_validation(validation, output_dir)
    else:
        logger.info("--- Validation Gate ---")
        skip_cpu = (residency_mode != "cpu_hit")
        validation = run_validation_gate(runner, workload, skip_cpu_hit=skip_cpu, residency_mode=residency_mode)
        save_validation(validation, output_dir)

        if not validation["all_passed"]:
            logger.error("Validation gate FAILED. Aborting measurements.")
            runner.cleanup()
            return

        if residency_mode == "cpu_hit" and validation.get("cpu_hit_supported") is False:
            logger.warning("CPU-resident hit not supported. Labeling as unsupported.")
            result = {
                "status": "unsupported",
                "validation": validation,
                "model": model_id,
                "context_length": context_length,
                "prefix_ratio": prefix_ratio,
                "residency_mode": residency_mode,
            }
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "unsupported.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            runner.cleanup()
            return

    # Warmup
    logger.info("--- Warmup (%d reps) ---", n_warmup)
    for i in range(n_warmup):
        if prefix_ratio > 0.0 and residency_mode != "recompute":
            runner.reset_prefix_cache()
            runner.warmup_prefix(workload.get_prefix_ids())
        segment = workload.get_segment(i % n_repeats)
        sp = runner.get_sampling_params(max_tokens=1)
        runner.llm.generate([segment.to_tokens_prompt()], sp)
        logger.info("Warmup %d/%d done", i + 1, n_warmup)

    # Measured runs
    logger.info("--- Measured Runs (%d reps) ---", n_repeats)
    results = []

    for rep in range(n_repeats):
        logger.info("Run %d/%d — mode=%s, ratio=%.1f%%",
                    rep + 1, n_repeats, residency_mode, prefix_ratio * 100)

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

        segment = workload.get_segment(rep)
        timing = runner.measure_request(segment)

        # Compute derived metrics per doc §8.5
        prefix_tokens = len(segment.prefix_ids)
        suffix_tokens = len(segment.suffix_ids)

        result = {
            "experiment": "exp2-shared-prefix-scaling",
            "model": model_id,
            "context_length": context_length,
            "prefix_ratio": prefix_ratio,
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "residency_mode": residency_mode,
            "rep": rep,
            "ttft_ms": timing.ttft_ms,
            "t_send": timing.t_send,
            "t_first_token": timing.t_first_token,
            "kv_stats_before": timing.kv_stats_before,
            "kv_stats_after": timing.kv_stats_after,
            "gpu_analysis": timing.gpu_analysis,
            "gpu_samples": [[t, b] for t, b in timing.gpu_samples],
            "engine_config": runner.engine_config,
            "hardware": {"gpu_id": gpu_id},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)
        save_raw_result(result, output_dir, rep)

        logger.info(
            "  TTFT=%.2f ms, GPU delta=%.1f MB, KV usage=%.3f",
            timing.ttft_ms,
            timing.gpu_analysis.get("delta_mb", 0),
            timing.kv_stats_after.get("usage", 0),
        )

    # Summary
    ttfts = [r["ttft_ms"] for r in results]
    ttfts_sorted = sorted(ttfts)
    n = len(ttfts_sorted)
    median = ttfts_sorted[n // 2]
    p90 = ttfts_sorted[min(int(n * 0.9), n - 1)]

    summary = {
        "model": model_id,
        "context_length": context_length,
        "prefix_ratio": prefix_ratio,
        "prefix_tokens": int(context_length * prefix_ratio),
        "suffix_tokens": context_length - int(context_length * prefix_ratio),
        "residency_mode": residency_mode,
        "n_repeats": n_repeats,
        "median_ttft_ms": round(median, 3),
        "p90_ttft_ms": round(p90, 3),
        "min_ttft_ms": round(min(ttfts), 3),
        "max_ttft_ms": round(max(ttfts), 3),
    }

    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("--- Summary ---")
    logger.info("Median TTFT: %.2f ms", median)
    logger.info("P90 TTFT: %.2f ms", p90)
    logger.info("Min: %.2f ms, Max: %.2f ms", min(ttfts), max(ttfts))
    logger.info("Saved: %s", summary_path)

    runner.cleanup()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Shared-Prefix Scaling"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--prefix-ratio", type=float, required=True)
    parser.add_argument(
        "--residency", required=True,
        help="Comma-separated modes: recompute,gpu_hit,cpu_hit"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--kv-offloading-size", type=int, default=None)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-monitor-interval-ms", type=int, default=10)
    parser.add_argument("--base-text-path", default="")
    args = parser.parse_args()

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

    modes = [m.strip() for m in args.residency.split(",")]

    for mode in modes:
        pct_label = int(args.prefix_ratio * 100)
        mode_output_dir = Path(args.output_dir) / mode
        run_single(
            model_id=model_id,
            model_path=args.model_path,
            context_length=args.context_length,
            prefix_ratio=args.prefix_ratio,
            residency_mode=mode,
            output_dir=str(mode_output_dir),
            log_dir=log_dir,
            n_warmup=args.n_warmup,
            n_repeats=args.n_repeats,
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
