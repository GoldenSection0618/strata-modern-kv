"""Experiment 2 (SGLang path): Shared-Prefix Scaling.

Fixed context length, sweep prefix ratio x residency.  Mirrors the vLLM
``run_exp2.py`` flow over the public SGLang HTTP server.

Usage:
    python3 sglang_hicache/run_exp2.py \\
        --model qwen --model-path /path/to/model \\
        --context-length 32768 --prefix-ratio 0.5 --residency cpu_hit \\
        --output-dir results/sglang/exp2/qwen/32768-50pct-cpu_hit
"""

from __future__ import annotations

import argparse
import json
import logging
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
    save_raw_result,
    save_summary,
    save_unsupported,
    save_validation,
    utc_run_tag,
    utcnow,
)
from sglang_hicache.metrics import diff_snapshots
from sglang_hicache.residency import (
    prepare_cpu_hit,
    prepare_gpu_hit,
    prepare_recompute,
)
from sglang_hicache.session import server_metadata, start_server_session
from sglang_hicache.summary import summarize_ttft
from sglang_hicache.validation import run_validation_gate
from sglang_hicache.workload import SGLangWorkload, checkpoint_tokenize_fallback

logger = logging.getLogger("sglang-exp2")


def setup_logging(log_dir: str, model_label: str, ctx: int, ratio: float, mode: str) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pct = int(ratio * 100)
    log_file = log_dir / f"sglang-exp2-{model_label}-{ctx}-{pct}pct-{mode}-{ts}.log"
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


def _prepare_residency(mode, client, workload, snapshot_fn, seed):
    if mode == "recompute":
        prepare_recompute(client)
        return None
    if mode == "gpu_hit":
        prepare_gpu_hit(client, workload.get_prefix_ids())
        return None
    if mode == "cpu_hit":
        return prepare_cpu_hit(
            client,
            workload.get_prefix_ids(),
            snapshot_fn,
            context_length=workload.total_tokens,
            seed=seed,
        )
    raise ValueError(f"unknown residency mode: {mode}")


def run_single(args) -> int:
    model_label = args.model_id.split("/")[-1].lower().replace(".", "")
    setup_logging(args.log_dir, model_label, args.context_length, args.prefix_ratio, args.residency)

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

    prefix_tokens = int(args.context_length * args.prefix_ratio)
    logger.info("=" * 60)
    logger.info("Experiment 2 (SGLang): Shared-Prefix Scaling")
    logger.info("Model:     %s (%s)", args.model_id, args.model_path)
    logger.info("Context:   %d tokens (fixed)", args.context_length)
    logger.info("Prefix:    %.1f%% (%d shared / %d unique)",
                args.prefix_ratio * 100, prefix_tokens,
                args.context_length - prefix_tokens)
    logger.info("Residency: %s", args.residency)
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
    workload = SGLangWorkload(
        tokenize_fn=checkpoint_tokenize_fallback(client.tokenize, args.model_path),
        total_tokens=args.context_length,
        prefix_ratio=args.prefix_ratio,
        n_reps=args.n_repeats,
        base_text=base_text,
    )

    metadata = server_metadata(server_config, args.log_dir)
    metadata.update({
        "experiment": "exp2-shared-prefix-scaling",
        "model": args.model_id,
        "model_path": args.model_path,
        "context_length": args.context_length,
        "prefix_ratio": args.prefix_ratio,
        "prefix_tokens": len(workload.get_prefix_ids()),
        "suffix_tokens": args.context_length - len(workload.get_prefix_ids()),
        "residency_mode": args.residency,
        "run_tag": args.run_tag,
        "n_warmup": args.n_warmup,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "timestamp": utcnow(),
    })
    save_metadata(metadata, args.output_dir)

    # Validation gate — skip for 0% ratio (no shared prefix to validate).
    if args.prefix_ratio == 0.0:
        logger.info("Skipping validation gate for 0%% prefix ratio")
        validation = {
            "checks": [],
            "all_passed": True,
            "cpu_hit_supported": None,
            "runtime": "sglang",
            "note": "0% prefix ratio - no cache hit to validate",
        }
        save_validation(validation, args.output_dir)
    else:
        logger.info("--- Validation Gate ---")
        validation = run_validation_gate(
            client,
            server_config,
            input_ids=workload.get_segment(0).prompt_token_ids,
            prefix_ids=workload.get_prefix_ids(),
            snapshot_fn=snapshot_fn,
            context_length=args.context_length,
            seed=args.seed,
        )
        save_validation(validation, args.output_dir)

        if not validation["all_passed"]:
            logger.error("Validation gate FAILED. Aborting measurements.")
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
                    "prefix_ratio": args.prefix_ratio,
                    "residency_mode": args.residency,
                },
                args.output_dir,
            )
            return

    # --- warmup ---
    logger.info("--- Warmup (%d reps) ---", args.n_warmup)
    for i in range(args.n_warmup):
        if args.prefix_ratio > 0.0:
            _prepare_residency(args.residency, client, workload, snapshot_fn, args.seed)
        else:
            prepare_recompute(client)
        segment = workload.get_segment(i % args.n_repeats)
        r = client.generate(segment.prompt_token_ids, max_new_tokens=1, request_id=100 + i)
        if not r.ok:
            logger.warning("warmup request %d failed: %s", i, r.error[:200])
        logger.info("Warmup %d/%d done", i + 1, args.n_warmup)

    # --- measured reps ---
    logger.info("--- Measured Runs (%d reps) ---", args.n_repeats)
    ttfts: list[float] = []
    for rep in range(args.n_repeats):
        prep = _prepare_residency(args.residency, client, workload, snapshot_fn, args.seed)
        segment = workload.get_segment(rep)
        before = snapshot_fn()
        result = client.generate(segment.prompt_token_ids, max_new_tokens=1, request_id=200 + rep)
        after = snapshot_fn()
        delta = diff_snapshots(after, before)

        if result.ok:
            ttfts.append(result.ttft_ms)
        else:
            logger.error("measured request %d failed: %s", rep, result.error[:300])
            ttfts.append(float("nan"))

        raw = {
            "experiment": "exp2-shared-prefix-scaling",
            "runtime": "sglang",
            "model": args.model_id,
            "context_length": args.context_length,
            "prefix_ratio": args.prefix_ratio,
            "prefix_tokens": len(segment.prefix_ids),
            "suffix_tokens": len(segment.suffix_ids),
            "residency_mode": args.residency,
            "rep": rep,
            "ttft_ms": round(result.ttft_ms, 3),
            "t_send": result.t_send,
            "t_first_token": result.t_first_token,
            "response": {
                "ok": result.ok,
                "error": result.error,
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cached_tokens": result.cached_tokens,
                "cached_tokens_details": result.cached_tokens_details,
                "output_token_id": result.output_token_id,
            },
            "cache_stats_before": before.to_dict(),
            "cache_stats_after": after.to_dict(),
            "cache_stats_delta": delta.to_dict(),
            "cpu_hit_prep": prep.to_dict() if prep is not None else None,
            "engine_config": server_config.engine_config_dict(),
            "hardware": metadata.get("hardware"),
            "timestamp": utcnow(),
        }
        save_raw_result(raw, args.output_dir, rep)
        logger.info(
            "Run %d/%d: TTFT=%.2f ms, cached=%d, details=%s",
            rep + 1, args.n_repeats, result.ttft_ms,
            result.cached_tokens, json.dumps(result.cached_tokens_details),
        )

    valid = [t for t in ttfts if t == t]
    summary = {
        "runtime": "sglang",
        "model": args.model_id,
        "context_length": args.context_length,
        "prefix_ratio": args.prefix_ratio,
        "prefix_tokens": len(workload.get_prefix_ids()),
        "suffix_tokens": args.context_length - len(workload.get_prefix_ids()),
        "residency_mode": args.residency,
        **summarize_ttft(valid, n_repeats=args.n_repeats),
    }
    save_summary(summary, args.output_dir)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2 (SGLang): Shared-Prefix Scaling"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--prefix-ratio", type=float, required=True)
    parser.add_argument("--residency", required=True,
                        help="recompute | gpu_hit | cpu_hit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-repeats", type=int, default=10)
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
    parser.add_argument("--smoke", action="store_true",
                        help="Small smoke settings (1 warmup, 2 reps)")
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
        args.n_repeats = 2

    run_tag = utc_run_tag(args.run_tag)
    args.run_tag = run_tag
    args.output_dir = str(
        run_output_dir(args.output_dir, run_tag, residency=args.residency)
    )

    sys.exit(run_single(args))


if __name__ == "__main__":
    main()
