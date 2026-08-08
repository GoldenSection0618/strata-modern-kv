"""Analysis script for Experiment 2 raw results.

Computes aggregate stats, net reuse benefit, CPU restore penalty,
and service stall ratio across prefix ratio × residency mode.

Usage:
    python3 analysis/exp2_analysis.py --input-dir results/exp2/qwen/ --output-dir results/exp2/processed/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger("exp2-analysis")


def load_raw_results(input_dir: Path) -> list[dict]:
    results = []
    for json_path in sorted(input_dir.rglob("rep_*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            results.append(data)
        except Exception as e:
            logger.warning("Failed to load %s: %s", json_path, e)
    return results


def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p90": None, "p50": None, "min": None, "max": None, "stdev": None, "n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "median": round(statistics.median(s), 3),
        "p90": round(s[min(int(n * 0.9), n - 1)], 3),
        "p50": round(s[n // 2], 3),
        "min": round(min(s), 3),
        "max": round(max(s), 3),
        "stdev": round(statistics.stdev(s), 3) if n > 1 else 0.0,
        "n": n,
    }


def aggregate_results(raw: list[dict]) -> list[dict]:
    """Group by (model, context_length, prefix_ratio, residency_mode)."""
    groups: dict[tuple, list[float]] = {}
    for r in raw:
        key = (r["model"], r["context_length"], r["prefix_ratio"], r["residency_mode"])
        groups.setdefault(key, []).append(r["ttft_ms"])

    rows = []
    for (model, ctx, ratio, mode), ttfts in sorted(groups.items()):
        stats = compute_stats(ttfts)
        rows.append({
            "model": model,
            "context_length": ctx,
            "prefix_ratio": ratio,
            "prefix_tokens": int(ctx * ratio),
            "suffix_tokens": ctx - int(ctx * ratio),
            "residency_mode": mode,
            **stats,
        })
    return rows


def compute_reuse_benefit(rows: list[dict]) -> list[dict]:
    """Compute net reuse benefit, speedup, and CPU restore penalty per ratio.

    Per doc §8.5:
      Net Reuse Benefit = TTFT_recompute - TTFT_CPU-hit
      Reuse Speedup = TTFT_recompute / TTFT_CPU-hit
      CPU Restore Penalty = TTFT_CPU-hit - TTFT_GPU-hit
      Service Stall Ratio = I/O stall / service time
    """
    # Index by (model, context_length, prefix_ratio)
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["model"], row["context_length"], row["prefix_ratio"])
        by_key.setdefault(key, {})[row["residency_mode"]] = row

    out = []
    for (model, ctx, ratio), modes in sorted(by_key.items()):
        rec = modes.get("recompute", {})
        gpu = modes.get("gpu_hit", {})
        cpu = modes.get("cpu_hit", {})

        rec_med = rec.get("median")
        gpu_med = gpu.get("median")
        cpu_med = cpu.get("median")

        row = {
            "model": model,
            "context_length": ctx,
            "prefix_ratio": ratio,
            "prefix_tokens": int(ctx * ratio),
            "recompute_ttft_ms": rec_med,
            "gpu_hit_ttft_ms": gpu_med,
            "cpu_hit_ttft_ms": cpu_med,
        }

        # Net reuse benefit & speedup (CPU-resident)
        if rec_med is not None and cpu_med is not None:
            row["net_reuse_benefit_ms"] = round(rec_med - cpu_med, 3)
            row["reuse_speedup"] = round(rec_med / cpu_med, 3) if cpu_med > 0 else None
        else:
            row["net_reuse_benefit_ms"] = None
            row["reuse_speedup"] = None

        # GPU-resident benefit (upper bound of reuse without I/O cost)
        if rec_med is not None and gpu_med is not None:
            row["gpu_reuse_benefit_ms"] = round(rec_med - gpu_med, 3)
        else:
            row["gpu_reuse_benefit_ms"] = None

        # CPU restore penalty
        if gpu_med is not None and cpu_med is not None:
            row["cpu_restore_penalty_ms"] = round(cpu_med - gpu_med, 3)
            # Service stall ratio: I/O stall / service time
            # I/O stall ≈ cpu_restore_penalty; service time ≈ cpu_hit TTFT
            row["service_stall_ratio"] = round(
                (cpu_med - gpu_med) / cpu_med, 4
            ) if cpu_med > 0 else None
        else:
            row["cpu_restore_penalty_ms"] = None
            row["service_stall_ratio"] = None

        out.append(row)

    return out


def aggregate_gpu_transfer(raw: list[dict]) -> list[dict]:
    """Aggregate GPU transfer data from raw results."""
    rows = []
    for r in raw:
        ga = r.get("gpu_analysis", {})
        if ga:
            rows.append({
                "model": r["model"],
                "context_length": r["context_length"],
                "prefix_ratio": r["prefix_ratio"],
                "prefix_tokens": r.get("prefix_tokens", int(r["context_length"] * r["prefix_ratio"])),
                "residency_mode": r["residency_mode"],
                "rep": r["rep"],
                "gpu_delta_mb": ga.get("delta_mb"),
                "restore_duration_ms": ga.get("restore_duration_ms"),
                "restore_bandwidth_mbs": ga.get("restore_bandwidth_mbs"),
            })
    return rows


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path)


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 2 raw results")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_results(input_dir)
    logger.info("Loaded %d raw results", len(raw))

    if not raw:
        logger.error("No raw results found")
        return

    # Aggregate TTFT
    agg = aggregate_results(raw)
    write_csv(
        agg, output_dir / "ttft_vs_prefix_ratio.csv",
        ["model", "context_length", "prefix_ratio", "prefix_tokens",
         "suffix_tokens", "residency_mode", "median", "p90", "p50",
         "min", "max", "stdev", "n"],
    )

    # Reuse benefit decomposition
    benefit = compute_reuse_benefit(agg)
    write_csv(
        benefit, output_dir / "reuse_benefit.csv",
        ["model", "context_length", "prefix_ratio", "prefix_tokens",
         "recompute_ttft_ms", "gpu_hit_ttft_ms", "cpu_hit_ttft_ms",
         "net_reuse_benefit_ms", "reuse_speedup", "gpu_reuse_benefit_ms",
         "cpu_restore_penalty_ms", "service_stall_ratio"],
    )

    # GPU transfer
    gpu = aggregate_gpu_transfer(raw)
    if gpu:
        write_csv(
            gpu, output_dir / "gpu_transfer.csv",
            ["model", "context_length", "prefix_ratio", "prefix_tokens",
             "residency_mode", "rep", "gpu_delta_mb",
             "restore_duration_ms", "restore_bandwidth_mbs"],
        )

    logger.info("Analysis complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
