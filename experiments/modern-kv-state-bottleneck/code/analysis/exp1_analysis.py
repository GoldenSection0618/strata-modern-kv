"""Analysis script for Experiment 1 raw results.

Reads raw JSON files, computes aggregate statistics, and writes
processed CSV files.  Figures are generated separately to keep
dependencies minimal.

Usage:
    python3 analysis/exp1_analysis.py --input-dir results/exp1/qwen/ --output-dir results/exp1/processed/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger("exp1-analysis")


def load_raw_results(input_dir: Path) -> list[dict]:
    """Load all rep_*.json files from a directory tree."""
    results = []
    for json_path in sorted(input_dir.rglob("rep_*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            results.append(data)
        except Exception as e:
            logger.warning("Failed to load %s: %s", json_path, e)
    return results


def compute_stats(values: list[float]) -> dict:
    """Compute median, P90, min, max, stdev."""
    if not values:
        return {"median": None, "p90": None, "min": None, "max": None, "stdev": None}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "median": round(statistics.median(sorted_vals), 3),
        "p90": round(sorted_vals[min(int(n * 0.9), n - 1)], 3),
        "p50": round(sorted_vals[n // 2], 3),
        "min": round(min(sorted_vals), 3),
        "max": round(max(sorted_vals), 3),
        "stdev": round(statistics.stdev(sorted_vals), 3) if n > 1 else 0.0,
        "n": n,
    }


def aggregate_results(raw_results: list[dict]) -> list[dict]:
    """Group by (model, context_length, residency_mode) and compute stats."""
    groups: dict[tuple, list[float]] = {}

    for r in raw_results:
        key = (r["model"], r["context_length"], r["residency_mode"])
        groups.setdefault(key, []).append(r["ttft_ms"])

    rows = []
    for (model, ctx, mode), ttfts in sorted(groups.items()):
        stats = compute_stats(ttfts)
        rows.append({
            "model": model,
            "context_length": ctx,
            "residency_mode": mode,
            **stats,
        })

    return rows


def compute_decomposition(rows: list[dict]) -> list[dict]:
    """Compute TTFT decomposition: compute, io_stall, reuse_saving."""
    # Index by (model, context_length)
    by_ctx: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["model"], row["context_length"])
        by_ctx.setdefault(key, {})[row["residency_mode"]] = row

    decomposed = []
    for (model, ctx), modes in sorted(by_ctx.items()):
        rec = modes.get("recompute", {})
        gpu = modes.get("gpu_hit", {})
        cpu = modes.get("cpu_hit", {})

        compute = rec.get("median")
        gpu_ttft = gpu.get("median")
        cpu_ttft = cpu.get("median")

        row = {
            "model": model,
            "context_length": ctx,
            "recompute_ttft_ms": compute,
            "gpu_hit_ttft_ms": gpu_ttft,
            "cpu_hit_ttft_ms": cpu_ttft,
        }

        if compute is not None and gpu_ttft is not None:
            row["reuse_saving_ms"] = round(compute - gpu_ttft, 3)
        else:
            row["reuse_saving_ms"] = None

        if gpu_ttft is not None and cpu_ttft is not None:
            row["io_stall_ms"] = round(cpu_ttft - gpu_ttft, 3)
            row["cpu_restore_penalty_ms"] = round(cpu_ttft - gpu_ttft, 3)
        else:
            row["io_stall_ms"] = None
            row["cpu_restore_penalty_ms"] = None

        decomposed.append(row)

    return decomposed


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Experiment 1 raw results"
    )
    parser.add_argument("--input-dir", required=True, help="Root dir with model subdirs")
    parser.add_argument("--output-dir", required=True, help="Output dir for processed CSV")
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

    # Aggregate
    agg_rows = aggregate_results(raw)
    write_csv(
        agg_rows, output_dir / "ttft_vs_context.csv",
        ["model", "context_length", "residency_mode", "median", "p90", "p50", "min", "max", "stdev", "n"],
    )

    # Decomposition
    decomp_rows = compute_decomposition(agg_rows)
    write_csv(
        decomp_rows, output_dir / "ttft_decomposition.csv",
        ["model", "context_length", "recompute_ttft_ms", "gpu_hit_ttft_ms",
         "cpu_hit_ttft_ms", "reuse_saving_ms", "io_stall_ms", "cpu_restore_penalty_ms"],
    )

    # GPU analysis aggregation
    gpu_rows = []
    for r in raw:
        ga = r.get("gpu_analysis", {})
        if ga:
            gpu_rows.append({
                "model": r["model"],
                "context_length": r["context_length"],
                "residency_mode": r["residency_mode"],
                "rep": r["rep"],
                "gpu_delta_mb": ga.get("delta_mb"),
                "restore_duration_ms": ga.get("restore_duration_ms"),
                "restore_bandwidth_mbs": ga.get("restore_bandwidth_mbs"),
            })
    if gpu_rows:
        write_csv(
            gpu_rows, output_dir / "gpu_transfer.csv",
            ["model", "context_length", "residency_mode", "rep",
             "gpu_delta_mb", "restore_duration_ms", "restore_bandwidth_mbs"],
        )

    logger.info("Analysis complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
