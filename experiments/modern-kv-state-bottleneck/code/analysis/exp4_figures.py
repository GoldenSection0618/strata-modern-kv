"""Experiment 4: Cross-Model Bottleneck Comparison — figure generation.

Reads the processed CSVs produced by exp4_synthesis.py and generates
the four core cross-model figures (design doc section 12):

  fig1_context_footprint.png   Context Length -> TTFT by residency (per model)
  fig2_reuse_benefit.png       Shared Prefix -> Reuse Benefit vs Restore Cost
  fig3_load_sensitivity.png    Normalized Load -> TTFT / throughput (Exp 3)
  fig4_bottleneck_composition.png  Matched points -> TTFT composition

Figures only use processed data (never hand-entered numbers).  If the
required data is missing (e.g. hit modes not yet collected, Experiment 3
not run), the affected figure is skipped with a warning instead of
crashing.  Figure labels are English to avoid font issues.

Requires: matplotlib (not installed in the server qwen env; run this
script where matplotlib is available, e.g. a local environment).

Usage:
    python3 analysis/exp4_figures.py \
        --input-dir results/exp4/processed/ \
        --output-dir results/exp4/figures/
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger("exp4-figures")

# Model display labels
MODEL_LABELS = {"qwen": "Qwen3.5-9B", "gemma4": "Gemma 4 12B"}

# Colors
COLORS = {"recompute": "#d62728", "gpu_hit": "#1f77b4", "cpu_hit": "#2ca02c"}


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        logger.warning("Missing processed file: %s", path)
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _require_columns(rows: list[dict], cols: list[str]) -> bool:
    if not rows:
        return False
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        logger.warning("Columns missing in data: %s", missing)
        return False
    return True


def fig1_context_ttft(rows: list[dict], out_path: Path):
    """Context Length -> median TTFT by residency mode, one line per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not _require_columns(
        rows, ["model", "context_length", "residency_mode", "median_ttft_ms"]
    ):
        return

    by_model_mode: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for r in rows:
        med = _f(r.get("median_ttft_ms"))
        if med is None:
            continue
        key = (r["model"], r["residency_mode"])
        by_model_mode.setdefault(key, []).append((int(r["context_length"]), med))

    if not by_model_mode:
        logger.warning("fig1: no usable TTFT data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, mode), pts in sorted(by_model_mode.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = f"{MODEL_LABELS.get(model, model)} ({mode})"
        color = COLORS.get(mode)
        ax.plot(xs, ys, marker="o", label=label, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4096, 8192, 16384, 32768])
    ax.set_xticklabels(["4K", "8K", "16K", "32K"])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Median TTFT (ms)")
    ax.set_title("Exp4: Context Length -> TTFT by Residency")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("fig1 -> %s", out_path)


def fig2_reuse_benefit(rows: list[dict], out_path: Path):
    """Shared Prefix -> Net Reuse Benefit / CPU Restore Penalty / speedup."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not _require_columns(
        rows, ["model", "prefix_ratio", "net_reuse_benefit_ms",
               "cpu_restore_penalty_ms", "reuse_speedup"]
    ):
        return

    # Group by model; use first context length that has data.
    by_model: dict[str, list[tuple[float, dict]]] = {}
    for r in rows:
        nrb = _f(r.get("net_reuse_benefit_ms"))
        if nrb is None:
            continue
        by_model.setdefault(r["model"], []).append(
            (_f(r["prefix_ratio"]), r)
        )

    if not by_model:
        logger.warning("fig2: no reuse-benefit data (need cpu_hit/gpu_hit)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("net_reuse_benefit_ms", "Net Reuse Benefit (ms)"),
        ("cpu_restore_penalty_ms", "CPU Restore Penalty (ms)"),
        ("reuse_speedup", "Reuse Speedup (x)"),
    ]
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        label = MODEL_LABELS.get(model, model)
        for ax, (key, ylabel) in zip(axes, metrics):
            ys = [_f(p[1].get(key)) for p in pts]
            ax.plot(xs, ys, marker="o", label=label)
            ax.set_xlabel("Shared-prefix ratio")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].legend()
    fig.suptitle("Exp4: Reuse Benefit vs Restore Cost")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("fig2 -> %s", out_path)


def fig3_load_sensitivity(rows: list[dict], out_path: Path):
    """Normalized Load -> TTFT p50 / achieved throughput (Experiment 3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not _require_columns(
        rows, ["model", "normalized_load", "ttft_p50_mean",
               "achieved_throughput_mean"]
    ):
        return

    by_model: dict[str, list[tuple[float, dict]]] = {}
    for r in rows:
        nl = _f(r.get("normalized_load"))
        if nl is None:
            continue
        by_model.setdefault(r["model"], []).append((nl, r))

    if not by_model:
        logger.warning("fig3: no Experiment 3 load data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        label = MODEL_LABELS.get(model, model)
        axes[0].plot(
            xs, [_f(p[1].get("ttft_p50_mean")) for p in pts],
            marker="o", label=label,
        )
        axes[1].plot(
            xs, [_f(p[1].get("achieved_throughput_mean")) for p in pts],
            marker="o", label=label,
        )
    axes[0].set_xlabel("Normalized load (offered / capacity)")
    axes[0].set_ylabel("TTFT p50 (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Normalized load (offered / capacity)")
    axes[1].set_ylabel("Achieved throughput (req/s)")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle("Exp4: Load Sensitivity (Experiment 3)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("fig3 -> %s", out_path)


def fig4_bottleneck_composition(rows: list[dict], out_path: Path):
    """Matched points -> stacked TTFT composition (compute / IO stall)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not _require_columns(
        rows, ["workload", "model", "ttft_ms", "compute_path_ms",
               "io_stall_ms"]
    ):
        return

    # Keep only rows with a full decomposition (parts_available=True).
    usable = [r for r in rows if r.get("parts_available") == "True"]
    if not usable:
        logger.warning(
            "fig4: no decomposition data (needs cpu_hit + gpu_hit at "
            "matched points)"
        )
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    bottoms = []
    for r in usable:
        comp = _f(r.get("compute_path_ms")) or 0.0
        stall = _f(r.get("io_stall_ms")) or 0.0
        label = f"{r['workload']}\n{MODEL_LABELS.get(r['model'], r['model'])}"
        labels.append(label)
        ax.bar(label, comp, color="#1f77b4", label="compute path" if not bottoms else "")
        ax.bar(label, stall, bottom=comp, color="#ff7f0e",
               label="I/O stall" if not bottoms else "")
        bottoms.append(comp + stall)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Exp4: TTFT Composition at Matched Points")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("fig4 -> %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Experiment 4 cross-model figures"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with exp4 processed CSVs")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for figure PNGs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # matplotlib is imported lazily inside each figure function; verify
    # availability up front so the user gets a clear message.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.error(
            "matplotlib is required but not installed. Install it in the "
            "environment that runs this script, e.g.: pip install matplotlib"
        )
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx_rows = read_csv(input_dir / "context_scaling.csv")
    reuse_rows = read_csv(input_dir / "reuse_benefit.csv")
    load_rows = read_csv(input_dir / "load_sensitivity.csv")
    comp_rows = read_csv(input_dir / "bottleneck_composition.csv")

    fig1_context_ttft(ctx_rows, output_dir / "fig1_context_ttft.png")
    fig2_reuse_benefit(reuse_rows, output_dir / "fig2_reuse_benefit.png")
    fig3_load_sensitivity(load_rows, output_dir / "fig3_load_sensitivity.png")
    fig4_bottleneck_composition(comp_rows, output_dir / "fig4_bottleneck_composition.png")

    logger.info("Figures done -> %s", output_dir)


if __name__ == "__main__":
    main()
