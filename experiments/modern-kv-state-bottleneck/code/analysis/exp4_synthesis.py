"""Experiment 4: Cross-Model Bottleneck Comparison — synthesis script.

Reuses Experiment 1/2/3 summary results and produces cross-model
processed tables (CSV).  This is a pure analysis script: it does not
run any measurement.

Design doc: experiments/modern-kv-state-bottleneck/docs/04-cross-model-bottleneck-comparison.md

Outputs (into --output-dir):
  context_scaling.csv        model x context_length x residency_mode TTFT
  reuse_benefit.csv          model x prefix_ratio reuse metrics
  load_sensitivity.csv       model x normalized load (Experiment 3)
  bottleneck_composition.csv representative matched points TTFT composition
  synthesis_report.json      data availability + interpretation summary

Usage:
    python3 analysis/exp4_synthesis.py \
        --exp1-dir results/exp1/ \
        --exp2-dir results/exp2/ \
        --exp3-dir results/exp3/ \
        --output-dir results/exp4/processed/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("exp4-synthesis")

# ---------------------------------------------------------------------------
# Model identifier normalization
# ---------------------------------------------------------------------------

_MODEL_ALIASES = {
    "qwen": "qwen",
    "qwen3.5-9b": "qwen",
    "Qwen/Qwen3.5-9B": "qwen",
    "Qwen3.5-9B": "qwen",
    "gemma4": "gemma4",
    "gemma": "gemma4",
    "gemma-4-12b": "gemma4",
    "google/gemma-4-12B": "gemma4",
    "Gemma-4-12B": "gemma4",
}

# Experiment 1 fixes shared-prefix ratio at 50% (design doc section 4).
EXP1_FIXED_PREFIX_RATIO = 0.5

# Residency mode order used in outputs.
MODES = ["recompute", "gpu_hit", "cpu_hit"]

# Matched validation points (design doc section 4): (label, ctx, ratio).
MATCHED_POINTS = [
    ("short-light", 8192, 0.50),
    ("long-light", 32768, 0.50),
    ("long-high-reuse", 32768, 0.75),
]


def normalize_model(model: str) -> str:
    """Map raw model identifier from any summary.json to a canonical label."""
    key = model.strip()
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]
    # Fallback: lowercase prefix match
    lowered = key.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma4"
    logger.warning("Unknown model identifier %r, using as-is", model)
    return key


# ---------------------------------------------------------------------------
# Summary discovery
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _parse_ctx_pct_mode_from_dirname(dirname: str) -> tuple[int | None, float | None, str | None]:
    """Parse '32768-25pct-cpu_hit' style dir names (Experiment 2).

    Returns (context_length, prefix_ratio, residency_mode) or Nones.
    """
    m = re.match(r"^(\d+)-(\d+)pct-(recompute|gpu_hit|cpu_hit)$", dirname)
    if not m:
        return None, None, None
    ctx = int(m.group(1))
    ratio = int(m.group(2)) / 100.0
    mode = m.group(3)
    return ctx, ratio, mode


def discover_summaries(exp_dir: Path, experiment: str = "exp2") -> list[dict]:
    """Recursively find summary.json files and attach parsed config.

    Args:
        exp_dir: results/exp1/ or results/exp2/ directory.
        experiment: "exp1" or "exp2".  exp1 summary.json has NO
            prefix_ratio field and its dirs are {model}/{ctx}/{mode}/...
            (design fixes ratio at 50%); exp2 dirs are
            {model}/{ctx}-{pct}pct-{mode}/... and do carry prefix_ratio.

    Returns list of dicts with keys:
      model, context_length, prefix_ratio, residency_mode, n_repeats,
      median_ttft_ms, p90_ttft_ms, min_ttft_ms, max_ttft_ms,
      summary_path, experiment
    """
    found: list[dict] = []
    if not exp_dir.is_dir():
        logger.warning("Experiment dir does not exist: %s", exp_dir)
        return found

    for summary_path in sorted(exp_dir.rglob("summary.json")):
        data = _load_json(summary_path)
        if not data:
            continue

        # Residency mode: prefer JSON field, fall back to parent dir name.
        mode = data.get("residency_mode")
        if not mode:
            mode = summary_path.parent.name

        # Context length: prefer JSON field, fall back to dirname parsing.
        ctx = data.get("context_length")
        ratio = data.get("prefix_ratio")

        # Experiment 2 style dirname: '32768-25pct-recompute'
        if ctx is None or ratio is None:
            parts = summary_path.parts
            # Walk up: .../results/exp2/qwen/32768-25pct-recompute/recompute/summary.json
            candidate = None
            for part in reversed(parts[:-1]):
                parsed = _parse_ctx_pct_mode_from_dirname(part)
                if parsed[0] is not None:
                    candidate = part
                    break
            if candidate is not None:
                pctx, pratio, pmode = _parse_ctx_pct_mode_from_dirname(candidate)
                if ctx is None:
                    ctx = pctx
                if ratio is None and pratio is not None:
                    ratio = pratio
                if mode is None:
                    mode = pmode

        # Experiment 1: prefix ratio fixed at 50% by design.
        if ratio is None and experiment == "exp1":
            ratio = EXP1_FIXED_PREFIX_RATIO

        if ctx is None or ratio is None or mode is None:
            logger.warning(
                "Cannot fully parse config from %s (ctx=%s ratio=%s mode=%s); skipping",
                summary_path, ctx, ratio, mode,
            )
            continue

        found.append({
            "model": normalize_model(data.get("model", "unknown")),
            "context_length": int(ctx),
            "prefix_ratio": float(ratio),
            "residency_mode": mode,
            "n_repeats": data.get("n_repeats"),
            "median_ttft_ms": data.get("median_ttft_ms"),
            "p90_ttft_ms": data.get("p90_ttft_ms"),
            "min_ttft_ms": data.get("min_ttft_ms"),
            "max_ttft_ms": data.get("max_ttft_ms"),
            "summary_path": str(summary_path),
            "experiment": experiment,
        })
    return found


def discover_exp3(exp3_dir: Path) -> list[dict]:
    """Discover Experiment 3 summaries (per load point).

    exp3 summary.json is a LIST of per-rate aggregated dicts, so this
    returns one flat record per load point, enriched with the model /
    residency from the enclosing directory tree.
    """
    found: list[dict] = []
    if not exp3_dir.is_dir():
        logger.warning("Experiment 3 dir does not exist: %s", exp3_dir)
        return found

    for summary_path in sorted(exp3_dir.rglob("summary.json")):
        data = _load_json(summary_path)
        if not data or not isinstance(data, list):
            logger.warning("exp3 summary %s is not a list; skipping", summary_path)
            continue

        # Model/residency from path: .../results/exp3/qwen/32k-50pct-cpu_hit/summary.json
        rel = summary_path.relative_to(exp3_dir)
        parts = list(rel.parts)
        model_label = parts[0] if parts else "unknown"
        mode = None
        ctx = None
        ratio = None
        for part in parts:
            m = re.match(r"^(\d+)k-(\d+)pct-(recompute|gpu_hit|cpu_hit)$", part)
            if m:
                ctx = int(m.group(1)) * 1024
                ratio = int(m.group(2)) / 100.0
                mode = m.group(3)
                break
            if part in MODES:
                mode = part

        for point in data:
            rec = dict(point)
            rec["model"] = normalize_model(model_label)
            rec["residency_mode"] = mode or rec.get("residency_mode")
            rec["context_length"] = ctx
            rec["prefix_ratio"] = ratio
            rec["summary_path"] = str(summary_path)
            found.append(rec)
    return found


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def build_context_scaling(all_summaries: list[dict]) -> list[dict]:
    """model x context_length x residency_mode -> TTFT stats (Experiment 1)."""
    rows = []
    for s in sorted(
        all_summaries,
        key=lambda r: (r["model"], r["context_length"], r["residency_mode"]),
    ):
        rows.append({
            "model": s["model"],
            "context_length": s["context_length"],
            "residency_mode": s["residency_mode"],
            "n_repeats": s["n_repeats"],
            "median_ttft_ms": s["median_ttft_ms"],
            "p90_ttft_ms": s["p90_ttft_ms"],
            "min_ttft_ms": s["min_ttft_ms"],
            "max_ttft_ms": s["max_ttft_ms"],
            "summary_path": s["summary_path"],
        })
    return rows


def build_reuse_benefit(all_summaries: list[dict]) -> list[dict]:
    """model x context_length x prefix_ratio reuse metrics (Experiment 2)."""
    # Index by (model, ctx, ratio) -> {mode: summary}
    by_key: dict[tuple, dict[str, dict]] = {}
    for s in all_summaries:
        key = (s["model"], s["context_length"], s["prefix_ratio"])
        by_key.setdefault(key, {})[s["residency_mode"]] = s

    rows = []
    for (model, ctx, ratio), modes in sorted(by_key.items()):
        rec = modes.get("recompute", {})
        gpu = modes.get("gpu_hit", {})
        cpu = modes.get("cpu_hit", {})

        rec_med = rec.get("median_ttft_ms")
        gpu_med = gpu.get("median_ttft_ms")
        cpu_med = cpu.get("median_ttft_ms")

        row = {
            "model": model,
            "context_length": ctx,
            "prefix_ratio": ratio,
            "prefix_tokens": int(ctx * ratio),
            "suffix_tokens": int(ctx * (1 - ratio)),
            "recompute_ttft_ms": rec_med,
            "gpu_hit_ttft_ms": gpu_med,
            "cpu_hit_ttft_ms": cpu_med,
        }

        # Net Reuse Benefit / Reuse Speedup (CPU-resident hierarchical reuse)
        if rec_med is not None and cpu_med is not None:
            row["net_reuse_benefit_ms"] = round(rec_med - cpu_med, 3)
            row["reuse_speedup"] = (
                round(rec_med / cpu_med, 4) if cpu_med > 0 else None
            )
        else:
            row["net_reuse_benefit_ms"] = None
            row["reuse_speedup"] = None

        # GPU-resident reuse benefit (upper bound without restore cost)
        if rec_med is not None and gpu_med is not None:
            row["gpu_reuse_benefit_ms"] = round(rec_med - gpu_med, 3)
        else:
            row["gpu_reuse_benefit_ms"] = None

        # CPU restore penalty and service stall ratio
        if gpu_med is not None and cpu_med is not None:
            row["cpu_restore_penalty_ms"] = round(cpu_med - gpu_med, 3)
            row["service_stall_ratio"] = (
                round((cpu_med - gpu_med) / cpu_med, 4) if cpu_med > 0 else None
            )
        else:
            row["cpu_restore_penalty_ms"] = None
            row["service_stall_ratio"] = None

        # Reuse Realization Ratio = Net Reuse Benefit / Avoidable Recompute Time.
        # Avoidable recompute time is approximated by GPU-resident reuse
        # benefit (recompute - gpu_hit).  Not reported when denominator is
        # tiny or missing (design doc section 8).
        nrb = row.get("net_reuse_benefit_ms")
        arb = row.get("gpu_reuse_benefit_ms")
        if nrb is not None and arb is not None and arb >= 1.0:
            row["reuse_realization_ratio"] = round(nrb / arb, 4)
        else:
            row["reuse_realization_ratio"] = None

        row["data_complete"] = all(v is not None for v in (rec_med, gpu_med, cpu_med))
        rows.append(row)
    return rows


def build_load_sensitivity(exp3_records: list[dict]) -> list[dict]:
    """model x offered_rate / normalized_load (Experiment 3)."""
    rows = []
    for r in sorted(
        exp3_records,
        key=lambda x: (x.get("model", ""), x.get("normalized_load", 0)),
    ):
        rows.append({
            "model": r.get("model"),
            "context_length": r.get("context_length"),
            "prefix_ratio": r.get("prefix_ratio"),
            "residency_mode": r.get("residency_mode"),
            "offered_rate": r.get("offered_rate"),
            "normalized_load": r.get("normalized_load"),
            "achieved_throughput_mean": r.get("achieved_throughput_mean"),
            "ttft_p50_mean": r.get("ttft_p50_mean"),
            "ttft_p90_mean": r.get("ttft_p90_mean"),
            "ttft_p99_mean": r.get("ttft_p99_mean"),
            "queueing_p50_mean": r.get("queueing_p50_mean"),
            "queueing_p90_mean": r.get("queueing_p90_mean"),
            "service_p50_mean": r.get("service_p50_mean"),
            "active_concurrency_max": r.get("active_concurrency_max"),
            "summary_path": r.get("summary_path"),
        })
    return rows


def build_bottleneck_composition(all_summaries: list[dict]) -> list[dict]:
    """Representative matched points -> TTFT composition.

    Low-load experiments 1/2 have negligible queueing by design, so:
      TTFT = service time = compute path + non-overlapped I/O stall + other
    compute path is approximated by GPU-resident hit TTFT (lower bound for
    reuse), or by recompute TTFT when gpu_hit is unavailable.  I/O stall is
    cpu_hit - gpu_hit.  'other' is the residual (measured TTFT - parts).
    """
    rows = []
    for label, mctx, mratio in MATCHED_POINTS:
        for model in sorted({s["model"] for s in all_summaries}):
            # Find matching summaries for this model/ctx/ratio.
            key = (model, mctx, mratio)
            modes: dict[str, dict] = {}
            for s in all_summaries:
                if (s["model"], s["context_length"], s["prefix_ratio"]) == key:
                    modes[s["residency_mode"]] = s
            rec = modes.get("recompute", {}).get("median_ttft_ms")
            gpu = modes.get("gpu_hit", {}).get("median_ttft_ms")
            cpu = modes.get("cpu_hit", {}).get("median_ttft_ms")

            row = {
                "workload": label,
                "context_length": mctx,
                "prefix_ratio": mratio,
                "model": model,
                "queueing_ms": None,  # low-load by design
            }
            if cpu is not None and gpu is not None:
                # compute path = gpu-resident TTFT (reuse without restore cost)
                # I/O stall = cpu-hit TTFT - gpu-hit TTFT (restore cost)
                # 'other' cannot be separated from TTFT alone -> None
                row["ttft_ms"] = cpu
                row["compute_path_ms"] = gpu
                row["io_stall_ms"] = round(cpu - gpu, 3)
                row["other_ms"] = None
                row["parts_available"] = True
            elif rec is not None:
                # No hit data: report recompute TTFT, parts unavailable.
                row["ttft_ms"] = rec
                row["compute_path_ms"] = None
                row["io_stall_ms"] = None
                row["other_ms"] = None
                row["parts_available"] = False
            else:
                row["ttft_ms"] = None
                row["compute_path_ms"] = None
                row["io_stall_ms"] = None
                row["other_ms"] = None
                row["parts_available"] = False
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    exp1_summaries: list[dict],
    exp2_summaries: list[dict],
    exp3_records: list[dict],
    models: set[str],
) -> dict:
    """Data-availability and interpretation summary."""
    models_sorted = sorted(models)

    def modes_available(sums: list[dict], model: str) -> dict:
        ms = {}
        for s in sums:
            if s["model"] == model:
                ms.setdefault(s["residency_mode"], 0)
                ms[s["residency_mode"]] += 1
        return ms

    report = {
        "generated_by": "exp4_synthesis.py",
        "note": (
            "Experiment 4 reuses Experiments 1-3; no new measurements. "
            "Cross-model differences are descriptive, not causal."
        ),
        "models_found": models_sorted,
        "exp1_summaries_per_model": {
            m: modes_available(exp1_summaries, m) for m in models_sorted
        },
        "exp2_summaries_per_model": {
            m: modes_available(exp2_summaries, m) for m in models_sorted
        },
        "exp3_load_points_per_model": {
            m: sum(1 for r in exp3_records if r.get("model") == m)
            for m in models_sorted
        },
        "matched_points": [
            {"workload": w, "context_length": c, "prefix_ratio": r}
            for w, c, r in MATCHED_POINTS
        ],
        "interpretation_guide": {
            "A_trends_consistent": (
                "Both models show rising restore pressure / I/O stall with "
                "context/reuse/load -> Strata problem is cross-model stable."
            ),
            "B_same_problem_different_severity": (
                "Both models have state-loading cost, but trigger region "
                "differs -> model-dependent severity/conditions."
            ),
            "C_only_one_model_bottlenecked": (
                "Strata motivation is NOT a unified modern-hybrid problem."
            ),
            "D_compute_dominated": (
                "No significant non-overlapped state stall -> original Strata "
                "motivation weakened on this model/runtime/hardware."
            ),
        },
    }
    return report


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: cross-model synthesis (analysis only)"
    )
    parser.add_argument("--exp1-dir", required=True)
    parser.add_argument("--exp2-dir", required=True)
    parser.add_argument("--exp3-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    exp1_dir = Path(args.exp1_dir)
    exp2_dir = Path(args.exp2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover
    exp1_summaries = discover_summaries(exp1_dir, experiment="exp1")
    exp2_summaries = discover_summaries(exp2_dir, experiment="exp2")
    logger.info("exp1 summaries: %d, exp2 summaries: %d", len(exp1_summaries), len(exp2_summaries))

    all_exp12 = exp1_summaries + exp2_summaries
    models = {s["model"] for s in all_exp12}

    exp3_records: list[dict] = []
    if args.exp3_dir:
        exp3_records = discover_exp3(Path(args.exp3_dir))
        logger.info("exp3 load points: %d", len(exp3_records))
        models |= {r.get("model") for r in exp3_records if r.get("model")}

    # Context scaling (Experiment 1)
    ctx_rows = build_context_scaling(exp1_summaries)
    write_csv(
        ctx_rows, output_dir / "context_scaling.csv",
        ["model", "context_length", "residency_mode", "n_repeats",
         "median_ttft_ms", "p90_ttft_ms", "min_ttft_ms", "max_ttft_ms",
         "summary_path"],
    )

    # Reuse benefit (Experiment 2)
    reuse_rows = build_reuse_benefit(exp2_summaries)
    write_csv(
        reuse_rows, output_dir / "reuse_benefit.csv",
        ["model", "context_length", "prefix_ratio", "prefix_tokens",
         "suffix_tokens", "recompute_ttft_ms", "gpu_hit_ttft_ms",
         "cpu_hit_ttft_ms", "net_reuse_benefit_ms", "reuse_speedup",
         "gpu_reuse_benefit_ms", "cpu_restore_penalty_ms",
         "service_stall_ratio", "reuse_realization_ratio", "data_complete"],
    )

    # Load sensitivity (Experiment 3)
    load_rows = build_load_sensitivity(exp3_records)
    write_csv(
        load_rows, output_dir / "load_sensitivity.csv",
        ["model", "context_length", "prefix_ratio", "residency_mode",
         "offered_rate", "normalized_load", "achieved_throughput_mean",
         "ttft_p50_mean", "ttft_p90_mean", "ttft_p99_mean",
         "queueing_p50_mean", "queueing_p90_mean", "service_p50_mean",
         "active_concurrency_max", "summary_path"],
    )

    # Bottleneck composition (matched points)
    comp_rows = build_bottleneck_composition(all_exp12)
    write_csv(
        comp_rows, output_dir / "bottleneck_composition.csv",
        ["workload", "context_length", "prefix_ratio", "model",
         "ttft_ms", "queueing_ms", "compute_path_ms", "io_stall_ms",
         "other_ms", "parts_available"],
    )

    # Report
    report = build_report(exp1_summaries, exp2_summaries, exp3_records, models)
    report_path = output_dir / "synthesis_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote %s", report_path)

    logger.info("Experiment 4 synthesis complete -> %s", output_dir)


if __name__ == "__main__":
    main()
