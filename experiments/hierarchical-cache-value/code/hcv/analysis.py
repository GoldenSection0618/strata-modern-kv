"""Deterministic raw -> processed -> results processing.

Reads only from raw run directories (``<results_root>/<experiment>/
run-*/``), filters by ``validation.json`` (reportability) and per-run
validity, computes derived metrics, and writes processed datasets plus
summary CSV tables under ``processed/`` / ``results/`` of a *separate*
output root.  Raw directories are never modified.

Derived quantities (per docs/02, docs/03):

* recomputation reduction = recompute(gpu_only) - recompute(hierarchical)
* relative TTFT improvement = (ttft_gpu_only - ttft_hier) / ttft_gpu_only
* throughput gain = throughput_hier / throughput_gpu_only - 1
* CPU-tier contribution = cpu_hit / (gpu_hit + cpu_hit)
* reuse capture ratio = effective reused volume / total reusable opportunity

The processing is deterministic: stable ordering, fixed rounding, no
randomness, and every output row carries its raw ``run_tag``.
"""

from __future__ import annotations

import sys

import csv
import json
import os
import statistics
from pathlib import Path

from hcv.config import (
    ARCH_GPU_ONLY,
    ARCH_HIERARCHICAL,
    GATE_FULL,
)

# ---------------------------------------------------------------------------
# Raw-run discovery
# ---------------------------------------------------------------------------


def find_runs(results_root: str, experiment: str) -> list[dict]:
    """Discover raw runs for an experiment; each entry has run_tag + paths."""
    base = Path(results_root) / experiment
    runs = []
    if not base.is_dir():
        return runs
    for run_dir in sorted(base.glob("run-*")):
        runs.append({
            "run_tag": run_dir.name,
            "dir": str(run_dir),
            "metadata": _read_json(run_dir / "metadata.json"),
            "validation": _read_json(run_dir / "validation.json"),
            "cell_summary": _read_json(run_dir / "results" / "cell_summary.json"),
            "measurements": _read_jsonl(run_dir / "raw" / "measurements.jsonl"),
        })
    return runs


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def gate_status_of(run: dict) -> str:
    v = run.get("validation", {})
    return v.get("status", "missing")


def run_validity(run: dict) -> tuple[bool, str]:
    """Per-run validity for aggregation (metadata + gate)."""
    md = run.get("metadata", {})
    if gate_status_of(run) != GATE_FULL:
        return False, f"hierarchy gate not full ({gate_status_of(run)})"
    if md.get("validity_status") != "valid":
        return False, f"run validity_status={md.get('validity_status')}"
    return True, ""


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def derive_pair(gpu: dict, hier: dict) -> dict:
    """Derive paired comparison metrics (GPU-only vs hierarchical)."""
    out: dict = {}
    for key in ("gpu_hit_tokens", "cpu_hit_tokens", "recomputed_tokens",
                "restore_tokens", "restore_bytes", "ttft_p50_ms", "ttft_p90_ms",
                "throughput_req_per_s"):
        g = gpu.get(key)
        h = hier.get(key)
        out[f"{key}_gpu_only"] = g
        out[f"{key}_hierarchical"] = h

    rg, rh = gpu.get("recomputed_tokens"), hier.get("recomputed_tokens")
    if rg is not None and rh is not None:
        out["recomputation_reduction"] = round(rg - rh, 3)
        out["relative_recomputation_reduction"] = (
            round((rg - rh) / rg, 4) if rg else None
        )

    tg, th = gpu.get("ttft_p50_ms"), hier.get("ttft_p50_ms")
    if tg and th:
        out["relative_ttft_improvement"] = round((tg - th) / tg, 4)

    xg, xh = gpu.get("throughput_req_per_s"), hier.get("throughput_req_per_s")
    if xg and xh:
        out["throughput_gain"] = round(xh / xg - 1.0, 4)

    gh, ch = hier.get("gpu_hit_tokens"), hier.get("cpu_hit_tokens")
    if gh is not None and ch is not None and (gh + ch) > 0:
        out["cpu_tier_contribution"] = ch / (gh + ch)
    return out


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": round(values[0], 4), "std": 0.0, "n": 1}
    return {
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4),
        "n": len(values),
    }


def aggregate_over_reps(records: list[dict], key: str) -> dict:
    """Aggregate a metric across repetition records (mean/std/n)."""
    vals = [r[key] for r in records if r.get(key) is not None]
    return _mean_std(vals)


# ---------------------------------------------------------------------------
# Experiment processors
# ---------------------------------------------------------------------------


def process_exp1(results_root: str) -> dict:
    """Exp1: aggregate by (architecture, initial_state) cell."""
    runs = find_runs(results_root, "exp1")
    cells: dict = {}
    excluded: list[dict] = []
    for run in runs:
        md = run.get("metadata", {})
        arch = md.get("architecture")
        state = md.get("cache_initial_state")
        cell_key = f"{arch}/{state}"
        valid, reason = run_validity(run)
        reps = [r for r in run.get("measurements", []) if r.get("kind") == "repetition"
                and r.get("validity", {}).get("valid")]
        if not valid or not reps:
            excluded.append({"run_tag": run["run_tag"], "cell": cell_key,
                             "reason": reason or "no valid repetitions"})
            continue
        cell = cells.setdefault(cell_key, {"architecture": arch, "initial_state": state,
                                           "runs": []})
        agg = {
            "gpu_hit_tokens": aggregate_over_reps([r["summary"] for r in reps], "gpu_hit_tokens"),
            "cpu_hit_tokens": aggregate_over_reps([r["summary"] for r in reps], "cpu_hit_tokens"),
            "recomputed_tokens": aggregate_over_reps([r["summary"] for r in reps], "recomputed_tokens"),
            "eviction_tokens": aggregate_over_reps([r["summary"] for r in reps], "eviction_tokens"),
            "restore_tokens": aggregate_over_reps([r["summary"] for r in reps], "restore_tokens"),
            "ttft_p50_ms": aggregate_over_reps([r["summary"] for r in reps], "ttft_p50_ms"),
            "ttft_p90_ms": aggregate_over_reps([r["summary"] for r in reps], "ttft_p90_ms"),
            "throughput_req_per_s": aggregate_over_reps([r["summary"] for r in reps],
                                                       "throughput_req_per_s"),
        }
        cell["runs"].append({"run_tag": run["run_tag"], "aggregate": agg,
                             "phases": [r["summary"].get("phases", {}) for r in reps]})
    return {"cells": cells, "excluded": excluded}


def process_exp2(results_root: str) -> dict:
    """Exp2: aggregate by (architecture, pressure_label) + derived pairs."""
    runs = find_runs(results_root, "exp2")
    points: dict = {}
    excluded: list[dict] = []
    for run in runs:
        md = run.get("metadata", {})
        arch = md.get("architecture")
        label = md.get("pressure_label") or _label_from_run(run)
        key = f"{arch}/{label}"
        valid, reason = run_validity(run)
        reps = [r for r in run.get("measurements", []) if r.get("kind") == "pressure_point"
                and r.get("validity", {}).get("valid")]
        if not valid or not reps:
            excluded.append({"run_tag": run["run_tag"], "point": key,
                             "reason": reason or "no valid repetitions"})
            continue
        pt = points.setdefault(key, {"architecture": arch, "pressure_label": label,
                                     "budget": md.get("gpu_cache_budget"), "runs": []})
        agg = {
            "gpu_hit_tokens": aggregate_over_reps([r["summary"] for r in reps], "gpu_hit_tokens"),
            "cpu_hit_tokens": aggregate_over_reps([r["summary"] for r in reps], "cpu_hit_tokens"),
            "recomputed_tokens": aggregate_over_reps([r["summary"] for r in reps], "recomputed_tokens"),
            "restore_tokens": aggregate_over_reps([r["summary"] for r in reps], "restore_tokens"),
            "ttft_p50_ms": aggregate_over_reps([r["summary"] for r in reps], "ttft_p50_ms"),
            "throughput_req_per_s": aggregate_over_reps([r["summary"] for r in reps],
                                                       "throughput_req_per_s"),
        }
        pt["runs"].append({"run_tag": run["run_tag"], "aggregate": agg})
        pt["eviction_observed"] = any(
            r.get("eviction_evidence", {}).get("eviction_observed") for r in reps
        )
    # derived paired metrics per pressure label
    curves = {}
    for label in sorted({k.split("/")[1] for k in points}):
        gpu_key = f"{ARCH_GPU_ONLY}/{label}"
        hier_key = f"{ARCH_HIERARCHICAL}/{label}"
        gp = points.get(gpu_key)
        hp = points.get(hier_key)
        if gp and hp:
            g = gp["runs"][0]["aggregate"]
            h = hp["runs"][0]["aggregate"]
            curves[label] = {
                "pressure_label": label,
                "budget_gpu_only": gp["budget"],
                "budget_hierarchical": hp["budget"],
                **derive_pair({k: v.get("mean") for k, v in g.items()},
                              {k: v.get("mean") for k, v in h.items()}),
            }
    return {"points": points, "curves": curves, "excluded": excluded}


def _label_from_run(run: dict) -> str:
    for rec in run.get("measurements", []):
        if rec.get("kind") == "pressure_point":
            return rec.get("pressure_label", "Unknown")
    return "Unknown"


def process_exp3(results_root: str) -> dict:
    """Exp3: aggregate by (architecture, configured reuse) + locality check."""
    runs = find_runs(results_root, "exp3")
    levels: dict = {}
    excluded: list[dict] = []
    traces: dict = {}
    for run in runs:
        md = run.get("metadata", {})
        arch = md.get("architecture")
        configured = md.get("configured_revisit_fraction")
        if configured is None:
            configured = _reuse_from_run(run)
        key = f"{arch}/{configured}"
        valid, reason = run_validity(run)
        reps = [r for r in run.get("measurements", []) if r.get("kind") == "reuse_point"
                and r.get("validity", {}).get("valid")]
        if not valid or not reps:
            excluded.append({"run_tag": run["run_tag"], "point": key,
                             "reason": reason or "no valid repetitions"})
            continue
        lv = levels.setdefault(key, {"architecture": arch,
                                     "configured_revisit_fraction": configured,
                                     "budget": md.get("gpu_cache_budget"), "runs": []})
        agg = {
            "actual_reuse": _mean_std([r["actual_reuse_request_weighted"] for r in reps]),
            "gpu_hit_tokens": aggregate_over_reps([r["summary"] for r in reps], "gpu_hit_tokens"),
            "cpu_hit_tokens": aggregate_over_reps([r["summary"] for r in reps], "cpu_hit_tokens"),
            "recomputed_tokens": aggregate_over_reps([r["summary"] for r in reps], "recomputed_tokens"),
            "restore_tokens": aggregate_over_reps([r["summary"] for r in reps], "restore_tokens"),
            "ttft_p50_ms": aggregate_over_reps([r["summary"] for r in reps], "ttft_p50_ms"),
            "throughput_req_per_s": aggregate_over_reps([r["summary"] for r in reps],
                                                       "throughput_req_per_s"),
        }
        lv["runs"].append({"run_tag": run["run_tag"], "aggregate": agg})
        # locality evidence: reuse distance summary recorded per run
        lv["reuse_distance"] = reps[0].get("reuse_distance")
        traces[configured] = {"run_tag": run["run_tag"], "trace_id": md.get("trace_id", "")}
    # cross-level locality-unchanged check on saved traces
    locality = _check_locality(results_root, traces)
    curves = {}
    for configured in sorted({k.split("/")[1] for k in levels}, key=float):
        gpu_key = f"{ARCH_GPU_ONLY}/{configured}"
        hier_key = f"{ARCH_HIERARCHICAL}/{configured}"
        gp, hp = levels.get(gpu_key), levels.get(hier_key)
        if gp and hp:
            g = gp["runs"][0]["aggregate"]
            h = hp["runs"][0]["aggregate"]
            curves[configured] = {
                "configured_revisit_fraction": float(configured),
                "actual_reuse": g.get("actual_reuse", {}).get("mean"),
                **derive_pair({k: v.get("mean") if isinstance(v, dict) else v
                               for k, v in g.items()},
                              {k: v.get("mean") if isinstance(v, dict) else v
                               for k, v in h.items()}),
            }
    return {"levels": levels, "curves": curves, "locality": locality, "excluded": excluded}


def _reuse_from_run(run: dict) -> float:
    for rec in run.get("measurements", []):
        if rec.get("kind") == "reuse_point":
            return rec.get("configured_revisit_fraction", 0.0)
    return 0.0


def _check_locality(results_root: str, traces: dict) -> dict:
    """Compare saved traces across reuse levels (ordering/hotspot/distance)."""
    from hcv.workload import load_trace, validate_locality_unchanged

    out = {"checked": False, "pairs": [], "errors": []}
    fracs = sorted({float(f) for f in traces}, reverse=True)
    if len(fracs) < 2:
        out["errors"].append("fewer than two reuse levels to compare")
        return out
    hi = fracs[0]
    for lo in fracs[1:]:
        try:
            hi_trace = load_trace(_trace_path(results_root, traces[hi]["run_tag"],
                                              traces[hi]["trace_id"]))
            lo_trace = load_trace(_trace_path(results_root, traces[lo]["run_tag"],
                                              traces[lo]["trace_id"]))
            validate_locality_unchanged(hi_trace, lo_trace)
            out["pairs"].append({"higher": hi, "lower": lo, "ok": True})
        except Exception as e:  # noqa: BLE001
            out["pairs"].append({"higher": hi, "lower": lo, "ok": False,
                                 "error": str(e)})
            out["errors"].append(f"locality drift between {hi} and {lo}: {e}")
    out["checked"] = bool(out["pairs"])
    return out


def _trace_path(results_root: str, run_tag: str, trace_id: str) -> str:
    base = Path(results_root) / "exp3" / run_tag / "raw" / "traces"
    return str(base / f"trace-{trace_id}.json")


def process_exp4(results_root: str) -> dict:
    """Exp4: representative-point summary from points.json + gate status."""
    runs = find_runs(results_root, "exp4")
    summary = {"points": {}, "selection_rule": None, "runs": runs}
    rule_path = Path(results_root) / "exp4" / "representative_selection.json"
    if rule_path.is_file():
        summary["selection_rule"] = json.loads(rule_path.read_text())
    for run in runs:
        points = _read_json(Path(run["dir"]) / "results" / "points.json")
        for point, rec in points.items():
            row = {"run_tag": run["run_tag"], "point": point,
                   "gate_status": gate_status_of(run),
                   "budget": rec.get("mem_fraction_static"),
                   "actual_reuse": rec.get("actual_reuse")}
            for arch, reps in rec.get("cells", {}).items():
                if reps:
                    row[f"{arch}_recomputed_tokens"] = reps[0].get("recomputed_tokens")
                    row[f"{arch}_cpu_hit_tokens"] = reps[0].get("cpu_hit_tokens")
                    row[f"{arch}_preemption_windows"] = reps[0].get("preemption_windows")
            summary["points"][point] = row
    return summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_processed(out_root: str, name: str, data: dict) -> str:
    path = os.path.join(out_root, "processed", f"{name}.json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def write_csv(out_root: str, name: str, rows: list[dict]) -> str:
    path = os.path.join(out_root, "results", f"{name}.csv")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("(no rows)\n")
        return path
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in cols})
    return path


def run_all(results_root: str, out_root: str) -> dict:
    """Process all experiments into processed/ + results/ tables."""
    outputs = {}
    e1 = process_exp1(results_root)
    outputs["exp1"] = write_processed(out_root, "exp1_aggregate", e1)
    rows1 = []
    for key, cell in e1["cells"].items():
        for r in cell["runs"]:
            rows1.append({"cell": key, "run_tag": r["run_tag"],
                          **{f"{k}_mean": v.get("mean") for k, v in r["aggregate"].items()}})
    outputs["exp1_table"] = write_csv(out_root, "exp1_baseline", rows1)

    e2 = process_exp2(results_root)
    outputs["exp2"] = write_processed(out_root, "exp2_aggregate", e2)
    rows2 = [{"pressure_label": k, **{kk: vv for kk, vv in v.items() if kk != "pressure_label"}}
             for k, v in sorted(e2["curves"].items())]
    outputs["exp2_table"] = write_csv(out_root, "exp2_pressure_curve", rows2)

    e3 = process_exp3(results_root)
    outputs["exp3"] = write_processed(out_root, "exp3_aggregate", e3)
    rows3 = [{"configured_revisit_fraction": k, **{kk: vv for kk, vv in v.items()
                                                   if kk != "configured_revisit_fraction"}}
             for k, v in sorted(e3["curves"].items(), key=lambda kv: float(kv[0]))]
    outputs["exp3_table"] = write_csv(out_root, "exp3_reuse_curve", rows3)

    e4 = process_exp4(results_root)
    outputs["exp4"] = write_processed(out_root, "exp4_representative", e4)
    rows4 = []
    for point, row in sorted(e4["points"].items()):
        rows4.append({"point": point, **row})
    outputs["exp4_table"] = write_csv(out_root, "exp4_cross_model", rows4)
    return outputs


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--out-root", required=True)
    ns = p.parse_args(argv)
    outputs = run_all(ns.results_root, ns.out_root)
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
