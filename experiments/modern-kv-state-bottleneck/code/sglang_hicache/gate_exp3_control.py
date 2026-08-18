#!/usr/bin/env python3
"""Strict semantic/result gate for one Exp3 control run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path):
    require(path.is_file(), f"missing required artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--mode", choices=("recompute", "gpu_hit"), required=True)
    parser.add_argument("--expected-formal-points", type=int, default=3)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--expected-requests", type=int, default=30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    require(run_dir.is_dir(), f"control run directory does not exist: {run_dir}")
    metadata = load_json(run_dir / "metadata.json")
    validation = load_json(run_dir / "validation.json")
    summary = load_json(run_dir / "summary.json")

    require(metadata.get("run_tag") == args.run_tag, "metadata run_tag mismatch")
    require(metadata.get("residency_mode") == args.mode, "residency mode mismatch")
    require(int(metadata.get("context_length", 0)) == 32768, "context is not 32768")
    require(int(metadata.get("prefix_pool_size", 0)) == 12, "prefix_pool_size is not 12")
    require(int(metadata.get("n_measured", 0)) == args.expected_requests, "n_measured mismatch")
    require(int(metadata.get("n_repeats", 0)) == args.expected_repeats, "n_repeats mismatch")
    require(metadata.get("control_mode") is True, "control_mode is not true")
    require(validation.get("all_passed") is True, "validation gate did not pass")

    require(isinstance(summary, list), "summary is not a list")
    require(len(summary) == args.expected_formal_points, "formal summary point count mismatch")
    require(
        all(row.get("residency_dominance_ok") is True for row in summary),
        "one or more formal points failed residency dominance",
    )

    raw_files = sorted((run_dir / "raw").glob("load_*_rep_*.json"))
    expected_raw = args.expected_formal_points * args.expected_repeats
    require(len(raw_files) == expected_raw, f"raw repetition count mismatch: {len(raw_files)} != {expected_raw}")
    for raw_path in raw_files:
        raw = load_json(raw_path)
        require(raw.get("residency_dominance_ok") is True, f"dominance failed in {raw_path.name}")
        records = raw.get("records")
        require(isinstance(records, list), f"records missing in {raw_path.name}")
        require(len(records) == args.expected_requests, f"request count mismatch in {raw_path.name}")
        require(all(r.get("ok") is True for r in records), f"failed request in {raw_path.name}")

    print(f"EXP3 CONTROL GATE PASS: mode={args.mode} run={run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"EXP3 CONTROL GATE FAIL: {exc}", flush=True)
        raise SystemExit(1) from exc
