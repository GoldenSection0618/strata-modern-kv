#!/usr/bin/env python3
"""Strict semantic gate for an Exp3 primary run.

Slurm COMPLETED is necessary but not sufficient: the runner can finish after
writing an unsupported result. This gate validates the exact run directory
and emits the frozen rates consumed by dependent controls only on success.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


def load_json(path: Path):
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_positive(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    require(math.isfinite(number) and number > 0, f"{label} must be finite and > 0")
    return number


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--rates-file", required=True)
    parser.add_argument("--expected-calibration-points", type=int, default=7)
    parser.add_argument("--expected-formal-points", type=int, default=7)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--expected-requests", type=int, default=30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rates_file = Path(args.rates_file).resolve()
    require(run_dir.is_dir(), f"primary run directory does not exist: {run_dir}")

    metadata = load_json(run_dir / "metadata.json")
    validation = load_json(run_dir / "validation.json")
    calibration = load_json(run_dir / "calibration.json")
    summary = load_json(run_dir / "summary.json")

    require(metadata.get("run_tag") == args.run_tag, "metadata run_tag mismatch")
    require(metadata.get("residency_mode") == "cpu_hit", "primary residency is not cpu_hit")
    require(int(metadata.get("context_length", 0)) == 32768, "primary context is not 32768")
    require(int(metadata.get("prefix_pool_size", 0)) == 12, "prefix_pool_size is not 12")
    require(int(metadata.get("n_measured", 0)) == args.expected_requests, "n_measured mismatch")
    require(int(metadata.get("n_repeats", 0)) == args.expected_repeats, "n_repeats mismatch")
    require(validation.get("all_passed") is True, "validation gate did not pass")

    probes = calibration.get("probes")
    require(isinstance(probes, list), "calibration probes missing")
    require(len(probes) == args.expected_calibration_points, "calibration point count mismatch")
    require(
        all(p.get("residency_dominance_ok") is True for p in probes),
        "one or more calibration probes failed residency dominance",
    )

    rates = calibration.get("sweep_rates")
    require(isinstance(rates, list), "calibration sweep_rates missing")
    require(len(rates) == args.expected_formal_points, "frozen rate count mismatch")
    numeric_rates = [finite_positive(v, f"sweep_rates[{i}]") for i, v in enumerate(rates)]

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

    rates_file.parent.mkdir(parents=True, exist_ok=True)
    rates_text = ",".join(format(rate, ".12g") for rate in numeric_rates) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".frozen-rates-", dir=rates_file.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rates_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, rates_file)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    print(f"EXP3 PRIMARY GATE PASS: {run_dir}")
    print(f"FROZEN_RATES={rates_text.strip()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"EXP3 PRIMARY GATE FAIL: {exc}", flush=True)
        raise SystemExit(1) from exc
