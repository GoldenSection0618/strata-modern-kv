"""Capacity calibration for Experiment 3.

Estimates sustainable capacity by probing the system at increasing
offered rates.  The sustainable capacity is the highest offered rate
at which achieved throughput continues to track offered load without
a persistently growing queue.

The calibration result is a set of load points (fractions of sustainable
capacity) that will be used in the formal sweep.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from profiling.load_driver import (
    AsyncLoadDriver,
    LoadDriverConfig,
    summarize_records,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a capacity calibration run."""

    model_id: str
    residency_mode: str
    context_length: int
    # Per-probe-rate summary
    probes: list[dict] = field(default_factory=list)
    # Estimated sustainable capacity (req/s)
    sustainable_capacity: float = 0.0
    # Final load points for formal sweep (offered rates in req/s)
    sweep_rates: list[float] = field(default_factory=list)
    # Normalized load points (fractions of sustainable capacity)
    normalized_loads: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "residency_mode": self.residency_mode,
            "context_length": self.context_length,
            "probes": self.probes,
            "sustainable_capacity": round(self.sustainable_capacity, 3),
            "sweep_rates": [round(r, 3) for r in self.sweep_rates],
            "normalized_loads": [round(l, 3) for l in self.normalized_loads],
        }


def run_calibration(
    llm,
    sampling_params,
    prompts: list[dict],
    model_id: str,
    residency_mode: str,
    context_length: int,
    probe_rates: list[float],
    n_requests_per_probe: int = 15,
    concurrency_ceiling: int = 64,
    output_dir: str | None = None,
) -> CalibrationResult:
    """Run capacity calibration.

    Sends a small batch of requests at each probe rate and measures
    achieved throughput vs offered rate.  The sustainable capacity is
    estimated as the highest rate where achieved ≈ offered.
    """
    import asyncio

    result = CalibrationResult(
        model_id=model_id,
        residency_mode=residency_mode,
        context_length=context_length,
    )

    logger.info("=== Capacity Calibration ===")
    logger.info("Probe rates: %s", probe_rates)

    for rate in probe_rates:
        config = LoadDriverConfig(
            offered_rate=rate,
            n_requests=n_requests_per_probe,
            concurrency_ceiling=concurrency_ceiling,
            arrival_pattern="poisson",
        )
        driver = AsyncLoadDriver(llm, sampling_params, prompts, config)
        records = asyncio.run(driver.run())
        summary = summarize_records(records)

        # Check if achieved tracks offered
        achieved = summary["achieved_throughput"]
        ratio = achieved / rate if rate > 0 else 1.0
        is_tracking = ratio > 0.85  # 85% tracking threshold

        probe = {
            "offered_rate": rate,
            "achieved_throughput": achieved,
            "tracking_ratio": round(ratio, 3),
            "is_tracking": is_tracking,
            "ttft_p50_ms": summary["ttft_p50_ms"],
            "ttft_p90_ms": summary["ttft_p90_ms"],
            "queueing_p50_ms": summary["queueing_p50_ms"],
            "queueing_p90_ms": summary["queueing_p90_ms"],
            "active_concurrency_max": summary["active_concurrency_max"],
        }
        result.probes.append(probe)

        logger.info(
            "  rate=%.1f → achieved=%.1f (ratio=%.2f, tracking=%s), "
            "TTFT p50=%.1f p90=%.1f, queue p50=%.1f",
            rate, achieved, ratio, is_tracking,
            summary["ttft_p50_ms"], summary["ttft_p90_ms"],
            summary["queueing_p50_ms"],
        )

    # Estimate sustainable capacity: highest tracking rate
    tracking_rates = [p["offered_rate"] for p in result.probes if p["is_tracking"]]
    if tracking_rates:
        result.sustainable_capacity = max(tracking_rates)
    else:
        # All probes failed tracking; use the lowest probe as conservative estimate
        result.sustainable_capacity = min(probe_rates) if probe_rates else 1.0
        logger.warning(
            "No probe tracked offered rate. Using conservative capacity=%.1f",
            result.sustainable_capacity,
        )

    # Generate sweep load points: ~6-8 points covering low to overload
    cap = result.sustainable_capacity
    normalized_loads = [0.25, 0.50, 0.70, 0.85, 1.00, 1.15, 1.30]
    result.normalized_loads = normalized_loads
    result.sweep_rates = [round(cap * nl, 2) for nl in normalized_loads]

    logger.info(
        "Sustainable capacity: %.2f req/s", result.sustainable_capacity
    )
    logger.info("Sweep rates: %s", result.sweep_rates)

    # Save calibration result
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "calibration.json"
        path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Calibration saved: %s", path)

    return result
