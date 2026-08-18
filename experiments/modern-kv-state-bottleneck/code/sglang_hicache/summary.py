"""Pure summary statistics shared by the SGLang Exp1-3 entry points.

Key conventions are byte-compatible with the existing vLLM path
(``profiling.load_driver.summarize_records`` / ``run_exp3.save_summary``)
so that raw/summary outputs from either runtime feed the same analysis
scripts (``analysis/exp{1,2}_analysis.py``, ``analysis/exp4_synthesis.py``).

Percentile convention (nearest-rank, preserved from the vLLM path):

    idx = int(len(sorted_vals) * p)
    value = sorted_vals[min(idx, len(sorted_vals) - 1)]
"""

from __future__ import annotations

from typing import Optional


def percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted list (vLLM-compatible)."""
    if not sorted_vals:
        raise ValueError("percentile of empty list")
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def summarize_ttft(ttfts: list[float], n_repeats: int | None = None) -> dict:
    """Median/P90/min/max summary with the same keys as the vLLM path."""
    if not ttfts:
        return {
            "median_ttft_ms": None,
            "p90_ttft_ms": None,
            "min_ttft_ms": None,
            "max_ttft_ms": None,
            "n_repeats": n_repeats,
        }
    s = sorted(ttfts)
    return {
        "median_ttft_ms": round(percentile(s, 0.50), 3),
        "p90_ttft_ms": round(percentile(s, 0.90), 3),
        "min_ttft_ms": round(s[0], 3),
        "max_ttft_ms": round(s[-1], 3),
        "n_repeats": n_repeats,
    }


def summarize_records(records: list[dict]) -> dict:
    """Summarize per-request load-driver records (vLLM-compatible keys).

    ``records`` are dicts with the fields produced by the SGLang load
    driver (request_id, t_arrival, t_start, t_first_token, t_complete)
    or the vLLM ``RequestRecord.to_dict()`` shape.
    """
    if not records:
        return {"error": "no records"}

    ttfts = sorted(r["ttft_ms"] for r in records)
    queueings = sorted(r["queueing_ms"] for r in records)
    services = sorted(r["service_ms"] for r in records)
    n = len(ttfts)

    # Active concurrency: max overlap of [t_start, t_first_token] intervals.
    events = []
    for r in records:
        events.append((r["t_start"], 1))
        events.append((r["t_first_token"], -1))
    events.sort()
    max_concurrency = 0
    current = 0
    for _, delta in events:
        current += delta
        max_concurrency = max(max_concurrency, current)

    # Achieved throughput: completed requests / total time span.
    t_min = min(r["t_arrival"] for r in records)
    t_max = max(r["t_complete"] for r in records)
    duration = t_max - t_min
    achieved_throughput = n / duration if duration > 0 else 0.0

    # Mean active concurrency via Little's law: L = rate * mean service time.
    if duration > 0:
        active_concurrency_mean = sum(services) / (duration * 1000)
    else:
        active_concurrency_mean = 0.0

    return {
        "n_requests": n,
        "offered_rate": round(n / duration, 3) if duration > 0 else 0.0,
        "achieved_throughput": round(achieved_throughput, 3),
        "active_concurrency_max": max_concurrency,
        "active_concurrency_mean": round(active_concurrency_mean, 2),
        "ttft_p50_ms": round(percentile(ttfts, 0.50), 3),
        "ttft_p90_ms": round(percentile(ttfts, 0.90), 3),
        "ttft_p99_ms": round(percentile(ttfts, 0.99), 3),
        "ttft_min_ms": round(ttfts[0], 3),
        "ttft_max_ms": round(ttfts[-1], 3),
        "ttft_mean_ms": round(sum(ttfts) / n, 3),
        "queueing_p50_ms": round(percentile(queueings, 0.50), 3),
        "queueing_p90_ms": round(percentile(queueings, 0.90), 3),
        "queueing_p99_ms": round(percentile(queueings, 0.99), 3),
        "queueing_mean_ms": round(sum(queueings) / n, 3),
        "service_p50_ms": round(percentile(services, 0.50), 3),
        "service_p90_ms": round(percentile(services, 0.90), 3),
        "service_mean_ms": round(sum(services) / n, 3),
        "duration_s": round(duration, 3),
    }


def _mean_or_none(values: list) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _max_or_none(values: list) -> Optional[float]:
    if not values:
        return None
    return round(max(values), 3)


def aggregate_load_summaries(all_summaries: list[dict]) -> list[dict]:
    """Aggregate per-rep load summaries by offered_rate (vLLM-compatible).

    Mirrors ``run_exp3.save_summary`` output keys so
    ``analysis/exp4_synthesis.py`` can consume SGLang Exp3 results.
    """
    by_rate: dict[float, list[dict]] = {}
    for s in all_summaries:
        by_rate.setdefault(s["offered_rate"], []).append(s)

    def _tier_ratios(reps: list[dict], key: str) -> list[float]:
        values = [
            r.get("tier_breakdown", {}).get(key)
            for r in reps
            if isinstance(r.get("tier_breakdown"), dict)
        ]
        return [v for v in values if v is not None]

    aggregated = []
    for rate in sorted(by_rate.keys()):
        reps = by_rate[rate]
        aggregated.append({
            "offered_rate": rate,
            "normalized_load": reps[0].get("normalized_load", 0.0),
            "n_reps": len(reps),
            "achieved_throughput_mean": _mean_or_none(
                [r.get("achieved_throughput", 0) for r in reps]
            ),
            "ttft_p50_mean": _mean_or_none([r.get("ttft_p50_ms", 0) for r in reps]),
            "ttft_p90_mean": _mean_or_none([r.get("ttft_p90_ms", 0) for r in reps]),
            "ttft_p99_mean": _mean_or_none([r.get("ttft_p99_ms", 0) for r in reps]),
            "ttft_p50_max": _max_or_none([r.get("ttft_p50_ms", 0) for r in reps]),
            "ttft_p90_max": _max_or_none([r.get("ttft_p90_ms", 0) for r in reps]),
            "ttft_p99_max": _max_or_none([r.get("ttft_p99_ms", 0) for r in reps]),
            "queueing_p50_mean": _mean_or_none([r.get("queueing_p50_ms", 0) for r in reps]),
            "queueing_p90_mean": _mean_or_none([r.get("queueing_p90_ms", 0) for r in reps]),
            "service_p50_mean": _mean_or_none([r.get("service_p50_ms", 0) for r in reps]),
            "service_p90_mean": _mean_or_none([r.get("service_p90_ms", 0) for r in reps]),
            "active_concurrency_max": _max_or_none(
                [r.get("active_concurrency_max", 0) for r in reps]
            ),
            "tier_device_ratio_mean": _mean_or_none(_tier_ratios(reps, "device_ratio")),
            "tier_host_ratio_mean": _mean_or_none(_tier_ratios(reps, "host_ratio")),
            "residency_dominance_ok": all(
                r.get("residency_dominance_ok", True) for r in reps
            ),
            "unsupported": any(r.get("unsupported", False) for r in reps),
        })
    return aggregated


def calibrate_sustainable_capacity(
    probes: list[dict], tracking_ratio: float = 0.85
) -> float:
    """Estimate sustainable capacity from probe summaries (vLLM-compatible).

    ``probes`` are dicts with ``offered_rate`` and ``achieved_throughput``.
    Capacity = highest offered rate whose achieved/offered tracking ratio
    is at least ``tracking_ratio``; falls back to the lowest probe rate.
    """
    tracking = [
        p["offered_rate"]
        for p in probes
        if p.get("achieved_throughput") is not None
        and p["offered_rate"] > 0
        and p["achieved_throughput"] / p["offered_rate"] >= tracking_ratio
    ]
    if tracking:
        return max(tracking)
    return min((p["offered_rate"] for p in probes), default=1.0)


def sweep_rates_from_capacity(capacity: float) -> list[float]:
    """Default 7-point normalized-load sweep (vLLM-compatible)."""
    normalized = [0.25, 0.50, 0.70, 0.85, 1.00, 1.15, 1.30]
    return [round(capacity * nl, 2) for nl in normalized]
