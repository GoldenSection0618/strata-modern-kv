"""Unit tests for percentile / load summaries and calibration math."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang_hicache.summary import (  # noqa: E402
    aggregate_load_summaries,
    calibrate_sustainable_capacity,
    percentile,
    summarize_records,
    sweep_rates_from_capacity,
)

from fixtures import sample_records  # noqa: E402


class TestPercentile(unittest.TestCase):
    def test_nearest_rank_convention(self):
        # Matches the vLLM path exactly: idx = int(n*p), min(idx, n-1).
        # For n=10: p50 -> idx=5 -> 6th element; p90/p99 -> idx=9 -> max.
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.assertEqual(percentile(vals, 0.50), 6.0)   # idx=5
        self.assertEqual(percentile(vals, 0.90), 10.0)  # idx=9 -> max
        self.assertEqual(percentile(vals, 0.99), 10.0)  # idx=9 -> max
        self.assertEqual(percentile(vals, 0.0), 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            percentile([], 0.5)


class TestSummarizeRecords(unittest.TestCase):
    def test_summary_keys_and_consistency(self):
        records = sample_records(n=10, base_ttft=100.0)
        s = summarize_records(records)
        self.assertEqual(s["n_requests"], 10)
        self.assertEqual(s["ttft_min_ms"], round(min(r["ttft_ms"] for r in records), 3))
        self.assertEqual(s["ttft_max_ms"], round(max(r["ttft_ms"] for r in records), 3))
        # queueing is fixed at 10 ms in the fixture
        self.assertEqual(s["queueing_mean_ms"], 10.0)
        self.assertEqual(s["service_mean_ms"], round(s["ttft_mean_ms"] - 10.0, 3))
        for key in ("ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
                    "achieved_throughput", "active_concurrency_max",
                    "active_concurrency_mean", "duration_s"):
            self.assertIn(key, s)

    def test_empty_records(self):
        self.assertEqual(summarize_records([]), {"error": "no records"})

    def test_littles_law(self):
        # All requests complete in 100 ms service with 10 ms queueing:
        # L = rate * mean_service.  duration ~= n * interval + last service.
        records = sample_records(n=10, base_ttft=100.0)
        s = summarize_records(records)
        self.assertGreater(s["active_concurrency_mean"], 0.0)


class TestCalibration(unittest.TestCase):
    def test_capacity_is_highest_tracking_rate(self):
        probes = [
            {"offered_rate": 1.0, "achieved_throughput": 1.0},
            {"offered_rate": 2.0, "achieved_throughput": 1.9},   # ratio 0.95
            {"offered_rate": 4.0, "achieved_throughput": 2.0},   # ratio 0.50
            {"offered_rate": 8.0, "achieved_throughput": 2.1},   # ratio 0.26
        ]
        self.assertEqual(calibrate_sustainable_capacity(probes), 2.0)

    def test_no_tracking_falls_back_to_lowest(self):
        probes = [
            {"offered_rate": 1.0, "achieved_throughput": 0.5},
            {"offered_rate": 2.0, "achieved_throughput": 0.6},
        ]
        self.assertEqual(calibrate_sustainable_capacity(probes), 1.0)

    def test_sweep_rates(self):
        rates = sweep_rates_from_capacity(4.0)
        self.assertEqual(len(rates), 7)
        self.assertEqual(rates[0], 1.0)   # 0.25 x
        self.assertEqual(rates[4], 4.0)   # 1.00 x
        self.assertEqual(rates[-1], 5.2)  # 1.30 x


class TestAggregateLoadSummaries(unittest.TestCase):
    def test_grouping_and_mean(self):
        summaries = [
            {"offered_rate": 1.0, "normalized_load": 0.25, "achieved_throughput": 1.0,
             "ttft_p50_ms": 10.0, "ttft_p90_ms": 12.0, "ttft_p99_ms": 15.0,
             "queueing_p50_ms": 0.1, "queueing_p90_ms": 0.2,
             "service_p50_ms": 9.0, "service_p90_ms": 11.0,
             "active_concurrency_max": 1},
            {"offered_rate": 1.0, "normalized_load": 0.25, "achieved_throughput": 1.1,
             "ttft_p50_ms": 11.0, "ttft_p90_ms": 13.0, "ttft_p99_ms": 16.0,
             "queueing_p50_ms": 0.2, "queueing_p90_ms": 0.3,
             "service_p50_ms": 10.0, "service_p90_ms": 12.0,
             "active_concurrency_max": 2},
            {"offered_rate": 2.0, "normalized_load": 0.5, "achieved_throughput": 1.9,
             "ttft_p50_ms": 20.0, "ttft_p90_ms": 25.0, "ttft_p99_ms": 30.0,
             "queueing_p50_ms": 1.0, "queueing_p90_ms": 2.0,
             "service_p50_ms": 18.0, "service_p90_ms": 22.0,
             "active_concurrency_max": 3},
        ]
        agg = aggregate_load_summaries(summaries)
        self.assertEqual(len(agg), 2)
        self.assertEqual(agg[0]["offered_rate"], 1.0)
        self.assertEqual(agg[0]["n_reps"], 2)
        self.assertEqual(agg[0]["achieved_throughput_mean"], 1.05)
        self.assertEqual(agg[0]["ttft_p50_mean"], 10.5)
        self.assertEqual(agg[0]["ttft_p50_max"], 11.0)
        self.assertEqual(agg[0]["active_concurrency_max"], 2)
        self.assertEqual(agg[1]["offered_rate"], 2.0)
        self.assertEqual(agg[1]["n_reps"], 1)

    def test_empty(self):
        self.assertEqual(aggregate_load_summaries([]), [])


if __name__ == "__main__":
    unittest.main()
