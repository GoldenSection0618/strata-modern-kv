"""Unit tests for the concurrent HTTP load driver (fake client, no network).

Verifies queueing/service/TTFT separation under load and the
concurrency-ceiling enforcement.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang_hicache.load_driver import HttpLoadDriver, LoadDriverConfig  # noqa: E402
from sglang_hicache.summary import summarize_records  # noqa: E402


class _FakeClient:
    """In-memory client with a configurable service latency."""

    def __init__(self, latency: float = 0.1):
        self.latency = latency

    def generate(self, prompt, max_new_tokens=1, request_id=0, timeout_s=None):
        time.sleep(self.latency)
        return type("R", (), {
            "ok": True,
            "error": "",
            "prompt_tokens": len(prompt),
            "completion_tokens": 1,
            "cached_tokens": 0,
            "cached_tokens_details": None,
            "output_token_id": 7,
            "text": "",
            "t_send": 0.0,
            "t_first_token": 0.0,
            "t_complete": 0.0,
        })()


def _run(rate, n, ceiling, latency=0.05, seed=1) -> dict:
    client = _FakeClient(latency=latency)
    recs = HttpLoadDriver(
        client,
        [[1, 2, 3]] * n,
        LoadDriverConfig(rate, n, ceiling, seed=seed),
    ).run()
    return summarize_records([r.to_dict() for r in recs])


class TestLoadDriver(unittest.TestCase):
    def test_low_load_no_queueing(self):
        # capacity = ceiling / service = 8 / 0.05 = 160 req/s; 5 req/s is light.
        s = _run(5.0, 20, 8, latency=0.05)
        self.assertEqual(s["n_requests"], 20)
        self.assertLess(s["queueing_p50_ms"], 5.0)
        self.assertAlmostEqual(s["service_p50_ms"], 50.0, delta=10.0)
        # TTFT ~= queueing + service
        self.assertAlmostEqual(s["ttft_p50_ms"], s["service_p50_ms"], delta=10.0)

    def test_overload_queueing_grows_and_ttft_decomposes(self):
        s = _run(400.0, 30, 4, latency=0.05)
        self.assertGreater(s["queueing_p50_ms"], 20.0)
        self.assertGreater(s["ttft_p50_ms"], s["service_p50_ms"])
        # TTFT = queueing + service (within tolerance)
        self.assertAlmostEqual(
            s["ttft_p50_ms"], s["queueing_p50_ms"] + s["service_p50_ms"], delta=15.0
        )

    def test_concurrency_ceiling_enforced(self):
        s = _run(400.0, 30, 4, latency=0.05)
        self.assertLessEqual(s["active_concurrency_max"], 4)
        self.assertLessEqual(s["active_concurrency_mean"], 4.0)

    def test_records_have_required_keys(self):
        client = _FakeClient(latency=0.01)
        recs = HttpLoadDriver(
            client, [[1, 2, 3]] * 5, LoadDriverConfig(10.0, 5, 4, seed=2)
        ).run()
        self.assertEqual(len(recs), 5)
        first = recs[0].to_dict()
        for key in (
            "request_id", "t_arrival", "t_start", "t_first_token", "t_complete",
            "queueing_ms", "ttft_ms", "service_ms", "total_ms", "ok",
        ):
            self.assertIn(key, first, f"missing key {key}")


if __name__ == "__main__":
    unittest.main()
