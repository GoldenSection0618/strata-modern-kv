"""Unit tests for the Prometheus metrics parser and typed snapshots.

Pure stdlib: parses synthetic scrape text, verifies the typed
CacheStats snapshot, the missing-metric => None rule, and before/after
delta semantics.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang_hicache.metrics import (  # noqa: E402
    CacheStatsDelta,
    diff_snapshots,
    metric_value,
    parse_prometheus_text,
    select_values,
    snapshot_from_scrape,
)

from fixtures import (  # noqa: E402
    PROM_TEXT_AFTER_CPU_HIT,
    PROM_TEXT_BASE,
    PROM_TEXT_SPARSE,
)


class TestParser(unittest.TestCase):
    def test_parses_counters_and_gauges(self):
        scrape = parse_prometheus_text(PROM_TEXT_BASE)
        self.assertEqual(
            metric_value(scrape, "sglang:max_total_num_tokens"), 471258.0
        )
        self.assertEqual(metric_value(scrape, "sglang:cache_hit_rate"), 0.33)
        # Label-filtered selection.
        self.assertEqual(
            metric_value(
                scrape, "sglang:prefill_effective_tokens_total", mode="host_hit"
            ),
            10.0,
        )
        self.assertIsNone(
            metric_value(
                scrape, "sglang:prefill_effective_tokens_total", mode="nope"
            )
        )

    def test_histogram_folding(self):
        scrape = parse_prometheus_text(PROM_TEXT_BASE)
        self.assertEqual(
            metric_value(scrape, "sglang:time_to_first_token_seconds_count"),
            None,  # bucket/count folded into histograms, not samples
        )
        hist = scrape.histograms["sglang:time_to_first_token_seconds"]
        self.assertEqual(hist.sum, 0.25)
        self.assertEqual(hist.count, 2.0)

    def test_escaped_label_values(self):
        text = 'm{label="a\\"b\\\\c\\n"} 1\n'
        scrape = parse_prometheus_text(text)
        self.assertEqual(scrape.samples["m"][0].labels["label"], 'a"b\\c\n')

    def test_special_values(self):
        text = 'm{v="x"} +Inf\nm{v="y"} -Inf\nm{v="z"} 1.5e3\n'
        scrape = parse_prometheus_text(text)
        self.assertEqual(select_values(scrape, "m", v="x"), [float("inf")])
        self.assertEqual(select_values(scrape, "m", v="y"), [float("-inf")])
        self.assertEqual(select_values(scrape, "m", v="z"), [1500.0])

    def test_comments_and_junk_lines_ignored(self):
        text = "# TYPE m counter\nm 3\n\nnot-a-sample\nm 4\n"
        scrape = parse_prometheus_text(text)
        self.assertEqual(select_values(scrape, "m"), [3.0, 4.0])


class TestSnapshot(unittest.TestCase):
    def test_typed_snapshot_fields(self):
        s = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_BASE), timestamp=1.0)
        self.assertEqual(s.prefill_input_tokens, 100.0)
        self.assertEqual(s.prefill_device_hit_tokens, 50.0)
        self.assertEqual(s.prefill_host_hit_tokens, 10.0)
        self.assertEqual(s.prefill_storage_hit_tokens, 0.0)  # present, value 0
        # load_back summed over pools
        self.assertEqual(s.load_back_tokens_total, 20.0)
        self.assertEqual(s.hicache_backup_tokens_total, 15.0)
        self.assertEqual(s.hicache_host_used_tokens, 500.0)
        self.assertEqual(s.hicache_host_total_tokens, 2000.0)
        self.assertEqual(s.cached_tokens_total, 70.0)
        self.assertEqual(s.cached_tokens_device, 60.0)
        self.assertEqual(s.cached_tokens_host, 10.0)
        self.assertEqual(s.ttft_seconds_sum, 0.25)
        self.assertEqual(s.ttft_seconds_count, 2.0)
        # kv free = available + evictable
        self.assertEqual(s.kv_free_tokens, 7000.0)

    def test_missing_metric_is_none_not_zero(self):
        s = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_SPARSE))
        self.assertEqual(s.prefill_input_tokens, 5.0)
        self.assertEqual(s.prefill_host_hit_tokens, None)
        self.assertEqual(s.load_back_tokens_total, None)
        self.assertEqual(s.hicache_host_used_tokens, None)
        self.assertEqual(s.ttft_seconds_count, None)
        self.assertIsNone(s.kv_free_tokens)  # kv_available missing
        self.assertEqual(s.max_total_num_tokens, 1000.0)

    def test_snapshot_serialization(self):
        s = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_BASE), timestamp=1.0)
        d = s.to_dict()
        self.assertEqual(d["prefill_host_hit_tokens"], 10.0)
        self.assertIn("raw", d)
        self.assertIn("families", d["raw"])


class TestDelta(unittest.TestCase):
    def test_delta_counters_and_gauges(self):
        before = snapshot_from_scrape(
            parse_prometheus_text(PROM_TEXT_BASE), timestamp=1.0
        )
        after = snapshot_from_scrape(
            parse_prometheus_text(PROM_TEXT_AFTER_CPU_HIT), timestamp=2.0
        )
        delta = diff_snapshots(after, before)
        self.assertEqual(delta.get("prefill_host_hit_tokens"), 30.0)
        self.assertEqual(delta.get("load_back_tokens_total"), 45.0)
        self.assertEqual(delta.get("hicache_host_used_tokens"), -200.0)
        self.assertEqual(delta.get("prefill_device_hit_tokens"), 16384.0)
        self.assertIsInstance(delta, CacheStatsDelta)

    def test_delta_missing_side_is_none(self):
        before = snapshot_from_scrape(
            parse_prometheus_text(PROM_TEXT_SPARSE), timestamp=1.0
        )
        after = snapshot_from_scrape(
            parse_prometheus_text(PROM_TEXT_BASE), timestamp=2.0
        )
        delta = diff_snapshots(after, before)
        # present in 'after' but missing in 'before' => unsupported
        self.assertIsNone(delta.get("load_back_tokens_total"))
        self.assertIsNone(delta.get("hicache_host_used_tokens"))
        # present on both sides => computed
        self.assertEqual(delta.get("prefill_input_tokens"), 95.0)

    def test_delta_serialization(self):
        before = snapshot_from_scrape(
            parse_prometheus_text(PROM_TEXT_BASE), timestamp=1.0
        )
        after = snapshot_from_scrape(
            parse_prometheus_text(PROM_TEXT_BASE), timestamp=2.0
        )
        d = diff_snapshots(after, before).to_dict()
        self.assertEqual(d["deltas"]["prefill_input_tokens"], 0.0)
        self.assertIn("timestamp", d)


if __name__ == "__main__":
    unittest.main()
