"""Unit tests for the Exp3 prefix pool: construction, deterministic
scheduling, tier classification, aggregation, and dominance decisions.

All pure-Python: a fake tokenizer provides the corpus, so no CUDA,
SGLang, network, or model weights are required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang_hicache.prefix_pool import (  # noqa: E402
    HIT_DOMINANCE_THRESHOLD,
    PrefixPool,
    PrefixFamily,
    aggregate_tier_hits,
    aggregate_tier_metric_delta,
    build_prefix_families,
    classify_tier,
    dominance_decision,
    schedule_families,
)


def _fake_tokenizer(text: str) -> list[int]:
    """Deterministic word-level tokenizer (pure)."""
    return [hash(w) & 0xFFFFFFFF for w in text.split()]


#: Corpus long enough to host several 128-token families.
_CORPUS = " ".join(f"token{i}" for i in range(20000))


class TestBuildPrefixFamilies(unittest.TestCase):
    def test_deterministic(self):
        a = build_prefix_families(_fake_tokenizer, 64, 64, 8, seed=7)
        b = build_prefix_families(_fake_tokenizer, 64, 64, 8, seed=7)
        self.assertEqual(
            [f.prefix_ids for f in a], [f.prefix_ids for f in b]
        )

    def test_distinct_prefixes(self):
        families = build_prefix_families(_fake_tokenizer, 64, 64, 8, seed=7)
        seen = {tuple(f.prefix_ids) for f in families}
        self.assertEqual(len(seen), 8)
        for f in families:
            self.assertEqual(len(f.prefix_ids), 64)
            self.assertEqual(len(f.suffix_pool), 1)
            self.assertEqual(len(f.suffix_pool[0]), 64)

    def test_seed_changes_placement(self):
        a = build_prefix_families(_fake_tokenizer, 64, 64, 8, seed=1)
        b = build_prefix_families(_fake_tokenizer, 64, 64, 8, seed=2)
        self.assertNotEqual(
            [f.prefix_ids for f in a], [f.prefix_ids for f in b]
        )

    def test_high_occupancy_layout_is_guaranteed(self):
        # Regression for job 1274953: random rejection placement failed for
        # 12 x 32K families even though the corpus had sufficient capacity.
        family_tokens = 32768
        corpus_tokens = 417921

        def tokenizer(_text: str) -> list[int]:
            return list(range(corpus_tokens))

        families = build_prefix_families(
            tokenizer,
            prefix_len=family_tokens // 2,
            suffix_len=family_tokens // 2,
            n_families=12,
            seed=42,
        )
        self.assertEqual(len(families), 12)
        prompts = [f.prefix_ids + f.suffix_pool[0] for f in families]
        spans = [(min(p), max(p) + 1) for p in prompts]
        for i, left in enumerate(spans):
            for right in spans[i + 1 :]:
                self.assertTrue(left[1] <= right[0] or right[1] <= left[0])

    def test_multiple_suffixes_reserve_full_family_span(self):
        families = build_prefix_families(
            _fake_tokenizer, 64, 32, 8, n_suffixes=3, seed=7
        )
        used = []
        for family in families:
            self.assertEqual(len(family.suffix_pool), 3)
            used.extend(tuple(x) for x in [family.prefix_ids, *family.suffix_pool])
        self.assertEqual(len(used), len(set(used)))

    def test_periodic_corpus_still_produces_distinct_prefix_content(self):
        # Equal 32K spacing aliases this periodic corpus. Randomized slack
        # gaps must break the alias while retaining non-overlapping spans.
        period = list(range(257))
        corpus = (period * ((417921 // len(period)) + 1))[:417921]

        families = build_prefix_families(
            lambda _text: corpus,
            prefix_len=16384,
            suffix_len=16384,
            n_families=12,
            seed=42,
        )
        prefixes = {tuple(f.prefix_ids) for f in families}
        self.assertEqual(len(prefixes), 12)

    def test_families_must_differ_in_first_cache_page(self):
        # Regression for A100 job 1292591: whole 16K prefixes differed, but
        # several shared the first radix-cache page. The first host restore
        # consequently made four later families device hits (20/80 split).
        common_head = [7] * 64
        blocks = []
        for i in range(20):
            blocks.extend(common_head + [i] * 64)

        with self.assertRaisesRegex(ValueError, "cache-page-distinct"):
            build_prefix_families(
                lambda _text: blocks,
                prefix_len=96,
                suffix_len=32,
                n_families=20,
                seed=42,
            )

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            build_prefix_families(_fake_tokenizer, 64, 64, 0)
        with self.assertRaises(ValueError):
            build_prefix_families(_fake_tokenizer, 0, 64, 4)


class TestScheduleFamilies(unittest.TestCase):
    def test_deterministic_round_robin(self):
        self.assertEqual(schedule_families(10, 4), [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        self.assertEqual(schedule_families(10, 4), schedule_families(10, 4))

    def test_covers_all_families(self):
        sched = schedule_families(100, 8)
        self.assertEqual(set(sched), set(range(8)))
        for i in range(8):
            self.assertEqual(sched.count(i), 100 // 8 + (1 if i < 100 % 8 else 0))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            schedule_families(10, 0)


class TestPrefixPool(unittest.TestCase):
    def setUp(self):
        self.families = build_prefix_families(_fake_tokenizer, 64, 64, 8, seed=7)
        self.pool = PrefixPool(self.families, seed=7)

    def test_prompt_for_cycles_families(self):
        p0 = self.pool.prompt_for(0)
        p8 = self.pool.prompt_for(8)
        self.assertEqual(p0, p8)  # same family, same suffix -> same prompt
        p1 = self.pool.prompt_for(1)
        self.assertNotEqual(p0[:64], p1[:64])  # distinct prefixes

    def test_prompts_deterministic_and_ordered(self):
        a = self.pool.prompts(20)
        b = self.pool.prompts(20)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)

    def test_metadata(self):
        md = self.pool.to_metadata()
        self.assertEqual(md["prefix_pool_size"], 8)
        self.assertEqual(md["prefix_tokens_per_family"], 64)


class TestTierClassification(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify_tier({"device": 64, "host": 0}), "device")
        self.assertEqual(classify_tier({"device": 0, "host": 64}), "host")
        self.assertEqual(classify_tier({"device": 32, "host": 32}), "mixed")
        self.assertEqual(classify_tier({"device": 0, "host": 0}), "none")
        self.assertEqual(classify_tier(None), "none")
        self.assertEqual(classify_tier({}), "none")

    def test_aggregate_counts_and_ratios(self):
        records = [
            {"cached_tokens_details": {"device": 64, "host": 0}},
            {"cached_tokens_details": {"device": 64, "host": 0}},
            {"cached_tokens_details": {"device": 0, "host": 64}},
            {"cached_tokens_details": None},
        ]
        agg = aggregate_tier_hits(records)
        self.assertEqual(agg["device"], 2)
        self.assertEqual(agg["host"], 1)
        self.assertEqual(agg["none"], 1)
        self.assertEqual(agg["total_hits"], 3)
        self.assertEqual(agg["device_ratio"], round(2 / 3, 4))
        self.assertEqual(agg["host_ratio"], round(1 / 3, 4))

    def test_mixed_hits_conservatively_reduce_pure_tier_ratios(self):
        records = [
            {"cached_tokens_details": {"device": 0, "host": 64}},
            {"cached_tokens_details": {"device": 32, "host": 32}},
            {"cached_tokens_details": {"device": 32, "host": 32}},
        ]
        agg = aggregate_tier_hits(records)
        self.assertEqual(agg["mixed"], 2)
        self.assertEqual(agg["total_hits"], 3)
        self.assertEqual(agg["host_ratio"], round(1 / 3, 4))
        self.assertEqual(agg["device_ratio"], 0.0)

    def test_aggregate_no_evidence_is_none(self):
        agg = aggregate_tier_hits([{"cached_tokens_details": None}])
        self.assertEqual(agg["total_hits"], 0)
        self.assertIsNone(agg["device_ratio"])
        self.assertIsNone(agg["host_ratio"])

    def test_window_metric_delta_ratios(self):
        from sglang_hicache.metrics import CacheStatsDelta

        delta = CacheStatsDelta(deltas={
            "prefill_device_hit_tokens": 100,
            "prefill_host_hit_tokens": 900,
            "load_back_tokens_total": 900,
        })
        agg = aggregate_tier_metric_delta(delta)
        self.assertEqual(agg["evidence_source"], "window_metric_delta_with_load_back")
        self.assertEqual(agg["total_hit_tokens"], 1000)
        self.assertEqual(agg["device_ratio"], 0.1)
        self.assertEqual(agg["host_ratio"], 0.9)
        self.assertTrue(agg["metrics_supported"])

    def test_concurrent_load_back_preserves_window_start_residency(self):
        from sglang_hicache.metrics import CacheStatsDelta

        # A100 job 1292807: all five 16K prefixes were restored H->D, but
        # four were admitted after restoration and reported as device hits.
        delta = CacheStatsDelta(deltas={
            "prefill_device_hit_tokens": 65536,
            "prefill_host_hit_tokens": 16384,
            "load_back_tokens_total": 81930,
        })
        agg = aggregate_tier_metric_delta(delta)
        self.assertEqual(agg["total_hit_tokens"], 81920)
        self.assertEqual(agg["host_tokens"], 81920)
        self.assertEqual(agg["device_tokens"], 0)
        self.assertEqual(agg["host_ratio"], 1.0)

    def test_window_metric_missing_peer_is_unsupported(self):
        from sglang_hicache.metrics import CacheStatsDelta

        delta = CacheStatsDelta(deltas={
            "prefill_device_hit_tokens": None,
            "prefill_host_hit_tokens": 900,
        })
        agg = aggregate_tier_metric_delta(delta)
        ok, label, reason = dominance_decision(agg, "cpu_hit")
        self.assertFalse(ok)
        self.assertEqual(label, "unsupported")
        self.assertIn("missing", reason)


class TestDominanceDecision(unittest.TestCase):
    def test_cpu_hit_host_dominates_passes(self):
        agg = aggregate_tier_hits(
            [{"cached_tokens_details": {"device": 0, "host": 64}}] * 9
            + [{"cached_tokens_details": {"device": 64, "host": 0}}]
        )
        ok, label, reason = dominance_decision(agg, "cpu_hit")
        self.assertTrue(ok)
        self.assertEqual(label, "cpu_hit")
        self.assertIn("host hits dominate", reason)

    def test_cpu_hit_mostly_gpu_is_unsupported(self):
        # 9 device hits, 1 host hit: requested cpu_hit must NOT pass and
        # must be labelled unsupported, never silently called cpu_hit.
        agg = aggregate_tier_hits(
            [{"cached_tokens_details": {"device": 64, "host": 0}}] * 9
            + [{"cached_tokens_details": {"device": 0, "host": 64}}]
        )
        ok, label, reason = dominance_decision(agg, "cpu_hit")
        self.assertFalse(ok)
        self.assertEqual(label, "unsupported")
        self.assertIn("must not be labelled cpu_hit", reason)

    def test_gpu_hit_device_dominates_passes(self):
        agg = aggregate_tier_hits(
            [{"cached_tokens_details": {"device": 64, "host": 0}}] * 10
        )
        ok, label, _ = dominance_decision(agg, "gpu_hit")
        self.assertTrue(ok)
        self.assertEqual(label, "gpu_hit")

    def test_gpu_hit_host_dominated_is_unsupported(self):
        agg = aggregate_tier_hits(
            [{"cached_tokens_details": {"device": 0, "host": 64}}] * 10
        )
        ok, label, _ = dominance_decision(agg, "gpu_hit")
        self.assertFalse(ok)
        self.assertEqual(label, "unsupported")

    def test_no_evidence_is_unsupported(self):
        agg = aggregate_tier_hits([{"cached_tokens_details": None}])
        ok, label, reason = dominance_decision(agg, "cpu_hit")
        self.assertFalse(ok)
        self.assertEqual(label, "unsupported")
        self.assertIn("unsupported, not zero", reason)

    def test_threshold_controls_decision(self):
        agg = aggregate_tier_hits(
            [{"cached_tokens_details": {"device": 0, "host": 64}}] * 7
            + [{"cached_tokens_details": {"device": 64, "host": 0}}] * 3
        )
        # host_ratio = 0.7 -> fails at 0.8, passes at 0.6
        ok_high, _, _ = dominance_decision(agg, "cpu_hit", threshold=0.8)
        ok_low, _, _ = dominance_decision(agg, "cpu_hit", threshold=0.6)
        self.assertFalse(ok_high)
        self.assertTrue(ok_low)

    def test_default_threshold_is_documented(self):
        self.assertEqual(HIT_DOMINANCE_THRESHOLD, 0.8)

    def test_recompute_no_dominance_requirement(self):
        ok, label, _ = dominance_decision(
            aggregate_tier_hits([]), "recompute"
        )
        self.assertTrue(ok)
        self.assertEqual(label, "recompute")

    def test_unknown_mode_unsupported(self):
        ok, label, _ = dominance_decision(
            aggregate_tier_hits(
                [{"cached_tokens_details": {"device": 1, "host": 0}}]
            ),
            "nope",
        )
        self.assertFalse(ok)
        self.assertEqual(label, "unsupported")


if __name__ == "__main__":
    unittest.main()
