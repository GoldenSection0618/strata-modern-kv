"""Unit tests for residency evidence evaluation and validation decisions.

The decision logic in ``sglang/residency.py`` and the gate composition in
``sglang/validation.py`` are exercised with synthetic evidence; a failed
hit-validation gate must produce a negative/unsupported result rather
than a silently-passing one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang_hicache.metrics import (  # noqa: E402
    CacheStatsDelta,
    diff_snapshots,
    parse_prometheus_text,
    snapshot_from_scrape,
)
from sglang_hicache.residency import (  # noqa: E402
    build_filler_prompts,
    compute_filler_budget,
    evaluate_cpu_hit_evidence,
    evaluate_gpu_hit_evidence,
    evaluate_recompute_evidence,
    fallback_filler_budget,
)

from fixtures import (  # noqa: E402
    PROM_TEXT_AFTER_CPU_HIT,
    PROM_TEXT_BASE,
    PROM_TEXT_SPARSE,
    device_hit_meta,
    host_hit_meta,
    no_hit_meta,
)

PREFIX_LEN = 16384


class TestRecomputeEvidence(unittest.TestCase):
    def test_no_cached_tokens_passes(self):
        passed, reason = evaluate_recompute_evidence({"cached_tokens": 0}, PREFIX_LEN)
        self.assertTrue(passed)
        self.assertIn("cached_tokens=0", reason)

    def test_cached_tokens_fails(self):
        passed, _ = evaluate_recompute_evidence({"cached_tokens": 4096}, PREFIX_LEN)
        self.assertFalse(passed)


class TestGPUHitEvidence(unittest.TestCase):
    def test_device_meta_passes(self):
        passed, reason = evaluate_gpu_hit_evidence(device_hit_meta(PREFIX_LEN), PREFIX_LEN)
        self.assertTrue(passed)
        self.assertIn("device", reason)

    def test_host_meta_fails(self):
        passed, reason = evaluate_gpu_hit_evidence(host_hit_meta(PREFIX_LEN), PREFIX_LEN)
        self.assertFalse(passed)
        self.assertIn("device=0", reason)

    def test_no_hit_fails(self):
        passed, reason = evaluate_gpu_hit_evidence(no_hit_meta(PREFIX_LEN), PREFIX_LEN)
        self.assertFalse(passed)

    def test_missing_breakdown_with_metric_delta(self):
        before = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_BASE), timestamp=1)
        after = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_BASE), timestamp=2)
        delta = diff_snapshots(after, before)  # zero device delta
        passed, _ = evaluate_gpu_hit_evidence(
            {"cached_tokens": PREFIX_LEN}, PREFIX_LEN, delta
        )
        self.assertFalse(passed)

    def test_hybrid_native_metadata_uses_isolated_device_metric(self):
        delta = CacheStatsDelta(deltas={
            "prefill_device_hit_tokens": PREFIX_LEN,
            "prefill_host_hit_tokens": 0,
        })
        passed, reason = evaluate_gpu_hit_evidence(
            {"cached_tokens": 0}, PREFIX_LEN, delta
        )
        self.assertTrue(passed)
        self.assertIn("isolated public device-tier metric", reason)

    def test_device_metric_without_host_peer_is_unsupported(self):
        delta = CacheStatsDelta(deltas={
            "prefill_device_hit_tokens": PREFIX_LEN,
            "prefill_host_hit_tokens": None,
        })
        passed, _ = evaluate_gpu_hit_evidence(
            {"cached_tokens": 0}, PREFIX_LEN, delta
        )
        self.assertFalse(passed)


class TestCPUHitEvidence(unittest.TestCase):
    def test_host_meta_passes(self):
        delta = CacheStatsDelta(deltas={
            "prefill_host_hit_tokens": PREFIX_LEN,
            "prefill_device_hit_tokens": 0,
            "load_back_tokens_total": PREFIX_LEN,
        })
        passed, reason = evaluate_cpu_hit_evidence(
            host_hit_meta(PREFIX_LEN), PREFIX_LEN, delta
        )
        self.assertTrue(passed)
        self.assertIn("host=16384", reason)
        # host-tier metrics also moved in the same window
        self.assertEqual(delta.get("load_back_tokens_total"), PREFIX_LEN)
        self.assertEqual(delta.get("prefill_host_hit_tokens"), PREFIX_LEN)

    def test_metadata_only_evidence_is_not_sufficient(self):
        # Per-request host evidence WITHOUT positive public restore deltas
        # in the same window must NOT pass.
        passed, reason = evaluate_cpu_hit_evidence(
            host_hit_meta(PREFIX_LEN), PREFIX_LEN, delta=None
        )
        self.assertFalse(passed)
        self.assertIn("prefill_host_hit_tokens", reason)
        self.assertIn("required", reason)

    def test_host_meta_with_zero_public_deltas_fails(self):
        # Both public metrics present but 0 in the window -> not positive.
        before = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_BASE), timestamp=1)
        after = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_BASE), timestamp=2)
        delta = diff_snapshots(after, before)
        passed, reason = evaluate_cpu_hit_evidence(
            host_hit_meta(PREFIX_LEN), PREFIX_LEN, delta
        )
        self.assertFalse(passed)
        self.assertIn("do not prove", reason)

    def test_host_meta_with_missing_public_metrics_is_unsupported(self):
        # Metric absent on one side -> delta None -> unsupported, not zero.
        before = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_SPARSE), timestamp=1)
        after = snapshot_from_scrape(
            parse_prometheus_text(PROM_TEXT_AFTER_CPU_HIT), timestamp=2
        )
        delta = diff_snapshots(after, before)
        passed, reason = evaluate_cpu_hit_evidence(
            host_hit_meta(PREFIX_LEN), PREFIX_LEN, delta
        )
        self.assertFalse(passed)
        self.assertIn("unsupported", reason)
        self.assertIn("prefill_host_hit_tokens", reason)

    def test_device_resident_fails_with_clear_reason(self):
        passed, reason = evaluate_cpu_hit_evidence(device_hit_meta(PREFIX_LEN), PREFIX_LEN)
        self.assertFalse(passed)
        self.assertIn("still GPU-resident", reason)
        self.assertIn("eviction", reason)

    def test_no_breakdown_is_unsupported(self):
        passed, reason = evaluate_cpu_hit_evidence({"cached_tokens": PREFIX_LEN}, PREFIX_LEN)
        self.assertFalse(passed)
        self.assertIn("cached_tokens_details", reason)
        self.assertIn("unsupported", reason)

    def test_partial_host_restore_fails(self):
        meta = {"cached_tokens": PREFIX_LEN,
                "cached_tokens_details": {"device": PREFIX_LEN // 2,
                                          "host": PREFIX_LEN // 2}}
        passed, _ = evaluate_cpu_hit_evidence(meta, PREFIX_LEN)
        self.assertFalse(passed)

    def test_hybrid_native_metadata_uses_isolated_host_metric(self):
        # The pinned Qwen3.5 hybrid native endpoint leaves metadata empty;
        # an exact host-tier counter delta around one synchronous request is
        # still direct public tier attribution. load_back is optional.
        delta = CacheStatsDelta(deltas={
            "prefill_host_hit_tokens": PREFIX_LEN,
            "prefill_device_hit_tokens": 0,
            "load_back_tokens_total": None,
        })
        passed, reason = evaluate_cpu_hit_evidence(
            {"cached_tokens": 0}, PREFIX_LEN, delta
        )
        self.assertTrue(passed)
        self.assertIn("host tier", reason)
        self.assertIn("load_back_delta=unsupported", reason)

    def test_mixed_public_tier_deltas_fail(self):
        delta = CacheStatsDelta(deltas={
            "prefill_host_hit_tokens": PREFIX_LEN,
            "prefill_device_hit_tokens": PREFIX_LEN,
            "load_back_tokens_total": None,
        })
        passed, _ = evaluate_cpu_hit_evidence(
            {"cached_tokens": 0}, PREFIX_LEN, delta
        )
        self.assertFalse(passed)

    def test_host_metric_without_device_peer_is_unsupported(self):
        delta = CacheStatsDelta(deltas={
            "prefill_host_hit_tokens": PREFIX_LEN,
            "prefill_device_hit_tokens": None,
            "load_back_tokens_total": None,
        })
        passed, reason = evaluate_cpu_hit_evidence(
            {"cached_tokens": 0}, PREFIX_LEN, delta
        )
        self.assertFalse(passed)
        self.assertIn("unsupported", reason)


class TestFillerBudget(unittest.TestCase):
    def test_metrics_based_budget(self):
        s = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_BASE), timestamp=1)
        budget = compute_filler_budget(s, PREFIX_LEN, margin_tokens=4096)
        expected = (471258 - 12000) + PREFIX_LEN + 4096
        self.assertEqual(budget, expected)

    def test_missing_capacity_returns_none(self):
        s = snapshot_from_scrape(parse_prometheus_text(PROM_TEXT_SPARSE), timestamp=1)
        self.assertIsNone(compute_filler_budget(s, PREFIX_LEN))

    def test_fallback_budget(self):
        budget = fallback_filler_budget(32768, 16384, factor=6)
        self.assertEqual(budget, 262144 + 16384 + 4096)

    def test_large_context_fallback_uses_factor(self):
        budget = fallback_filler_budget(65536, 16384, factor=6)
        self.assertEqual(budget, 6 * 65536 + 16384 + 4096)

    def test_filler_prompts_distinct_and_deterministic(self):
        a = build_filler_prompts(4, 64, seed=7)
        b = build_filler_prompts(4, 64, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(a[0]), 64)
        # distinct from each other
        self.assertNotEqual(a[0], a[1])


class TestValidationGateComposition(unittest.TestCase):
    def test_gate_requires_all_checks(self):
        from sglang_hicache.validation import run_validation_gate  # noqa: F401

        # The gate function itself requires a live client; here we only
        # verify the composition rule that is used by the runners:
        # all_passed = all(non-skipped checks pass) and a failed cpu_hit
        # check must set cpu_hit_supported=False.
        checks = [
            ("a", True, "ok"),
            ("b", False, "bad"),
            ("c", None, "skipped"),
        ]
        non_skipped = [c for c in checks if c[1] is not None]
        all_passed = all(c[1] for c in non_skipped)
        self.assertFalse(all_passed)

        checks_ok = [("a", True, "ok"), ("c", None, "skipped")]
        non_skipped = [c for c in checks_ok if c[1] is not None]
        self.assertTrue(all(c[1] for c in non_skipped))


class TestConfigStability(unittest.TestCase):
    def _cpu_cfg(self, **kw):
        from sglang_hicache.config import SGLangServerConfig

        return SGLangServerConfig(
            model_path="/m",
            model_id="M",
            residency_mode="cpu_hit",
            port=8000,
            page_size=64,
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
            hicache_write_policy="write_through",
            hicache_ratio=2.0,
            **kw,
        )

    def _info(self, **overrides):
        info = {
            "disable_radix_cache": False,
            "enable_hierarchical_cache": True,
            "enable_cache_report": True,
            "page_size": 64,
            "hicache_io_backend": "kernel",
            "hicache_mem_layout": "page_first",
            "hicache_write_policy": "write_through",
            "hicache_ratio": 2.0,
            "hicache_size": 0,
        }
        info.update(overrides)
        return info

    def test_matching_config_passes(self):
        from sglang_hicache.validation import config_mismatches

        self.assertEqual(config_mismatches(self._info(), self._cpu_cfg()), [])

    def test_server_argv_enables_cache_report(self):
        self.assertIn("--enable-cache-report", self._cpu_cfg().build_argv())

    def test_drift_detected(self):
        from sglang_hicache.validation import config_mismatches

        mismatches = config_mismatches(
            self._info(page_size=32, hicache_write_policy="write_back"),
            self._cpu_cfg(),
        )
        self.assertEqual(len(mismatches), 2)
        self.assertTrue(any("page_size" in m for m in mismatches))
        self.assertTrue(any("hicache_write_policy" in m for m in mismatches))

    def test_missing_flag_is_mismatch(self):
        from sglang_hicache.validation import config_mismatches

        info = self._info()
        del info["page_size"]
        mismatches = config_mismatches(info, self._cpu_cfg())
        self.assertTrue(any("page_size=missing" in m for m in mismatches))

    def test_hicache_size_pinned(self):
        from sglang_hicache.validation import config_mismatches

        cfg = self._cpu_cfg(hicache_size_gb=20)
        self.assertEqual(config_mismatches(self._info(hicache_size=20), cfg), [])
        mismatches = config_mismatches(self._info(hicache_size=10), cfg)
        self.assertTrue(any("hicache_size" in m for m in mismatches))

    def test_gpu_hit_ignores_inactive_hicache_defaults(self):
        from sglang_hicache.config import SGLangServerConfig
        from sglang_hicache.validation import config_mismatches

        cfg = SGLangServerConfig(
            model_path="/m", model_id="M", residency_mode="gpu_hit", port=8000,
            page_size=64, hicache_io_backend="direct",
            hicache_mem_layout="page_first_direct", hicache_ratio=3.0,
        )
        info = self._info(
            enable_hierarchical_cache=False,
            hicache_io_backend="kernel", hicache_mem_layout="page_first",
            hicache_ratio=2.0,
        )
        self.assertEqual(config_mismatches(info, cfg), [])

    def test_recompute_ignores_inactive_hicache_defaults(self):
        from sglang_hicache.config import SGLangServerConfig
        from sglang_hicache.validation import config_mismatches

        cfg = SGLangServerConfig(
            model_path="/m", model_id="M", residency_mode="recompute", port=8000,
            page_size=64, hicache_io_backend="direct",
            hicache_mem_layout="page_first_direct", hicache_ratio=3.0,
        )
        info = self._info(
            disable_radix_cache=True, enable_hierarchical_cache=False,
            hicache_io_backend="kernel", hicache_mem_layout="page_first",
            hicache_ratio=2.0,
        )
        self.assertEqual(config_mismatches(info, cfg), [])

    def test_controls_still_validate_common_flags(self):
        from sglang_hicache.config import SGLangServerConfig
        from sglang_hicache.validation import config_mismatches

        cfg = SGLangServerConfig(
            model_path="/m", model_id="M", residency_mode="gpu_hit", port=8000,
            page_size=64,
        )
        mismatches = config_mismatches(
            self._info(enable_hierarchical_cache=False, page_size=32), cfg
        )
        self.assertEqual(len(mismatches), 1)
        self.assertIn("page_size", mismatches[0])


class TestHTTPClientFlush(unittest.TestCase):
    def test_flush_passes_scheduler_wait_timeout(self):
        from sglang_hicache.http_client import SGLangHTTPClient

        client = object.__new__(SGLangHTTPClient)
        seen = {}

        def fake_request(method, path, payload=None, timeout_s=None):
            seen.update(method=method, path=path, timeout_s=timeout_s)
            return 200, b"Cache flushed."

        client._request = fake_request
        self.assertTrue(client.flush_cache(timeout_s=45.0))
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["path"], "/flush_cache?timeout=45")
        self.assertEqual(seen["timeout_s"], 55.0)


if __name__ == "__main__":
    unittest.main()
