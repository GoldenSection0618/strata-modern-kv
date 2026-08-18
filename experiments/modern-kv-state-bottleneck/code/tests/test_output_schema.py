"""Unit tests for output-schema behavior of the SGLang run entry points.

Verifies that raw reps, summaries, metadata, and validation outputs
carry the required keys (runtime, residency mode, exact workload
params, and the pinned HiCache provenance: hicache_io_backend,
hicache_mem_layout, page size, write policy, host cache size/ratio,
SGLang commit) and keep raw/summary/validation/metadata separation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang_hicache.config import SGLangServerConfig  # noqa: E402
from sglang_hicache.io import (  # noqa: E402
    save_metadata,
    save_raw_result,
    save_summary,
    save_unsupported,
    save_validation,
)
from sglang_hicache.summary import summarize_ttft  # noqa: E402


def _engine_dict(**overrides) -> dict:
    cfg = SGLangServerConfig(
        model_path="/models/m",
        model_id="M",
        residency_mode="cpu_hit",
        port=0,
        sglang_commit="7120f3ee13de565cc737e0598110e7f7603c4e9f",
    )
    return cfg.engine_config_dict() | overrides


class TestServerConfig(unittest.TestCase):
    def test_residency_argv_mapping(self):
        cfg = SGLangServerConfig(
            model_path="/m", model_id="M", residency_mode="recompute", port=8000
        )
        argv = cfg.build_argv()
        self.assertIn("--disable-radix-cache", argv)
        self.assertNotIn("--enable-hierarchical-cache", argv)
        self.assertIn("--enable-metrics", argv)
        self.assertIn("--port", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "8000")

    def test_cpu_hit_argv_mapping(self):
        cfg = SGLangServerConfig(
            model_path="/m",
            model_id="M",
            residency_mode="cpu_hit",
            port=8000,
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
            hicache_write_policy="write_through",
            hicache_ratio=2.0,
            page_size=64,
        )
        argv = cfg.build_argv()
        for flag in (
            "--enable-hierarchical-cache",
            "--hicache-ratio", "2.0",
            "--hicache-io-backend", "kernel",
            "--hicache-mem-layout", "page_first",
            "--hicache-write-policy", "write_through",
            "--page-size", "64",
        ):
            self.assertIn(flag, argv)
        self.assertNotIn("--disable-radix-cache", argv)

    def test_gpu_hit_argv_mapping(self):
        cfg = SGLangServerConfig(
            model_path="/m", model_id="M", residency_mode="gpu_hit", port=8000
        )
        argv = cfg.build_argv()
        self.assertNotIn("--disable-radix-cache", argv)
        self.assertNotIn("--enable-hierarchical-cache", argv)

    def test_hicache_size_overrides_ratio(self):
        cfg = SGLangServerConfig(
            model_path="/m",
            model_id="M",
            residency_mode="cpu_hit",
            port=8000,
            hicache_size_gb=20,
            hicache_ratio=2.0,
        )
        argv = cfg.build_argv()
        self.assertIn("--hicache-size", argv)
        self.assertEqual(argv[argv.index("--hicache-size") + 1], "20")
        self.assertNotIn("--hicache-ratio", argv)


class TestMetadataSchema(unittest.TestCase):
    def test_required_provenance_keys_present(self):
        d = _engine_dict()
        for key in (
            "runtime",
            "residency_mode",
            "hicache_io_backend",
            "hicache_mem_layout",
            "hicache_write_policy",
            "hicache_ratio",
            "hicache_size_gb",
            "page_size",
            "sglang_commit",
            "enable_hierarchical_cache",
            "enable_prefix_caching",
            "mem_fraction_static",
        ):
            self.assertIn(key, d, f"missing key {key}")
        self.assertEqual(d["runtime"], "sglang")
        self.assertEqual(
            d["sglang_commit"], "7120f3ee13de565cc737e0598110e7f7603c4e9f"
        )
        # Page size is explicitly pinned (never left to runtime default).
        self.assertEqual(d["page_size"], 64)


class TestOutputSeparation(unittest.TestCase):
    def test_files_written_to_expected_names(self, tmp=".tmp_test_schema"):
        import shutil

        root = Path(tmp)
        shutil.rmtree(root, ignore_errors=True)
        try:
            save_metadata({"a": 1}, root)
            save_validation({"all_passed": True}, root)
            save_summary({"median_ttft_ms": 1.0}, root)
            save_raw_result({"ttft_ms": 1.0}, root, 0)
            save_unsupported({"status": "unsupported"}, root)
            names = {p.name for p in root.rglob("*") if p.is_file()}
            self.assertIn("metadata.json", names)
            self.assertIn("validation.json", names)
            self.assertIn("summary.json", names)
            self.assertIn("rep_00.json", names)
            self.assertIn("unsupported.json", names)
            # raw lives under raw/
            self.assertTrue((root / "raw" / "rep_00.json").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestSummarySchema(unittest.TestCase):
    def test_ttft_summary_keys(self):
        s = summarize_ttft([1.0, 2.0, 3.0], n_repeats=3)
        for key in ("median_ttft_ms", "p90_ttft_ms", "min_ttft_ms", "max_ttft_ms",
                    "n_repeats"):
            self.assertIn(key, s)
        self.assertEqual(s["median_ttft_ms"], 2.0)
        self.assertEqual(s["min_ttft_ms"], 1.0)
        self.assertEqual(s["max_ttft_ms"], 3.0)
        self.assertEqual(s["n_repeats"], 3)

    def test_empty_ttft_summary_uses_none(self):
        s = summarize_ttft([], n_repeats=10)
        self.assertIsNone(s["median_ttft_ms"])
        self.assertIsNone(s["p90_ttft_ms"])
        self.assertEqual(s["n_repeats"], 10)


if __name__ == "__main__":
    unittest.main()
