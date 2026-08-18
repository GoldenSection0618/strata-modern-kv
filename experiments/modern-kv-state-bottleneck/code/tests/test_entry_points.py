"""Entry-point import tests for the SGLang Exp1-3 implementation.

Covers the two regression classes the supervisor review flagged:

1. **Package shadowing** — the local experiment package must be named
   ``sglang_hicache`` (never ``sglang``), so a child
   ``python -m sglang.launch_server`` started from ``code/`` resolves the
   *upstream* SGLang package.  The server launch module constant must stay
   exactly ``sglang.launch_server``.
2. **Missing imports** — all three entry points must import cleanly and
   their residency-preparation helpers must be bound to the real functions
   (Exp2 previously imported only ``prepare_recompute`` while dispatching
   to ``prepare_gpu_hit``/``prepare_cpu_hit``).
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sglang_hicache  # noqa: E402


class TestPackageShadowing(unittest.TestCase):
    def test_local_package_is_sglang_hicache(self):
        self.assertTrue(
            sglang_hicache.__file__.endswith(
                str(Path("sglang_hicache") / "__init__.py")
            ),
            sglang_hicache.__file__,
        )

    def test_no_local_sglang_package_on_code_dir(self):
        # code/ must not contain a package named `sglang` — otherwise a
        # child `python -m sglang.launch_server` started from code/ would
        # resolve the local package and fail to load upstream SGLang.
        code_dir = Path(sglang_hicache.__file__).resolve().parent.parent
        self.assertFalse(
            (code_dir / "sglang").is_dir(),
            "code/ must not contain a local 'sglang' package (shadowing)",
        )
        self.assertTrue((code_dir / "sglang_hicache").is_dir())

    def test_launch_module_stays_upstream(self):
        from sglang_hicache.server_lifecycle import LAUNCH_MODULE

        self.assertEqual(LAUNCH_MODULE, "sglang.launch_server")
        # The argv prefix produced by the server lifecycle must be
        # [python, "-m", "sglang.launch_server", ...].
        from sglang_hicache.server_lifecycle import SGLangServerProcess

        proc = SGLangServerProcess(argv=["--model-path", "/m"], stdout_path="/tmp/x.out",
                                   stderr_path="/tmp/x.err")
        argv = [proc.python_executable, "-m", LAUNCH_MODULE, *proc.argv]
        self.assertEqual(argv[1:3], ["-m", "sglang.launch_server"])
        self.assertNotIn("sglang_hicache", argv[1:3])


class TestEntryPointImports(unittest.TestCase):
    """All three entry points import cleanly and dispatch to real helpers."""

    MODULES = ["run_exp1", "run_exp2", "run_exp3"]

    def test_entry_points_import_cleanly(self):
        for name in self.MODULES:
            mod = importlib.import_module(f"sglang_hicache.{name}")
            self.assertTrue(hasattr(mod, "main"))
            self.assertTrue(hasattr(mod, "run_single"))

    def test_exp2_residency_helpers_are_bound(self):
        # Regression: run_exp2 previously imported only prepare_recompute
        # while _prepare_residency dispatched to prepare_gpu_hit /
        # prepare_cpu_hit -> NameError at measurement time.
        from sglang_hicache import residency
        from sglang_hicache import run_exp2

        self.assertIs(run_exp2.prepare_gpu_hit, residency.prepare_gpu_hit)
        self.assertIs(run_exp2.prepare_cpu_hit, residency.prepare_cpu_hit)
        self.assertIs(run_exp2.prepare_recompute, residency.prepare_recompute)

    def test_exp1_and_exp3_residency_helpers_are_bound(self):
        from sglang_hicache import residency
        from sglang_hicache import run_exp1
        from sglang_hicache import run_exp3

        # Exp1 dispatches to the single-prefix helpers.
        self.assertIs(run_exp1.prepare_gpu_hit, residency.prepare_gpu_hit)
        self.assertIs(run_exp1.prepare_cpu_hit, residency.prepare_cpu_hit)
        self.assertIs(run_exp1.prepare_recompute, residency.prepare_recompute)
        # Exp3 dispatches to the pool-aware variants (prefix pool semantics).
        self.assertIs(run_exp3.prepare_gpu_hit_pool, residency.prepare_gpu_hit_pool)
        self.assertIs(run_exp3.prepare_cpu_hit_pool, residency.prepare_cpu_hit_pool)
        self.assertIs(run_exp3.prepare_recompute, residency.prepare_recompute)

    def test_exp3_pool_helpers_are_bound(self):
        from sglang_hicache import prefix_pool
        from sglang_hicache import run_exp3

        self.assertIs(run_exp3.build_prefix_families, prefix_pool.build_prefix_families)
        self.assertIs(run_exp3.aggregate_tier_hits, prefix_pool.aggregate_tier_hits)
        self.assertIs(
            run_exp3.aggregate_tier_metric_delta,
            prefix_pool.aggregate_tier_metric_delta,
        )
        self.assertIs(run_exp3.dominance_decision, prefix_pool.dominance_decision)


class TestGenerateUsesNativeInputIds(unittest.TestCase):
    """Exact-token generation must use the native /generate with input_ids."""

    def test_generate_sends_exact_input_ids(self):
        import json

        from sglang_hicache.http_client import SGLangHTTPClient

        client = SGLangHTTPClient("http://127.0.0.1:1")
        captured = {}

        def fake_request(method, path, payload=None, timeout_s=None):
            captured["method"] = method
            captured["path"] = path
            captured["payload"] = payload
            body = json.dumps({
                "text": "x",
                "meta_info": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "cached_tokens_details": {"device": 0, "host": 0},
                    "output_token_logprobs": [[0.0, 7, "x"]],
                },
            }).encode("utf-8")
            return 200, body

        client._request = fake_request
        result = client.generate([101, 102, 103], max_new_tokens=1, request_id=5)
        self.assertTrue(result.ok)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/generate")
        self.assertEqual(captured["payload"]["input_ids"], [101, 102, 103])
        self.assertEqual(captured["payload"]["sampling_params"]["max_new_tokens"], 1)
        self.assertFalse(captured["payload"]["stream"])


if __name__ == "__main__":
    unittest.main()
