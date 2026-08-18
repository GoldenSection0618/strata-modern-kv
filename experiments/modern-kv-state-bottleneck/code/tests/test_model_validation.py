"""Unit tests for model-checkpoint validation (check_model_arch).

``check_model_arch`` must compare the resolved /model_info model path
against the expected checkpoint under canonical-path resolution only, and
a nonempty wrong model must fail.  These tests exercise the pure
comparison (:func:`model_path_matches`) and the gate's behavior with a
fake client.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang_hicache.validation import (  # noqa: E402
    CHECK_MODEL_ARCH,
    check_model_arch,
    model_path_matches,
    resolve_model_path,
)


class TestModelPathMatches(unittest.TestCase):
    def test_exact_match_passes(self):
        ok, reason = model_path_matches("/models/qwen", "/models/qwen")
        self.assertTrue(ok)
        self.assertIn("matches", reason)

    def test_canonical_path_resolution_passes(self):
        # Trailing slash and dot components resolve to the same realpath.
        ok, _ = model_path_matches("/models/qwen/", "/models/./qwen")
        self.assertTrue(ok)

    def test_nonempty_wrong_model_fails(self):
        ok, reason = model_path_matches("/models/qwen", "/models/gemma")
        self.assertFalse(ok)
        self.assertIn("does not match", reason)

    def test_expected_empty_fails(self):
        ok, reason = model_path_matches("/models/qwen", "")
        self.assertFalse(ok)
        self.assertIn("no expected model path", reason)

    def test_reported_empty_fails(self):
        ok, reason = model_path_matches("", "/models/qwen")
        self.assertFalse(ok)
        self.assertIn("empty model_path", reason)

    def test_symlink_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real_model"
            real.mkdir()
            link = Path(tmp) / "link_model"
            link.symlink_to(real, target_is_directory=True)
            ok, _ = model_path_matches(str(link), str(real))
            self.assertTrue(ok)
            self.assertEqual(resolve_model_path(str(link)), str(real.resolve()))

    def test_hf_id_does_not_match_filesystem_path(self):
        # An HF-style identifier must NOT silently pass against a pinned
        # absolute checkpoint path.
        ok, _ = model_path_matches("Qwen/Qwen3.5-9B", "/abs/models/Qwen3.5-9B")
        self.assertFalse(ok)


class _FakeModelInfoClient:
    """Fake client whose /model_info returns a fixed payload."""

    def __init__(self, model_path=None, error=None):
        self.model_path = model_path
        self.error = error

    def model_info(self):
        if self.error is not None:
            raise self.error
        info = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
        if self.model_path is not None:
            info["model_path"] = self.model_path
        return info


class TestCheckModelArch(unittest.TestCase):
    def test_matching_model_passes(self):
        name, passed, detail = check_model_arch(
            _FakeModelInfoClient(model_path="/models/qwen"), "/models/qwen"
        )
        self.assertEqual(name, CHECK_MODEL_ARCH)
        self.assertTrue(passed)

    def test_wrong_nonempty_model_fails(self):
        name, passed, detail = check_model_arch(
            _FakeModelInfoClient(model_path="/models/gemma"), "/models/qwen"
        )
        self.assertEqual(name, CHECK_MODEL_ARCH)
        self.assertFalse(passed)
        self.assertIn("does not match", detail)
        self.assertIn("gemma", detail)

    def test_missing_model_path_fails(self):
        name, passed, detail = check_model_arch(
            _FakeModelInfoClient(model_path=None), "/models/qwen"
        )
        self.assertFalse(passed)
        self.assertIn("missing model_path", detail)

    def test_model_info_error_fails(self):
        name, passed, _ = check_model_arch(
            _FakeModelInfoClient(error=RuntimeError("boom")), "/models/qwen"
        )
        self.assertFalse(passed)

    def test_canonical_difference_passes(self):
        name, passed, _ = check_model_arch(
            _FakeModelInfoClient(model_path="/models/../models/qwen"), "/models/qwen"
        )
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
