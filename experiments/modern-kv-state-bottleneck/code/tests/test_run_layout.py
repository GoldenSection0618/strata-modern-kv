"""Unit tests for the run-tagged output layout (requirement: preserve every
run).  Repeated runs must never overwrite raw files: each run gets a
``run-<tag>`` directory derived from a UTC timestamp + SLURM job id (with a
user ``RUN_TAG`` override), and analysis keeps recursive discovery.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang_hicache.io import (  # noqa: E402
    run_output_dir,
    save_metadata,
    save_raw_result,
    utc_run_tag,
)

TAG_RE = re.compile(r"^\d{8}T\d{6}Z(-job\d+)?$")


class TestRunTag(unittest.TestCase):
    def test_default_tag_format(self):
        tag = utc_run_tag()
        self.assertRegex(tag, TAG_RE)

    def test_override_wins(self):
        self.assertEqual(utc_run_tag("my-custom-tag"), "my-custom-tag")

    def test_slurm_job_id_included(self, monkeypatch_env=None):
        old = os.environ.get("SLURM_JOB_ID")
        os.environ["SLURM_JOB_ID"] = "424242"
        try:
            tag = utc_run_tag()
        finally:
            if old is None:
                os.environ.pop("SLURM_JOB_ID", None)
            else:
                os.environ["SLURM_JOB_ID"] = old
        self.assertRegex(tag, r"^\d{8}T\d{6}Z-job424242$")

    def test_tags_are_unique_per_call(self):
        a = utc_run_tag("t1")
        b = utc_run_tag("t2")
        self.assertNotEqual(a, b)


class TestRunOutputDir(unittest.TestCase):
    def test_residency_nested(self):
        out = run_output_dir("/base/results", "20260809T120000Z-job1", residency="cpu_hit")
        self.assertEqual(
            out, Path("/base/results/cpu_hit/run-20260809T120000Z-job1")
        )

    def test_plain(self):
        out = run_output_dir("/base/results", "tag123")
        self.assertEqual(out, Path("/base/results/run-tag123"))

    def test_repeated_runs_do_not_overwrite(self):
        root = Path(".tmp_test_run_layout")
        shutil.rmtree(root, ignore_errors=True)
        try:
            dir_a = run_output_dir(root, "tag-A", residency="cpu_hit")
            dir_b = run_output_dir(root, "tag-B", residency="cpu_hit")
            save_metadata({"run_tag": "tag-A"}, dir_a)
            save_raw_result({"ttft_ms": 1.0}, dir_a, 0)
            save_metadata({"run_tag": "tag-B"}, dir_b)
            save_raw_result({"ttft_ms": 2.0}, dir_b, 0)

            self.assertTrue((dir_a / "metadata.json").is_file())
            self.assertTrue((dir_a / "raw" / "rep_00.json").is_file())
            self.assertTrue((dir_b / "metadata.json").is_file())
            self.assertTrue((dir_b / "raw" / "rep_00.json").is_file())
            # Each run kept its own raw file -> nothing was overwritten.
            self.assertIn("1.0", (dir_a / "raw" / "rep_00.json").read_text(encoding="utf-8"))
            self.assertIn("2.0", (dir_b / "raw" / "rep_00.json").read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_recursive_discovery_finds_run_dirs(self):
        # Analysis uses rglob; run-tagged dirs must remain discoverable.
        root = Path(".tmp_test_run_layout_disc")
        shutil.rmtree(root, ignore_errors=True)
        try:
            out = run_output_dir(root, "xyz", residency="recompute")
            save_metadata({"a": 1}, out)
            save_raw_result({"b": 2}, out, 3)
            found = sorted(
                p.relative_to(root) for p in root.rglob("*") if p.is_file()
            )
            names = {str(p) for p in found}
            self.assertIn("recompute/run-xyz/metadata.json", names)
            self.assertIn("recompute/run-xyz/raw/rep_03.json", names)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
