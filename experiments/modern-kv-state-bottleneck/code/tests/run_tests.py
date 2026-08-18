#!/usr/bin/env python3
"""Tiny unittest runner for the SGLang-path pure-Python tests.

Usage (from the code/ directory):

    python3 tests/run_tests.py            # or
    bash tests/run_tests.sh

No pytest, CUDA, SGLang, network, or model weights required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

TEST_MODULES = [
    "test_metrics_parser",
    "test_output_schema",
    "test_summaries",
    "test_validation",
    "test_load_driver",
    "test_entry_points",
    "test_model_validation",
    "test_prefix_pool",
    "test_run_layout",
]


def main() -> int:
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for mod in TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(mod))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
