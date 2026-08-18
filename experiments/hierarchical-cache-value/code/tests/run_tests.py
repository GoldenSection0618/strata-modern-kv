#!/usr/bin/env python3
"""Pure-Python unit test runner (no pytest dependency).

Usage (compute node, inside Slurm)::

    cd <code dir>
    python tests/run_tests.py

Discovers every ``test_*`` function in every ``tests/test_*.py`` module,
runs it, and reports pass/fail counts.  Exits 0 only when all tests
pass.  Tests never start servers, never touch GPUs, and never import
torch or sglang.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback


def main() -> int:
    code_dir = str(pathlib.Path(__file__).resolve().parent.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    test_dir = pathlib.Path(__file__).resolve().parent
    modules = sorted(p.stem for p in test_dir.glob("test_*.py"))
    total = 0
    failed = 0
    failures: list[str] = []
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        funcs = sorted(
            name for name in dir(mod)
            if name.startswith("test_") and callable(getattr(mod, name))
        )
        for func_name in funcs:
            total += 1
            try:
                getattr(mod, func_name)()
                print(f"PASS {mod_name}.{func_name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                failures.append(f"{mod_name}.{func_name}: {e}")
                print(f"FAIL {mod_name}.{func_name}: {e}")
                traceback.print_exc()

    print(f"\n{total - failed}/{total} tests passed")
    if failed:
        print("failures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
