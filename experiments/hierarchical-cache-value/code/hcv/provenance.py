"""Environment provenance: marker checks, JIT toolchain, runtime identity.

Static checks (pure, executed by a minimal dry-run Slurm job):

* ``check_markers`` — the canonical prefix must contain the six marker
  files (``sglang_commit.txt``, ``provenance.json``, ``pip_freeze.txt``,
  ``conda_list.txt``, ``cu129_complete.txt``, ``jit_toolchain_complete.txt``)
  and ``sglang_commit.txt`` must match the pinned commit.
* ``check_jit_toolchain`` — the prefix-local gcc/g++/nvcc executables
  must exist and be executable.

Runtime checks (compute node only, via the sbatch preflight):

* ``collect_runtime_provenance`` — runs the canonical python to print
  torch/sglang/nvcc/g++ versions and records them into run metadata.
  This is never a substitute for the real BF16/JIT preflight performed
  by the sbatch before server launch.

Environment/dry-run success is NOT full-hierarchy proof: satisfying
these checks only establishes environment conformance (see
``hcv.hierarchy`` for the mechanism gate).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from hcv.config import ENV_MARKER_FILES, JIT_TOOLCHAIN_BINS


def check_markers(env_dir: str, pinned_commit: str) -> tuple[bool, list[str]]:
    """Verify the marker files exist and the commit file matches.

    Returns (ok, errors).  A missing marker or a commit mismatch is a
    hard failure; the canonical prefix must never be silently swapped
    for another environment.
    """
    errors: list[str] = []
    root = Path(env_dir)
    for marker in ENV_MARKER_FILES:
        path = root / marker
        if not path.is_file():
            errors.append(f"missing marker: {path}")
    commit_file = root / "sglang_commit.txt"
    if commit_file.is_file():
        try:
            actual = commit_file.read_text(encoding="utf-8").strip()
        except OSError as e:
            errors.append(f"cannot read {commit_file}: {e}")
            actual = ""
        if actual != pinned_commit:
            errors.append(
                f"sglang_commit.txt={actual!r} != pinned {pinned_commit!r}"
            )
    return (not errors), errors


def check_jit_toolchain(env_dir: str) -> tuple[bool, list[str]]:
    """Verify the prefix-local JIT compiler executables exist and run."""
    errors: list[str] = []
    root = Path(env_dir)
    for rel in JIT_TOOLCHAIN_BINS:
        path = root / rel
        if not path.is_file():
            errors.append(f"missing JIT toolchain binary: {path}")
        elif not os.access(path, os.X_OK):
            errors.append(f"JIT toolchain binary not executable: {path}")
    return (not errors), errors


def check_model_path(model_path: str) -> tuple[bool, list[str]]:
    """Verify the model cache directory exists and contains weight files.

    Accepts both the HuggingFace snapshot layout
    (``snapshots/<rev>/...``) and the flat layout used by this project's
    cache (files directly under the model dir).
    """
    errors: list[str] = []
    root = Path(model_path)
    if not root.is_dir():
        errors.append(f"model path not a directory: {model_path}")
        return False, errors

    def _has_weights(d: Path) -> bool:
        return any(
            p.name.endswith(".safetensors") or p.name.endswith(".bin")
            or p.name == "model.safetensors.index.json"
            for p in d.iterdir()
        )

    snapshots = root / "snapshots"
    if snapshots.is_dir():
        revisions = [p for p in snapshots.iterdir() if p.is_dir()]
        if not revisions:
            errors.append(f"model snapshots dir empty: {snapshots}")
        elif not any(_has_weights(r) for r in revisions):
            errors.append(f"no weight files under snapshots: {snapshots}")
    elif not _has_weights(root):
        errors.append(f"no weight files found under model dir: {model_path}")
    return (not errors), errors


def check_log_path(log_dir: str) -> tuple[bool, list[str]]:
    """Verify the log directory exists and is writable."""
    errors: list[str] = []
    path = Path(log_dir)
    if not path.is_dir():
        errors.append(f"log dir missing: {log_dir}")
    elif not os.access(path, os.W_OK):
        errors.append(f"log dir not writable: {log_dir}")
    return (not errors), errors


def collect_runtime_provenance(env_dir: str, python_bin: Optional[str] = None) -> dict:
    """Run the canonical python to collect runtime identity facts.

    Executes on the compute node inside the sbatch (never on the login
    node).  Returns a dict of version facts; failures are recorded as
    error strings so the caller can decide severity.
    """
    py = python_bin or str(Path(env_dir) / "bin" / "python")
    script = (
        "import json,sys;"
        "out={};"
        "try:\n"
        " import torch; out['torch']=torch.__version__; out['torch_cuda']=torch.version.cuda;"
        " out['cuda_available']=bool(torch.cuda.is_available())\n"
        "except Exception as e: out['torch_error']=str(e)\n"
        "try:\n"
        " import sglang; out['sglang']=getattr(sglang,'__version__','?')\n"
        "except Exception as e: out['sglang_error']=str(e)\n"
        "try:\n"
        " import sglang_kernel; out['sglang_kernel']=getattr(sglang_kernel,'__version__','?')\n"
        "except Exception as e: out['sglang_kernel_error']=str(e)\n"
        "print(json.dumps(out,sort_keys=True))"
    )
    try:
        proc = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": f"cannot run {py}: {e}"}
    if proc.returncode != 0:
        return {"error": f"provenance script rc={proc.returncode}: {proc.stderr[-500:]}"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        return {"error": f"cannot parse provenance output: {e}"}


def print_env_identity(env_dir: str) -> None:
    """Human-readable environment identity dump (sbatch echo)."""
    print(f"env_dir        = {env_dir}")
    print(f"python         = {env_dir}/bin/python")
    commit_file = Path(env_dir) / "sglang_commit.txt"
    if commit_file.is_file():
        print(f"sglang_commit = {commit_file.read_text().strip()}")


def main(argv=None) -> int:
    """CLI: ``python -m hcv.provenance check --env-dir ... --commit ...``.

    Exits 0 only when both marker and toolchain checks pass.  Used by
    every ``ylh-hcv-*`` sbatch before launching a server.
    """
    import argparse

    p = argparse.ArgumentParser(prog="hcv.provenance")
    sub = p.add_subparsers(dest="command", required=True)
    chk = sub.add_parser("check")
    chk.add_argument("--env-dir", required=True)
    chk.add_argument("--commit", required=True)
    ns = p.parse_args(argv)

    ok, errors = check_markers(ns.env_dir, ns.commit)
    jit_ok, jit_errors = check_jit_toolchain(ns.env_dir)
    ok = ok and jit_ok
    errors = errors + jit_errors
    print_env_identity(ns.env_dir)
    if not ok:
        for e in errors:
            print(f"PROVENANCE ERROR: {e}", file=sys.stderr)
        return 1
    print("PROVENANCE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
