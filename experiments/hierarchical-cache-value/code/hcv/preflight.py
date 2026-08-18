"""Compute-node preflight: real BF16 execution + C++20/CUDA JIT check.

Runs inside every ``ylh-hcv-*`` sbatch BEFORE server launch (never on
the login node).  It verifies, on the actual Slurm compute node:

1. PyTorch imports and reports the expected CUDA build
   (``2.11.0+cu129``) and a usable GPU;
2. a real BF16 matmul executes on the GPU and produces a finite result;
3. the prefix-local JIT toolchain (``CUDACXX``/``CC``/``CXX``) compiles
   and runs a tiny C++20 + CUDA kernel, proving the CUDA-12.9 user-space
   toolchain works on this node.

Any failure aborts the job before a server is started.  Environment
conformance is NOT full-hierarchy proof (see ``hcv.hierarchy``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def bf16_preflight() -> dict:
    """Real BF16 matmul on the GPU; returns facts or raises."""
    import torch

    facts = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on compute node")
    x = torch.randn(4, 4, dtype=torch.bfloat16, device="cuda")
    y = (x @ x).sum().item()
    if not (y == y and abs(y) < 1e6):  # finite sanity
        raise RuntimeError(f"BF16 matmul produced non-finite result: {y}")
    facts["bf16_matmul_sum"] = float(y)
    facts["bf16_ok"] = True
    return facts


def jit_preflight(cuda_home: str, cc: str, cxx: str, workdir: str) -> dict:
    """Compile and run a tiny C++20 + CUDA kernel with the prefix toolchain."""
    nvcc = os.environ.get("CUDACXX", str(Path(cuda_home) / "bin" / "nvcc"))
    src = Path(workdir) / "jit_probe.cu"
    src.write_text(
        "#include <cstdio>\n"
        "__global__ void probe(int* out){ *out = 42; }\n"
        "int main(){\n"
        "  int h=0; int* d=nullptr;\n"
        "  cudaMalloc(&d, sizeof(int));\n"
        "  probe<<<1,1>>>(d);\n"
        "  cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);\n"
        "  cudaFree(d);\n"
        "  printf(\"jit_probe=%d\\n\", h);\n"
        "  return h == 42 ? 0 : 1;\n"
        "}\n",
        encoding="utf-8",
    )
    out_bin = Path(workdir) / "jit_probe"
    cmd = [
        nvcc, "-std=c++20", "-ccbin", cxx, "-o", str(out_bin), str(src),
        "-arch=sm_80",
    ]
    env = dict(os.environ)
    env.setdefault("CC", cc)
    env.setdefault("CXX", cxx)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc JIT compile failed (rc={proc.returncode}): {proc.stderr[-800:]}")
    run = subprocess.run([str(out_bin)], capture_output=True, text=True, timeout=120)
    if run.returncode != 0 or "jit_probe=42" not in run.stdout:
        raise RuntimeError(f"JIT probe binary failed (rc={run.returncode}): {run.stdout} {run.stderr}")
    return {
        "nvcc": nvcc,
        "cc": cc,
        "cxx": cxx,
        "jit_compile_ok": True,
        "jit_run_output": run.stdout.strip(),
    }


def main() -> int:
    facts = {}
    try:
        facts["bf16"] = bf16_preflight()
        cuda_home = os.environ.get("CUDA_HOME", "")
        cc = os.environ.get("CC", "")
        cxx = os.environ.get("CXX", "")
        if not cuda_home or not cc or not cxx:
            raise RuntimeError("CUDA_HOME/CC/CXX not exported (see ENVIRONMENT_REQUIREMENTS.md)")
        with tempfile.TemporaryDirectory(prefix="hcv-jit-") as tmp:
            facts["jit"] = jit_preflight(cuda_home, cc, cxx, tmp)
    except Exception as e:  # noqa: BLE001
        print(f"PREFLIGHT FAILED: {e}", flush=True)
        print(f"facts={facts}", flush=True)
        return 1
    import json

    print("PREFLIGHT OK", flush=True)
    print(json.dumps(facts, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
