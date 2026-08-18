#!/bin/bash
#SBATCH -J ylh-sglang-cu129-bootstrap
#SBATCH -p i80m512l40
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# Native conda/pip installation only. Containers are forbidden on this cluster.
# RUN=0 is a read-only plan. RUN=1 creates a new cu129 prefix.
# RESUME=1 may continue only a verified incomplete prefix and never reruns
# conda create, deletes files, or modifies the read-only SGLang source clone.

set -euo pipefail
source "$HOME/yanglihan/env.sh"
module load miniforge3/24.11.2_1
eval "$(conda shell.bash hook)"

RUN="${RUN:-0}"
RESUME="${RESUME:-0}"
SGLANG_ENV_DIR="${SGLANG_ENV_DIR:-$DL_ROOT/envs/sglang-hicache-cu129-torch211}"
SGLANG_SOURCE="${SGLANG_SOURCE:-$DL_ROOT/projects/sglang}"
SGLANG_REFERENCE_HEAD="${SGLANG_REFERENCE_HEAD:-7120f3ee13de565cc737e0598110e7f7603c4e9f}"
SGLANG_COMMIT="${SGLANG_COMMIT:-4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63}"
PYTHON_VER="${PYTHON_VER:-3.12}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DL_ROOT/cache/pip}"
WHEEL_CACHE="${WHEEL_CACHE:-$DL_ROOT/cache/wheels/sglang-cu129}"
SGLANG_BUILD_RUST_EXTS="none"
SGLANG_VERSION="0.5.6.post3.dev8468+g4ad990ba7"

TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCHAUDIO_VERSION="2.11.0"
SGLANG_KERNEL_VERSION="0.4.5"
SGL_DEEP_GEMM_VERSION="0.1.5.post1"
CU_INDEX="129"
SGLANG_KERNEL_WHEEL="$WHEEL_CACHE/sglang_kernel-${SGLANG_KERNEL_VERSION}+cu129-cp310-abi3-manylinux2014_x86_64.whl"
SGL_DEEP_GEMM_WHEEL="$WHEEL_CACHE/sgl_deep_gemm-${SGL_DEEP_GEMM_VERSION}+cu129-py3-none-manylinux2014_x86_64.whl"
TORCH_WHEEL="$WHEEL_CACHE/torch-${TORCH_VERSION}+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
TORCHVISION_WHEEL="$WHEEL_CACHE/torchvision-${TORCHVISION_VERSION}+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
TORCHAUDIO_WHEEL="$WHEEL_CACHE/torchaudio-${TORCHAUDIO_VERSION}+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
SGLANG_SOURCE_ARCHIVE="$WHEEL_CACHE/sglang-${SGLANG_COMMIT}.tar.gz"

if [ ! -d "$SGLANG_SOURCE/.git" ]; then
    echo "ERROR: pinned SGLang clone missing at $SGLANG_SOURCE" >&2
    exit 1
fi
ACTUAL_COMMIT=$(git -C "$SGLANG_SOURCE" rev-parse HEAD)
if [ "$ACTUAL_COMMIT" != "$SGLANG_REFERENCE_HEAD" ]; then
    echo "ERROR: SGLang reference clone is $ACTUAL_COMMIT, expected $SGLANG_REFERENCE_HEAD" >&2
    exit 1
fi
if ! git -C "$SGLANG_SOURCE" cat-file -e "${SGLANG_COMMIT}^{commit}"; then
    echo "ERROR: pinned install commit is absent from the reference clone: $SGLANG_COMMIT" >&2
    exit 1
fi
if [ -n "$(git -C "$SGLANG_SOURCE" status --porcelain)" ]; then
    echo "ERROR: pinned SGLang clone is not clean; refusing to build" >&2
    git -C "$SGLANG_SOURCE" status --porcelain >&2
    exit 1
fi

for wheel in \
    "$SGLANG_KERNEL_WHEEL" \
    "$SGL_DEEP_GEMM_WHEEL" \
    "$TORCH_WHEEL" \
    "$TORCHVISION_WHEEL" \
    "$TORCHAUDIO_WHEEL" \
    "$SGLANG_SOURCE_ARCHIVE"; do
    if [ ! -f "$wheel" ]; then
        echo "ERROR: required cached wheel missing: $wheel" >&2
        exit 1
    fi
done
tar -tzf "$SGLANG_SOURCE_ARCHIVE" >/dev/null

if [ "$RUN" != "1" ]; then
    echo "=== DRY RUN: no changes ==="
    echo "target=$SGLANG_ENV_DIR"
    echo "source=$SGLANG_SOURCE"
    echo "reference_head=$SGLANG_REFERENCE_HEAD"
    echo "commit=$SGLANG_COMMIT"
    echo "cuda_dependency_branch=cu129"
    echo "torch=$TORCH_VERSION+cu129"
    echo "wheel_cache=$WHEEL_CACHE"
    echo "source_archive=$SGLANG_SOURCE_ARCHIVE"
    echo "cached_wheels=verified"
    echo "generic_cuda13_cleanup=enabled"
    echo "nvdisasm_tool_exception=13.3.73"
    echo "jit_toolchain=conda cuda-nvcc 12.9 + gxx_linux-64 12"
    echo "containers=forbidden"
    echo "Execute with: RUN=1 sbatch $0"
    exit 0
fi

CREATE_ENV=1
if [ -d "$SGLANG_ENV_DIR" ]; then
    if [ -x "$SGLANG_ENV_DIR/bin/python" ] \
        && [ -f "$SGLANG_ENV_DIR/sglang_commit.txt" ] \
        && [ "$(cat "$SGLANG_ENV_DIR/sglang_commit.txt")" = "$SGLANG_COMMIT" ] \
        && [ -f "$SGLANG_ENV_DIR/cu129_complete.txt" ] \
        && [ -f "$SGLANG_ENV_DIR/jit_toolchain_complete.txt" ] \
        && [ -x "$SGLANG_ENV_DIR/bin/nvcc" ] \
        && [ -x "$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-g++" ] \
        && "$SGLANG_ENV_DIR/bin/python" -c \
          'import importlib.metadata as m, sglang, torch; forbidden={"nvidia-cuda-cccl","nvidia-cuda-crt","nvidia-cuda-nvcc","nvidia-cuda-nvrtc","nvidia-cuda-runtime","nvidia-nvvm"}; bad=[d.metadata.get("Name","").lower() for d in m.distributions() if d.metadata.get("Name","").lower() in forbidden and int(d.version.split(".",1)[0]) >= 13]; assert torch.version.cuda.startswith("12."); assert "+cu129" in torch.__version__; assert m.version("nvidia-cuda-nvdisasm") == "13.3.73"; assert not bad, bad' \
          >/dev/null 2>&1; then
        echo "Exact completed cu129 environment already exists; no changes made."
        exit 0
    fi
    if [ "$RESUME" = "1" ] \
        && [ -x "$SGLANG_ENV_DIR/bin/python" ] \
        && [ -f "$SGLANG_ENV_DIR/conda-meta/history" ]; then
        echo "RESUME=1: verified incomplete conda prefix; skipping conda create."
        CREATE_ENV=0
    else
        echo "ERROR: target exists but is not a completed cu129 environment." >&2
        echo "Use RESUME=1 only for a known interrupted bootstrap; nothing was changed." >&2
        exit 1
    fi
fi

if [ "$CREATE_ENV" = "1" ]; then
    conda create -y -p "$SGLANG_ENV_DIR" "python=$PYTHON_VER"
fi
conda activate "$SGLANG_ENV_DIR"
PYTHON="$SGLANG_ENV_DIR/bin/python"
export PATH="$SGLANG_ENV_DIR/bin:$PATH"
# Do not inherit qwen/gemma runtime libraries injected by env.sh.
export LD_LIBRARY_PATH="$SGLANG_ENV_DIR/lib"
export PIP_CACHE_DIR WHEEL_CACHE SGLANG_BUILD_RUST_EXTS
export GIT_OPTIONAL_LOCKS=0
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_VERSION"

# Materialize only the package sources from the locally cached official commit
# archive.  This avoids network access and missing-object fetches from the
# cluster's partial reference clone.
SOURCE_STAGE="${SLURM_TMPDIR:-/tmp}/sglang-cu129-source-${SLURM_JOB_ID}"
mkdir -p "$SOURCE_STAGE"
ARCHIVE_PREFIX="sglang-${SGLANG_COMMIT}"
tar -xzf "$SGLANG_SOURCE_ARCHIVE" \
    --strip-components=1 \
    -C "$SOURCE_STAGE" \
    "$ARCHIVE_PREFIX/python" \
    "$ARCHIVE_PREFIX/README.md" \
    "$ARCHIVE_PREFIX/LICENSE"
sed -i 's/cuda-python>=13\.0/cuda-python>=12,<13/' "$SOURCE_STAGE/python/pyproject.toml"
sed -i 's/flashinfer_python\[cu13\]/flashinfer_python[cu12]/' "$SOURCE_STAGE/python/pyproject.toml"
sed -i 's/nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' "$SOURCE_STAGE/python/pyproject.toml"

"$PYTHON" -m pip install --upgrade pip wheel "setuptools<82"

if ! "$PYTHON" -c \
  'import torch; assert "+cu129" in torch.__version__; assert torch.version.cuda.startswith("12.")' \
  >/dev/null 2>&1; then
    # Build a dependency-only metadata stub. The three substitutions are the
    # exact CUDA-12 branch maintained upstream. The read-only SGLang clone is
    # never edited and no container is used.
    STUB_ROOT="${SLURM_TMPDIR:-/tmp}/sglang-cu129-deps-${SLURM_JOB_ID}"
    mkdir -p "$STUB_ROOT/sglang"
    cp "$SOURCE_STAGE/python/pyproject.toml" "$STUB_ROOT/pyproject.toml"
    cp "$SOURCE_STAGE/python/setup.py" "$STUB_ROOT/setup.py"
    cp "$SOURCE_STAGE/README.md" "$STUB_ROOT/README.md"
    cp "$SOURCE_STAGE/LICENSE" "$STUB_ROOT/LICENSE"
    cp "$SOURCE_STAGE/python/sglang/version.py" "$STUB_ROOT/sglang/version.py"
    touch "$STUB_ROOT/sglang/__init__.py"
    sed -i 's/cuda-python>=13\.0/cuda-python>=12,<13/' "$STUB_ROOT/pyproject.toml"
    sed -i 's/flashinfer_python\[cu13\]/flashinfer_python[cu12]/' "$STUB_ROOT/pyproject.toml"
    sed -i 's/nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' "$STUB_ROOT/pyproject.toml"

    # Resolve the base text-serving dependency set against the official cu129
    # PyTorch index. Optional extras are intentionally excluded.
    "$PYTHON" -m pip install \
        --extra-index-url "https://download.pytorch.org/whl/cu${CU_INDEX}" \
        "$TORCH_WHEEL" \
        "$TORCHVISION_WHEEL" \
        "$TORCHAUDIO_WHEEL" \
        "$SGLANG_KERNEL_WHEEL" \
        "$SGL_DEEP_GEMM_WHEEL" \
        "$STUB_ROOT"

    # Remove any CUDA-13-only distributions left by dependency extras.
    mapfile -t CU13_PACKAGES < <("$PYTHON" - <<'PY'
from importlib import metadata
names = set()
for dist in metadata.distributions():
    name = (dist.metadata.get("Name") or "").strip()
    if name.lower().endswith("-cu13"):
        names.add(name)
for name in sorted(names):
    print(name)
PY
    )
    if [ "${#CU13_PACKAGES[@]}" -gt 0 ]; then
        "$PYTHON" -m pip uninstall -y "${CU13_PACKAGES[@]}"
    fi

else
    echo "Verified torch cu129 already present; skipping resolved base dependency reinstall."
fi

# humming-kernels' cu13 extra uses generic NVIDIA distribution names whose
# names do not end in -cu13.  They are not needed by the text HiCache path and
# must not coexist with the pinned CUDA-12 runtime.
mapfile -t GENERIC_CUDA13_PACKAGES < <("$PYTHON" - <<'PY'
from importlib import metadata

forbidden = {
    "nvidia-cuda-cccl",
    "nvidia-cuda-crt",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-nvvm",
}
for dist in metadata.distributions():
    name = (dist.metadata.get("Name") or "").lower()
    if name in forbidden and int(dist.version.split(".", 1)[0]) >= 13:
        print(dist.metadata.get("Name"))
PY
)
if [ "${#GENERIC_CUDA13_PACKAGES[@]}" -gt 0 ]; then
    "$PYTHON" -m pip uninstall -y "${GENERIC_CUDA13_PACKAGES[@]}"
fi

# Driver 525.125.06 can run the cluster-validated 2.11 cu129 stack.  The newer
# 2.13 cu129 build initializes CUDA but fails its first tensor operation with
# cudaErrorSymbolNotFound on these nodes, so pin the working ABI chain.
if ! "$PYTHON" -c \
  "import torch; assert torch.__version__ == '${TORCH_VERSION}+cu129'" \
  >/dev/null 2>&1; then
    "$PYTHON" -m pip install --force-reinstall \
        --index-url "https://download.pytorch.org/whl/cu${CU_INDEX}" \
        "$TORCH_WHEEL" \
        "$TORCHVISION_WHEEL" \
        "$TORCHAUDIO_WHEEL"
else
    echo "Verified exact cluster-compatible torch $TORCH_VERSION+cu129 already present."
fi

# Repair resolver choices that the PyTorch index can otherwise downgrade.
"$PYTHON" -m pip install --upgrade \
    "numpy>=1.25,<2.4" \
    "cuda-pathfinder>=1.5.4,<2" \
    "cuda-bindings~=12.9.7" \
    "setuptools<82" \
    helion \
    "nvidia-cuda-nvdisasm==13.3.73" \
    "torchcodec==0.11.1"

"$PYTHON" -m pip install --force-reinstall --no-deps \
    "$SGLANG_KERNEL_WHEEL" \
    "$SGL_DEEP_GEMM_WHEEL"
"$PYTHON" -m pip uninstall -y sgl-deep-ep || true

# Install the exact pinned Python source with the CUDA-12 metadata branch.
# The staged copy is job-local: the read-only checkout remains untouched and
# optional Rust services are not built.
"$PYTHON" -m pip uninstall -y sglang
"$PYTHON" -m pip install --no-deps "$SOURCE_STAGE/python"
"$PYTHON" -m pip check

# SGLang JITs CUDA extensions at server start. Rocky 8's system GCC 8.5 does
# not provide the C++20 <version> header required by the pinned source, so the
# complete environment includes a user-prefix compiler and CUDA 12.9 nvcc.
conda install -y -p "$SGLANG_ENV_DIR" \
  -c nvidia/label/cuda-12.9.1 -c conda-forge \
  "gxx_linux-64=12" "cuda-nvcc=12.9"
NVCC="$SGLANG_ENV_DIR/bin/nvcc"
CXX="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-g++"
BUILD_DIR="${SLURM_TMPDIR:-/tmp}/sglang-jit-toolchain-${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
printf '%s\n' '#include <version>' 'int main() { return 0; }' > "$BUILD_DIR/cxx20.cpp"
printf '%s\n' '#include <version>' '__global__ void k() {}' 'int main() { return 0; }' > "$BUILD_DIR/cuda20.cu"
"$CXX" -std=c++20 -c "$BUILD_DIR/cxx20.cpp" -o "$BUILD_DIR/cxx20.o"
"$NVCC" -ccbin "$CXX" -std=c++20 -Xcompiler -fPIC \
  -c "$BUILD_DIR/cuda20.cu" -o "$BUILD_DIR/cuda20.o"
"$CXX" "$BUILD_DIR/cuda20.o" \
  -L"$SGLANG_ENV_DIR/targets/x86_64-linux/lib/stubs" -lcuda \
  -shared -o "$BUILD_DIR/cuda20.so"
NVCC_VERSION="$($NVCC --version | tail -n 1)"
GXX_FULL_VERSION="$($CXX -dumpfullversion)"

export SGLANG_ENV_DIR SGLANG_SOURCE SGLANG_REFERENCE_HEAD SGLANG_COMMIT TORCH_VERSION SGLANG_VERSION NVCC_VERSION GXX_FULL_VERSION
"$PYTHON" - <<'PY'
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess

import torch
import sglang
import sgl_kernel  # noqa: F401
import flashinfer  # noqa: F401

assert sglang.__version__ == os.environ["SGLANG_VERSION"], sglang.__version__
assert "+cu129" in torch.__version__, torch.__version__
assert torch.__version__ == os.environ["TORCH_VERSION"] + "+cu129", torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12."), torch.version.cuda
assert torch.cuda.is_available(), "CUDA unavailable"
device = torch.cuda.get_device_name(0)
x = torch.randn((32, 32), device="cuda", dtype=torch.bfloat16)
value = (x @ x).float().sum().item()

bad = []
generic_cuda13_names = {
    "nvidia-cuda-cccl",
    "nvidia-cuda-crt",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-nvvm",
}
for dist in md.distributions():
    name = (dist.metadata.get("Name") or "").lower()
    version = dist.version
    if name.endswith("-cu13") or "+cu130" in version:
        bad.append(f"{name}=={version}")
    if name in generic_cuda13_names and int(version.split(".", 1)[0]) >= 13:
        bad.append(f"{name}=={version}")
assert not bad, f"CUDA-13 distributions remain: {bad}"
cuda_python = md.version("cuda-python")
assert int(cuda_python.split(".", 1)[0]) < 13, cuda_python
nvdisasm = md.version("nvidia-cuda-nvdisasm")
assert nvdisasm == "13.3.73", nvdisasm

source = os.environ["SGLANG_SOURCE"]
status = subprocess.run(
    ["git", "-C", source, "status", "--porcelain"],
    capture_output=True, text=True, check=True,
).stdout.strip() or "(clean)"
driver = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    capture_output=True, text=True, check=True,
).stdout.strip().splitlines()[0]
payload = {
    "pinned_commit": os.environ["SGLANG_COMMIT"],
    "source_clone": source,
    "source_reference_head": os.environ["SGLANG_REFERENCE_HEAD"],
    "source_git_status_porcelain": status,
    "installed_sglang_version": sglang.__version__,
    "torch_version": torch.__version__,
    "torch_version_reason": "cluster driver 525 compatibility",
    "torch_cuda": torch.version.cuda,
    "cuda_python": cuda_python,
    "nvdisasm_tool_version": nvdisasm,
    "jit_cuda_home": os.environ["SGLANG_ENV_DIR"],
    "jit_nvcc_version": os.environ["NVCC_VERSION"],
    "jit_host_compiler": "x86_64-conda-linux-gnu-g++",
    "jit_host_compiler_version": os.environ["GXX_FULL_VERSION"],
    "jit_cxx_standard": "c++20",
    "jit_toolchain_source": "conda user prefix",
    "gpu": device,
    "driver": driver,
    "bf16_matmul_sum": value,
    "cuda_dependency_branch": "cu129",
    "sglang_build_rust_exts": os.environ["SGLANG_BUILD_RUST_EXTS"],
    "container_used": False,
    "wheel_cache": os.environ["WHEEL_CACHE"],
}
env_dir = Path(os.environ["SGLANG_ENV_DIR"])
(env_dir / "provenance.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
(env_dir / "sglang_commit.txt").write_text(
    os.environ["SGLANG_COMMIT"] + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY

"$PYTHON" -m pip freeze > "$SGLANG_ENV_DIR/pip_freeze.txt"
conda list -p "$SGLANG_ENV_DIR" > "$SGLANG_ENV_DIR/conda_list.txt"
printf '%s\n' "nvcc=$NVCC_VERSION" "gxx=$GXX_FULL_VERSION" \
  > "$SGLANG_ENV_DIR/jit_toolchain_complete.txt"
printf '%s\n' "verified cu129 environment" > "$SGLANG_ENV_DIR/cu129_complete.txt"
echo "Environment ready: $SGLANG_ENV_DIR"
