#!/bin/bash
#SBATCH --job-name=ylh-sglang-jit-toolchain
#SBATCH --partition=i80m512l40
#SBATCH --nodelist=smtg6001
#SBATCH --account=humx_lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

set -euo pipefail

source "$HOME/yanglihan/env.sh"
module load miniforge3/24.11.2_1
eval "$(conda shell.bash hook)"

RUN="${RUN:-0}"
TARGET_ENV="${SGLANG_ENV_DIR:-$DL_ROOT/envs/sglang-hicache-cu129-torch211}"
GXX_VERSION="12"
CUDA_NVCC_VERSION="12.9"

test -x "$TARGET_ENV/bin/python"
test -f "$TARGET_ENV/conda-meta/history"
test -f "$TARGET_ENV/cu129_complete.txt"

echo "target=$TARGET_ENV"
echo "gxx_linux-64=$GXX_VERSION"
echo "cuda-nvcc=$CUDA_NVCC_VERSION"
echo "channels=nvidia/label/cuda-12.9.1,conda-forge"

if [ "$RUN" != "1" ]; then
    echo "=== DRY RUN: solver only; no packages installed ==="
    conda install --dry-run -p "$TARGET_ENV" \
      -c nvidia/label/cuda-12.9.1 -c conda-forge \
      "gxx_linux-64=$GXX_VERSION" "cuda-nvcc=$CUDA_NVCC_VERSION"
    echo "Execute with: RUN=1 sbatch $0"
    exit 0
fi

conda install -y -p "$TARGET_ENV" \
  -c nvidia/label/cuda-12.9.1 -c conda-forge \
  "gxx_linux-64=$GXX_VERSION" "cuda-nvcc=$CUDA_NVCC_VERSION"

NVCC="$TARGET_ENV/bin/nvcc"
CC="$TARGET_ENV/bin/x86_64-conda-linux-gnu-gcc"
CXX="$TARGET_ENV/bin/x86_64-conda-linux-gnu-g++"
test -x "$NVCC"
test -x "$CC"
test -x "$CXX"

BUILD_DIR="${SLURM_TMPDIR:-/tmp}/sglang-jit-toolchain-${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
printf '%s\n' '#include <version>' 'int main() { return 0; }' > "$BUILD_DIR/cxx20.cpp"
printf '%s\n' '#include <version>' '__global__ void k() {}' 'int main() { return 0; }' > "$BUILD_DIR/cuda20.cu"
"$CXX" -std=c++20 -c "$BUILD_DIR/cxx20.cpp" -o "$BUILD_DIR/cxx20.o"
"$NVCC" -ccbin "$CXX" -std=c++20 -Xcompiler -fPIC \
  -c "$BUILD_DIR/cuda20.cu" -o "$BUILD_DIR/cuda20.o"
"$CXX" "$BUILD_DIR/cuda20.o" \
  -L"$TARGET_ENV/targets/x86_64-linux/lib/stubs" -lcuda \
  -shared -o "$BUILD_DIR/cuda20.so"

NVCC_VERSION="$($NVCC --version | tail -n 1)"
GXX_FULL_VERSION="$($CXX -dumpfullversion)"
export TARGET_ENV NVCC_VERSION GXX_FULL_VERSION
"$TARGET_ENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

target = Path(os.environ["TARGET_ENV"])
path = target / "provenance.json"
payload = json.loads(path.read_text()) if path.exists() else {}
payload.update(
    {
        "jit_cuda_home": str(target),
        "jit_nvcc_version": os.environ["NVCC_VERSION"],
        "jit_host_compiler": "x86_64-conda-linux-gnu-g++",
        "jit_host_compiler_version": os.environ["GXX_FULL_VERSION"],
        "jit_cxx_standard": "c++20",
        "jit_toolchain_source": "conda user prefix",
    }
)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
conda list -p "$TARGET_ENV" > "$TARGET_ENV/conda_list.txt"
printf '%s\n' "nvcc=$NVCC_VERSION" "gxx=$GXX_FULL_VERSION" \
  > "$TARGET_ENV/jit_toolchain_complete.txt"

echo "C++20 host compile: OK"
echo "CUDA C++20 compile: OK"
echo "CUDA driver-stub link: OK"
echo "nvcc=$NVCC_VERSION"
echo "gxx=$GXX_FULL_VERSION"
