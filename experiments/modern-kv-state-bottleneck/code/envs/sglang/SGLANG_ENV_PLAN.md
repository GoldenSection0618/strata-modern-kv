# SGLang / HiCache CUDA 12.9 Environment

> Status: environment and user-space JIT toolchain verified on compute nodes.
> Exp1-3 measurements must use the exact prefix and commit below.

## 1. Canonical environment

```text
prefix: /share01/hpc/humxlab_intern/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211
SGLang commit: 4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63
SGLang version: 0.5.6.post3.dev8468+g4ad990ba7
Torch: 2.11.0+cu129
sglang-kernel: 0.4.5+cu129
sgl-deep-gemm: 0.1.5.post1+cu129
cuda-nvcc: 12.9.86 (inside the prefix)
host compiler: g++ 12.4.0 (inside the prefix)
Rust extensions: disabled for the Python text/HiCache path
container: not used
```

The read-only clone at `~/yanglihan/dl-stack/projects/sglang` is a reference
checkout at `7120f3ee13de565cc737e0598110e7f7603c4e9f`. It is intentionally not
edited and is not the installed commit. The installed source is the cached
archive of `4ad990ba…`, selected immediately before upstream commit
`434e64628` migrated the CUDA PyTorch stack to Torch 2.13. The selected commit
still contains the complete public HiCache server flags, host layouts,
metrics, and `benchmark/hicache/` implementation required by Exp1-3.

## 2. Why this combination is required

- Current SGLang dependency metadata resolves to CUDA 13/Torch 2.13, which is
  not usable with the cluster's driver 525 A100 nodes.
- Replacing only the CUDA 13 wheel with Torch `2.13.0+cu129` is still invalid:
  CUDA initializes, but the first real tensor operation fails with
  `cudaErrorSymbolNotFound`.
- Torch `2.11.0+cu129` passes a real BF16 matmul on driver 525 and has matching
  cu129 SGLang kernel/DeepGEMM wheels.
- SGLang performs CUDA JIT compilation during server startup. Rocky 8's system
  GCC 8.5 lacks the C++20 `<version>` header, so the prefix must also contain
  `cuda-nvcc 12.9.86` and `g++ 12.4.0`.

The older `envs/sglang` and partial `envs/sglang-cu129` prefixes are retained
as failure evidence. Do not delete, repair in place, or select them in a
runner.

## 3. Canonical scripts and dry-run policy

All installation, dependency resolution, compilation, and GPU checks run in
complete Slurm jobs. Nothing is installed on the login node.

```bash
cd ~/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck/code/envs/sglang

# Default is read-only: validates source archive/wheels and prints the plan.
sbatch bootstrap_sglang_cu129_env.sbatch

# Only for creating/resuming the canonical prefix after reviewing dry run.
RUN=1 sbatch bootstrap_sglang_cu129_env.sbatch

# Standalone toolchain repair/verification; also dry-run by default.
sbatch install_sglang_jit_toolchain.sbatch
RUN=1 sbatch install_sglang_jit_toolchain.sbatch
```

`bootstrap_sglang_cu129_env.sbatch` never runs `conda create` over an existing
prefix. A verified incomplete prefix requires explicit `RUN=1 RESUME=1`; a
completed prefix is accepted only when Python, the pinned commit, CUDA 12.9
markers, JIT marker, nvcc, g++, and imports all match.

The standalone toolchain script installs only into the canonical prefix and
compiles both:

1. a C++20 source containing `#include <version>` with conda g++;
2. a CUDA C++20 source containing the same header with conda nvcc and that g++
   as `-ccbin`.

## 4. Required runner environment

Every Exp1-3 sbatch file pins the prefix and exports these values before the
SGLang child process starts:

```bash
export CUDA_HOME="$SGLANG_ENV_DIR"
export CC="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-g++"
export CUDACXX="$SGLANG_ENV_DIR/bin/nvcc"
export NVCC_CCBIN="$CXX"
export LIBRARY_PATH="$SGLANG_ENV_DIR/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

This prevents JIT discovery from falling back to `/usr/local/cuda/bin/nvcc`
and system GCC 8.5. Before loading model weights, an actual run also performs
a BF16 matmul on the GPU allocated by Slurm. `torch.cuda.is_available()` alone
is not an adequate health check.

## 5. Provenance files

The prefix contains:

| File | Meaning |
|---|---|
| `sglang_commit.txt` | exact installed source commit |
| `provenance.json` | SGLang/Torch/CUDA, driver/GPU/BF16, JIT compiler, source status, Rust and container choices |
| `pip_freeze.txt` | Python package snapshot |
| `conda_list.txt` | conda package/toolchain snapshot |
| `cu129_complete.txt` | core CUDA 12.9 environment passed validation |
| `jit_toolchain_complete.txt` | C++20 host and CUDA compilation passed |

Ordinary Git/version provenance is used; no file hashes are generated.

## 6. Verified job evidence

| Job | Result |
|---|---|
| `1273225` | candidate commit: Torch 2.11 metadata, HiCache args/layouts/benchmarks present, no DeepEP requirement |
| `1273282` | canonical environment: pip check, import, L40 BF16 matmul, provenance all passed |
| `1273322` | conda solver dry run for nvcc 12.9 + g++ 12 succeeded |
| `1273330` | standalone toolchain script dry run succeeded |
| `1273336` | toolchain install; C++20 and CUDA C++20 compilation succeeded |
| `1273379` | repeated toolchain validation including CUDA driver-stub link succeeded |
| `1273382-1273384` | final Exp1-3 runner dry runs with canonical env/toolchain/link path succeeded |
| `1273468` | 27 targeted residency/config evidence tests passed |
| `1273477-1273479` | final-code Exp1-3 runner dry runs passed in parallel |
| `1273483` | Exp1 recompute smoke: validation PASS, summary written |
| `1273484` | Exp2 gpu_hit smoke: validation PASS (`device_hit_delta=2048`, host=0) |
| `1273485` | Exp3 cpu_hit smoke: validation PASS (`host_hit_delta=2048`, device=0); all 7 formal load points passed dominance |
| `1273491` | full pure-Python test suite before peer-tier missing-metric tightening: 111 tests passed |
| `1273494` | final full pure-Python test suite: 113 tests passed |

Failed jobs remain useful evidence: `1273296`/`1273298` exposed an unhealthy
GPU on `smtg5002`; `1273297` completed model/cache initialization and exposed
the system GCC 8.5 JIT failure; `1273318` confirmed GPU0 `[Unknown Error]`;
`1273349-1273351` verified the new compiler could build all FlashInfer objects
and then exposed the missing Conda driver-stub link path. They are not reported
as experimental measurements.

`1273387/1273388` and `1273440/1273441` are preserved negative runs from
before the pinned Qwen3.5 hybrid evidence fallback and full write-through
eviction preparation were implemented. Slurm `COMPLETED` alone did not make
their failed validation outputs usable.

## 7. Boundaries

- No Docker, Podman, Singularity, system driver changes, `/usr/local/cuda`
  changes, sudo, or system package installation.
- `sgl-deep-ep` is not installed; it is not required by the selected standard
  Python text/HiCache path.
- `nvidia-cuda-nvdisasm 13.3.73` is an offline disassembly tool required by
  the Cutlass dependency, not a linked CUDA 13 runtime. Generic CUDA 13
  runtime/compiler Python distributions are rejected by validation.
- Formal results are valid only after the residency validation gates described
  in `06-sglang-execution-path.md` pass. A successfully started server alone
  is not a result.
