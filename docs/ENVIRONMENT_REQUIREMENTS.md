# Environment Requirements

> Verified against the project environment configuration on 2026-08-10.

This document defines the environment requirements used by this repository. It records the expected runtime layout, package versions, compatibility settings, and validation procedure. Experiment-level mechanism validation remains separate: satisfying these environment requirements does not by itself prove that a hierarchical-cache, hybrid-state restore, page-size, I/O, or scheduler mechanism is valid.

## 1. Environment isolation

Serving runtimes use separate conda prefixes under:

```text
~/yanglihan/dl-stack/envs/
├── qwen
├── gemma4
└── sglang-hicache-cu129-torch211
```

Required separation:

- `envs/qwen` is the vLLM environment for `Qwen/Qwen3.5-9B`.
- `envs/gemma4` is the vLLM environment for `google/gemma-4-12B-it`.
- `envs/sglang-hicache-cu129-torch211` is the canonical SGLang / HiCache environment for the current KV/state and hierarchical-cache experiments.
- SGLang dependencies must not be installed into either vLLM environment.
- SGLang experiments must not fall back to `envs/qwen` or `envs/gemma4`.
- The older `envs/sglang` and partial `envs/sglang-cu129` prefixes are retained as failure evidence and must not be selected, repaired in place, or deleted by experiment runners.

## 2. vLLM software stack

`qwen` and `gemma4` use the same required package versions:

| Component | Required version |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.11.0+cu129` |
| vLLM | `0.26.0+cu129` |
| Triton | `3.6.0` |
| Transformers | `5.14.1` |

The vLLM artifact must be the explicit CUDA-12.9 wheel:

```text
~/yanglihan/dl-stack/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

Do not replace it with the un-suffixed/default vLLM 0.26.0 CUDA-13 wheel. Adding a CUDA-12.9 PyTorch index does not change the CUDA target of the vLLM native wheel itself.

The wheel filename must retain its Python/platform tags. Renaming it to a generic `vllm-0.26.0+cu129.whl` is invalid for pip/uv wheel-tag validation.

## 3. Compute-node conda loading

Environment creation, package installation, import checks, and GPU validation must run on compute nodes through complete `sbatch` jobs.

Use:

```bash
source "$HOME/yanglihan/env.sh"
module load miniforge3/24.11.2_1
eval "$(conda shell.bash hook)"
conda activate "$DL_ROOT/envs/<name>"
```

Rules:

- `env.sh` may inject another serving environment. Explicitly activate the intended target prefix before running its commands.
- Do not use `source "$TARGET_ENV/etc/profile.d/conda.sh"`; ordinary prefix environments are not required to provide that file.
- When shell activation is unnecessary, prefer `$TARGET_ENV/bin/python -m pip` and `$TARGET_ENV/bin/python` over bare `python` / `pip`.
- Existing prefixes must not be recreated by default. Recovery logic must first validate `bin/python` and `conda-meta/history`, then skip creation when the prefix is usable.

## 4. Shared environment variables

The project uses:

```bash
export DL_ROOT="$HOME/yanglihan/dl-stack"
export PIP_CACHE_DIR="$DL_ROOT/cache/pip"
export HF_HOME="$DL_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$DL_ROOT/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$DL_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$DL_ROOT/cache/torch"
```

Machine-local paths and credentials must not be embedded into portable experiment configuration except as explicit runtime metadata.

## 5. CUDA and driver constraints

The A100 cluster baseline uses NVIDIA driver `525.60.13`. `nvidia-smi` reports `CUDA Version: 12.0`; this field is a driver capability indicator, not the installed user-space toolkit version.

The required vLLM stack uses PyTorch and vLLM CUDA-12.9 binaries. The vLLM native library must resolve CUDA 12 runtime libraries, not `libcudart.so.13`.

For every reported run, record:

- GPU model and form factor;
- NVIDIA driver;
- PyTorch version and `torch.version.cuda`;
- serving-runtime version / commit / build source;
- resolved native CUDA dependencies when relevant.

Do not infer runtime compatibility from `nvidia-smi`'s nominal CUDA field alone.

## 6. libstdc++ compatibility

Rocky 8.6 system `libstdc++.so.6` is insufficient for the required Python/runtime stack when `CXXABI_1.3.15` is needed.

Required handling:

1. install `libstdcxx-ng` in each relevant conda environment;
2. place the environment library directory in `LD_LIBRARY_PATH` before Python starts.

A `sitecustomize.py` change to `LD_LIBRARY_PATH` is not an acceptable substitute because it runs after process startup and therefore cannot reliably change already-resolved dynamic-linker dependencies.

## 7. vLLM FlashInfer sampler setting

Both vLLM environments must use:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

This avoids the observed FlashInfer sampling-kernel JIT failure involving the CUB `FlagHeads` API on the project A100 software stack and falls back to the PyTorch sampler.

The variable name is exact. Do not use `VLLM_FLASHINFER_SAMPLING` or `VLLM_FLASHINFER_SAMPLER`.

Treat this as a fixed compatibility setting across paired vLLM experiments unless sampler implementation itself is the explicit independent variable.

## 8. vLLM environment validation

A vLLM prefix must pass all of the following on a compute node before experiment execution:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch; x=torch.randn(4,4,dtype=torch.bfloat16,device='cuda'); print((x@x).sum())"
python -c "from vllm import LLM; print('OK')"
```

Expected package/runtime identity for `qwen` and `gemma4` includes:

```text
PyTorch: 2.11.0+cu129
CUDA build: 12.9
GPU: NVIDIA A100-SXM4-40GB on the A100 baseline node
```

Check the vLLM native dependency explicitly:

```bash
ldd "$DL_ROOT/envs/qwen/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so" | grep cudart
```

The resolved dependency must be CUDA 12 rather than `libcudart.so.13`.

## 9. SGLang / HiCache requirements

SGLang / HiCache experiments must use:

```text
prefix: /share01/hpc/humxlab_intern/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211
SGLang: 0.5.6.post3.dev8468+g4ad990ba7
installed commit: 4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63
PyTorch: 2.11.0+cu129
sglang-kernel: 0.4.5+cu129
sgl-deep-gemm: 0.1.5.post1+cu129
cuda-nvcc: 12.9.86
host compiler: g++ 12.4.0
```

The read-only clone at `~/yanglihan/dl-stack/projects/sglang` remains a reference checkout at `7120f3ee13de565cc737e0598110e7f7603c4e9f`; it is not the installed source revision. The installed build is pinned immediately before the upstream CUDA-13 / PyTorch-2.13 migration. Do not mix SGLang and vLLM dependencies.

Every SGLang sbatch file must set the canonical prefix and its user-space JIT toolchain before starting the server:

```bash
export SGLANG_ENV_DIR="$DL_ROOT/envs/sglang-hicache-cu129-torch211"
export PYTHON_BIN="$SGLANG_ENV_DIR/bin/python"
export SGLANG_COMMIT="4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63"
export CUDA_HOME="$SGLANG_ENV_DIR"
export CC="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-g++"
export CUDACXX="$SGLANG_ENV_DIR/bin/nvcc"
export NVCC_CCBIN="$CXX"
export LIBRARY_PATH="$SGLANG_ENV_DIR/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

The prefix must provide matching `sglang_commit.txt`, `provenance.json`, `pip_freeze.txt`, `conda_list.txt`, `cu129_complete.txt`, and `jit_toolchain_complete.txt` evidence. A real BF16 operation and the C++20/CUDA JIT checks run on a Slurm compute node; login-node import or compilation is not an acceptable substitute.

For formal experiments, the pinned SGLang build must additionally pass the mechanism-specific checks required by the relevant experiment, including as applicable:

- exact checkpoint launch;
- native-kernel loading;
- attention backend and supported page sizes;
- HiCache enablement and effective configuration;
- `direct` / `kernel` I/O paths;
- full hybrid-state save / restore coverage;
- recurrent-state eviction / restore correctness;
- resolved GPU / CPU allocation by state group;
- numerical consistency;
- scheduler semantics.

Environment conformance and mechanism capability are separate validation layers.

## 10. Reproducibility metadata

Every formal run must record at least:

- environment identifier: `qwen`, `gemma4`, or `sglang-hicache-cu129-torch211`;
- exact model identifier and revision;
- runtime version / source commit / build artifact;
- Python, PyTorch, Triton, Transformers, and serving-runtime versions when applicable;
- driver and PyTorch CUDA build;
- relevant environment overrides, including `VLLM_USE_FLASHINFER_SAMPLER=0` for vLLM;
- precision / cache dtype;
- mechanism-specific capability status.

If these requirements change, update this document together with any affected assumptions in `TECHNICAL_BASELINE.md` before comparing measurements across environment revisions.
