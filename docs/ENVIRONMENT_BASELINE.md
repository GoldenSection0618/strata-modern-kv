# Environment Baseline

> Last verified from the deployed project environments: 2026-08-09

This document records the **currently deployed cluster environment baseline** used by this repository. It is intentionally separate from experiment-level capability validation. A serving environment being installed and importable does not by itself prove that a specific hierarchical-cache, hybrid-state restore, page-size, I/O, or scheduler mechanism is valid for a reported experiment.

## 1. Environment layout

The serving runtimes are isolated by conda prefix under:

```text
~/yanglihan/dl-stack/envs/
├── qwen
├── gemma4
└── sglang
```

Runtime isolation is mandatory:

- `envs/qwen` serves Qwen3.5-9B through vLLM.
- `envs/gemma4` serves Gemma 4 12B-it through vLLM.
- `envs/sglang` is dedicated to SGLang / HiCache and is installed from a pinned read-only source revision.
- SGLang dependencies must not be installed into either vLLM environment.
- SGLang experiments must not silently fall back to `envs/qwen` or `envs/gemma4`.

## 2. vLLM environment versions

`qwen` and `gemma4` currently use the same software stack:

| Component | Version |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.11.0+cu129` |
| vLLM | `0.26.0+cu129` |
| Triton | `3.6.0` |
| Transformers | `5.14.1` |

The active vLLM binary is the explicit CUDA-12.9 release wheel:

```text
~/yanglihan/dl-stack/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

The previously attempted un-suffixed/default vLLM 0.26.0 CUDA-13 path is retained only as failure history. It is **not** the current execution environment.

## 3. Cluster loading convention

Environment creation, installation, import checks, and GPU validation are executed on compute nodes through complete `sbatch` jobs.

```bash
source "$HOME/yanglihan/env.sh"
module load miniforge3/24.11.2_1
eval "$(conda shell.bash hook)"
conda activate "$DL_ROOT/envs/<name>"
```

Rules:

- `env.sh` may inject an existing serving environment. Explicitly activate the intended prefix before running experiment commands.
- Do not assume a prefix environment contains `etc/profile.d/conda.sh`.
- When shell activation is unnecessary, prefer the target prefix explicitly, for example `$TARGET_ENV/bin/python` and `$TARGET_ENV/bin/python -m pip`.
- Existing prefixes are not recreated by default. Recovery scripts must validate `bin/python` and `conda-meta/history` before deciding whether creation can be skipped.

## 4. Shared paths

`env.sh` provides the project storage roots:

```bash
export DL_ROOT="$HOME/yanglihan/dl-stack"
export PIP_CACHE_DIR="$DL_ROOT/cache/pip"
export HF_HOME="$DL_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$DL_ROOT/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$DL_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$DL_ROOT/cache/torch"
```

Machine-local paths remain outside version-controlled experiment configuration unless they are recorded as environment metadata.

## 5. CUDA and driver interpretation

The inspected A100 node reports NVIDIA driver `525.60.13`; `nvidia-smi` reports `CUDA Version: 12.0`. The latter is a driver capability field, not the installed userspace toolkit version.

The current vLLM environments use PyTorch / vLLM CUDA-12.9 binaries and have been selected specifically to avoid the CUDA-13 native-extension path that failed on this cluster.

For every reported run, record the actual driver, PyTorch CUDA build, runtime build source, GPU model, and native-extension status. A nominal CUDA version alone is not sufficient metadata.

## 6. libstdc++ compatibility

The Rocky 8.6 system `libstdc++.so.6` is too old for the deployed Python/runtime stack and can trigger missing `CXXABI_1.3.15` errors.

The deployed workaround is:

1. install `libstdcxx-ng` in each relevant conda environment;
2. ensure the environment library directory is present in `LD_LIBRARY_PATH` **before Python starts**.

The project does not use a `sitecustomize.py` workaround for this problem because modifying `LD_LIBRARY_PATH` after Python startup is too late for already-resolved dynamic-linker dependencies.

## 7. FlashInfer sampler policy for vLLM

Both vLLM model environments currently disable the FlashInfer sampler:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

This avoids the observed FlashInfer sampling-kernel JIT failure involving the CUB `FlagHeads` API on the current A100 software stack. The runtime falls back to the PyTorch sampler.

Do not substitute similarly named variables. The project baseline uses exactly `VLLM_USE_FLASHINFER_SAMPLER=0`.

This is a runtime-compatibility setting, not an experimental optimization. It should remain fixed across paired vLLM comparisons unless sampler implementation itself becomes an explicit experimental variable.

## 8. Mandatory environment validation

A freshly created or repaired vLLM environment must pass all of the following on a compute node before it is accepted as the environment baseline:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch; x=torch.randn(4,4,dtype=torch.bfloat16,device='cuda'); print((x@x).sum())"
python -c "from vllm import LLM; print('OK')"
```

The native vLLM library must also be checked for the CUDA runtime it resolves:

```bash
ldd "$DL_ROOT/envs/qwen/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so" | grep cudart
```

The expected deployed vLLM path resolves CUDA 12 runtime libraries rather than `libcudart.so.13`.

## 9. SGLang / HiCache status

The SGLang / HiCache environment is already established in the dedicated `envs/sglang` prefix from a pinned source revision. Therefore repository documents must not describe "building a non-Docker CUDA-12-compatible SGLang environment" as unfinished work.

Formal SGLang experiments still require **mechanism-specific validation** on the pinned build, including as applicable:

- exact checkpoint launch;
- native-kernel loading;
- attention backend and supported page sizes;
- HiCache enablement and effective configuration;
- `direct` / `kernel` I/O paths;
- full hybrid-state save / restore coverage;
- recurrent-state eviction and restore correctness;
- resolved GPU / CPU state-group allocation;
- numerical consistency;
- scheduler semantics.

Environment installation status and mechanism capability status must remain separate fields.

## 10. Reproducibility contract

Every formal run should record at least:

- environment name: `qwen`, `gemma4`, or `sglang`;
- exact model identifier and revision;
- runtime version / source commit / build artifact;
- Python, PyTorch and runtime versions;
- driver and PyTorch CUDA build;
- relevant environment overrides, including `VLLM_USE_FLASHINFER_SAMPLER=0` for the vLLM baseline;
- precision / cache dtype;
- mechanism-specific capability status.

If the deployed environment changes, update this document and the relevant runtime status in `TECHNICAL_BASELINE.md` before treating new measurements as comparable with earlier runs.