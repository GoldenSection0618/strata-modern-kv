#!/bin/bash
# ============================================================================
# Common environment for all ylh-hcv-* sbatch jobs.
# Must be sourced AFTER "$HOME/yanglihan/env.sh".
#
# Exports the prefix-local CUDA/JIT variables required by
# docs/ENVIRONMENT_REQUIREMENTS.md section 9, and resolves the canonical
# SGLang environment (sglang-hicache-cu129-torch211).  No fallback to
# envs/sglang, envs/sglang-cu129, qwen or gemma4 is ever allowed.
# ============================================================================

set -euo pipefail

DL_ROOT="${DL_ROOT:-$HOME/yanglihan/dl-stack}"
export DL_ROOT

SGLANG_ENV_DIR="$DL_ROOT/envs/sglang-hicache-cu129-torch211"
export SGLANG_ENV_DIR
export PYTHON_BIN="$SGLANG_ENV_DIR/bin/python"
export SGLANG_COMMIT="4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63"
export SGLANG_VERSION="0.5.6.post3.dev8468+g4ad990ba7"

# Prefix-local user-space CUDA/JIT toolchain (ENVIRONMENT_REQUIREMENTS.md 9).
export CUDA_HOME="$SGLANG_ENV_DIR"
export CC="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-g++"
export CUDACXX="$SGLANG_ENV_DIR/bin/nvcc"
export NVCC_CCBIN="$CXX"
export LIBRARY_PATH="$SGLANG_ENV_DIR/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
# libstdcxx-ng from the prefix must precede system libstdc++ (CXXABI_1.3.15).
export LD_LIBRARY_PATH="$SGLANG_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

# Experiment code directory: explicit override from the submit script wins;
# otherwise derive from this file's location (works in any worktree).
CODE_DIR="${HCV_CODE_DIR:-$DL_ROOT/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}"
export CODE_DIR
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Results root (raw runs live under the worktree; .gitignore excludes them).
export RESULTS_ROOT="${RESULTS_ROOT:-$CODE_DIR/../results}"
export LOG_DIR="${LOG_DIR:-$HOME/logs}"

# Config file per experiment (user override wins).
export CONFIG_PATH="${CONFIG_PATH:-$CODE_DIR/configs}"
