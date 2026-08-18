#!/bin/bash
#SBATCH -J ylh-sglang-exp1-RUN
#SBATCH -p i56m512A100
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# ============================================================================
# Experiment 1 (SGLang path): Context Length Scaling
# ============================================================================
# Usage (full sweep):
#   sbatch --export=MODEL=qwen,CTX=4096,MODE=recompute run_exp1_sglang.sbatch
#   bash submit_exp1_sglang.sh qwen
#
# Smoke (small settings, separate from the full sweep):
#   SMOKE=1 bash submit_exp1_sglang.sh qwen smoke
#
# Overrides: HICACHE_* flags, PAGE_SIZE, MEM_FRACTION, READY_TIMEOUT,
# SGLANG_ENV_DIR, RUN_TAG.
#
# NOTE on partition/GPU: the PARTITION/GPU environment variables are honored
# ONLY by the submit scripts (submit_exp1_sglang.sh passes -p/--gres to
# sbatch).  The #SBATCH directives below are fixed; they do NOT read
# PARTITION/GPU from the environment.  To change partition or GPU count for a
# direct `sbatch run_exp1_sglang.sbatch` call, edit the #SBATCH -p/--gres
# lines or use the submit script.
# ============================================================================

set -euo pipefail
source "$HOME/yanglihan/env.sh"

DL_ROOT="$HOME/yanglihan/dl-stack"
EXP_ROOT="$DL_ROOT/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
CODE_DIR="$EXP_ROOT/code"
LOG_DIR="/share01/hpc/humxlab_intern/yanglihan/logs"
RESULTS_ROOT="$EXP_ROOT/results/sglang/exp1"

# --- pinned SGLang commit (provenance) ---
export SGLANG_COMMIT="${SGLANG_COMMIT:-4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63}"

# --- configuration from environment ---
MODEL="${MODEL:-qwen}"
CTX="${CTX:-4096}"
MODE="${MODE:-recompute}"
N_WARMUP="${N_WARMUP:-3}"
N_REPEATS="${N_REPEATS:-10}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PAGE_SIZE="${PAGE_SIZE:-64}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
HICACHE_RATIO="${HICACHE_RATIO:-2.0}"
HICACHE_SIZE="${HICACHE_SIZE:-0}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
# Unique run tag: user RUN_TAG override wins; otherwise UTC timestamp +
# SLURM job id (a fresh tag per job -> repeated runs never overwrite).
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"

# --- model path resolution ---
case "$MODEL" in
  qwen|Qwen)
    MODEL_PATH="$DL_ROOT/cache/huggingface/models--Qwen--Qwen3.5-9B"
    MODEL_ID="Qwen/Qwen3.5-9B"
    ;;
  gemma|Gemma)
    MODEL_PATH="$DL_ROOT/cache/huggingface/models--google--gemma-4-12B-it"
    MODEL_ID="google/gemma-4-12B"
    ;;
  *)
    echo "ERROR: Unknown MODEL=$MODEL (use 'qwen' or 'gemma')" >&2
    exit 1
    ;;
esac

# --- SGLang environment resolution (dedicated env REQUIRED) --------------
SGLANG_ENV_DIR="${SGLANG_ENV_DIR:-$DL_ROOT/envs/sglang-hicache-cu129-torch211}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$SGLANG_ENV_DIR/bin/python" ]; then
        PYTHON_BIN="$SGLANG_ENV_DIR/bin/python"
    else
        echo "ERROR: dedicated SGLang environment not found at $SGLANG_ENV_DIR" >&2
        echo "  A missing dedicated SGLang env is a HARD FAILURE; the qwen env" >&2
        echo "  python must not be used for SGLang runs (wrong dependency tree)." >&2
        echo "  See $CODE_DIR/envs/sglang/SGLANG_ENV_PLAN.md for the bootstrap plan." >&2
        exit 1
    fi
fi
export PATH="$SGLANG_ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$SGLANG_ENV_DIR/lib"

# --- user-space CUDA JIT toolchain (system GCC 8.5 lacks C++20 <version>) ---
export CUDA_HOME="$SGLANG_ENV_DIR"
export CC="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$SGLANG_ENV_DIR/bin/x86_64-conda-linux-gnu-g++"
export CUDACXX="$SGLANG_ENV_DIR/bin/nvcc"
export NVCC_CCBIN="$CXX"
export LIBRARY_PATH="$SGLANG_ENV_DIR/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
for tool in "$CC" "$CXX" "$CUDACXX"; do
    if [ ! -x "$tool" ]; then
        echo "ERROR: required user-space JIT tool missing: $tool" >&2
        echo "  Run code/envs/sglang/install_sglang_jit_toolchain.sbatch first." >&2
        exit 1
    fi
done
if [ ! -f "$SGLANG_ENV_DIR/jit_toolchain_complete.txt" ]; then
    echo "ERROR: JIT toolchain completion marker is missing." >&2
    exit 1
fi

echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sglang; print('sglang', sglang.__version__)" || {
    echo "ERROR: sglang is not importable in $PYTHON_BIN" >&2
    echo "  See $CODE_DIR/envs/sglang/SGLANG_ENV_PLAN.md" >&2
    exit 1
}
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model directory missing: $MODEL_PATH" >&2
    exit 1
fi
if [ ! -f "$SGLANG_ENV_DIR/cu129_complete.txt" ] \
    || [ ! -f "$SGLANG_ENV_DIR/sglang_commit.txt" ] \
    || [ "$(cat "$SGLANG_ENV_DIR/sglang_commit.txt")" != "$SGLANG_COMMIT" ]; then
    echo "ERROR: SGLang environment marker/commit mismatch: $SGLANG_ENV_DIR" >&2
    exit 1
fi
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN OK: model, environment, import, commit, JIT toolchain, and log path verified; server not started."
    exit 0
fi

"$PYTHON_BIN" - <<'PY'
import torch

assert torch.cuda.is_available(), "allocated GPU is unavailable to PyTorch"
name = torch.cuda.get_device_name(0)
x = torch.randn((32, 32), device="cuda", dtype=torch.bfloat16)
value = (x @ x).float().sum().item()
print(f"GPU preflight OK: {name}; BF16 matmul sum={value}")
PY

# --- smoke vs full ---
SMOKE_ARG=""
if [ "$SMOKE" = "1" ]; then
    N_WARMUP=1
    N_REPEATS=2
    SMOKE_ARG="--smoke"
fi

# --- output (run-tagged; every job writes its own run-<tag> dir) ---
OUTPUT_DIR="$RESULTS_ROOT/${MODEL}/${CTX}"

echo "================================================"
echo "Experiment 1 (SGLang): Context Length Scaling"
echo "Model:      $MODEL_ID"
echo "Context:    $CTX tokens"
echo "Mode:       $MODE"
echo "Smoke:      $SMOKE"
echo "SGLang:     commit=$SGLANG_COMMIT"
echo "HiCache:    ratio=$HICACHE_RATIO size=$HICACHE_SIZE "
echo "            io=$HICACHE_IO_BACKEND layout=$HICACHE_MEM_LAYOUT "
echo "            write=$HICACHE_WRITE_POLICY page=$PAGE_SIZE"
echo "Output:     $OUTPUT_DIR (run tag: $RUN_TAG)"
echo "Log:        $LOG_DIR"
echo "SLURM Job:  $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================"

cd "$CODE_DIR"
"$PYTHON_BIN" sglang_hicache/run_exp1.py \
    --model "$MODEL" \
    --model-path "$MODEL_PATH" \
    --context-length "$CTX" \
    --residency "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --run-tag "$RUN_TAG" \
    --n-warmup "$N_WARMUP" \
    --n-repeats "$N_REPEATS" \
    --mem-fraction-static "$MEM_FRACTION" \
    --page-size "$PAGE_SIZE" \
    --tp-size "$TP_SIZE" \
    --hicache-ratio "$HICACHE_RATIO" \
    --hicache-size "$HICACHE_SIZE" \
    --hicache-io-backend "$HICACHE_IO_BACKEND" \
    --hicache-mem-layout "$HICACHE_MEM_LAYOUT" \
    --hicache-write-policy "$HICACHE_WRITE_POLICY" \
    --ready-timeout "$READY_TIMEOUT" \
    $SMOKE_ARG

echo ""
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
