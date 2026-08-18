#!/bin/bash
#SBATCH -J ylh-sglang-exp3-RUN
#SBATCH -p i56m512A100
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# ============================================================================
# Experiment 3 (SGLang path): Request-Rate Scaling / Concurrency Pressure
# ============================================================================
# Primary (cpu_hit, full calibration + 7-point sweep):
#   sbatch --export=MODEL=qwen,MODE=cpu_hit run_exp3_sglang.sbatch
#   bash submit_exp3_sglang.sh qwen primary
#
# Controls (recompute/gpu_hit, representative points, reuse primary
# calibration rates via FROZEN_RATES):
#   FROZEN_RATES=1.2,2.4,3.4,4.1,4.9,5.6,6.3 bash submit_exp3_sglang.sh qwen control
#
# Smoke (small settings, separate from the full sweep):
#   SMOKE=1 bash submit_exp3_sglang.sh qwen smoke
#
# Overrides: HICACHE_* flags, PAGE_SIZE, MEM_FRACTION, CONCURRENCY,
# PREFIX_POOL_SIZE, HIT_DOMINANCE_THRESHOLD, READY_TIMEOUT, SGLANG_ENV_DIR,
# RUN_TAG.
#
# NOTE on partition/GPU: the PARTITION/GPU environment variables are honored
# ONLY by the submit scripts (submit_exp3_sglang.sh passes -p/--gres to
# sbatch).  The #SBATCH directives below are fixed; they do NOT read
# PARTITION/GPU from the environment.  To change partition or GPU count for a
# direct `sbatch run_exp3_sglang.sbatch` call, edit the #SBATCH -p/--gres
# lines or use the submit script.
# ============================================================================

set -euo pipefail
source "$HOME/yanglihan/env.sh"

DL_ROOT="$HOME/yanglihan/dl-stack"
EXP_ROOT="$DL_ROOT/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
CODE_DIR="$EXP_ROOT/code"
LOG_DIR="/share01/hpc/humxlab_intern/yanglihan/logs"
RESULTS_ROOT="$EXP_ROOT/results/sglang/exp3"

# --- pinned SGLang commit (provenance) ---
export SGLANG_COMMIT="${SGLANG_COMMIT:-4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63}"

# --- configuration from environment ---
MODEL="${MODEL:-qwen}"
MODE="${MODE:-cpu_hit}"
CTX="${CTX:-32768}"
RATIO="${RATIO:-0.5}"
N_WARMUP="${N_WARMUP:-5}"
N_MEASURED="${N_MEASURED:-30}"
N_REPEATS="${N_REPEATS:-3}"
CONCURRENCY="${CONCURRENCY:-64}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PAGE_SIZE="${PAGE_SIZE:-64}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
HICACHE_RATIO="${HICACHE_RATIO:-3.0}"
HICACHE_SIZE="${HICACHE_SIZE:-0}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
MAX_RUNNING_REQS="${MAX_RUNNING_REQS:-}"
# Prefix-pool / dominance parameters (pinned in metadata; see run_exp3.py).
PREFIX_POOL_SIZE="${PREFIX_POOL_SIZE:-12}"
HIT_DOMINANCE_THRESHOLD="${HIT_DOMINANCE_THRESHOLD:-0.8}"
# Unique run tag: user RUN_TAG override wins; otherwise UTC timestamp +
# SLURM job id (a fresh tag per job -> repeated runs never overwrite).
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"

# --- calibration control ---
CAL_RATES="${CAL_RATES:-0.5,1.0,2.0,4.0,6.0,8.0,12.0}"
FROZEN_RATES="${FROZEN_RATES:-}"
CONTROL="${CONTROL:-0}"

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
    N_MEASURED=5
    N_REPEATS=1
    CONCURRENCY=4
    SMOKE_ARG="--smoke"
fi

# --- control / frozen rates ---
CONTROL_ARG=""
if [ "$CONTROL" = "1" ]; then
    CONTROL_ARG="--control-mode"
fi
FROZEN_ARG=""
if [ -n "$FROZEN_RATES" ]; then
    FROZEN_ARG="--skip-calibration --frozen-rates $FROZEN_RATES"
fi
MAX_RUNNING_ARG=""
if [ -n "$MAX_RUNNING_REQS" ]; then
    MAX_RUNNING_ARG="--max-running-requests $MAX_RUNNING_REQS"
fi

# --- output (run-tagged; exp4-synthesis-compatible directory name) ---
CTX_K=$((CTX / 1024))
PCT_LABEL=$(awk -v r="$RATIO" 'BEGIN { printf "%d", r * 100 }')
OUTPUT_DIR="$RESULTS_ROOT/${MODEL}/${CTX_K}k-${PCT_LABEL}pct-${MODE}"

echo "================================================"
echo "Experiment 3 (SGLang): Request-Rate Scaling"
echo "Model:      $MODEL_ID"
echo "Context:    $CTX tokens (fixed), prefix $RATIO ($PCT_LABEL%)"
echo "Mode:       $MODE"
echo "Control:    $CONTROL"
echo "Frozen:     ${FROZEN_RATES:-none (calibrate)}"
echo "Smoke:      $SMOKE"
echo "SGLang:     commit=$SGLANG_COMMIT"
echo "HiCache:    ratio=$HICACHE_RATIO size=$HICACHE_SIZE "
echo "            io=$HICACHE_IO_BACKEND layout=$HICACHE_MEM_LAYOUT "
echo "            write=$HICACHE_WRITE_POLICY page=$PAGE_SIZE"
echo "Load:       ceiling=$CONCURRENCY measured=$N_MEASURED reps=$N_REPEATS"
echo "Prefix pool: $PREFIX_POOL_SIZE families; dominance threshold $HIT_DOMINANCE_THRESHOLD"
echo "Output:     $OUTPUT_DIR (run tag: $RUN_TAG)"
echo "Log:        $LOG_DIR"
echo "SLURM Job:  $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================"

cd "$CODE_DIR"
"$PYTHON_BIN" sglang_hicache/run_exp3.py \
    --model "$MODEL" \
    --model-path "$MODEL_PATH" \
    --residency "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --run-tag "$RUN_TAG" \
    --context-length "$CTX" \
    --prefix-ratio "$RATIO" \
    --prefix-pool-size "$PREFIX_POOL_SIZE" \
    --hit-dominance-threshold "$HIT_DOMINANCE_THRESHOLD" \
    --n-warmup "$N_WARMUP" \
    --n-measured "$N_MEASURED" \
    --n-repeats "$N_REPEATS" \
    --concurrency-ceiling "$CONCURRENCY" \
    --mem-fraction-static "$MEM_FRACTION" \
    --page-size "$PAGE_SIZE" \
    --tp-size "$TP_SIZE" \
    --hicache-ratio "$HICACHE_RATIO" \
    --hicache-size "$HICACHE_SIZE" \
    --hicache-io-backend "$HICACHE_IO_BACKEND" \
    --hicache-mem-layout "$HICACHE_MEM_LAYOUT" \
    --hicache-write-policy "$HICACHE_WRITE_POLICY" \
    --calibration-rates "$CAL_RATES" \
    --ready-timeout "$READY_TIMEOUT" \
    $MAX_RUNNING_ARG \
    $CONTROL_ARG \
    $FROZEN_ARG \
    $SMOKE_ARG

echo ""
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
