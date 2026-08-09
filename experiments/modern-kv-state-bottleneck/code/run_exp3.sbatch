#!/bin/bash
#SBATCH -J ylh-exp3-RUN
#SBATCH -p i56m512A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# Experiment 3: Request-Rate Scaling
# Usage: sbatch --export=MODEL=qwen,MODE=cpu_hit run_exp3.sbatch
#        sbatch --export=MODEL=qwen,MODE=recompute,CONTROL=1,FROZEN_RATES=1.2,2.4,3.4,4.1,4.9,5.6,6.3 run_exp3.sbatch

set -euo pipefail
source "$HOME/yanglihan/env.sh"

# IMPORTANT: flashinfer sampler JIT compilation fails on this CUDA/CUB
# version (cub 300302: BlockAdjacentDifference has no member "FlagHeads").
# vLLM 0.26.0 enables VLLM_USE_FLASHINFER_SAMPLER by default, so we MUST
# disable it here.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Use spawn for the vLLM EngineCore subprocess: fork can fail with
# "Cannot re-initialize CUDA in forked subprocess" when multiple vLLM
# tasks share a node.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DL_ROOT="$HOME/yanglihan/dl-stack"
EXP_ROOT="$DL_ROOT/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
LOG_DIR="$DL_ROOT/projects/strata-modern-kv/logs"
CODE_DIR="$EXP_ROOT/code"

# --- Configuration from environment ---
MODEL="${MODEL:-qwen}"
MODE="${MODE:-cpu_hit}"
CTX="${CTX:-32768}"
RATIO="${RATIO:-0.5}"
N_WARMUP="${N_WARMUP:-5}"
N_MEASURED="${N_MEASURED:-30}"
N_REPEATS="${N_REPEATS:-3}"
CONCURRENCY="${CONCURRENCY:-64}"
KV_OFFLOAD="${KV_OFFLOAD:-0}"
GPU_UTIL="${GPU_UTIL:-0.90}"
CONTROL="${CONTROL:-0}"

# --- Calibration rates (comma-separated) ---
CAL_RATES="${CAL_RATES:-0.5,1.0,2.0,4.0,6.0,8.0,12.0}"

# --- Frozen sweep rates: reuse a previous calibration (skip calibration) ---
FROZEN_RATES="${FROZEN_RATES:-}"

# --- Model path resolution ---
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
    echo "Unknown MODEL: $MODEL" >&2
    exit 1
    ;;
esac

# --- KV offloading ---
KV_OFFLOAD_ARG=""
if [ "$KV_OFFLOAD" -gt 0 ] 2>/dev/null; then
    KV_OFFLOAD_ARG="--kv-offloading-size $KV_OFFLOAD"
fi

# --- Control mode ---
CONTROL_ARG=""
if [ "$CONTROL" = "1" ]; then
    CONTROL_ARG="--control-mode"
fi

# --- Frozen rates (skip calibration) ---
FROZEN_ARG=""
if [ -n "$FROZEN_RATES" ]; then
    FROZEN_ARG="--skip-calibration --frozen-rates $FROZEN_RATES"
fi

# --- Output directory ---
OUTPUT_DIR="$EXP_ROOT/results/exp3/${MODEL}/${CTX}-${MODE}"

echo "================================================"
echo "Experiment 3: Request-Rate Scaling"
echo "Model:     $MODEL_ID"
echo "Context:   $CTX tokens (fixed)"
echo "Prefix:    $RATIO (50%)"
echo "Residency: $MODE"
echo "Control:   $CONTROL"
echo "Frozen:    ${FROZEN_RATES:-none (calibrate)}"
echo "Output:    $OUTPUT_DIR"
echo "Warmup:    $N_WARMUP, Measured: $N_MEASURED, Repeats: $N_REPEATS"
echo "Concurrency ceiling: $CONCURRENCY"
echo "SLURM Job: $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "Start:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================"

cd "$CODE_DIR"
python3 run_exp3.py \
    --model "$MODEL" \
    --model-path "$MODEL_PATH" \
    --residency "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --context-length "$CTX" \
    --prefix-ratio "$RATIO" \
    --n-warmup "$N_WARMUP" \
    --n-measured "$N_MEASURED" \
    --n-repeats "$N_REPEATS" \
    --concurrency-ceiling "$CONCURRENCY" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --calibration-rates "$CAL_RATES" \
    $KV_OFFLOAD_ARG \
    $CONTROL_ARG \
    $FROZEN_ARG

echo ""
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
