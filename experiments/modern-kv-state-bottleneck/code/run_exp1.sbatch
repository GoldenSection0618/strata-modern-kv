#!/bin/bash
#SBATCH -J ylh-exp1-RUN
#SBATCH -p i56m512A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# Experiment 1: Context Length Scaling
# Usage: sbatch --export=MODEL=qwen,CTX=4096,MODE=recompute run_exp1.sbatch

set -euo pipefail
source "$HOME/yanglihan/env.sh"

# IMPORTANT: flashinfer sampler JIT compilation fails on this CUDA/CUB
# version (cub 300302: BlockAdjacentDifference has no member "FlagHeads").
# vLLM 0.26.0 enables VLLM_USE_FLASHINFER_SAMPLER by default, so we MUST
# disable it here. NOTE: VLLM_ATTENTION_BACKEND is NOT a recognized env var
# in vLLM 0.26.0 (it logs "Unknown vLLM environment variable") - do not use it.
export VLLM_USE_FLASHINFER_SAMPLER=0

DL_ROOT="$HOME/yanglihan/dl-stack"
EXP_ROOT="$DL_ROOT/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
LOG_DIR="$DL_ROOT/projects/strata-modern-kv/logs"

# --- Configuration from environment ---
MODEL="${MODEL:-qwen}"
CTX="${CTX:-4096}"
MODE="${MODE:-recompute}"
N_WARMUP="${N_WARMUP:-3}"
N_REPEATS="${N_REPEATS:-10}"
KV_OFFLOAD="${KV_OFFLOAD:-0}"
GPU_UTIL="${GPU_UTIL:-0.90}"

# --- Model path mapping ---
if [ "$MODEL" = "qwen" ]; then
    MODEL_ID="Qwen/Qwen3.5-9B"
    MODEL_PATH="$DL_ROOT/cache/huggingface/models--Qwen--Qwen3.5-9B"
elif [ "$MODEL" = "gemma" ]; then
    MODEL_ID="google/gemma-4-12B"
    MODEL_PATH="$DL_ROOT/cache/huggingface/models--google--gemma-4-12B-it"
else
    echo "ERROR: Unknown MODEL=$MODEL (use 'qwen' or 'gemma')"
    exit 1
fi

# --- KV offloading ---
KV_OFFLOAD_ARG=""
if [ "$KV_OFFLOAD" -gt 0 ] 2>/dev/null; then
    KV_OFFLOAD_ARG="--kv-offloading-size $KV_OFFLOAD"
fi

# --- Output ---
OUTPUT_DIR="$EXP_ROOT/results/exp1/${MODEL}/${CTX}/${MODE}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "================================================"
echo "Experiment 1: Context Length Scaling"
echo "Model:   $MODEL_ID"
echo "Context: $CTX tokens"
echo "Mode:    $MODE"
echo "Output:  $OUTPUT_DIR"
echo "Log:     $LOG_DIR"
echo "================================================"

python3 "$EXP_ROOT/code/run_exp1.py" \
    --model "$MODEL" \
    --model-path "$MODEL_PATH" \
    --context-length "$CTX" \
    --residency "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --n-warmup "$N_WARMUP" \
    --n-repeats "$N_REPEATS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    $KV_OFFLOAD_ARG

echo "Done."
