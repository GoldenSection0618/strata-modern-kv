#!/bin/bash
#SBATCH -J ylh-exp2-RUN
#SBATCH -p i56m512A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# Experiment 2: Shared-Prefix Scaling
# Usage: sbatch --export=MODEL=qwen,CTX=32768,RATIO=0.5,MODE=recompute run_exp2.sbatch

set -euo pipefail
source "$HOME/yanglihan/env.sh"

DL_ROOT="$HOME/yanglihan/dl-stack"
EXP_ROOT="$DL_ROOT/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
LOG_DIR="$DL_ROOT/projects/strata-modern-kv/logs"
CODE_DIR="$EXP_ROOT/code"

# --- Configuration from environment ---
MODEL="${MODEL:-qwen}"
CTX="${CTX:-32768}"
RATIO="${RATIO:-0.5}"
MODE="${MODE:-recompute}"
N_WARMUP="${N_WARMUP:-3}"
N_REPEATS="${N_REPEATS:-10}"

# --- Model path resolution ---
MODEL_PATH=""
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

# --- Output directory ---
PCT_LABEL=$(awk -v r="$RATIO" 'BEGIN { printf "%d", r * 100 }')
OUTPUT_DIR="$EXP_ROOT/results/exp2/${MODEL}/${CTX}-${PCT_LABEL}pct-${MODE}"

echo "=== Experiment 2: Shared-Prefix Scaling ==="
echo "Model:     $MODEL_ID ($MODEL_PATH)"
echo "Context:   $CTX tokens (fixed)"
echo "Prefix:    $RATIO ($PCT_LABEL%)"
echo "Residency: $MODE"
echo "Output:    $OUTPUT_DIR"
echo "Warmup:    $N_WARMUP, Repeats: $N_REPEATS"
echo "SLURM Job: $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "Start:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$CODE_DIR"
python3 run_exp2.py \
  --model "$MODEL" \
  --model-path "$MODEL_PATH" \
  --context-length "$CTX" \
  --prefix-ratio "$RATIO" \
  --residency "$MODE" \
  --output-dir "$OUTPUT_DIR" \
  --log-dir "$LOG_DIR" \
  --n-warmup "$N_WARMUP" \
  --n-repeats "$N_REPEATS"

echo ""
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
