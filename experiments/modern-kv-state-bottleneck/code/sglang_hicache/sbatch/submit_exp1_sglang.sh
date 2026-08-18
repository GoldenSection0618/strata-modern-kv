#!/bin/bash
# Submit SGLang Experiment 1 jobs for a model.
#
# Usage:
#   bash submit_exp1_sglang.sh qwen           # full sweep (4 ctx x 3 modes)
#   bash submit_exp1_sglang.sh qwen smoke     # smoke settings (2 reps)
#   PARTITION=i80m512l40 bash submit_exp1_sglang.sh gemma   # L40 partition
#
# Env overrides: PARTITION, GPU, HICACHE_*, PAGE_SIZE, MEM_FRACTION.

set -euo pipefail

MODEL="${1:-qwen}"
PHASE="${2:-full}"  # full | smoke

EXP_ROOT="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
SBATCH_FILE="$EXP_ROOT/code/sglang_hicache/sbatch/run_exp1_sglang.sbatch"
PARTITION="${PARTITION:-i56m512A100}"
GPU="${GPU:-1}"

CONTEXTS=(4096 8192 16384 32768)
MODES=(recompute gpu_hit cpu_hit)

if [ "$PHASE" = "smoke" ]; then
    CONTEXTS=(4096)
    export SMOKE=1
fi

submitted=0
for ctx in "${CONTEXTS[@]}"; do
    for mode in "${MODES[@]}"; do
        job_name="ylh-sglang-exp1-${MODEL}-${ctx}-${mode}"
        echo "Submitting: $job_name (partition=$PARTITION, gpu=$GPU)"
        sbatch \
            -p "$PARTITION" \
            --gres="gpu:${GPU}" \
            --export=ALL,MODEL="${MODEL}",CTX="${ctx}",MODE="${mode}",SMOKE="${SMOKE:-0}" \
            -J "$job_name" \
            "$SBATCH_FILE"
        submitted=$((submitted + 1))
    done
done

echo "Submitted $submitted jobs for model=$MODEL phase=$PHASE"
