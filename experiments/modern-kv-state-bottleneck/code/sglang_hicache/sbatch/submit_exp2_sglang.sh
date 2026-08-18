#!/bin/bash
# Submit SGLang Experiment 2 jobs for a model.
#
# Usage:
#   bash submit_exp2_sglang.sh qwen           # full sweep (5 ratios x 3 modes)
#   bash submit_exp2_sglang.sh qwen smoke     # smoke settings (2 reps)
#   PARTITION=i80m512l40 bash submit_exp2_sglang.sh gemma   # L40 partition
#
# Env overrides: PARTITION, GPU, CTX, HICACHE_*, PAGE_SIZE, MEM_FRACTION.

set -euo pipefail

MODEL="${1:-qwen}"
PHASE="${2:-full}"  # full | smoke

EXP_ROOT="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
SBATCH_FILE="$EXP_ROOT/code/sglang_hicache/sbatch/run_exp2_sglang.sbatch"
PARTITION="${PARTITION:-i56m512A100}"
GPU="${GPU:-1}"
CTX="${CTX:-32768}"

RATIOS=(0.0 0.25 0.5 0.75 0.875)
MODES=(recompute gpu_hit cpu_hit)

if [ "$PHASE" = "smoke" ]; then
    RATIOS=(0.5)
    MODES=(recompute)
    export SMOKE=1
fi

submitted=0
for ratio in "${RATIOS[@]}"; do
    for mode in "${MODES[@]}"; do
        # 0% prefix ratio: no shared prefix -> recompute only (per design).
        if awk -v r="$ratio" 'BEGIN { exit !(r == 0.0 && "'"$mode"'" != "recompute") }'; then
            echo "Skipping $ratio $mode (0% ratio only runs recompute)"
            continue
        fi
        pct_label=$(awk -v r="$ratio" 'BEGIN { printf "%d", r * 100 }')
        job_name="ylh-sglang-exp2-${MODEL}-${CTX}-${pct_label}pct-${mode}"
        echo "Submitting: $job_name (partition=$PARTITION, gpu=$GPU)"
        sbatch \
            -p "$PARTITION" \
            --gres="gpu:${GPU}" \
            --export=ALL,MODEL="${MODEL}",CTX="${CTX}",RATIO="${ratio}",MODE="${mode}",SMOKE="${SMOKE:-0}" \
            -J "$job_name" \
            "$SBATCH_FILE"
        submitted=$((submitted + 1))
    done
done

echo "Submitted $submitted jobs for model=$MODEL phase=$PHASE"
