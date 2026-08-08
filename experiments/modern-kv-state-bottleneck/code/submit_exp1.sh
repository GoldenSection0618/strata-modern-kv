#!/bin/bash
# Submit all Experiment 1 jobs for a model.
#
# Usage:
#   bash submit_exp1.sh qwen
#   bash submit_exp1.sh gemma

set -euo pipefail

MODEL="${1:-qwen}"
EXP_ROOT="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"

CONTEXTS=(4096 8192 16384 32768)
MODES=(recompute gpu_hit cpu_hit)

# CPU-resident hit needs kv_offloading enabled
# Start with a moderate value; adjust based on validation results
KV_OFFLOAD_CPU=5

submitted=0
for ctx in "${CONTEXTS[@]}"; do
    for mode in "${MODES[@]}"; do
        export MODEL CTX="$ctx" MODE="$mode"

        if [ "$mode" = "cpu_hit" ]; then
            export KV_OFFLOAD="$KV_OFFLOAD_CPU"
        else
            export KV_OFFLOAD=0
        fi

        job_name="ylh-exp1-${MODEL}-${ctx}-${mode}"
        echo "Submitting: $job_name"

        sbatch -J "$job_name" "$EXP_ROOT/code/run_exp1.sbatch"
        submitted=$((submitted + 1))
    done
done

echo "Submitted $submitted jobs for model=$MODEL"
