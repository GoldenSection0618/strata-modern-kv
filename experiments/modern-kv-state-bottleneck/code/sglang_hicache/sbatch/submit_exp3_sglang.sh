#!/bin/bash
# Submit SGLang Experiment 3 jobs for a model.
#
# Usage:
#   bash submit_exp3_sglang.sh qwen            # primary cpu_hit (full calibration + sweep)
#   bash submit_exp3_sglang.sh qwen primary    # cpu_hit only
#   bash submit_exp3_sglang.sh qwen control    # recompute + gpu_hit controls
#   bash submit_exp3_sglang.sh qwen smoke      # smoke settings (1 rep, 5 reqs)
#
# To make control conditions reuse the primary calibration, export
# FROZEN_RATES before submitting controls, e.g.:
#   FROZEN_RATES=1.2,2.4,3.4,4.1,4.9,5.6,6.3 bash submit_exp3_sglang.sh qwen control
#
# Env overrides: PARTITION, GPU, CTX, RATIO, HICACHE_*, PAGE_SIZE, MEM_FRACTION.

set -euo pipefail

MODEL="${1:-qwen}"
PHASE="${2:-all}"   # all | primary | control | smoke

EXP_ROOT="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
SBATCH_FILE="$EXP_ROOT/code/sglang_hicache/sbatch/run_exp3_sglang.sbatch"
PARTITION="${PARTITION:-i56m512A100}"
GPU="${GPU:-1}"
CTX="${CTX:-32768}"
RATIO="${RATIO:-0.5}"
FROZEN_RATES="${FROZEN_RATES:-}"

# --- Primary: cpu_hit (full calibration + 7-point sweep) ---
if [ "$PHASE" = "all" ] || [ "$PHASE" = "primary" ]; then
    jobname="ylh-sglang-exp3-${MODEL}-cpu_hit"
    echo "Submitting primary: $jobname"
    sbatch \
        -p "$PARTITION" \
        --gres="gpu:${GPU}" \
        --export=ALL,MODEL="${MODEL}",MODE=cpu_hit,CTX="${CTX}",RATIO="${RATIO}" \
        -J "$jobname" \
        "$SBATCH_FILE"
    echo "Submitted: $jobname"
fi

# --- Control: recompute + gpu_hit (representative points only) ---
if [ "$PHASE" = "all" ] || [ "$PHASE" = "control" ]; then
    for ctrl_mode in recompute gpu_hit; do
        jobname="ylh-sglang-exp3-${MODEL}-${ctrl_mode}"
        echo "Submitting control: $jobname"
        if [ -n "$FROZEN_RATES" ]; then
            sbatch \
                -p "$PARTITION" \
                --gres="gpu:${GPU}" \
                --export=ALL,MODEL="${MODEL}",MODE="${ctrl_mode}",CTX="${CTX}",RATIO="${RATIO}",CONTROL=1,FROZEN_RATES="${FROZEN_RATES}" \
                -J "$jobname" \
                "$SBATCH_FILE"
            echo "  (using frozen rates: $FROZEN_RATES)"
        else
            sbatch \
                -p "$PARTITION" \
                --gres="gpu:${GPU}" \
                --export=ALL,MODEL="${MODEL}",MODE="${ctrl_mode}",CTX="${CTX}",RATIO="${RATIO}",CONTROL=1 \
                -J "$jobname" \
                "$SBATCH_FILE"
            echo "  (NOTE: no FROZEN_RATES - control runs its own calibration)"
        fi
        echo "Submitted: $jobname"
    done
fi

# --- Smoke: minimal settings on the primary mode ---
if [ "$PHASE" = "smoke" ]; then
    jobname="ylh-sglang-exp3-${MODEL}-smoke"
    echo "Submitting smoke: $jobname"
    sbatch \
        -p "$PARTITION" \
        --gres="gpu:${GPU}" \
        --export=ALL,MODEL="${MODEL}",MODE=cpu_hit,CTX="${CTX}",RATIO="${RATIO}",SMOKE=1 \
        -J "$jobname" \
        "$SBATCH_FILE"
    echo "Submitted: $jobname"
fi

echo ""
echo "Experiment 3 (SGLang) submission complete (model=$MODEL phase=$PHASE)."
echo "  Primary (cpu_hit): full calibration + 7-point sweep"
echo "  Controls (recompute, gpu_hit): representative points only"
