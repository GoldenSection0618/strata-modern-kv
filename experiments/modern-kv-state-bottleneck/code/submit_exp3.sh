#!/bin/bash
# Submit all Experiment 3 tasks for a given model.
#
# Usage:
#   bash submit_exp3.sh qwen           # full sweep (cpu_hit primary + controls)
#   bash submit_exp3.sh qwen primary    # cpu_hit only (full calibration + sweep)
#   bash submit_exp3.sh qwen control     # recompute + gpu_hit controls only
#
# To make control conditions reuse the primary calibration, export
# FROZEN_RATES before submitting controls, e.g.:
#   FROZEN_RATES=1.2,2.4,3.4,4.1,4.9,5.6,6.3 bash submit_exp3.sh qwen control

set -euo pipefail

MODEL="${1:-qwen}"
PHASE="${2:-all}"   # all | primary | control

CTX="${CTX:-32768}"
RATIO="${RATIO:-0.5}"
FROZEN_RATES="${FROZEN_RATES:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Primary: cpu_hit (full calibration + 7-point sweep) ---
if [ "$PHASE" = "all" ] || [ "$PHASE" = "primary" ]; then
    jobname="ylh-exp3-${MODEL}-cpu_hit"
    echo "Submitting primary: $jobname"
    sbatch \
        --export=MODEL="${MODEL}",MODE=cpu_hit,CTX="${CTX}",RATIO="${RATIO}" \
        -J "$jobname" \
        "$SCRIPT_DIR/run_exp3.sbatch"
    echo "Submitted: $jobname"
fi

# --- Control: recompute + gpu_hit (representative points only) ---
if [ "$PHASE" = "all" ] || [ "$PHASE" = "control" ]; then
    for ctrl_mode in recompute gpu_hit; do
        jobname="ylh-exp3-${MODEL}-${ctrl_mode}"
        echo "Submitting control: $jobname"
        if [ -n "$FROZEN_RATES" ]; then
            sbatch \
                --export=MODEL="${MODEL}",MODE="${ctrl_mode}",CTX="${CTX}",RATIO="${RATIO}",CONTROL=1,FROZEN_RATES="${FROZEN_RATES}" \
                -J "$jobname" \
                "$SCRIPT_DIR/run_exp3.sbatch"
            echo "  (using frozen rates: $FROZEN_RATES)"
        else
            sbatch \
                --export=MODEL="${MODEL}",MODE="${ctrl_mode}",CTX="${CTX}",RATIO="${RATIO}",CONTROL=1 \
                -J "$jobname" \
                "$SCRIPT_DIR/run_exp3.sbatch"
            echo "  (NOTE: no FROZEN_RATES — control will run its own calibration)"
        fi
        echo "Submitted: $jobname"
    done
fi

echo ""
echo "Experiment 3 submission complete."
echo "  Primary (cpu_hit): full calibration + 7-point sweep"
echo "  Controls (recompute, gpu_hit): representative points only (low/sat/overload)"
