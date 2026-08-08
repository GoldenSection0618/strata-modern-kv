#!/bin/bash
# Submit all Experiment 3 tasks for a given model.
#
# Usage:
#   bash submit_exp3.sh qwen           # full sweep (cpu_hit primary + controls)
#   bash submit_exp3.sh qwen primary    # cpu_hit only (full calibration + sweep)
#   bash submit_exp3.sh qwen control     # recompute + gpu_hit controls only

set -euo pipefail

MODEL="${1:-qwen}"
PHASE="${2:-all}"   # all | primary | control

CTX="${CTX:-32768}"
RATIO="${RATIO:-0.5}"

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

# --- Control: recompute (representative points only) ---
if [ "$PHASE" = "all" ] || [ "$PHASE" = "control" ]; then
    jobname="ylh-exp3-${MODEL}-recompute"
    echo "Submitting control: $jobname"
    sbatch \
        --export=MODEL="${MODEL}",MODE=recompute,CTX="${CTX}",RATIO="${RATIO}",CONTROL=1 \
        -J "$jobname" \
        "$SCRIPT_DIR/run_exp3.sbatch"
    echo "Submitted: $jobname"

# --- Control: gpu_hit (representative points only) ---
    jobname="ylh-exp3-${MODEL}-gpu_hit"
    echo "Submitting control: $jobname"
    sbatch \
        --export=MODEL="${MODEL}",MODE=gpu_hit,CTX="${CTX}",RATIO="${RATIO}",CONTROL=1 \
        -J "$jobname" \
        "$SCRIPT_DIR/run_exp3.sbatch"
    echo "Submitted: $jobname"
fi

echo ""
echo "Experiment 3 submission complete."
echo "  Primary (cpu_hit): full calibration + 7-point sweep"
echo "  Controls (recompute, gpu_hit): representative points only (low/sat/overload)"
