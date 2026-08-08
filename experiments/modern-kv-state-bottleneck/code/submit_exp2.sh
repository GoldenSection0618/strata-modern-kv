#!/bin/bash
# Submit all Experiment 2 tasks for a given model.
# Usage: bash submit_exp2.sh qwen    (or: gemma)
#
# Submits 5 ratios × 3 modes = 15 tasks (0% ratio skips cpu_hit/gpu_hit
# validation but still runs recompute as baseline).

set -euo pipefail

MODEL="${1:-qwen}"
CTX="${CTX:-32768}"

RATIOS="0.0 0.25 0.50 0.75 0.875"
MODES="recompute gpu_hit cpu_hit"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for ratio in $RATIOS; do
  for mode in $MODES; do
    # 0% ratio: only recompute makes sense (no prefix to reuse)
    if [ "$ratio" = "0.0" ] && [ "$mode" != "recompute" ]; then
      echo "Skip: ratio=0.0 mode=$mode (no shared prefix)"
      continue
    fi

    pct=$(awk -v r="$ratio" 'BEGIN { printf "%d", r * 100 }')
    jobname="ylh-exp2-${MODEL}-${CTX}-${pct}pct-${mode}"

    sbatch \
      --export=MODEL="${MODEL}",CTX="${CTX}",RATIO="${ratio}",MODE="${mode}",N_WARMUP=3,N_REPEATS=10 \
      -J "$jobname" \
      "$SCRIPT_DIR/run_exp2.sbatch"
    echo "Submitted: $jobname"
  done
done
