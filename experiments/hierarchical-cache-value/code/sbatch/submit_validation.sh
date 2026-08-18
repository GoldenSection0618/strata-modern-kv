#!/bin/bash
# Lightweight submit for the hierarchy capability gate.
#   bash submit_validation.sh                # full gate, MODEL=qwen
#   MODEL=gemma bash submit_validation.sh    # secondary model gate
#   SMOKE=1 bash submit_validation.sh        # smoke gate
#   DRY_RUN=1 bash submit_validation.sh      # minimal static-validation job
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HCV_CODE_DIR="$(dirname "$HERE")"

MODEL="${MODEL:-qwen}"
SMOKE="${SMOKE:-0}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    sbatch --export=ALL,HCV_CODE_DIR,MODEL="$MODEL",SMOKE="$SMOKE",EXPERIMENT=validation \
        "$HERE/run_dry_run.sbatch"
else
    sbatch --export=ALL,HCV_CODE_DIR,MODEL="$MODEL",SMOKE="$SMOKE" \
        "$HERE/run_validation.sbatch"
fi
