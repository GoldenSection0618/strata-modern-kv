#!/bin/bash
# Lightweight submit for Experiment 4 (two phases, dependency chained):
#   1. freeze the V0/V1/V2 selection rule from primary results;
#   2. run the matched validation on the secondary model.
#   bash submit_exp4.sh                 # freeze + run (gemma)
#   SMOKE=1 bash submit_exp4.sh         # smoke run phase
#   DRY_RUN=1 bash submit_exp4.sh       # submit static-validation chain
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HCV_CODE_DIR="$(dirname "$HERE")"

MODEL="${MODEL:-gemma}"
SMOKE="${SMOKE:-0}"

submitted_jobs=()
rollback_on_error() {
    status=$?
    if [ "$status" -ne 0 ] && [ "${#submitted_jobs[@]}" -gt 0 ]; then scancel "${submitted_jobs[@]}" || true; fi
    exit "$status"
}
trap rollback_on_error EXIT

if [ "${DRY_RUN:-0}" = "1" ]; then freeze_runner="$HERE/run_dry_run.sbatch"; run_runner="$HERE/run_dry_run.sbatch"; else freeze_runner="$HERE/run_exp4_freeze.sbatch"; run_runner="$HERE/run_exp4.sbatch"; fi
FREEZE_JOB=$(sbatch --export=ALL,HCV_CODE_DIR,MODEL="$MODEL",EXPERIMENT=exp4 \
    --parsable "$freeze_runner")
FREEZE_JOB="${FREEZE_JOB%%;*}"
submitted_jobs+=("$FREEZE_JOB")
echo "exp4 freeze job: $FREEZE_JOB"

RUN_JOB=$(sbatch --parsable --kill-on-invalid-dep=yes \
    --export=ALL,HCV_CODE_DIR,MODEL="$MODEL",SMOKE="$SMOKE",EXPERIMENT=exp4 \
    --dependency=afterok:"$FREEZE_JOB" "$run_runner")
RUN_JOB="${RUN_JOB%%;*}"
submitted_jobs+=("$RUN_JOB")
trap - EXIT
