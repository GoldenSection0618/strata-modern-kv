#!/bin/bash
# Lightweight submit for Experiment 2: calibration first, then all
# pressure points chained with a Slurm dependency (afterok).
#   bash submit_exp2.sh                 # full sweep (calibration + points)
#   SMOKE=1 bash submit_exp2.sh         # smoke sweep
#   DRY_RUN=1 bash submit_exp2.sh       # submit static-validation chain
#   PRESSURE=Low bash submit_exp2.sh    # single point (calibration still runs)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HCV_CODE_DIR="$(dirname "$HERE")"

MODEL="${MODEL:-qwen}"
SMOKE="${SMOKE:-0}"
PRESSURE="${PRESSURE:-}"
LADDER="${LADDER:-Low Medium High}"

submitted_jobs=()
rollback_on_error() {
    status=$?
    if [ "$status" -ne 0 ] && [ "${#submitted_jobs[@]}" -gt 0 ]; then scancel "${submitted_jobs[@]}" || true; fi
    exit "$status"
}
trap rollback_on_error EXIT

if [ "${DRY_RUN:-0}" = "1" ]; then calib_runner="$HERE/run_dry_run.sbatch"; point_runner="$HERE/run_dry_run.sbatch"; else calib_runner="$HERE/run_exp2_calibration.sbatch"; point_runner="$HERE/run_exp2.sbatch"; fi
CALIB_JOB=$(sbatch --export=ALL,HCV_CODE_DIR,MODEL="$MODEL",SMOKE="$SMOKE",EXPERIMENT=exp2 \
    --parsable "$calib_runner")
CALIB_JOB="${CALIB_JOB%%;*}"
submitted_jobs+=("$CALIB_JOB")
echo "calibration job: $CALIB_JOB"

if [ -n "$PRESSURE" ]; then
    LADDER="$PRESSURE"
fi
previous_job="$CALIB_JOB"
for LABEL in $LADDER; do
    for ARCH in gpu_only hierarchical; do
        job=$(sbatch --parsable --kill-on-invalid-dep=yes \
            --export=ALL,HCV_CODE_DIR,MODEL="$MODEL",SMOKE="$SMOKE",EXPERIMENT=exp2,ARCH="$ARCH",PRESSURE="$LABEL" \
            --dependency=afterok:"$previous_job" "$point_runner")
        job="${job%%;*}"
        submitted_jobs+=("$job")
        previous_job="$job"
    done
done
trap - EXIT
