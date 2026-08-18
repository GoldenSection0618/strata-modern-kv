#!/bin/bash
# Lightweight submit for Experiment 3: reuse sweep at the fixed pressure
# selected from Experiment 2 (needs the completed calibration).
#   bash submit_exp3.sh                 # full sweep (levels x architectures)
#   REUSE=0.50 bash submit_exp3.sh      # single reuse level
#   SMOKE=1 bash submit_exp3.sh         # smoke sweep
#   DRY_RUN=1 bash submit_exp3.sh       # login-node static validation
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HCV_CODE_DIR="$(dirname "$HERE")"

MODEL="${MODEL:-qwen}"
SMOKE="${SMOKE:-0}"
REUSE="${REUSE:-}"
PRESSURE="${PRESSURE:-High}"
LEVELS="${LEVELS:-0.00 0.25 0.50 0.75}"

if [ -n "$REUSE" ]; then
    LEVELS="$REUSE"
fi
submitted_jobs=()
rollback_on_error() {
    status=$?
    if [ "$status" -ne 0 ] && [ "${#submitted_jobs[@]}" -gt 0 ]; then scancel "${submitted_jobs[@]}" || true; fi
    exit "$status"
}
trap rollback_on_error EXIT
previous_job="${ROOT_DEPENDENCY:-}"
for LEVEL in $LEVELS; do
    for ARCH in gpu_only hierarchical; do
        if [ "${DRY_RUN:-0}" = "1" ]; then runner="$HERE/run_dry_run.sbatch"; else runner="$HERE/run_exp3.sbatch"; fi
        dep_args=()
        if [ -n "$previous_job" ]; then dep_args+=(--dependency="afterok:$previous_job" --kill-on-invalid-dep=yes); fi
        job=$(sbatch --parsable "${dep_args[@]}" \
            --export=ALL,HCV_CODE_DIR,MODEL="$MODEL",SMOKE="$SMOKE",EXPERIMENT=exp3,ARCH="$ARCH",REUSE="$LEVEL",PRESSURE="$PRESSURE" \
            "$runner")
        job="${job%%;*}"
        submitted_jobs+=("$job")
        previous_job="$job"
    done
done
trap - EXIT
