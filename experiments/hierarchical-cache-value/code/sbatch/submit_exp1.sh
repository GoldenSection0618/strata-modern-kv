#!/bin/bash
# Lightweight submit for Experiment 1 (all four cells).
# Architecture order is interleaved so machine drift does not
# systematically favor one side: gpu_only,cold -> hier,cold ->
# gpu_only,warm -> hier,warm.
#   bash submit_exp1.sh                 # full 4-cell matrix
#   SMOKE=1 bash submit_exp1.sh         # smoke cells
#   DRY_RUN=1 bash submit_exp1.sh       # submit minimal static-validation jobs
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HCV_CODE_DIR="$(dirname "$HERE")"

MODEL="${MODEL:-qwen}"
SMOKE="${SMOKE:-0}"
CONCURRENCY="${CONCURRENCY:-}"
CELL_LIMIT="${CELL_LIMIT:-0}"

submitted_jobs=()
rollback_on_error() {
    status=$?
    if [ "$status" -ne 0 ] && [ "${#submitted_jobs[@]}" -gt 0 ]; then
        scancel "${submitted_jobs[@]}" || true
    fi
    exit "$status"
}
trap rollback_on_error EXIT

previous_job="${ROOT_DEPENDENCY:-}"
cell_count=0
for CELL in "gpu_only cold" "hierarchical cold" "gpu_only warm" "hierarchical warm"; do
    if [ "$CELL_LIMIT" -gt 0 ] && [ "$cell_count" -ge "$CELL_LIMIT" ]; then break; fi
    set -- $CELL
    if [ "${DRY_RUN:-0}" = "1" ]; then runner="$HERE/run_dry_run.sbatch"; else runner="$HERE/run_exp1.sbatch"; fi
    dep_args=()
    if [ -n "$previous_job" ]; then dep_args+=(--dependency="afterok:$previous_job" --kill-on-invalid-dep=yes); fi
    job=$(sbatch --parsable "${dep_args[@]}" \
        --export=ALL,HCV_CODE_DIR,MODEL="$MODEL",SMOKE="$SMOKE",EXPERIMENT=exp1,CONCURRENCY="$CONCURRENCY",ARCH="$1",STATE="$2" \
        "$runner")
    job="${job%%;*}"
    submitted_jobs+=("$job")
    previous_job="$job"
    cell_count=$((cell_count + 1))
done
trap - EXIT
