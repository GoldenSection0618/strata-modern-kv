#!/bin/bash
# Submit Qwen Exp1 + Exp2 as one serialized A100 chain.
# This avoids concurrent CPU/DRAM/HiCache interference on the measurement node.

set -euo pipefail

EXP_ROOT="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
SBATCH_DIR="$EXP_ROOT/code/sglang_hicache/sbatch"
EXP1_RUNNER="$SBATCH_DIR/run_exp1_sglang.sbatch"
EXP2_RUNNER="$SBATCH_DIR/run_exp2_sglang.sbatch"
NODELIST="${NODELIST:-smtg5001}"
ROOT_DEPENDENCY="${ROOT_DEPENDENCY:-}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-exp1-exp2-qwen}"
PIPELINE_DIR="$EXP_ROOT/results/sglang/exp12-pipelines/$PIPELINE_TAG"
MANIFEST="$PIPELINE_DIR/jobs.txt"

mkdir -p "$PIPELINE_DIR"
printf 'pipeline_tag=%s\nnode=%s\nroot_dependency=%s\n' \
    "$PIPELINE_TAG" "$NODELIST" "$ROOT_DEPENDENCY" > "$MANIFEST"

submitted_jobs=()
rollback_on_error() {
    status=$?
    if [ "$status" -ne 0 ] && [ "${#submitted_jobs[@]}" -gt 0 ]; then
        echo "ERROR: partial submission; cancelling jobs from this invocation: ${submitted_jobs[*]}" >&2
        scancel "${submitted_jobs[@]}" || true
    fi
    exit "$status"
}
trap rollback_on_error EXIT

previous_job=""
submit_one() {
    label="$1"
    runner="$2"
    exports="$3"
    dependency=""
    if [ -n "$previous_job" ]; then
        dependency="afterok:$previous_job"
    elif [ -n "$ROOT_DEPENDENCY" ]; then
        dependency="$ROOT_DEPENDENCY"
    fi

    dep_args=()
    if [ -n "$dependency" ]; then
        dep_args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    fi
    job="$(sbatch --parsable \
        --nodelist="$NODELIST" \
        "${dep_args[@]}" \
        -J "ylh-$label" \
        --export="ALL,$exports" \
        "$runner")"
    job="${job%%;*}"
    submitted_jobs+=("$job")
    printf '%s=%s dependency=%s\n' "$label" "$job" "${dependency:-none}" | tee -a "$MANIFEST"
    previous_job="$job"
}

# Exp1: four context lengths x three residency modes.
for ctx in 4096 8192 16384 32768; do
    for mode in recompute gpu_hit cpu_hit; do
        label="exp1-qwen-${ctx}-${mode}"
        tag="${PIPELINE_TAG}-${label}"
        submit_one "$label" "$EXP1_RUNNER" \
            "MODEL=qwen,CTX=$ctx,MODE=$mode,RUN_TAG=$tag,SMOKE=0,DRY_RUN=0,HICACHE_RATIO=3,HICACHE_IO_BACKEND=direct,HICACHE_MEM_LAYOUT=page_first_direct"
    done
done

# Exp2: 0% only recompute; other four ratios run all three modes.
for spec in \
    '0.0:0:recompute' \
    '0.25:25:recompute' '0.25:25:gpu_hit' '0.25:25:cpu_hit' \
    '0.5:50:recompute' '0.5:50:gpu_hit' '0.5:50:cpu_hit' \
    '0.75:75:recompute' '0.75:75:gpu_hit' '0.75:75:cpu_hit' \
    '0.875:87:recompute' '0.875:87:gpu_hit' '0.875:87:cpu_hit'; do
    IFS=: read -r ratio pct mode <<< "$spec"
    label="exp2-qwen-32768-${pct}pct-${mode}"
    tag="${PIPELINE_TAG}-${label}"
    submit_one "$label" "$EXP2_RUNNER" \
        "MODEL=qwen,CTX=32768,RATIO=$ratio,MODE=$mode,RUN_TAG=$tag,SMOKE=0,DRY_RUN=0,HICACHE_RATIO=3,HICACHE_IO_BACKEND=direct,HICACHE_MEM_LAYOUT=page_first_direct"
done

printf 'last_job=%s\n' "$previous_job" | tee -a "$MANIFEST"
echo "PIPELINE_DIR=$PIPELINE_DIR"
trap - EXIT
