#!/bin/bash
# Submit the production Exp3 chain:
# primary cpu_hit -> strict semantic gate -> recompute + gpu_hit controls.

set -euo pipefail

MODEL="${1:-qwen}"
if [ "$MODEL" != "qwen" ]; then
    echo "ERROR: this validated production pipeline currently supports qwen only" >&2
    exit 2
fi

EXP_ROOT="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
SBATCH_DIR="$EXP_ROOT/code/sglang_hicache/sbatch"
RUNNER="$SBATCH_DIR/run_exp3_sglang.sbatch"
GATE="$SBATCH_DIR/gate_exp3_primary.sbatch"
CONTROL="$SBATCH_DIR/run_exp3_control_from_gate.sbatch"
CONTROL_GATE="$SBATCH_DIR/gate_exp3_control.sbatch"
NODELIST="${NODELIST:-smtg5001}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-exp3-qwen32k}"
PRIMARY_TAG="${PIPELINE_TAG}-primary"
PIPELINE_DIR="$EXP_ROOT/results/sglang/exp3/pipelines/$PIPELINE_TAG"
RATES_FILE="$PIPELINE_DIR/frozen_rates.txt"
PRIMARY_RUN_DIR="$EXP_ROOT/results/sglang/exp3/qwen/32k-50pct-cpu_hit/run-$PRIMARY_TAG"

mkdir -p "$PIPELINE_DIR"

submitted_jobs=()
rollback_on_error() {
    status=$?
    if [ "$status" -ne 0 ] && [ "${#submitted_jobs[@]}" -gt 0 ]; then
        echo "ERROR: pipeline submission failed; cancelling jobs submitted by this invocation: ${submitted_jobs[*]}" >&2
        scancel "${submitted_jobs[@]}" || true
    fi
    exit "$status"
}
trap rollback_on_error EXIT

primary_job="$(sbatch --parsable \
    --nodelist="$NODELIST" \
    -J ylh-sglang-exp3-qwen-primary \
    --export=ALL,MODEL=qwen,MODE=cpu_hit,CTX=32768,RATIO=0.5,RUN_TAG="$PRIMARY_TAG",SMOKE=0,DRY_RUN=0 \
    "$RUNNER")"
primary_job="${primary_job%%;*}"
submitted_jobs+=("$primary_job")

gate_job="$(sbatch --parsable \
    --dependency="afterok:$primary_job" \
    --kill-on-invalid-dep=yes \
    -J ylh-exp3-qwen-primary-gate \
    --export=ALL,PRIMARY_RUN_DIR="$PRIMARY_RUN_DIR",PRIMARY_RUN_TAG="$PRIMARY_TAG",RATES_FILE="$RATES_FILE" \
    "$GATE")"
gate_job="${gate_job%%;*}"
submitted_jobs+=("$gate_job")

recompute_job="$(sbatch --parsable \
    --dependency="afterok:$gate_job" \
    --kill-on-invalid-dep=yes \
    --nodelist="$NODELIST" \
    -J ylh-sglang-exp3-qwen-recompute \
    --export=ALL,MODEL=qwen,CTX=32768,RATIO=0.5,CONTROL_MODE=recompute,RATES_FILE="$RATES_FILE",CONTROL_RUN_TAG="${PIPELINE_TAG}-recompute" \
    "$CONTROL")"
recompute_job="${recompute_job%%;*}"
submitted_jobs+=("$recompute_job")

gpu_hit_job="$(sbatch --parsable \
    --dependency="afterok:$gate_job" \
    --kill-on-invalid-dep=yes \
    --nodelist="$NODELIST" \
    -J ylh-sglang-exp3-qwen-gpu_hit \
    --export=ALL,MODEL=qwen,CTX=32768,RATIO=0.5,CONTROL_MODE=gpu_hit,RATES_FILE="$RATES_FILE",CONTROL_RUN_TAG="${PIPELINE_TAG}-gpu_hit" \
    "$CONTROL")"
gpu_hit_job="${gpu_hit_job%%;*}"
submitted_jobs+=("$gpu_hit_job")

recompute_gate_job="$(sbatch --parsable \
    --dependency="afterok:$recompute_job" \
    --kill-on-invalid-dep=yes \
    -J ylh-exp3-qwen-recompute-gate \
    --export=ALL,CONTROL_MODE=recompute,CONTROL_RUN_TAG="${PIPELINE_TAG}-recompute",CONTROL_RUN_DIR="$EXP_ROOT/results/sglang/exp3/qwen/32k-50pct-recompute/run-${PIPELINE_TAG}-recompute" \
    "$CONTROL_GATE")"
recompute_gate_job="${recompute_gate_job%%;*}"
submitted_jobs+=("$recompute_gate_job")

gpu_hit_gate_job="$(sbatch --parsable \
    --dependency="afterok:$gpu_hit_job" \
    --kill-on-invalid-dep=yes \
    -J ylh-exp3-qwen-gpu_hit-gate \
    --export=ALL,CONTROL_MODE=gpu_hit,CONTROL_RUN_TAG="${PIPELINE_TAG}-gpu_hit",CONTROL_RUN_DIR="$EXP_ROOT/results/sglang/exp3/qwen/32k-50pct-gpu_hit/run-${PIPELINE_TAG}-gpu_hit" \
    "$CONTROL_GATE")"
gpu_hit_gate_job="${gpu_hit_gate_job%%;*}"
submitted_jobs+=("$gpu_hit_gate_job")

cat > "$PIPELINE_DIR/jobs.txt" <<EOF
pipeline_tag=$PIPELINE_TAG
primary_job=$primary_job
gate_job=$gate_job
recompute_job=$recompute_job
gpu_hit_job=$gpu_hit_job
recompute_gate_job=$recompute_gate_job
gpu_hit_gate_job=$gpu_hit_gate_job
primary_run_dir=$PRIMARY_RUN_DIR
rates_file=$RATES_FILE
EOF

echo "PIPELINE_TAG=$PIPELINE_TAG"
echo "PRIMARY_JOB=$primary_job"
echo "GATE_JOB=$gate_job dependency=afterok:$primary_job"
echo "RECOMPUTE_JOB=$recompute_job dependency=afterok:$gate_job"
echo "GPU_HIT_JOB=$gpu_hit_job dependency=afterok:$gate_job"
echo "RECOMPUTE_GATE_JOB=$recompute_gate_job dependency=afterok:$recompute_job"
echo "GPU_HIT_GATE_JOB=$gpu_hit_gate_job dependency=afterok:$gpu_hit_job"
echo "PIPELINE_DIR=$PIPELINE_DIR"
trap - EXIT
