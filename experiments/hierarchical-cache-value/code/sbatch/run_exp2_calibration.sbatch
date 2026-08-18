#!/bin/bash
#SBATCH -J ylh-hcv-exp2-calib
#SBATCH -p i56m512A100
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# ============================================================================
# Experiment 2 calibration: minimum GPU budget without preemption.
# Must complete successfully before any exp2/exp3/exp4 sweep job is
# submitted (they hard-fail without calibration.json).
#   sbatch run_exp2_calibration.sbatch
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

MODEL="${MODEL:-qwen}"
SMOKE="${SMOKE:-0}"
RUN_TAG="${RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
case "$MODEL" in qwen|gemma) ;; *) echo "ERROR: MODEL=$MODEL" >&2; exit 2 ;; esac

echo "== exp2 calibration | MODEL=$MODEL SMOKE=$SMOKE =="

if [ "${DRY_RUN:-0}" = "1" ]; then
    cd "$CODE_DIR"
    "$PYTHON_BIN" -m hcv.dry_run \
        --config "$CONFIG_PATH/exp2.json" \
        --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
        --model "$MODEL" --experiment exp2 --log-dir "$LOG_DIR" \
        $( [ "$SMOKE" = "1" ] && echo --smoke )
    exit $?
fi

"$PYTHON_BIN" -m hcv.provenance check --env-dir "$SGLANG_ENV_DIR" --commit "$SGLANG_COMMIT"
echo "== preflight =="
"$PYTHON_BIN" -m hcv.preflight || { echo "PREFLIGHT FAILED" >&2; exit 1; }

cd "$CODE_DIR"
"$PYTHON_BIN" -m hcv.calibrate \
    --config "$CONFIG_PATH/exp2.json" \
    --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
    --model "$MODEL" --experiment exp2 --log-dir "$LOG_DIR" \
    --ready-timeout "$READY_TIMEOUT" \
    $( [ "$SMOKE" = "1" ] && echo --smoke )

echo "== exp2 calibration done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
