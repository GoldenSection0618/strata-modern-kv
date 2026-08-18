#!/bin/bash
#SBATCH -J ylh-hcv-exp3
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
# Experiment 3 reuse-level point (one job per ARCH x REUSE).
#   sbatch --export=MODEL=qwen,ARCH=gpu_only,REUSE=0.50 run_exp3.sbatch
# Requires the completed calibration first.  REUSE: 0.00|0.25|0.50|0.75.
# Fixed pressure comes from the calibration floor (pressure_label).
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

MODEL="${MODEL:-qwen}"
ARCH="${ARCH:-gpu_only}"                   # gpu_only | hierarchical
REUSE="${REUSE:-0.50}"                     # revisit fraction for this level
PRESSURE="${PRESSURE:-High}"               # fixed exp2 pressure label
SMOKE="${SMOKE:-0}"
RUN_TAG="${RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

case "$ARCH" in gpu_only|hierarchical) ;; *) echo "ERROR: ARCH=$ARCH" >&2; exit 2 ;; esac
case "$MODEL" in qwen|gemma) ;; *) echo "ERROR: MODEL=$MODEL" >&2; exit 2 ;; esac

echo "== exp3 | MODEL=$MODEL ARCH=$ARCH REUSE=$REUSE PRESSURE=$PRESSURE =="

if [ "${DRY_RUN:-0}" = "1" ]; then
    cd "$CODE_DIR"
    "$PYTHON_BIN" -m hcv.dry_run \
        --config "$CONFIG_PATH/exp3.json" \
        --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
        --model "$MODEL" --experiment exp3 --log-dir "$LOG_DIR" \
        --architecture "$ARCH" --revisit-fraction "$REUSE" --pressure-label "$PRESSURE" \
        $( [ "$SMOKE" = "1" ] && echo --smoke )
    exit $?
fi

"$PYTHON_BIN" -m hcv.provenance check --env-dir "$SGLANG_ENV_DIR" --commit "$SGLANG_COMMIT"
echo "== preflight =="
"$PYTHON_BIN" -m hcv.preflight || { echo "PREFLIGHT FAILED" >&2; exit 1; }

cd "$CODE_DIR"
"$PYTHON_BIN" -m hcv.run_exp3 \
    --config "$CONFIG_PATH/exp3.json" \
    --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
    --model "$MODEL" --experiment exp3 --log-dir "$LOG_DIR" \
    --architecture "$ARCH" --revisit-fraction "$REUSE" --pressure-label "$PRESSURE" \
    --ready-timeout "$READY_TIMEOUT" \
    $( [ "$SMOKE" = "1" ] && echo --smoke )

echo "== exp3 point done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
