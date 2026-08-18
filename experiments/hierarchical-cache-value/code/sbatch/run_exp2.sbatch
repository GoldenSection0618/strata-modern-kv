#!/bin/bash
#SBATCH -J ylh-hcv-exp2
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
# Experiment 2 pressure point (one job per ARCH x PRESSURE).
#   sbatch --export=MODEL=qwen,ARCH=gpu_only,PRESSURE=Low run_exp2.sbatch
# Requires the completed calibration (run_exp2_calibration.sbatch) first.
# PRESSURE: Low | Medium | High | VeryHigh (VeryHigh optional).
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

MODEL="${MODEL:-qwen}"
ARCH="${ARCH:-gpu_only}"                   # gpu_only | hierarchical
PRESSURE="${PRESSURE:-Low}"                # Low|Medium|High|VeryHigh
SMOKE="${SMOKE:-0}"
RUN_TAG="${RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

case "$ARCH" in gpu_only|hierarchical) ;; *) echo "ERROR: ARCH=$ARCH" >&2; exit 2 ;; esac
case "$PRESSURE" in Low|Medium|High|VeryHigh) ;; *) echo "ERROR: PRESSURE=$PRESSURE" >&2; exit 2 ;; esac
case "$MODEL" in qwen|gemma) ;; *) echo "ERROR: MODEL=$MODEL" >&2; exit 2 ;; esac

echo "== exp2 | MODEL=$MODEL ARCH=$ARCH PRESSURE=$PRESSURE SMOKE=$SMOKE =="

if [ "${DRY_RUN:-0}" = "1" ]; then
    cd "$CODE_DIR"
    HCV_PRESSURE_LABEL="$PRESSURE" "$PYTHON_BIN" -m hcv.dry_run \
        --config "$CONFIG_PATH/exp2.json" \
        --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
        --model "$MODEL" --experiment exp2 --log-dir "$LOG_DIR" \
        --architecture "$ARCH" --pressure-label "$PRESSURE" \
        $( [ "$SMOKE" = "1" ] && echo --smoke )
    exit $?
fi

"$PYTHON_BIN" -m hcv.provenance check --env-dir "$SGLANG_ENV_DIR" --commit "$SGLANG_COMMIT"
echo "== preflight =="
"$PYTHON_BIN" -m hcv.preflight || { echo "PREFLIGHT FAILED" >&2; exit 1; }

cd "$CODE_DIR"
"$PYTHON_BIN" -m hcv.run_exp2 \
    --config "$CONFIG_PATH/exp2.json" \
    --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
    --model "$MODEL" --experiment exp2 --log-dir "$LOG_DIR" \
    --architecture "$ARCH" --pressure-label "$PRESSURE" \
    --ready-timeout "$READY_TIMEOUT" \
    $( [ "$SMOKE" = "1" ] && echo --smoke )

echo "== exp2 point done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
