#!/bin/bash
#SBATCH -J ylh-hcv-exp4-freeze
#SBATCH -p i56m512A100
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# ============================================================================
# Experiment 4 phase 1: freeze the V0/V1/V2 selection rule from primary
# (Experiments 2-3) results, BEFORE any secondary-model performance data.
# Requires the completed exp2 calibration.
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

MODEL="${MODEL:-gemma}"                    # secondary model for exp4
RUN_TAG="${RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"
case "$MODEL" in qwen|gemma) ;; *) echo "ERROR: MODEL=$MODEL" >&2; exit 2 ;; esac

echo "== exp4 freeze | MODEL=$MODEL =="

if [ "${DRY_RUN:-0}" = "1" ]; then
    cd "$CODE_DIR"
    "$PYTHON_BIN" -m hcv.dry_run \
        --config "$CONFIG_PATH/exp4.json" \
        --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
        --model "$MODEL" --experiment exp4 --log-dir "$LOG_DIR"
    exit $?
fi

"$PYTHON_BIN" -m hcv.provenance check --env-dir "$SGLANG_ENV_DIR" --commit "$SGLANG_COMMIT"

cd "$CODE_DIR"
"$PYTHON_BIN" -m hcv.run_exp4 --phase freeze \
    --config "$CONFIG_PATH/exp4.json" \
    --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
    --model "$MODEL" --experiment exp4 --log-dir "$LOG_DIR"

echo "== exp4 freeze done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
