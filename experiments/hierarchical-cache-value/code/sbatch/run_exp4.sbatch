#!/bin/bash
#SBATCH -J ylh-hcv-exp4
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
# Experiment 4 phase 2: matched V0/V1/V2 validation on the secondary model.
#   sbatch --export=MODEL=gemma run_exp4.sbatch
# Hard-fails when representative_selection.json was not frozen first.
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

MODEL="${MODEL:-gemma}"                    # secondary model for exp4
SMOKE="${SMOKE:-0}"
RUN_TAG="${RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
case "$MODEL" in qwen|gemma) ;; *) echo "ERROR: MODEL=$MODEL" >&2; exit 2 ;; esac

echo "== exp4 run | MODEL=$MODEL SMOKE=$SMOKE =="

if [ "${DRY_RUN:-0}" = "1" ]; then
    cd "$CODE_DIR"
    "$PYTHON_BIN" -m hcv.dry_run \
        --config "$CONFIG_PATH/exp4.json" \
        --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
        --model "$MODEL" --experiment exp4 --log-dir "$LOG_DIR" \
        $( [ "$SMOKE" = "1" ] && echo --smoke )
    exit $?
fi

# Hard gate: the frozen selection rule must exist before secondary perf data.
if [ ! -f "$RESULTS_ROOT/exp4/representative_selection.json" ]; then
    echo "ERROR: representative_selection.json missing; run run_exp4_freeze.sbatch first" >&2
    exit 3
fi

"$PYTHON_BIN" -m hcv.provenance check --env-dir "$SGLANG_ENV_DIR" --commit "$SGLANG_COMMIT"
echo "== preflight =="
"$PYTHON_BIN" -m hcv.preflight || { echo "PREFLIGHT FAILED" >&2; exit 1; }

cd "$CODE_DIR"
"$PYTHON_BIN" -m hcv.run_exp4 --phase run \
    --config "$CONFIG_PATH/exp4.json" \
    --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
    --model "$MODEL" --experiment exp4 --log-dir "$LOG_DIR" \
    --ready-timeout "$READY_TIMEOUT" \
    $( [ "$SMOKE" = "1" ] && echo --smoke )

echo "== exp4 run done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
