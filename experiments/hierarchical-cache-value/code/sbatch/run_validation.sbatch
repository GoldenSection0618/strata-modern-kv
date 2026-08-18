#!/bin/bash
#SBATCH -J ylh-hcv-validation
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
# Full-hierarchy capability gate (validation.json controls reportability).
#   sbatch --export=MODEL=qwen run_validation.sbatch
#   DRY_RUN=1 MODEL=qwen bash submit_validation.sh      (minimal Slurm dry run)
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

# --- user overrides resolved ONCE here (never silently overwritten) -------
MODEL="${MODEL:-qwen}"                 # qwen | gemma
SMOKE="${SMOKE:-0}"
RUN_TAG="${RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
case "$MODEL" in
  qwen|gemma) ;;
  *) echo "ERROR: MODEL=$MODEL invalid (qwen|gemma)" >&2; exit 2 ;;
esac

echo "=================================================="
echo "Hierarchy gate | MODEL=$MODEL SMOKE=$SMOKE"
echo "SGLang env     | $SGLANG_ENV_DIR"
echo "SGLang commit  | $SGLANG_COMMIT"
echo "Run tag        | $RUN_TAG"
echo "Results root   | $RESULTS_ROOT"
echo "Job            | ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-?}"
echo "=================================================="

# --- DRY_RUN: static validation only, no server, no GPU, no torch ---------
if [ "${DRY_RUN:-0}" = "1" ]; then
    cd "$CODE_DIR"
    "$PYTHON_BIN" -m hcv.dry_run \
        --config "$CONFIG_PATH/validation.json" \
        --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
        --model "$MODEL" --experiment validation --log-dir "$LOG_DIR" \
        $( [ "$SMOKE" = "1" ] && echo --smoke )
    exit $?
fi

# --- provenance markers (must match the pinned commit) ---------------------
"$PYTHON_BIN" -m hcv.provenance check --env-dir "$SGLANG_ENV_DIR" --commit "$SGLANG_COMMIT"

# --- compute-node preflight: real BF16 + C++20/CUDA JIT --------------------
echo "== preflight =="
"$PYTHON_BIN" -m hcv.preflight || { echo "PREFLIGHT FAILED" >&2; exit 1; }

cd "$CODE_DIR"
"$PYTHON_BIN" -m hcv.run_validation \
    --config "$CONFIG_PATH/validation.json" \
    --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
    --model "$MODEL" --experiment validation --log-dir "$LOG_DIR" \
    --ready-timeout "$READY_TIMEOUT" \
    $( [ "$SMOKE" = "1" ] && echo --smoke )

echo "== validation done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
