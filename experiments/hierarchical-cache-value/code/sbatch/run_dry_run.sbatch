#!/bin/bash
#SBATCH -J ylh-hcv-dry-run
#SBATCH -p i56m512A100
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

MODEL="${MODEL:-qwen}"
EXPERIMENT="${EXPERIMENT:-validation}"
ARCH="${ARCH:-gpu_only}"
STATE="${STATE:-warm}"
PRESSURE="${PRESSURE:-High}"
REUSE="${REUSE:-0.50}"
RUN_TAG="${RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID}}"

cd "$CODE_DIR"
args=(--config "$CONFIG_PATH/$EXPERIMENT.json" --results-root "$RESULTS_ROOT"
      --run-tag "$RUN_TAG" --model "$MODEL" --experiment "$EXPERIMENT"
      --log-dir "$LOG_DIR")
case "$EXPERIMENT" in
  validation) ;;
  exp1) args+=(--architecture "$ARCH" --initial-state "$STATE") ;;
  exp2) args+=(--architecture "$ARCH" --pressure-label "$PRESSURE") ;;
  exp3) args+=(--architecture "$ARCH" --pressure-label "$PRESSURE" --revisit-fraction "$REUSE") ;;
  exp4) ;;
  *) echo "ERROR: EXPERIMENT=$EXPERIMENT" >&2; exit 2 ;;
esac
if [ "${SMOKE:-0}" = "1" ]; then args+=(--smoke); fi
"$PYTHON_BIN" -m hcv.dry_run "${args[@]}"
