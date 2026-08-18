#!/bin/bash
#SBATCH -J ylh-sglang-exp3-control
#SBATCH -p i56m512A100
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

set -euo pipefail

: "${CONTROL_MODE:?CONTROL_MODE is required}"
: "${RATES_FILE:?RATES_FILE is required}"
: "${CONTROL_RUN_TAG:?CONTROL_RUN_TAG is required}"

case "$CONTROL_MODE" in
    recompute|gpu_hit) ;;
    *) echo "ERROR: invalid CONTROL_MODE=$CONTROL_MODE" >&2; exit 2 ;;
esac

if [ ! -s "$RATES_FILE" ]; then
    echo "ERROR: frozen rates file missing or empty: $RATES_FILE" >&2
    exit 3
fi
FROZEN_RATES="$(tr -d '[:space:]' < "$RATES_FILE")"
if [[ ! "$FROZEN_RATES" =~ ^[0-9]+([.][0-9]+)?(,[0-9]+([.][0-9]+)?){6}$ ]]; then
    echo "ERROR: invalid frozen rates: $FROZEN_RATES" >&2
    exit 4
fi

export MODEL="${MODEL:-qwen}"
export MODE="$CONTROL_MODE"
export CTX="${CTX:-32768}"
export RATIO="${RATIO:-0.5}"
export CONTROL=1
export FROZEN_RATES
export RUN_TAG="$CONTROL_RUN_TAG"
export SMOKE=0
export DRY_RUN=0

RUNNER="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck/code/sglang_hicache/sbatch/run_exp3_sglang.sbatch"
exec bash "$RUNNER"
