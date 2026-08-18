#!/bin/bash
#SBATCH -J ylh-exp3-primary-gate
#SBATCH -p cpu_amd_low
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

set -euo pipefail
source "$HOME/yanglihan/env.sh"

: "${PRIMARY_RUN_DIR:?PRIMARY_RUN_DIR is required}"
: "${PRIMARY_RUN_TAG:?PRIMARY_RUN_TAG is required}"
: "${RATES_FILE:?RATES_FILE is required}"

EXP_ROOT="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
PYTHON="$HOME/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211/bin/python"

exec "$PYTHON" "$EXP_ROOT/code/sglang_hicache/gate_exp3_primary.py" \
    --run-dir "$PRIMARY_RUN_DIR" \
    --run-tag "$PRIMARY_RUN_TAG" \
    --rates-file "$RATES_FILE" \
    --expected-calibration-points 7 \
    --expected-formal-points 7 \
    --expected-repeats 3 \
    --expected-requests 30
