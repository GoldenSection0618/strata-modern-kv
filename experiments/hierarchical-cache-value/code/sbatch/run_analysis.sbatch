#!/bin/bash
#SBATCH -J ylh-hcv-analysis
#SBATCH -p i56m512A100
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# ============================================================================
# Deterministic raw -> processed -> results processing (no GPU needed).
# Pure CPU work is still run through Slurm per cluster rules.
#   sbatch run_analysis.sbatch
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

echo "== analysis | results-root=$RESULTS_ROOT =="

cd "$CODE_DIR"
"$PYTHON_BIN" -m hcv.analysis \
    --results-root "$RESULTS_ROOT" \
    --out-root "$RESULTS_ROOT/processed_out"

echo "== analysis done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
