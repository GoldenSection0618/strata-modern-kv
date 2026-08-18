#!/bin/bash
#SBATCH -J ylh-hcv-tests
#SBATCH -p i56m512A100
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# ============================================================================
# Pure-Python unit tests (no GPU, no server).  Runs only inside Slurm per
# cluster rules; never on the login node.
#   sbatch run_tests.sbatch
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

echo "== tests | python=$PYTHON_BIN =="
cd "$CODE_DIR"
"$PYTHON_BIN" tests/run_tests.py
echo "== tests done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
