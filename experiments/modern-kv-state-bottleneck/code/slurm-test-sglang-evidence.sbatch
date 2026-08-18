#!/bin/bash
#SBATCH -J ylh-sglang-evidence-unit
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

EXP_ROOT="$HOME/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck"
PYTHON="$HOME/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211/bin/python"

cd "$EXP_ROOT"
PYTHONPATH=code:code/tests "$PYTHON" code/tests/run_tests.py
