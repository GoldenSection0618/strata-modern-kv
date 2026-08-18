#!/bin/bash
#SBATCH -J ylh-hcv-exp1
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
# Experiment 1 cell: Baseline Benefit.
#   sbatch --export=MODEL=qwen,ARCH=gpu_only,STATE=cold run_exp1.sbatch
# The submit script alternates architecture order across cells so machine
# drift does not systematically favor one architecture.
# ============================================================================
set -euo pipefail
source "$HOME/yanglihan/env.sh"
source "${HCV_CODE_DIR:-$HOME/yanglihan/dl-stack/projects/strata-modern-kv-hcache/experiments/hierarchical-cache-value/code}/sbatch/common_env.sh"

MODEL="${MODEL:-qwen}"                     # qwen | gemma
ARCH="${ARCH:-gpu_only}"                   # gpu_only | hierarchical
STATE="${STATE:-cold}"                     # cold | warm
SMOKE="${SMOKE:-0}"
N_REPEATS="${N_REPEATS:-}"
CONCURRENCY="${CONCURRENCY:-}"
RUN_TAG="${RUN_TAG:-run-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID:-local}}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

case "$ARCH" in gpu_only|hierarchical) ;; *) echo "ERROR: ARCH=$ARCH" >&2; exit 2 ;; esac
case "$STATE" in cold|warm) ;; *) echo "ERROR: STATE=$STATE" >&2; exit 2 ;; esac
case "$MODEL" in qwen|gemma) ;; *) echo "ERROR: MODEL=$MODEL" >&2; exit 2 ;; esac

echo "== exp1 | MODEL=$MODEL ARCH=$ARCH STATE=$STATE SMOKE=$SMOKE =="

if [ -n "$CONCURRENCY" ]; then export HCV_CONCURRENCY="$CONCURRENCY"; fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    cd "$CODE_DIR"
    "$PYTHON_BIN" -m hcv.dry_run \
        --config "$CONFIG_PATH/exp1.json" \
        --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
        --model "$MODEL" --experiment exp1 --log-dir "$LOG_DIR" \
        --architecture "$ARCH" --initial-state "$STATE" \
        $( [ "$SMOKE" = "1" ] && echo --smoke )
    exit $?
fi

"$PYTHON_BIN" -m hcv.provenance check --env-dir "$SGLANG_ENV_DIR" --commit "$SGLANG_COMMIT"
echo "== preflight =="
"$PYTHON_BIN" -m hcv.preflight || { echo "PREFLIGHT FAILED" >&2; exit 1; }

cd "$CODE_DIR"
"$PYTHON_BIN" -m hcv.run_exp1 \
    --config "$CONFIG_PATH/exp1.json" \
    --results-root "$RESULTS_ROOT" --run-tag "$RUN_TAG" \
    --model "$MODEL" --experiment exp1 --log-dir "$LOG_DIR" \
    --architecture "$ARCH" --initial-state "$STATE" \
    --ready-timeout "$READY_TIMEOUT" \
    $( [ -n "$N_REPEATS" ] && echo --n-repeats "$N_REPEATS" ) \
    $( [ "$SMOKE" = "1" ] && echo --smoke )

echo "== exp1 cell done: $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
