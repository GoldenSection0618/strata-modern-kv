#!/bin/bash
# Lightweight submit for deterministic processing and unit tests.
#   bash submit_analysis.sh     # raw -> processed -> results tables
#   bash submit_tests.sh        # pure-Python unit tests (compute node)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HCV_CODE_DIR="$(dirname "$HERE")"

case "${1:-}" in
  analysis)  sbatch --export=ALL,HCV_CODE_DIR "$HERE/run_analysis.sbatch" ;;
  tests)     sbatch --export=ALL,HCV_CODE_DIR "$HERE/run_tests.sbatch" ;;
  *)
    echo "usage: bash submit_analysis.sh (analysis|tests)"
    exit 2
    ;;
esac
