#!/bin/bash
# Run the SGLang-path pure-Python unit tests.
#
# Uses the newest available python3.10+ interpreter (the repository code
# uses modern type syntax); falls back to any python3 on PATH.  No
# packages are installed; only the stdlib is used.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

PY=""
for candidate in \
    "$HOME/yanglihan/dl-stack/envs/qwen/bin/python3.12" \
    "$HOME/yanglihan/dl-stack/envs/sglang/bin/python3" \
    python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: no python3.10+ interpreter found" >&2
    exit 1
fi

echo "Using interpreter: $PY"
exec "$PY" "$HERE/run_tests.py"
