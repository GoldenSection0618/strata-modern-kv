#!/bin/bash
#SBATCH -J ylh-sglang-env-bootstrap
#SBATCH -p i80m512l40
#SBATCH --account=humx_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.out
#SBATCH --error=/share01/hpc/humxlab_intern/yanglihan/logs/%x-%j.err

# ============================================================================
# PLAN: bootstrap the dedicated SGLang environment (NON-DESTRUCTIVE).
# ============================================================================
# Target:  $HOME/yanglihan/dl-stack/envs/sglang
# Source:  read-only SGLang clone pinned at commit
#          7120f3ee13de565cc737e0598110e7f7603c4e9f
#          ($HOME/yanglihan/dl-stack/projects/sglang)
#
# Safety:
#   * Creates ONLY the new envs/sglang conda prefix.
#   * NEVER runs `conda create` over an existing prefix: if the target dir
#     already exists, the job FAILS without modifying it unless it can prove
#     the exact completed environment is already installed (python binary +
#     matching sglang_commit.txt + pip_freeze.txt + importable sglang).
#   * DRY RUN by default: RUN=0 prints the plan and exits 0.
#     Execute with:  RUN=1 sbatch bootstrap_sglang_env.sbatch
#   * Installation is NOT executed by this repo's workers; run this job
#     only when the user explicitly authorizes it.
#
# Provenance (no checksums): Git commit + `git status` of the pinned clone,
# installed sglang version, and a full `pip freeze` snapshot.
# ============================================================================

set -euo pipefail
source "$HOME/yanglihan/env.sh"

RUN="${RUN:-0}"
RESUME="${RESUME:-0}"
SGLANG_ENV_DIR="${SGLANG_ENV_DIR:-$DL_ROOT/envs/sglang}"
SGLANG_SOURCE="${SGLANG_SOURCE:-$DL_ROOT/projects/sglang}"
SGLANG_COMMIT="${SGLANG_COMMIT:-7120f3ee13de565cc737e0598110e7f7603c4e9f}"
PYTHON_VER="${PYTHON_VER:-3.12}"
PIP_CACHE="${PIP_CACHE:-$DL_ROOT/cache/pip}"
SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"

# --- provenance checks that run even in dry-run mode ----------------------
if [ ! -d "$SGLANG_SOURCE/.git" ]; then
    echo "ERROR: SGLang source clone not found at $SGLANG_SOURCE" >&2
    exit 1
fi
ACTUAL_COMMIT=$(git -C "$SGLANG_SOURCE" rev-parse HEAD)
if [ "$ACTUAL_COMMIT" != "$SGLANG_COMMIT" ]; then
    echo "ERROR: source clone is at $ACTUAL_COMMIT, expected $SGLANG_COMMIT" >&2
    exit 1
fi
echo "Source clone verified at pinned commit $SGLANG_COMMIT"
echo "Source clone git status (porcelain):"
git -C "$SGLANG_SOURCE" status --porcelain || true

# --- existing-prefix safety: never conda-create over an existing dir ------
CREATE_ENV=1
if [ -d "$SGLANG_ENV_DIR" ]; then
    echo "ERROR: target environment directory already exists: $SGLANG_ENV_DIR" >&2
    echo "  The bootstrap will NOT run 'conda create' over an existing prefix." >&2
    if [ -x "$SGLANG_ENV_DIR/bin/python" ] \
        && [ -f "$SGLANG_ENV_DIR/sglang_commit.txt" ] \
        && [ "$(cat "$SGLANG_ENV_DIR/sglang_commit.txt")" = "$SGLANG_COMMIT" ] \
        && [ -f "$SGLANG_ENV_DIR/pip_freeze.txt" ]; then
        if "$SGLANG_ENV_DIR/bin/python" -c "import sglang" >/dev/null 2>&1; then
            echo "  The exact completed environment is already installed" >&2
            echo "  (python + matching sglang_commit.txt + pip_freeze.txt +" >&2
            echo "  importable sglang). Exiting 0 WITHOUT modifying anything." >&2
            exit 0
        fi
        echo "  sglang_commit.txt matches but 'import sglang' FAILS in that" >&2
        echo "  python; refusing to touch the existing directory." >&2
        exit 1
    fi
    if [ "$RESUME" = "1" ] \
        && [ -x "$SGLANG_ENV_DIR/bin/python" ] \
        && [ -f "$SGLANG_ENV_DIR/conda-meta/history" ]; then
        echo "  RESUME=1: verified an incomplete conda prefix; continuing" >&2
        echo "  without running conda create or deleting any files." >&2
        CREATE_ENV=0
    else
        echo "  Existing dir does not prove the pinned environment is installed;" >&2
        echo "  refusing to modify it. For a verified interrupted bootstrap," >&2
        echo "  rerun with RUN=1 RESUME=1; no files will be deleted." >&2
        exit 1
    fi
fi

if [ "$RUN" != "1" ]; then
    echo ""
    echo "=== DRY RUN (no changes made) ==="
    echo "Would create conda env at:   $SGLANG_ENV_DIR"
    echo "Would install from:          $SGLANG_SOURCE (commit $SGLANG_COMMIT)"
    echo "Would record provenance in:  $SGLANG_ENV_DIR/provenance.json"
    echo "                             $SGLANG_ENV_DIR/sglang_commit.txt"
    echo "                             $SGLANG_ENV_DIR/pip_freeze.txt"
    echo ""
    echo "To actually execute: RUN=1 sbatch $0"
    echo "See $DL_ROOT/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck/code/envs/sglang/SGLANG_ENV_PLAN.md"
    exit 0
fi

# --- execution (authorized only) ------------------------------------------
module load miniforge3/24.11.2_1

mkdir -p "$DL_ROOT/cache"

if [ "$CREATE_ENV" = "1" ]; then
    echo "Creating conda env at $SGLANG_ENV_DIR (python=$PYTHON_VER) ..."
    conda create -y -p "$SGLANG_ENV_DIR" "python=$PYTHON_VER"
else
    echo "Resuming installation in verified prefix $SGLANG_ENV_DIR"
fi
eval "$(conda shell.bash hook)"
conda activate "$SGLANG_ENV_DIR"
PYTHON="$SGLANG_ENV_DIR/bin/python"

# Install pinned SGLang from the local clone.  sglang-kernel and other
# hard deps are pulled from the configured index (proxy from env.sh).
export PIP_CACHE_DIR="$PIP_CACHE"
export SGLANG_BUILD_RUST_EXTS
echo "SGLANG_BUILD_RUST_EXTS=$SGLANG_BUILD_RUST_EXTS"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e "$SGLANG_SOURCE/python"

# --- provenance recording (commit + git status + version + pip freeze) ----
export SGLANG_ENV_DIR SGLANG_SOURCE SGLANG_COMMIT
"$PYTHON" - <<'EOF'
import json, os, pathlib, subprocess, sys

env_dir = os.environ["SGLANG_ENV_DIR"]
source = os.environ["SGLANG_SOURCE"]
commit = os.environ["SGLANG_COMMIT"]

git_status = subprocess.run(
    ["git", "-C", source, "status", "--porcelain"],
    capture_output=True, text=True,
).stdout.strip() or "(clean)"

try:
    import sglang
    installed_version = sglang.__version__
except Exception as e:  # pragma: no cover
    installed_version = f"unavailable: {e}"

provenance = {
    "pinned_commit": commit,
    "source_clone": source,
    "source_git_status_porcelain": git_status,
    "installed_sglang_version": installed_version,
    "python": sys.version.split()[0],
    "sglang_build_rust_exts": os.environ.get("SGLANG_BUILD_RUST_EXTS"),
}
out = pathlib.Path(env_dir) / "provenance.json"
out.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
pathlib.Path(env_dir, "sglang_commit.txt").write_text(commit + "\n", encoding="utf-8")
print("Provenance written to", out)
EOF

"$PYTHON" -m pip freeze > "$SGLANG_ENV_DIR/pip_freeze.txt"

echo "=== post-install verification ==="
"$SGLANG_ENV_DIR/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__); print('torch', torch.__version__)"
echo "commit file: $(cat "$SGLANG_ENV_DIR/sglang_commit.txt")"
echo "Done. Environment ready at $SGLANG_ENV_DIR"
