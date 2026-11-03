#!/usr/bin/env bash
# Tier B GPU runs the reviewers asked for, on a local CUDA machine (Linux or WSL2).
#
#   bash scripts/run-review-experiments.sh
#   bash scripts/run-review-experiments.sh --lane smoke
#   bash scripts/run-review-experiments.sh --skip-setup --scans "24 37 40"
#
# 1. installs uv if missing, then `uv sync --dev` and the pinned MASt3R
#    checkout -- in that order, because `uv sync` removes the editable
#    mast3r/dust3r installs
# 2. fetches the DTU scans by HTTP byte range (~0.25 GB a scan)
# 3. runs bench.run_tier_b: a smoke lane on 8 frames of the first scan, then
#    B2, B4 and B6 once per scan, merging results into bench/results/*.csv
#
# Safe to re-run after an interruption: finished scans are skipped. The
# environment, data, logs and run outputs live under $ZSHIFT_HOME
# (default ~/zshift); only the merged CSVs are written into the repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZSHIFT_HOME="${ZSHIFT_HOME:-$HOME/zshift}"

SCANS="24 37 40 55 63"
EXPERIMENTS="b2 b4 b6"
LANE="all"
SKIP_SETUP=0
RUN_ARGS=()

usage() {
  cat << 'EOF'
usage: run-review-experiments.sh [--scans "24 37 40"] [--experiments "b2 b4 b6"]
                                 [--lane all|smoke|full] [--skip-setup]
                                 [--with-scale] [--force] [--preflight-only]
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --scans) SCANS="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --lane) LANE="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --with-scale | --force | --preflight-only) RUN_ARGS+=("$1"); shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# The environment lives on the Linux filesystem, not in the repo: under WSL
# the repo sits on /mnt/c, shared with Windows (whose .venv is a different
# platform's environment), and /mnt/c I/O is several times slower than ext4.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$ZSHIFT_HOME/venv}"
# `uv pip install` (used by setup-mast3r.sh) targets VIRTUAL_ENV, not the
# project environment.
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
# The uv cache and the environment can be on different filesystems.
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

step() { printf '\n==> %s\n' "$*"; }

mkdir -p "$ZSHIFT_HOME"

if ! command -v uv > /dev/null; then
  step "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ "$SKIP_SETUP" -eq 0 ]; then
  step "uv sync --dev  ->  $UV_PROJECT_ENVIRONMENT"
  (cd "$REPO_ROOT" && uv sync --dev)
  step "MASt3R checkout + editable install"
  PYTHON="$PY" bash "$REPO_ROOT/scripts/setup-mast3r.sh"
fi
if [ ! -x "$PY" ]; then
  echo "no environment at $PY; run without --skip-setup" >&2
  exit 1
fi

cd "$REPO_ROOT"

step "fetch DTU scans: $SCANS"
# shellcheck disable=SC2086 # SCANS is a space-separated list on purpose
"$PY" -m bench.fetch_dtu --dest "$ZSHIFT_HOME/dtu" --scans $SCANS

step "Tier B ($LANE lane: $EXPERIMENTS)"
# shellcheck disable=SC2086 # EXPERIMENTS is a space-separated list on purpose
"$PY" -m bench.run_tier_b \
  --manifest "$ZSHIFT_HOME/dtu/manifest.json" \
  --work-dir "$ZSHIFT_HOME/tier_b" \
  --lane "$LANE" \
  --experiments $EXPERIMENTS \
  "${RUN_ARGS[@]}"

step "done"
echo "Merged CSVs: $REPO_ROOT/bench/results"
echo "Logs:        $ZSHIFT_HOME/tier_b/logs"
echo "Figures:     $PY -m bench.plots"
