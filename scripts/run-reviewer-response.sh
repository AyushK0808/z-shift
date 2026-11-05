#!/usr/bin/env bash
# Every experiment added for the reviewers' second round, in one command.
#
#   bash scripts/run-reviewer-response.sh
#   bash scripts/run-reviewer-response.sh --only "b12 b13"
#   bash scripts/run-reviewer-response.sh --skip-tests
#
# From Windows (the repo's scripts run inside WSL2):
#   wsl.exe -d Ubuntu --cd /mnt/c/Users/theof/projects/personal/z-shift --exec bash scripts/run-reviewer-response.sh
#
#   tests  the unit tests, which include the backend contract test
#   b10    refinement on the real DTU meshes B6 saved: object isolation, DTU normal
#          consistency, and both component filters timed with repeats  (CPU+GPU)
#   b11    CO3D teddybear scored against CO3D's reference cloud, building the
#          8- and 24-view runs first if they are missing                (GPU)
#   b12    synchronisation on the real DTU rig images with injected clocks (CPU)
#   b13    DUSt3R as a second reconstruction backend on the five DTU scans (GPU)
#
# Needs what run-review-experiments.sh sets up: the environment and the DTU scans
# under $ZSHIFT_HOME, and B6's outputs for B10. Results land in bench/results/.
# Each experiment is independent: one failing does not stop the rest, and the
# script exits non-zero at the end if any did.
#
# Time on an 8 GB laptop GPU: about 1.5 h with the default budget below, dominated
# by B10's split_bodies timing, which is capped per scene (SPLIT_BUDGET_S, default
# 1800 s) because that filter took half an hour on one real mesh. B13 downloads
# the DUSt3R checkpoint (~2.3 GB) on first use.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZSHIFT_HOME="${ZSHIFT_HOME:-$HOME/zshift}"
PY="${UV_PROJECT_ENVIRONMENT:-$ZSHIFT_HOME/venv}/bin/python"
SPLIT_BUDGET_S="${SPLIT_BUDGET_S:-1800}"
ONLY="tests b10 b11 b12 b13"
SKIP_TESTS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h | --help) sed -n 2,26p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ "$SKIP_TESTS" -eq 1 ] && ONLY="${ONLY/tests/}"

if [ ! -x "$PY" ]; then
  echo "no environment at $PY; run scripts/run-review-experiments.sh once first" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ZSHIFT_GT_CACHE_DIR="${ZSHIFT_GT_CACHE_DIR:-$ZSHIFT_HOME/tier_b/gt_cache}"
MANIFEST="$ZSHIFT_HOME/dtu/manifest.json"
LOGS="$ZSHIFT_HOME/tier_b/logs"
mkdir -p "$LOGS" "$ZSHIFT_GT_CACHE_DIR"
cd "$REPO_ROOT"

failed=()
step() { printf '\n==> %s  [%s]\n' "$1" "$(date +%H:%M:%S)"; }
wants() { [[ " $ONLY " == *" $1 "* ]]; }
# Run a command, log it, and remember a failure without stopping.
attempt() {
  local name="$1"
  shift
  # --line-buffered: grep holds its output in 4 KB blocks when it writes to a pipe, which
  # leaves the console and the log empty for minutes at a time. tr turns tqdm's carriage
  # returns into newlines so progress bars cannot swallow whole lines.
  "$@" 2>&1 | tr '\r' '\n' \
    | grep --line-buffered -v -E "autocast|^ +@|FutureWarning|it/s" \
    | tee -a "$LOGS/response_$name.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "!! $name failed (log: $LOGS/response_$name.log)"
    failed+=("$name")
    return 1
  fi
}

if wants tests; then
  step "unit tests"
  attempt tests "$PY" -m pytest tests -q -p no:cacheprovider
fi

if wants b10; then
  step "B10 refinement on the real DTU meshes"
  if [ -d "$ZSHIFT_HOME/tier_b/full/outputs/b6" ]; then
    attempt b10 "$PY" -u -m bench.exp_b10_real_mesh_refinement \
      --manifest "$MANIFEST" --outputs "$ZSHIFT_HOME/tier_b/full/outputs/b6" \
      --split-budget-s "$SPLIT_BUDGET_S" -v
  else
    echo "!! no B6 outputs under $ZSHIFT_HOME/tier_b/full/outputs/b6; run scripts/run-review-experiments.sh"
    failed+=(b10)
  fi
fi

if wants b11; then
  step "B11 CO3D teddybear accuracy"
  # B9's runs hold the raw mesh and point cloud B11 scores. Build whichever is
  # missing, skipping split_bodies, which only matters for Table II's timing.
  [ -f "$ZSHIFT_HOME/case_study/run/mesh.glb" ] || attempt b9_8 "$PY" -u -m bench.exp_b9_case_study \
    --data-dir "$ZSHIFT_HOME/co3d/teddybear" --n-images 8 --filters labels \
    --output-root "$ZSHIFT_HOME/case_study" --results-dir "$ZSHIFT_HOME/tier_b/scratch"
  [ -f "$ZSHIFT_HOME/case_study_24/run/mesh.glb" ] || attempt b9_24 "$PY" -u -m bench.exp_b9_case_study \
    --data-dir "$ZSHIFT_HOME/co3d/teddybear" --n-images 24 --filters labels --isolate-object \
    --output-root "$ZSHIFT_HOME/case_study_24" --results-dir "$ZSHIFT_HOME/tier_b/scratch"
  attempt b11 "$PY" -u -m bench.exp_b11_co3d_accuracy \
    --data-dir "$ZSHIFT_HOME/co3d/teddybear" \
    --runs "views8=$ZSHIFT_HOME/case_study/run" "views24=$ZSHIFT_HOME/case_study_24/run" -v
fi

if wants b12; then
  step "B12 synchronisation on the real DTU rig"
  attempt b12 "$PY" -u -m bench.exp_b12_dtu_sync --manifest "$MANIFEST" -v
fi

if wants b13; then
  step "B13 DUSt3R as a second backend"
  attempt b13 "$PY" -u -m bench.exp_b13_backend_swap --manifest "$MANIFEST" \
    --output-root "$ZSHIFT_HOME/tier_b/full/outputs/b13" -v
fi

step "summary"
"$PY" -m bench.summarize_response

if [ "${#failed[@]}" -gt 0 ]; then
  echo
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo
echo "done. Results: $REPO_ROOT/bench/results/{b10,b11,b12,b13}_*.csv"
