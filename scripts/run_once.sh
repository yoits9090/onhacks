#!/usr/bin/env bash
# run_once.sh — one-shot worker launcher for the frozen Chipoint experiment.
#
# DEFAULT (no --execute / --dry-run): prints the runnable stage commands and
# exits. Performs NO installs, downloads, model loads, or file writes.
#
# --execute: Linux + CUDA instance ONLY. Runs the frozen stage list once
# under a single wall-clock budget = min(experiment_config.json
# max_worker_minutes, deadline_utc - now), wrapped in GNU timeout
# (TERM, KILL after 30s) around the ENTIRE setup+download+experiment.
#
# The timeout kills THIS worker process tree only — it does NOT stop
# provider billing and is NOT a spending cap. Pod termination is a separate,
# explicitly approved step (see LAUNCH_PLAN.md). This script never creates,
# rents, or terminates cloud resources.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROOT="$DEFAULT_ROOT"
MODE="dry-run"
MIN_FREE_KIB=$((65 * 1024 * 1024))  # 65 GiB conservative floor (200 GB planned)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_once.sh [--dry-run|--execute] [--root PATH]

  (default)          print the stage commands; no side effects
  --dry-run          same as default
  --execute          run the stages once on THIS machine (Linux+CUDA only)
  --root PATH        project root (default: repo containing this script)
  -h, --help         this help

Stages executed by --execute, in order, all inside ONE wall-clock budget:
  1. bash scripts/setup_instance.sh
  2. .venv/bin/python scripts/fetch_assets.py --galleries --dinov3-weights --towers so400m256 gopt dinov3
  3. .venv/bin/python scripts/run_experiment.py all --root ROOT --config ROOT/experiment_config.json --run-dir ROOT/artifacts/experiment-v1
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --execute) MODE="execute"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    --root)
      [ $# -ge 2 ] || { echo "error: --root requires a path" >&2; exit 2; }
      ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if [ ! -d "$ROOT" ]; then
  echo "error: --root '$ROOT' is not a directory" >&2
  exit 2
fi
ROOT="$(cd "$ROOT" && pwd)"
CONFIG="$ROOT/experiment_config.json"
RUN_DIR="$ROOT/artifacts/experiment-v1"
LOG="$RUN_DIR/worker.log"
LOCK="$RUN_DIR/.worker.lock"
REPORT="$RUN_DIR/report.json"
EXAMPLES="$RUN_DIR/demo_examples/examples.json"

# A finished run needs BOTH the report and the exported demo examples.
# report-only means evaluate finished but export was interrupted -> repair.
INCOMPLETE=0
if [ -f "$REPORT" ] && [ -f "$EXAMPLES" ]; then
  echo "[run_once] experiment already completed: $REPORT and $EXAMPLES exist."
  echo "[run_once] nothing will be installed, downloaded, or run."
  echo "[run_once] results are already local (demo: scripts/serve_demo.py)."
  echo "[run_once] to re-pull from the worker: bash scripts/sync_results.sh <SSH_ALIAS>"
  exit 0
elif [ -f "$REPORT" ]; then
  INCOMPLETE=1
fi

if [ "$MODE" != "execute" ]; then
  if [ "$INCOMPLETE" = 1 ]; then
    cat <<EOF
[dry-run] $REPORT exists but $EXAMPLES is missing — the run finished
evaluate but not export (interrupted?). Only the CPU export stage is needed:
cd "$ROOT"
.venv/bin/python scripts/run_experiment.py export --root "$ROOT" --config "$ROOT/experiment_config.json" --run-dir "$ROOT/artifacts/experiment-v1"
[dry-run] re-run with --execute to perform this export-only repair
(no install, no download, no GPU required — needs .venv + data/ archive).
EOF
    exit 0
  fi
  cat <<EOF
[dry-run] no --execute: nothing will be installed, downloaded, or written.
[dry-run] on the Linux CUDA instance, run:
cd "$ROOT"
bash scripts/setup_instance.sh
.venv/bin/python scripts/fetch_assets.py --galleries --dinov3-weights --towers so400m256 gopt dinov3
.venv/bin/python scripts/run_experiment.py all --root "$ROOT" --config "$ROOT/experiment_config.json" --run-dir "$ROOT/artifacts/experiment-v1"
[dry-run] or re-run this script there with --execute (single worker, wall-clock capped).
EOF
  exit 0
fi

# ---- --execute: export-only repair (CPU, offline, no GPU gates) -----------
if [ "$INCOMPLETE" = 1 ]; then
  if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "error: $ROOT/.venv missing — the export repair needs the project" >&2
    echo "       venv. Run 'bash scripts/setup_instance.sh' on the host first" >&2
    echo "       (this script does not auto-install)." >&2
    exit 1
  fi
  if [ ! -f "$CONFIG" ]; then
    echo "error: config not found: $CONFIG" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR"
  # Best-effort single-worker lock: export is idempotent, so a missing flock
  # (e.g. macOS) warns instead of blocking recovery.
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK"
    if ! flock -n 9; then
      echo "error: another worker holds $LOCK; refusing a concurrent run." >&2
      exit 1
    fi
  else
    echo "[run_once] note: flock unavailable; continuing without worker lock"
  fi
  exec > >(tee -a "$LOG") 2>&1
  echo "[run_once] export-only repair: report.json present, examples.json missing."
  echo "[run_once] running: run_experiment.py export (CPU, offline)"
  rc=0
  (cd "$ROOT" && .venv/bin/python scripts/run_experiment.py export \
      --root "$ROOT" --config "$CONFIG" --run-dir "$RUN_DIR") || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[run_once] export repair complete; demo examples now present."
  else
    echo "[run_once] export repair FAILED rc=$rc — inspect $LOG."
  fi
  exit "$rc"
fi

# ---- --execute gates (all before any install/download/model load) ---------

if [ "$(uname -s)" != "Linux" ]; then
  echo "error: --execute is for the Linux CUDA instance only (detected $(uname -s))." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "error: nvidia-smi missing or no GPU visible; --execute requires a CUDA host." >&2
  exit 1
fi
echo "[run_once] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

if [ ! -f "$CONFIG" ]; then
  echo "error: config not found: $CONFIG" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 required to read $CONFIG" >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "error: GNU timeout (coreutils) required for the wall-clock cap." >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "error: flock (util-linux) required for the single-worker lock." >&2
  exit 1
fi

# Wall-clock budget = min(max_worker_minutes, deadline_utc - now); refuse elapsed.
read -r BUDGET_SEC REMAIN_SEC <<<"$(python3 - "$CONFIG" <<'PY'
import json, sys
from datetime import datetime, timezone
cfg = json.load(open(sys.argv[1]))
now = datetime.now(timezone.utc)
deadline = datetime.fromisoformat(cfg["deadline_utc"])
remain = (deadline - now).total_seconds()
budget = min(remain, float(cfg.get("max_worker_minutes", 180)) * 60)
print(int(max(0, budget)), int(max(0, remain)))
PY
)"
if [ "${BUDGET_SEC:-0}" -le 0 ]; then
  echo "error: experiment deadline/budget already elapsed (remain=${REMAIN_SEC}s); refusing to start." >&2
  exit 1
fi
echo "[run_once] wall-clock budget: ${BUDGET_SEC}s (deadline in ${REMAIN_SEC}s)"

AVAIL_KIB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if [ -z "$AVAIL_KIB" ] || [ "$AVAIL_KIB" -lt "$MIN_FREE_KIB" ]; then
  echo "error: only ${AVAIL_KIB:-unknown} KiB free under $ROOT; need >=65 GiB" >&2
  echo "       (200 GB planned: ~29 GB model assets + torch/cache + ~2.3 GB dataset)." >&2
  exit 1
fi
echo "[run_once] disk: $((AVAIL_KIB / 1024 / 1024)) GiB free (need >=65)"

# Single worker per run-dir; lock auto-releases when this script exits.
mkdir -p "$RUN_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "error: another worker holds $LOCK; refusing a concurrent run." >&2
  exit 1
fi
echo "[run_once] worker lock held: $LOCK"

# Everything from here (gates output above excluded) goes to worker.log too.
exec > >(tee -a "$LOG") 2>&1
echo "[run_once] logging stdout+stderr -> $LOG (append)"
echo "======================================================================"
echo " WARNING: the ${BUDGET_SEC}s timeout below kills THIS worker process"
echo " tree only. It does NOT stop provider billing and is NOT a spending"
echo " cap. Terminate the pod separately after export is verified."
echo "======================================================================"

# Frozen stage list — no design changes; run_experiment.py's own gates cover
# encoding sanity, so no separate smoke stage is repeated here.
worker_stages() {
  set -euo pipefail
  cd "$ONHACKS_ROOT"
  echo "=== [1/3] setup_instance.sh ==="
  bash scripts/setup_instance.sh   # ends with a torch CUDA check (post-install)
  echo "=== [2/3] fetch_assets.py --galleries --dinov3-weights ==="
  .venv/bin/python scripts/fetch_assets.py --galleries --dinov3-weights \
      --towers so400m256 gopt dinov3
  echo "=== [3/3] run_experiment.py all ==="
  .venv/bin/python scripts/run_experiment.py all \
      --root "$ONHACKS_ROOT" \
      --config "$ONHACKS_ROOT/experiment_config.json" \
      --run-dir "$ONHACKS_ROOT/artifacts/experiment-v1"
}
export -f worker_stages

rc=0
ONHACKS_ROOT="$ROOT" timeout --signal=TERM --kill-after=30 "$BUDGET_SEC" \
    bash -c 'worker_stages' || rc=$?

if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "[run_once] TIMEOUT: worker exceeded ${BUDGET_SEC}s and was killed."
  echo "[run_once] Provider billing continues until the pod is terminated."
elif [ "$rc" -ne 0 ]; then
  echo "[run_once] worker FAILED rc=$rc — inspect $LOG; completed cache batches may resume."
else
  echo "[run_once] worker completed all stages (rc=0)."
fi
echo "[run_once] pull results back: bash scripts/sync_results.sh <SSH_ALIAS> [--watch]"
exit "$rc"
