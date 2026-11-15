#!/usr/bin/env bash
# sync_results.sh — pull experiment artifacts from the worker over SSH.
#
# Read-only on the remote: rsync pull of <remote-root>/artifacts/experiment-v1/
# into the LOCAL artifacts/experiment-v1/ (the same path serve_demo.py and
# run_once.sh read, so pulled results are immediately demo-able). Never
# uploads, never deletes (no --delete flag is ever passed; stock macOS
# openrsync has no --no-delete and does not delete by default). Tolerates
# transient SSH loss in --watch mode. BatchMode + ConnectTimeout and
# StrictHostKeyChecking=yes: the host key MUST already be trusted in
# known_hosts — verify the fingerprint on your first manual 'ssh ALIAS'
# login; an unknown key fails hard instead of being auto-accepted. No
# secrets or model blobs are transferred — only the run directory.
#
# This script never creates, rents, or terminates cloud resources.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/artifacts/experiment-v1"

SSH_ALIAS=""
WATCH=0
REMOTE_ROOT="~/onhacks"
LOCAL_ROOT="$DEFAULT_LOCAL_ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/sync_results.sh SSH_ALIAS [--watch] [--remote-root PATH] [--local-root PATH]

  SSH_ALIAS        ssh config alias of the worker (ports/users/keys live in ~/.ssh/config)
  --watch          re-pull every 60s until Ctrl-C; transient SSH loss is tolerated
  --remote-root    remote project root   (default: ~/onhacks)
  --local-root     local results dir     (default: <repo>/artifacts/experiment-v1)
  -h, --help       this help

Pulls <remote-root>/artifacts/experiment-v1/ -> <local-root>/ via
rsync -av --partial --timeout=300 (excludes *.part temp files; never deletes).
Host key must already be trusted (StrictHostKeyChecking=yes + BatchMode):
run 'ssh ALIAS' once interactively and verify the fingerprint first.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH=1; shift ;;
    --remote-root)
      [ $# -ge 2 ] || { echo "error: --remote-root requires a path" >&2; exit 2; }
      REMOTE_ROOT="$2"; shift 2 ;;
    --local-root)
      [ $# -ge 2 ] || { echo "error: --local-root requires a path" >&2; exit 2; }
      LOCAL_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    *)
      if [ -z "$SSH_ALIAS" ]; then SSH_ALIAS="$1"; shift
      else echo "error: unexpected argument '$1'" >&2; usage >&2; exit 2; fi ;;
  esac
done

if [ -z "$SSH_ALIAS" ]; then
  echo "error: SSH_ALIAS is required (see --help)" >&2
  usage >&2
  exit 2
fi
case "$SSH_ALIAS" in
  -*|*://*|*@*) echo "error: SSH_ALIAS must be a plain ~/.ssh/config alias, got '$SSH_ALIAS'" >&2; exit 2 ;;
esac

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10
      -o StrictHostKeyChecking=yes
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
SRC="${SSH_ALIAS}:${REMOTE_ROOT}/artifacts/experiment-v1/"
DST="$LOCAL_ROOT/"

mkdir -p "$LOCAL_ROOT"

pull() {
  # No --delete anywhere: local files are never removed. openrsync (stock
  # macOS) lacks --no-delete; absence of --delete flags is the guarantee.
  rsync -av --partial --timeout=300 \
      -e "$(printf '%q ' "${SSH[@]}")" \
      --exclude '*.part' --exclude 'worker.log.part' --exclude '.worker.lock' \
      "$SRC" "$DST"
}

report_rc() {
  if [ "$1" -eq 0 ]; then
    echo "[sync] ok -> $LOCAL_ROOT"
  else
    # Any nonzero (incl. 23 partial/vanished, 30 timeout, ssh failure) is a
    # real failure — never masked as success.
    echo "[sync] FAILED rc=$1 (BatchMode ssh to '$SSH_ALIAS'; host key must be trusted)" >&2
  fi
}

if [ "$WATCH" -eq 1 ]; then
  trap 'echo; echo "[watch] interrupted — exiting (no deletes were made)"; exit 130' INT TERM
  echo "[watch] every 60s: $SRC -> $DST"
  echo "[watch] Ctrl-C to stop. Failed pulls are retried; nothing local is ever deleted."
  while true; do
    rc=0
    pull || rc=$?
    report_rc "$rc"
    sleep 60
  done
else
  rc=0
  pull || rc=$?
  report_rc "$rc"
  exit "$rc"
fi
