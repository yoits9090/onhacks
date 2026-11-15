#!/usr/bin/env bash
# Compatibility wrapper -> scripts/fetch_assets.py "$@"
#
# NOTE: the old behavior also downloaded OSV-5M benchmark images/datasets.
# That is intentionally REMOVED — benchmark data is not part of the
# street-arm asset fetch and must come from an authored manifest
# (see EVALUATION.md 'Data Separation'). This wrapper now only forwards
# explicit fetch_assets.py flags, e.g.:
#   ./scripts/fetch_data.sh --galleries --dinov3-weights --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
exec "$PY" scripts/fetch_assets.py "$@"
