#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Prefer the local virtual environment rather than the system interpreter.
if [[ -x .venv/bin/python ]]; then
  ATLAS_PYTHON="$PWD/.venv/bin/python"
else
  ATLAS_PYTHON="python3"
fi
if ! "$ATLAS_PYTHON" -c 'import numpy' >/dev/null 2>&1; then
  echo "NumPy is not available to $ATLAS_PYTHON."
  echo 'Install: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt'
  exit 2
fi
case "${1:-gui}" in
 gui|start) shift 2>/dev/null || true; echo 'Opening http://127.0.0.1:8765'; (sleep 1; xdg-open http://127.0.0.1:8765 >/dev/null 2>&1 || true) & exec "$ATLAS_PYTHON" -m atlas.cli serve "$@" ;;
 serve|train|generate|simulate|study-real|diagnose|import-file|migrate-from|logs|report|probe|doctor|study|demonstrate|respond|ask|learn-csv|predict-equation|compose-equations) exec "$ATLAS_PYTHON" -m atlas.cli "$@" ;;
 *) echo 'Usage: ./launch.sh [gui|serve|train|generate|simulate|study-real|diagnose|import-file|migrate-from|logs|report|probe|doctor|study|demonstrate|respond|ask|learn-csv|predict-equation|compose-equations]'; exit 2;;
esac
