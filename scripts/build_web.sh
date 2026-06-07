#!/usr/bin/env bash
# Build the moonlander wheel into web/assets/ for the Pyodide frontend.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$ROOT/web/assets"

# Prefer the project venv's python; fall back to system python3.
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

mkdir -p "$ASSETS"
# Remove any stale wheels so the frontend always installs the fresh one.
rm -f "$ASSETS"/*.whl

"$PY" -m build --wheel --outdir "$ASSETS" "$ROOT"

echo "Built: $(ls "$ASSETS"/*.whl)"
