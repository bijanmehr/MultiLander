#!/usr/bin/env bash
# Serve the web frontend locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Moon Lander: http://localhost:8000"
python3 -m http.server 8000 --directory "$ROOT/web"
