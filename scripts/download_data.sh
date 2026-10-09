#!/usr/bin/env bash
# Thin wrapper; see scripts/download_data.py (stdlib only, pinned commit, SHA-256 verified).
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" scripts/download_data.py "$@"
