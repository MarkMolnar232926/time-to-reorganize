#!/usr/bin/env bash
# End-to-end reproduction: download data -> audit -> pipeline -> figures -> numbers.
# Stages are added as phases complete; every README number/figure must come from here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

scripts/download_data.sh
"$PY" scripts/audit_data.py
"$PY" -m reorg.cli losses --out outputs/losses.csv
"$PY" -m reorg.cli episodes
"$PY" scripts/phase1_prototype.py --no-anim
"$PY" scripts/sensitivity.py
"$PY" scripts/check_readme.py README.md
