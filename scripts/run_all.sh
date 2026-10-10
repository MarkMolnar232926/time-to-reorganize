#!/usr/bin/env bash
# End-to-end reproduction: download data -> audit -> pipeline -> analyses -> figures -> numbers.
# Every number and figure in the README is regenerated here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
WORKERS="${WORKERS:-4}"

scripts/download_data.sh
"$PY" scripts/audit_data.py                          # docs/audit_generated.md
"$PY" -m reorg.cli losses --out outputs/losses.csv   # possession losses (D-004)
"$PY" -m reorg.cli episodes                          # outputs/episodes.csv, outputs/teams.csv
"$PY" scripts/phase1_prototype.py --no-anim          # docs/gate1_generated.md, CIF figures
"$PY" scripts/sensitivity.py                         # docs/sensitivity_generated.md
"$PY" scripts/confirmatory.py --workers "$WORKERS"   # docs/confirmatory_generated.md (H1-H5)
"$PY" scripts/sensitivity_effects.py                 # docs/sensitivity_effects_generated.md
"$PY" -m reorg.cli team-cards                        # outputs/teams/*.png
"$PY" scripts/check_readme.py README.md
