PY ?= .venv/bin/python

.PHONY: setup data audit test lint all docker confirmatory figures cards

setup:            ## create .venv from the lockfile
	uv sync --extra dev

data:             ## download SkillCorner open data (pinned commit) into data/
	scripts/download_data.sh

audit:
	$(PY) scripts/audit_data.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

confirmatory:     ## H1-H5 per the frozen analysis plan
	$(PY) scripts/confirmatory.py --workers 4

figures:          ## README Figure 1 and Table 1 (needs confirmatory outputs)
	$(PY) scripts/make_figures.py

cards:            ## coach-facing team cards
	$(PY) -m reorg.cli team-cards

all:              ## full reproduction
	PYTHON=$(PY) scripts/run_all.sh

docker:
	docker build -t reorg . && docker run --rm reorg
