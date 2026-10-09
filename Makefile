PY ?= .venv/bin/python

.PHONY: setup data audit test lint all docker

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

all:              ## full reproduction
	PYTHON=$(PY) scripts/run_all.sh

docker:
	docker build -t reorg . && docker run --rm reorg
