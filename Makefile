# Verityne — one-command workflows.
#
#   make setup      install everything (backend venv + frontend deps)
#   make pipeline   build the corpus, calibrate, score, train, evaluate
#   make demo       seed the gauntlet + red-team pool, then run both servers
#   make dev        run backend and frontend together

SHELL := /bin/bash
PY    := .venv/bin/python
PIP   := .venv/bin/pip
export HF_HOME := $(CURDIR)/storage/models/hf
export PYTHONPATH := $(CURDIR)/backend

.PHONY: help setup setup-backend setup-frontend pipeline dataset benchmark calibrate score train evaluate \
        gauntlet redteam backend frontend dev demo test clean clean-data fresh

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: setup-backend setup-frontend ## install backend + frontend dependencies

setup-backend: ## create the venv and install Python dependencies
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip wheel setuptools
	$(PIP) install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -q -r backend/requirements.txt
	@echo "backend ready"

setup-frontend: ## install dashboard dependencies
	cd frontend && npm install --no-audit --no-fund

# ---------------------------------------------------------------- data + models
dataset: ## build the synthetic evaluation corpus (300 packets with video)
	$(PY) backend/scripts/build_dataset.py --packets 300 --with-video --clean

benchmark: ## measure every candidate deepfake checkpoint on our own data
	$(PY) backend/scripts/benchmark_models.py

calibrate: ## fit the face-match band, spectral head and generator fingerprint
	$(PY) backend/scripts/calibrate.py

score: ## run all detectors over the corpus and cache the raw scores
	$(PY) backend/scripts/score_corpus.py --workers 4

train: ## fit the fusion layer on the training split
	$(PY) backend/scripts/train_fusion.py

evaluate: ## produce the held-out evaluation report
	$(PY) backend/scripts/evaluate.py

pipeline: dataset benchmark calibrate score train evaluate ## the whole data + model pipeline
	@echo "pipeline complete — see eval/metrics.json"

gauntlet: ## load the 10 genuine + 10 fraudulent demo fixtures
	$(PY) backend/scripts/seed_gauntlet.py --real 10 --fake 10

redteam: ## generate a held-out pool of unseen attacks for the live red-team button
	$(PY) backend/scripts/redteam_generate.py --count 12

# ---------------------------------------------------------------- running
backend: ## run the API on :8000
	.venv/bin/uvicorn verityne.main:app --host 0.0.0.0 --port 8000 --app-dir backend

frontend: ## run the dashboard on :3000
	cd frontend && npm run dev

dev: ## run both, streaming logs from each
	@trap 'kill 0' EXIT; \
	$(MAKE) backend & \
	$(MAKE) frontend & \
	wait

demo: gauntlet redteam ## seed demo data, then run everything
	@$(MAKE) dev

test: ## run the backend test suite
	.venv/bin/python -m pytest backend/tests -q

# ---------------------------------------------------------------- housekeeping
clean: ## remove generated heatmaps, uploads and the database
	rm -rf storage/uploads/* storage/heatmaps/* storage/verityne.db

clean-data: ## also remove the corpus and evaluation artefacts (keeps downloaded models)
	rm -rf datasets/corpus datasets/redteam datasets/manifest.json eval/*.json storage/models/*.joblib

fresh: clean clean-data pipeline gauntlet redteam ## nuke everything and rebuild end to end
