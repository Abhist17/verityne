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
        ablate gauntlet redteam backend frontend dev demo test clean clean-data fresh \
        real data-real calibrate-face real-docs eval-real-docs eval-real-video \
        indian-faces calibrate-linkage

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: setup-backend setup-frontend ## install backend + frontend dependencies

setup-backend: ## create the venv and install Python dependencies
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip wheel setuptools
	$(PIP) install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -q -r backend/requirements.txt
	# facenet-pytorch's declared pins are stale; see requirements-nodeps.txt.
	$(PIP) install -q --no-deps -r backend/requirements-nodeps.txt
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

ablate: ## what each detector is worth, and whether the corpus leaks its labels
	$(PY) backend/scripts/ablate_fusion.py

pipeline: dataset benchmark calibrate score train evaluate ablate ## the whole data + model pipeline
	@echo "pipeline complete — see eval/metrics.json and eval/ablation.json"

# ---------------------------------------------------------------- real data
# The synthetic corpus measures separability on data we generated. These targets
# measure the same detectors on data we did not: real photographs of real people,
# and identity documents that were physically printed and captured. See the
# "Measured on real data" section of the README.

data-real: ## download the public real datasets (LFW ~233 MB; MIDV-2020 must be fetched manually)
	$(PY) -c "from pathlib import Path; import sys; sys.path.insert(0,'backend/scripts'); \
	from calibrate_face_match_lfw import ensure_dataset; ensure_dataset(Path('datasets/lfw/lfw_home'))"
	@echo
	@echo "The Indian-faces shard (~430 MB) is fetched on demand by \`make indian-faces\`."
	@echo
	@echo "MIDV-2020 is not scriptable — it has no stable direct link. Fetch scan_upright.tar"
	@echo "and photo.tar from ftp://smartengines.com/midv-2020/ into datasets/midv2020/, then:"
	@echo "  mkdir -p datasets/midv2020/scan datasets/midv2020/photo"
	@echo "  tar -xf datasets/midv2020/scan_upright.tar -C datasets/midv2020/scan"
	@echo "  tar -xf datasets/midv2020/photo.tar      -C datasets/midv2020/photo"

calibrate-face: ## fit the identity threshold on LFW's 6,000 real pairs and apply it
	$(PY) backend/scripts/calibrate_face_match_lfw.py --apply

calibrate-linkage: ## fit the linkage threshold as a SEARCH, not a pair (needs LFW)
	$(PY) backend/scripts/calibrate_linkage_lfw.py --apply

real-docs: ## build the tamper set from real captured MIDV-2020 documents
	$(PY) backend/scripts/build_real_docs.py --per-capture 250

eval-real-docs: ## score ID forensics on real documents → eval/real_docs.json
	$(PY) backend/scripts/evaluate_real_docs.py

eval-real-video: ## score liveness on recorded deepfakes → eval/real_video.json
	@echo "Point --data at an extracted FaceForensics++, Celeb-DF v2 or DFDC-preview tree."
	@echo "FF++ and Celeb-DF are gated behind a signed request form; nothing here downloads them."
	$(PY) backend/scripts/evaluate_real_video.py --data $(or $(DATA),datasets/faceforensics)

indian-faces: ## is the selfie detector reading demography? real Indian faces vs FFHQ
	$(PY) backend/scripts/evaluate_indian_faces.py --n $(or $(N),300) --shards $(or $(SHARDS),1)

real: calibrate-face calibrate-linkage real-docs eval-real-docs indian-faces ## the real-data track that needs no gated access
	@echo "real-data track complete — see eval/face_match_lfw.json, eval/real_docs.json"
	@echo "and eval/indian_faces.json, eval/linkage_lfw.json"

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
