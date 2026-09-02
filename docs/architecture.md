[← Verityne](../README.md)

> How a packet moves through the system, what the dashboard shows, and where every file lives.

# Architecture

```
                       ┌──────────────────────────────────────┐
  selfie.jpg ─────────▶│                                      │
  id_document.jpg ────▶│   POST /verify   (FastAPI)           │
  liveness.mp4 ───────▶│                                      │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  stage one - run concurrently        │
                       │  ┌────────────┐ ┌─────────────────┐  │
                       │  │ selfie     │ │ id forensics    │  │
                       │  │ deepfake   │ │ (OCR + ELA +    │  │
                       │  └────────────┘ │  structure)     │  │
                       │  ┌────────────┐ └─────────────────┘  │
                       │  │ liveness   │ ┌─────────────────┐  │
                       │  │ video      │ │ metadata / EXIF │  │
                       │  └────────────┘ └─────────────────┘  │
                       │  ┌──────────────────────────────────┐│
                       │  │ behavioral biometrics            ││
                       │  │ (keystroke · pointer · locale)   ││
                       │  └──────────────────────────────────┘│
                       │        shared cache: decoded images, │
                       │        face crops, embeddings        │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  stage two - needs both face crops   │
                       │  face match (selfie vs ID portrait)  │
                       └──────────────────┬───────────────────┘
                                          │
              cross-cutting ──────────────┤
              • linkage (same face / same file, prior submissions)
              • generator fingerprint (which model made this fake)

  form telemetry ─────▶ POST /behavioral ─▶ features ─▶ detector 6
  (posted before the files, bound by token at /verify)
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  fusion: logistic regression over    │
                       │  10 features → Platt calibration     │
                       │  then max() with the un-modelled     │
                       │  evidence channels, each ceilinged   │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  policy: per-merchant thresholds     │
                       │  + abstention band                   │
                       │  → ACCEPT / REVIEW / REJECT          │
                       │  + explanation + heatmaps + audit    │
                       └──────────────────────────────────────┘
```

Stage one runs on a `ThreadPoolExecutor`, not `asyncio.gather`: torch and OpenCV
release the GIL inside their C extensions, so threads genuinely overlap. Async
over CPU-bound work would have bought nothing.

---

## The dashboard

Next.js 14 App Router, seven pages:

| Page | What it does |
| --- | --- |
| **Live Verify** (`/`) | Drag in a selfie, ID and liveness clip; get the verdict, the three reasons, the heatmaps and the per-detector breakdown. Also collects the form-fill telemetry Detector 6 reads, and says on screen that it is doing so. |
| **Gauntlet** (`/gauntlet`) | Runs 10 genuine + 10 fraudulent fixtures over server-sent events, scoring live. |
| **Metrics** (`/metrics`) | The held-out report rendered - ROC, per-attack recall, bias audit, Detector 6's evaluation, and the cost-of-friction curve with operator-tunable ₹ sliders. |
| **Corrections** (`/corrections`) | The audit trail: every belief measured and lost, what it cost, and which are still open. Each entry renders the evidence file and path its numbers came from. |
| **Attack Gallery** (`/attacks`) | Rejected submissions grouped by attack pattern, with the evidence that flagged each. |
| **Threat Intelligence** (`/threat`) | Fraud rings as a node-link graph, generator-fingerprint mix, attack-pattern counts and a live feed. Clicking a node filters the feed to that cluster. |
| **Review Queue** (`/review`) | Human-in-the-loop: everything that abstained, with accept/reject and an audit note. Tracks how often analysts agree with the model. |

---

## Repository layout

```
backend/
  verityne/
    detectors/        the six detectors + model loading, generator fingerprinting
      behavioral.py     detector 6: keystroke, pointer and locale features + rules
    utils/            ELA, spectral, OCR, Verhoeff, Grad-CAM, hashing, provenance
    api/              route modules: verify, behavioral, gauntlet, metrics, ops
    pipeline.py       orchestration (ThreadPool - torch and OpenCV release the GIL,
                      asyncio.gather over CPU-bound work would buy nothing)
    fusion.py         logistic regression + calibration + policy decisions
    explain.py        attack classification and natural-language narration
    linkage.py        cross-submission face and asset lookup
    config.py         paths, env, per-merchant policy loading
    policy.yaml       merchant thresholds, hot-reloadable
    db.py             SQLAlchemy models: submissions, detector results, telemetry, audit log
    schemas.py        pydantic request/response contracts
    main.py           FastAPI app, CORS, timing middleware, static mounts
  scripts/            dataset generation, benchmarking, calibration, training, evaluation
    ablate_fusion.py      what each detector is worth; audits the corpus for label leaks
    indian_faces.py       fetch real photographs of Indian people (third-party shards)
    evaluate_indian_faces.py  does the selfie detector read demography? Indian vs FFHQ
    midv2020.py           read MIDV-2020: rectify a card out of a photo, carry its annotations
    build_real_docs.py    derive a tamper set from real captured documents
    evaluate_real_docs.py score ID forensics on it, including tamper localisation
    calibrate_face_match_lfw.py  fit the identity threshold on LFW's 6,000 real pairs
    calibrate_linkage_lfw.py  fit the linkage threshold as a search, over every LFW pair
    real_video.py         read FaceForensics++ / Celeb-DF / DFDC, whichever is present
    evaluate_real_video.py score liveness on recorded deepfakes
  tests/              238 tests over the deterministic surface
  requirements.txt          resolvable pins
  requirements-nodeps.txt   facenet-pytorch, installed second with --no-deps
frontend/
  app/                Live Verify, Gauntlet, Metrics, Attack Gallery, Review Queue
  components/         DropZone, DetectorPanel, Nav, shared UI primitives
    RingGraph.tsx     fraud rings as a node-link diagram. Edge style is the
                      evidence type: a byte-identical file is drawn solid and
                      pulls tighter, a similarity link is dashed, so a proven
                      cluster never looks like an inferred one. Layout is
                      computed synchronously and seeded deterministically -
                      it does not depend on rAF and does not move between
                      reloads.
  lib/api.ts          typed API client
  lib/telemetry.ts    detector 6's client half: keystroke, pointer and focus
                      timing. Redacts key identity at source - printable keys
                      are reported as one placeholder character, so the
                      extractor can still count them without the applicant's
                      PAN ever leaving the browser as a keystroke log.
eval/                 metrics.json, model_benchmark.json, calibration.json,
                      fusion_training.json, face_match_lfw.json, real_docs.json,
                      ablation.json, ablation_leaked_corpus.json
                      - committed; these are the evidence
datasets/             generated corpus (git-ignored, rebuild with `make dataset`);
                      lfw/, midv2020/, real_docs/ also git-ignored - research
                      datasets are fetched, not vendored (`make data-real`)
storage/              uploads, heatmaps, model cache, SQLite (git-ignored)
.github/workflows/    CI: backend tests + dashboard typecheck and build
.dockerignore         keeps the 4 GB model cache out of the build context
```

---

---

[← Verityne](../README.md) - **Architecture** · [Corrections](corrections.md) · [Results](results.md) · [Measured on real data](real-data.md) · [Detector 6](behavioral.md) · [Fusion & policy](fusion.md) · [What it proves](evaluation.md) · [API & config](api.md)
