# Verityne

**Deepfake-aware KYC verification for payment platforms.**

[![CI](https://github.com/Abhist17/verityne/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhist17/verityne/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg)](https://www.python.org/)
[![Next.js 14](https://img.shields.io/badge/next.js-14-000000.svg)](https://nextjs.org/)

Five independent detectors, a calibrated fusion layer, and a human-readable
explanation behind every verdict — built on the assumption that the attacker has
Stable Diffusion and DeepFaceLab on their laptop.

```
POST /verify  →  { verdict, risk score, top 3 reasons, heatmaps, per-detector breakdown }
```

Held-out ROC-AUC **0.906** · 2.1 s per packet on GPU · every number in this file
is reproducible with `make pipeline`.

---

## Contents

- [The problem](#the-problem)
- [The honest version of what this is](#the-honest-version-of-what-this-is)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [The five detectors](#the-five-detectors)
- [Fusion and policy](#fusion-and-policy)
- [Results](#results)
- [What the evaluation does and does not prove](#what-the-evaluation-does-and-does-not-prove)
- [The dashboard](#the-dashboard)
- [API](#api)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Ethics](#ethics)
- [Deliberately not built](#deliberately-not-built)

---

## The problem

KYC verification was designed when faking an identity meant forging a plastic
card and finding a lookalike. That era ended. A photorealistic face of a person
who does not exist takes about four seconds and costs nothing; face-swap tooling
puts any face onto a "turn your head and blink" liveness video; and matched fake
PAN + selfie + liveness kits sell in Telegram groups for a few hundred rupees.

Every fake merchant that gets through becomes chargeback losses, laundering
exposure, and a regulatory problem for the platform that onboarded them.

## The honest version of what this is

We cannot out-model Jumio or IDfy on detector quality in a weekend, and this
README will not pretend otherwise. What is genuinely missing from the market is
an **evaluation harness that reports where deepfake detection actually fails** —
and that is what Verityne is built around. The system works; the measurement of
where it *doesn't* work is the contribution.

Concretely, the first thing this project did was benchmark the obvious
off-the-shelf answer, on its own data, before building anything on top of it:

| Selfie detector (measured on our training split, n=194) | ROC-AUC | ms/image |
| --- | --- | --- |
| `dima806/deepfake_vs_real_image_detection` | **0.416** | 7.6 |
| `prithivMLmods/Deep-Fake-Detector-v2-Model` | **0.609** | 8.0 |
| Frequency-domain head, fitted here | **0.964** (train) | 2.8 |

Both pretrained checkpoints are at or below chance. That is not a bug — their
label mappings were verified, and they are printed in
`eval/model_benchmark.json` — it is what happens when a model trained on
StyleGAN and face-swap video frames meets diffusion output. A project that had
shipped the popular checkpoint on reputation would have reported a confident
number for a model that was guessing.

So the architecture leans on the signals that generalise: **frequency-domain
artefacts** left by every upsampling generator, **image provenance** (is this
file the resolution it claims to be?), and **deterministic document checks** that
cannot hallucinate. The pretrained CNN is an ensemble member, not the system.

---

## Quick start

### Prerequisites

| | Version | Notes |
| --- | --- | --- |
| Python | 3.12 | 3.11 works; 3.13 is untested |
| Node.js | 20 LTS | for the Next.js dashboard |
| Disk | ~6 GB | ~4 GB of that is the Hugging Face model cache |
| GPU | optional | CUDA 12.4 if present; CPU is the automatic fallback |

The first run downloads model weights (FaceNet, the deepfake transformer,
EasyOCR) into `storage/models/hf`. Everything after that is offline.

```bash
make setup                      # venv + npm install
make pipeline                   # corpus → benchmark → calibrate → score → train → evaluate
make demo                       # seed gauntlet + red-team pool, then run both servers
```

Dashboard on <http://localhost:3000>, API docs on <http://localhost:8000/docs>.

Or with Docker:

```bash
cp .env.example .env
docker compose up               # dashboard :3000, API :8000
```

<details>
<summary>Running the pieces individually</summary>

```bash
make dataset      # 300 synthetic KYC packets with liveness clips
make benchmark    # measure candidate deepfake checkpoints on our data, pin the winner
make calibrate    # fit the face-match band, spectral head and generator fingerprint
make score        # run all detectors over the corpus, cache raw scores
make train        # fit the fusion layer (train split only)
make evaluate     # held-out report → eval/metrics.json
make gauntlet     # load 10 genuine + 10 fraudulent demo fixtures
make redteam      # generate unseen attacks for the live red-team button
make backend      # API only, on :8000
make frontend     # dashboard only, on :3000
make test         # backend test suite
make clean        # drop uploads, heatmaps and the database
make fresh        # nuke everything and rebuild end to end
```

`make pipeline` takes roughly 25–40 minutes on a CUDA GPU and considerably
longer on CPU — most of it is generating the corpus, not training.
</details>

---

## Architecture

```
                       ┌──────────────────────────────────────┐
  selfie.jpg ─────────▶│                                      │
  id_document.jpg ────▶│   POST /verify   (FastAPI)           │
  liveness.mp4 ───────▶│                                      │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  stage one — run concurrently        │
                       │  ┌────────────┐ ┌─────────────────┐  │
                       │  │ selfie     │ │ id forensics    │  │
                       │  │ deepfake   │ │ (OCR + ELA +    │  │
                       │  └────────────┘ │  structure)     │  │
                       │  ┌────────────┐ └─────────────────┘  │
                       │  │ liveness   │ ┌─────────────────┐  │
                       │  │ video      │ │ metadata / EXIF │  │
                       │  └────────────┘ └─────────────────┘  │
                       │        shared cache: decoded images, │
                       │        face crops, embeddings        │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  stage two — needs both face crops   │
                       │  face match (selfie vs ID portrait)  │
                       └──────────────────┬───────────────────┘
                                          │
              cross-cutting ──────────────┤
              • linkage (same face / same file, prior submissions)
              • generator fingerprint (which model made this fake)
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  fusion: logistic regression over    │
                       │  10 features → isotonic calibration  │
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

## The five detectors

| # | Detector | What it actually looks at |
| --- | --- | --- |
| 1 | **Selfie deepfake** | A pretrained transformer *and* a fitted frequency-domain head. They vote; when they disagree, confidence drops rather than one being picked. Grad-CAM shows which pixels drove the call. |
| 2 | **Liveness video** | Per-frame appearance plus three temporal signals a frame-level model cannot see: identity drift between consecutive frames, head-pose jitter, and optical-flow discontinuity at splice boundaries. |
| 3 | **ID forensics** | OCR with positional confusion repair, then *structural* validation — a PAN's 4th character is a holder-type code and its 5th is the surname initial; Aadhaar carries a Verhoeff check digit. Plus edge-normalised Error Level Analysis, and the printed portrait run through the deepfake classifier. |
| 4 | **Face match** | 512-d FaceNet embeddings, selfie vs the portrait on the card. Flagged in both directions: too low is impersonation, too high means the "selfie" is a copy of the ID photo. Decision band is **fitted from data** (`eval/calibration.json`), not hard-coded. |
| 5 | **Metadata / EXIF** | Deterministic provenance: generator tags, editor software, capture-to-submission age, device/resolution consistency, screen re-capture, and an upscale check that asks whether the file carries the detail its resolution claims. |

Beyond the five, two cross-cutting signals: **cross-submission linkage** (one
face onboarding under several names is a ring, and the embeddings are already
computed) and **generator fingerprinting** (which model made this fake — free
labels, because we generated the fakes ourselves; 89.2% train accuracy over
`stable_diffusion` / `faceswap` / `real`).

## Fusion and policy

A **logistic regression** over ten features (each detector's score *and* its
confidence). XGBoost is fitted alongside for comparison and the report prints
both — on the training split LR reaches 0.891 CV AUC against XGBoost's 0.857, so
LR ships: at this feature count they tie, and LR hands back a coefficient per
detector you can argue with.

| Feature | LR coefficient |
| --- | --- |
| `metadata_exif_score` | **+1.73** |
| `face_match_conf` | +0.91 |
| `face_match_score` | +0.89 |
| `id_forensics_score` | +0.88 |
| `liveness_video_score` | +0.44 |
| `selfie_deepfake_conf` | +0.43 |
| `metadata_exif_conf` | +0.16 |
| `selfie_deepfake_score` | +0.07 |
| `id_forensics_conf` | −0.00 |
| `liveness_video_conf` | −0.92 |

Isotonic calibration makes the output read as a probability, which is what the
policy thresholds and the cost model both assume it is.

Two things force a `REVIEW` that the raw score alone would not: the score
landing inside the abstention band around a threshold, and a required input
being missing or a detector erroring. Verityne would rather say "I am not sure"
than flip a coin on someone's livelihood.

Per-merchant thresholds live in `backend/verityne/policy.yaml` and hot-reload
via `POST /admin/reload-policy`:

```yaml
default:
  min_risk_for_reject: 0.75
  min_risk_for_review: 0.40
  abstain_band: 0.05

merchants:
  crypto_exchange_01:            # high-risk vertical: accept more friction
    min_risk_for_reject: 0.60
    avg_fraud_loss_inr: 250000
  gig_marketplace_02:            # low-value, high-volume: friction is expensive
    min_risk_for_reject: 0.88
    require_liveness: false
```

---

## Results

Everything below is from `eval/metrics.json`, computed on an **identity-disjoint
held-out split**: 106 packets, 51 fraudulent, 55 genuine. No face that trained
the fusion layer appears in these numbers.

### Headline

| | Value |
| --- | --- |
| Fusion ROC-AUC | **0.906** |
| Mean score, genuine | 0.248 |
| Mean score, fraud | 0.740 |

At the two shipped thresholds:

| Threshold | Precision | Recall | False accept | False reject | Accuracy |
| --- | --- | --- | --- | --- | --- |
| `REVIEW` @ 0.40 | 0.863 | 0.863 | 13.7% | 12.7% | 0.868 |
| `REJECT` @ 0.75 | 0.938 | 0.588 | 41.2% | **3.6%** | 0.783 |

Read those two rows together: the reject threshold is deliberately conservative.
It auto-rejects only 59% of fraud, but wrongly rejects 3.6% of genuine
merchants — the remaining fraud lands in the review queue rather than being
waved through, which is the whole point of a three-way verdict.

### Per detector

`auc_on_target_attacks` is the number that matters — a detector is not supposed
to catch attacks aimed at a different part of the packet.

| Detector | AUC (all) | AUC (target attacks) | Coverage | Errors |
| --- | --- | --- | --- | --- |
| Metadata / EXIF | 0.874 | **1.000** | 100% | 0 |
| Face match | 0.746 | 0.857 | 100% | 0 |
| Selfie deepfake | 0.650 | 0.844 | 100% | 0 |
| Liveness video | 0.604 | 0.560 | 100% | 0 |
| ID forensics | 0.526 | 0.640 | 100% | 0 |

### Per attack type, worst first

| Attack | n | Caught @ REVIEW | Caught @ REJECT | AUC vs genuine |
| --- | --- | --- | --- | --- |
| `synthetic_identity` | 7 | 57% | 57% | 0.823 |
| `tampered_document` | 5 | 60% | 60% | 0.689 |
| `invalid_document` | 4 | 75% | 25% | 0.836 |
| `reused_id_selfie` | 6 | 83% | 17% | 0.902 |
| `face_swap_liveness` | 5 | 100% | 40% | 0.940 |
| `stale_or_edited_media` | 8 | 100% | 38% | 0.946 |
| `generated_selfie` | 9 | 100% | 100% | 1.000 |
| `impersonation` | 7 | 100% | 100% | 1.000 |

Diffusion-generated selfies and impersonation are solved. Tampered documents are
the weakest class at 0.689 and the honest headline of this table.

### Split by capture mode

ELA reads an image's edit history; photographing a card re-encodes the frame and
largely erases it. Averaging photo and scan into one number would describe
neither population.

| Capture mode | n | ID forensics AUC | Tamper-only AUC | Fusion AUC |
| --- | --- | --- | --- | --- |
| Photo | 57 | 0.486 | 0.664 | 0.876 |
| Scan | 49 | 0.647 | 0.565 | 0.946 |

### Latency

| | ms |
| --- | --- |
| p50 | 6,361 |
| p90 | 7,414 |
| p99 | 9,169 |

Measured in batch mode with detectors running **sequentially**. The API runs
stage one concurrently: a live `POST /verify` against the corpus measured
**2,137 ms** end to end on a CUDA GPU. Per-detector medians: ID forensics 4,980
ms (OCR dominates), liveness 1,074 ms, selfie deepfake 275 ms, face match 52 ms,
metadata 30 ms.

### Bias audit

Bucketed by ITA° computed from selfie pixels. **This is a harness, not a
conclusion** — see the limitations below.

| Bucket | n | AUC | False reject | False accept |
| --- | --- | --- | --- | --- |
| dark | 15 | 0.946 | 0.0% | 75.0% |
| brown | 33 | 0.897 | 0.0% | 21.4% |
| tan | 17 | 0.944 | 11.1% | 12.5% |
| intermediate | 19 | 1.000 | 0.0% | 40.0% |
| light | 11 | 0.850 | 0.0% | 80.0% |
| very light | 10 | 0.800 | 20.0% | 40.0% |

---

## What the evaluation does and does not prove

Three deliberate choices keep the numbers honest:

- **No class-correlated shortcuts.** Genuine and fake assets go through the same
  capture simulation, and EXIF presence is mixed across both classes. Without
  that, a detector learns "PNG means fake" and reports a fraudulent 0.99 AUC.
- **Per-attack recall is reported, worst-first**, so a weak detector cannot hide
  behind a strong one in an aggregate.
- **Tamper detection is split by upload type**, because photo and scan
  populations behave differently and an average would describe neither.

### Known limitations, stated up front

- **Liveness clips are animated from stills.** The video numbers measure
  face-swap separability under identical capture conditions, not end-to-end
  liveness on real recordings. Replacing `backend/scripts/videos.py` with
  recorded clips is the single highest-value upgrade to this corpus.
- **Genuine selfie/ID pairs derive from one source photograph per identity**,
  re-captured to approximate a second photo. Their similarity therefore runs far
  higher than real pairs (mean cosine 0.943), and cosine similarity cannot
  separate a copied ID photo from a genuine pair here — duplicate separability
  AUC is 0.548, i.e. chance. Rather than fit a threshold that would buy
  duplicate recall by rejecting real merchants, the band is set above the
  genuine distribution and that attack is caught through image provenance
  instead.
- **The bias audit is a harness, not a conclusion.** It buckets by ITA° — an
  image-derived skin-tone proxy, confounded by lighting, and not self-reported
  demographic data — and 106 packets across six buckets cannot power the result.
  It is reported because deepfake detectors are known to degrade on darker skin
  and measuring that is the minimum bar, not because this run settles anything.
- **The ₹ figures are assumptions.** Fraud loss, merchant LTV, abandonment
  probability and base rate are operator inputs, exposed as sliders on the
  metrics page. A single headline "₹X saved" would be unfalsifiable.
- **The corpus is synthetic.** Everything above measures separability on data we
  generated. It is a lower bound on how much work production deployment is, not
  a substitute for it.

---

## The dashboard

Next.js 14 App Router, five pages:

| Page | What it does |
| --- | --- |
| **Live Verify** (`/`) | Drag in a selfie, ID and liveness clip; get the verdict, the three reasons, the heatmaps and the per-detector breakdown. |
| **Gauntlet** (`/gauntlet`) | Runs 10 genuine + 10 fraudulent fixtures over server-sent events, scoring live. |
| **Metrics** (`/metrics`) | The held-out report rendered — ROC, per-attack recall, bias audit, and the cost-of-friction curve with operator-tunable ₹ sliders. |
| **Attack Gallery** (`/attacks`) | Rejected submissions grouped by attack pattern, with the evidence that flagged each. |
| **Review Queue** (`/review`) | Human-in-the-loop: everything that abstained, with accept/reject and an audit note. Tracks how often analysts agree with the model. |

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /verify` | Score one KYC packet (multipart: `selfie`, `liveness_video`, `id_document`). |
| `POST /batch-verify` | Retroactive sweep — re-score history with the current model to find fakes that were let through. |
| `GET /submissions` · `GET /submissions/{id}` | Browse the audit log; full record for one submission. |
| `POST /submissions/{id}/rescore` | Re-run one stored packet against the current model. |
| `GET /gauntlet` · `POST /gauntlet/run` · `GET /gauntlet/stream` | Fixtures, batch scoreboard, and the SSE live feed. |
| `GET /metrics` | Held-out report plus live operational stats. |
| `GET /metrics/cost-curve` · `GET /metrics/thresholds` | Friction/fraud tradeoff, and where each merchant policy sits on it. |
| `GET /attacks` | Flagged submissions grouped by attack pattern. |
| `GET /review-queue` · `POST /review/{id}/decision` · `GET /review/agreement` | Human-in-the-loop queue, analyst decisions, model/analyst agreement rate. |
| `POST /redteam/generate` | Score a never-before-seen synthetic attack. |
| `GET /policy` · `POST /admin/reload-policy` | Effective merchant policy; hot-reload `policy.yaml`. |
| `GET /audit` | Raw audit event stream. |
| `GET /health` | Uptime, resolved device, warm model status, active policy. |

Auth is a single `X-API-Key` header — this is a demo, not a tenancy model.

<details>
<summary>Example request and response</summary>

```bash
curl -X POST http://localhost:8000/verify \
  -H "X-API-Key: verityne-demo-key" \
  -F "selfie=@selfie.jpg" \
  -F "id_document=@pan.jpg" \
  -F "liveness_video=@liveness.mp4" \
  -F "merchant_id=default" \
  -F "claimed_name=Ravi Kumar"
```

```jsonc
{
  "submission_id": "ed0f5f3eb2cf40e8bbf853ff8b0396fb",
  "verdict": "REJECT",
  "final_score": 0.92,
  "abstained": false,
  "attack_pattern": "reused_kyc_kit",
  "generator_guess": null,
  "top_reasons": [
    "This face matches 1 earlier submission(s) under the same name (possible duplicate application)",
    "The exact same image file was used in 4 earlier submission(s) — consistent with a purchased, pre-made KYC kit",
    "The id_document is 1012px wide but carries detail only to about 569px — it was enlarged from a much smaller source"
  ],
  "explanation": "This submission was rejected with an overall risk score of 0.92. …",
  "detector_breakdown": { "selfie_deepfake": {…}, "id_forensics": {…}, … },
  "heatmaps": { "selfie_deepfake": "/static/heatmaps/…png", … },
  "latency_ms": 2136.51,
  "fusion_model": "logreg",
  "policy": {…},
  "created_at": "2026-08-29T08:31:11.538746Z"
}
```
</details>

---

## Configuration

Copy `.env.example` to `.env`. Every value has a working default.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VERITYNE_API_KEY` | `verityne-demo-key` | Single-tenant demo auth (`X-API-Key`). |
| `VERITYNE_DB` | `sqlite:///storage/verityne.db` | Any SQLAlchemy URL; `postgresql+psycopg://…` works, schema is identical. |
| `VERITYNE_STORAGE` | `./storage` | Uploads, heatmaps, model cache, SQLite file. |
| `VERITYNE_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu`. |
| `VERITYNE_MAX_UPLOAD_MB` | `40` | Per-file upload ceiling. |
| `VERITYNE_MAX_VIDEO_FRAMES` | `24` | Frames sampled from a liveness clip. |
| `VERITYNE_FRAME_STRIDE` | `5` | Sampling stride within the clip. |
| `VERITYNE_WARMUP` | `1` | Load models at boot instead of on the first request. |
| `VERITYNE_CORS` | `http://localhost:3000` | Comma-separated allowed origins. |
| `VERITYNE_SLACK_WEBHOOK` | *(empty)* | High-confidence rejections POST here if set. |
| `VERITYNE_LOG` | `INFO` | Log level. |
| `HF_HOME` | `./storage/models/hf` | Keeps model weights inside the repo, not `~/.cache`. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Where the dashboard looks for the API. |
| `NEXT_PUBLIC_API_KEY` | `verityne-demo-key` | Key the dashboard sends. |

---

## Testing

```bash
make test                                  # 37 tests, ~0.5 s
.venv/bin/python -m pytest backend/tests -q -k verhoeff   # one group
cd frontend && npx tsc --noEmit && npm run build          # dashboard gates
```

The suite covers the deterministic parts — Verhoeff check digits, PAN structural
validation, OCR confusion repair, ELA tamper scoring, spectral features, fusion
arithmetic, policy decisions and the detector base contract. It deliberately
does *not* assert on model outputs: those belong in `eval/metrics.json`, where a
regression shows up as a number rather than a red test. CI
(`.github/workflows/ci.yml`) runs the same suite on CPU torch plus the dashboard
typecheck and production build.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `torch==2.6.0+cu124` won't install | You have no CUDA. Drop the local tag: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r backend/requirements.txt` for the rest. |
| First request takes 60 s+ | Model weights downloading. Set `VERITYNE_WARMUP=1` so the cost is paid at boot, and check `GET /health` for `models`. |
| `cv2.CascadeClassifier` missing | OpenCV 5 removed it and the face-detection fallback needs it. Stay on `opencv-python-headless==4.11.x`. |
| `/metrics` returns empty | `eval/metrics.json` is committed, but a `make clean-data` removes it. Re-run `make evaluate` (or the whole `make pipeline`). |
| Dashboard shows "failed to fetch" | `NEXT_PUBLIC_API_URL` points somewhere the browser can't reach, or the origin isn't in `VERITYNE_CORS`. |
| Gauntlet page is empty | Fixtures aren't loaded: `make gauntlet`. |
| CUDA out of memory | `VERITYNE_DEVICE=cpu`, or lower `VERITYNE_MAX_VIDEO_FRAMES`. |

---

## Repository layout

```
backend/
  verityne/
    detectors/        the five detectors + model loading, generator fingerprinting
    utils/            ELA, spectral, OCR, Verhoeff, Grad-CAM, hashing, provenance
    api/              route modules: verify, gauntlet, metrics, ops
    pipeline.py       orchestration (ThreadPool — torch and OpenCV release the GIL,
                      asyncio.gather over CPU-bound work would buy nothing)
    fusion.py         logistic regression + calibration + policy decisions
    explain.py        attack classification and natural-language narration
    linkage.py        cross-submission face and asset lookup
    config.py         paths, env, per-merchant policy loading
    policy.yaml       merchant thresholds, hot-reloadable
    db.py             SQLAlchemy models: submissions, detector results, audit log
    schemas.py        pydantic request/response contracts
    main.py           FastAPI app, CORS, timing middleware, static mounts
  scripts/            dataset generation, benchmarking, calibration, training, evaluation
  tests/              37 tests over the deterministic surface
frontend/
  app/                Live Verify, Gauntlet, Metrics, Attack Gallery, Review Queue
  components/         DropZone, DetectorPanel, Nav, shared UI primitives
  lib/api.ts          typed API client
eval/                 metrics.json, model_benchmark.json, calibration.json,
                      fusion_training.json — committed; these are the evidence
datasets/             generated corpus (git-ignored, rebuild with `make dataset`)
storage/              uploads, heatmaps, model cache, SQLite (git-ignored)
.github/workflows/    CI: backend tests + dashboard typecheck and build
```

---

## Ethics

**No real citizen's identity document is used anywhere in this project.** Every
card in the corpus is generated by `backend/scripts/idcards.py` from fictional
names and structurally-valid-but-unissued numbers. Real faces come from FFHQ, a
public research dataset; synthetic faces are generated locally.

Verityne never rejects unilaterally at low confidence — the abstention band
exists so borderline cases reach a human. Audit-log retention is configurable
per merchant, and every verdict is stored with the evidence that produced it,
because "the model said so" is not an acceptable answer to a merchant who was
blocked.

## Deliberately not built

Real multi-tenant auth, payment-provider integration, training a deepfake model
from scratch, a mobile SDK, multi-script OCR beyond English, and UIDAI biometric
integration (regulated, months of paperwork). Scope discipline, not oversight.

---

## License

MIT — see [LICENSE](LICENSE).
