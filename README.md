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

Held-out ROC-AUC **0.914** · 2.1 s per packet on GPU · every number in this file
is reproducible with `make pipeline`, and the real-data numbers with `make real`.

The face-identity threshold is fitted on **LFW** (98.2% ± 0.4% over its official
10-fold protocol) and tamper detection is measured on **real identity documents
that were printed, photographed and scanned** — see
[Measured on real data](#measured-on-real-data).

---

## Contents

- [The problem](#the-problem)
- [The honest version of what this is](#the-honest-version-of-what-this-is)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [The five detectors](#the-five-detectors)
- [Fusion and policy](#fusion-and-policy)
- [Results](#results)
- [Measured on real data](#measured-on-real-data)
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

The same habit, applied later to our own code on real data, found two more:
a face-identity threshold that would have rejected **half of genuine
applicants**, and a tamper check that fires on **1 of 500 real documents**.
Both are written up in [Measured on real data](#measured-on-real-data).

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

Python dependencies install in **two** steps, and `make setup` does both:

```bash
pip install -r backend/requirements.txt
pip install --no-deps -r backend/requirements-nodeps.txt
```

The second file holds `facenet-pytorch` alone. It pins `torch<2.3`, `numpy<2`
and `Pillow<10.3` — bounds that no longer hold and that the code does not
actually need — so leaving it in the first file makes pip fail outright with
`ResolutionImpossible`. Splitting it keeps the honest resolution for everything
else instead of unpinning the whole file to work around one stale package.

```bash
make setup                      # venv + npm install
make pipeline                   # corpus → benchmark → calibrate → score → train → evaluate
make real                       # calibrate on LFW + measure tamper detection on real documents
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

```bash
make data-real       # fetch LFW; prints the manual steps for MIDV-2020
make calibrate-face  # fit the identity threshold on LFW's 6,000 real pairs, and apply it
make real-docs       # build the tamper set from real captured documents
make eval-real-docs  # score ID forensics on it
make eval-real-video DATA=datasets/faceforensics   # needs gated access; see below
```

`make pipeline` takes roughly 25–40 minutes on a CUDA GPU and considerably
longer on CPU — most of it is generating the corpus, not training. `make real`
adds about 45 minutes, nearly all of it OCR on 500 documents.
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
| 3 | **ID forensics** | OCR with positional confusion repair, then *structural* validation — a PAN's 4th character is a holder-type code and its 5th is the surname initial; Aadhaar carries a Verhoeff check digit. Plus edge-normalised Error Level Analysis, and the printed portrait run through the deepfake classifier. **The ELA component is measured at chance on real captured documents** — see [Measured on real data](#measured-on-real-data); the structural checks are deterministic and unaffected. |
| 4 | **Face match** | 512-d FaceNet embeddings, selfie vs the portrait on the card. Flagged in both directions: too low is impersonation, too high means the "selfie" is a copy of the ID photo. The lower bound is **fitted on LFW's 6,000 real pairs** (`eval/face_match_lfw.json`, 98.2% accuracy); the upper bound still comes from the corpus, because LFW contains no documents. |
| 5 | **Metadata / EXIF** | Deterministic provenance: generator tags, editor software, capture-to-submission age, device/resolution consistency, screen re-capture, and an upscale check that asks whether the file carries the detail its resolution claims. |

Beyond the five, two cross-cutting signals: **cross-submission linkage** (one
face onboarding under several names is a ring, and the embeddings are already
computed) and **generator fingerprinting** (which model made this fake — free
labels, because we generated the fakes ourselves; 89.2% train accuracy over
`stable_diffusion` / `faceswap` / `real`).

## Fusion and policy

A **logistic regression** over ten features (each detector's score *and* its
confidence). XGBoost is fitted alongside for comparison and the report prints
both — on the training split LR reaches 0.881 CV AUC against XGBoost's 0.871, so
LR ships: at this feature count they tie, and LR hands back a coefficient per
detector you can argue with.

| Feature | LR coefficient |
| --- | --- |
| `metadata_exif_score` | **+1.64** |
| `id_forensics_score` | +0.89 |
| `face_match_conf` | +0.87 |
| `face_match_score` | +0.86 |
| `selfie_deepfake_conf` | +0.50 |
| `liveness_video_score` | +0.31 |
| `metadata_exif_conf` | +0.27 |
| `id_forensics_conf` | −0.00 |
| `selfie_deepfake_score` | −0.05 |
| `liveness_video_conf` | −0.18 |

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
| Fusion ROC-AUC | **0.914** |
| Mean score, genuine | 0.299 |
| Mean score, fraud | 0.810 |

At the two shipped thresholds:

| Threshold | Precision | Recall | False accept | False reject | Accuracy |
| --- | --- | --- | --- | --- | --- |
| `REVIEW` @ 0.40 | 0.746 | 0.922 | 7.8% | 29.1% | 0.811 |
| `REJECT` @ 0.75 | 0.943 | 0.647 | 35.3% | **3.6%** | 0.811 |

Read those two rows together: the reject threshold is deliberately conservative.
It auto-rejects 65% of fraud while wrongly rejecting 3.6% of genuine
merchants — the remaining fraud lands in the review queue rather than being
waved through, which is the whole point of a three-way verdict.

**The `REVIEW` row is worse than it was, and the reason is worth stating.** The
previous run put 12.7% of genuine merchants into review; this one puts 29.1%.
Ranking did not get worse — AUC rose from 0.906 to 0.914 — the *score
distribution* moved: mean genuine went from 0.248 to 0.299, so a fixed 0.40
threshold now sits lower on it. That is a direct consequence of correcting the
face-match band on real pairs (below), and it is mostly an artifact of this
corpus rather than of the fix. The corpus's genuine selfie/ID pairs derive from
one source photograph each and sit at cosine 0.943, near the top of a band whose
lower bound is now calibrated on what two *real* photographs of one person
score. Under an honest band, this corpus's own genuine pairs look faintly
duplicate-ish — which is the corpus artifact becoming visible, not a new defect.
`min_risk_for_review` in `policy.yaml` is a policy knob and has deliberately not
been re-tuned to hide the shift.

### Per detector

`auc_on_target_attacks` is the number that matters — a detector is not supposed
to catch attacks aimed at a different part of the packet.

| Detector | AUC (all) | AUC (target attacks) | Coverage | Errors |
| --- | --- | --- | --- | --- |
| Metadata / EXIF | 0.874 | **1.000** | 100% | 0 |
| Face match | 0.705 | 0.776 | 100% | 0 |
| Selfie deepfake | 0.650 | 0.844 | 100% | 0 |
| Liveness video | 0.604 | 0.560 | 100% | 0 |
| ID forensics | 0.573 | 0.702 | 100% | 0 |

### Per attack type, worst first

| Attack | n | Caught @ REVIEW | Caught @ REJECT | AUC vs genuine |
| --- | --- | --- | --- | --- |
| `tampered_document` | 5 | 60% | 60% | 0.798 |
| `synthetic_identity` | 7 | 71% | 57% | 0.809 |
| `invalid_document` | 4 | 100% | 25% | 0.905 |
| `face_swap_liveness` | 5 | 100% | 40% | 0.889 |
| `stale_or_edited_media` | 8 | 100% | 50% | 0.909 |
| `reused_id_selfie` | 6 | 100% | 50% | 0.938 |
| `generated_selfie` | 9 | 100% | 100% | 0.998 |
| `impersonation` | 7 | 100% | 100% | 1.000 |

Diffusion-generated selfies and impersonation are solved. Tampered documents are
still the weakest class and the honest headline of this table — which is exactly
why the next section stops measuring them on documents we drew ourselves, and
finds something considerably worse than 0.798.

> **On these numbers being different from a previous run.** They were
> regenerated after the face-match band was re-fitted on LFW, so `face_match`
> and everything downstream of it moved. `id_forensics` also moved (0.526 →
> 0.573) by more than that change explains. The detector is deterministic —
> verified by scoring the same documents twice — so the earlier figure came from
> a different model-resolution state in that run, not from noise. Rather than
> guess at which, the figures printed here are the ones the current code
> reproduces.

### Split by capture mode

ELA reads an image's edit history; photographing a card re-encodes the frame and
largely erases it. Averaging photo and scan into one number would describe
neither population.

| Capture mode | n | ID forensics AUC | Tamper-only AUC | Fusion AUC |
| --- | --- | --- | --- | --- |
| Photo | 57 | 0.555 | 0.688 | 0.911 |
| Scan | 49 | 0.674 | 0.696 | 0.916 |

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
| dark | 15 | 0.929 | 0.0% | 50.0% |
| brown | 33 | 0.913 | 0.0% | 28.6% |
| tan | 17 | 0.951 | 11.1% | 12.5% |
| intermediate | 19 | 0.989 | 0.0% | 30.0% |
| light | 11 | 0.883 | 0.0% | 60.0% |
| very light | 10 | 0.780 | 20.0% | 40.0% |

---

## Measured on real data

Everything in [Results](#results) is separability on a corpus we generated. That
was listed as a limitation; this section is the work of removing it where it
could be removed, and it produced the two most useful findings in the project —
both of them negative.

Reproduce with `make real`. Neither dataset here needs gated access.

| Track | Data | Report |
| --- | --- | --- |
| Face identity | LFW — 13,233 photographs of 5,749 real people | `eval/face_match_lfw.json` |
| ID documents | MIDV-2020 — real documents, printed then photographed and scanned | `eval/real_docs.json` |
| Liveness video | FaceForensics++ / Celeb-DF v2 / DFDC preview | *path built and tested; datasets gated, so no number is claimed* |

`GET /metrics/real` serves whichever of these exist, and names the ones that do
not rather than letting a missing dataset read as a zero.

### 1. Face identity, on LFW — and two thresholds that were badly wrong

The face-match band's lower bound and `linkage.SAME_PERSON` both answer one
question: are these two faces the same person. Both were hard-coded or fitted on
corpus pairs that cannot answer it. LFW's published 10-fold protocol is 6,000
pairs, half same-person, and every pair is two genuinely different photographs.
Threshold chosen on nine folds and scored on the tenth, images through the same
MTCNN crop and the same FaceNet weights the API uses.

| | Value |
| --- | --- |
| Accuracy, 10-fold | **0.9820 ± 0.0040** |
| ROC-AUC | **0.9901** |
| Fitted threshold | 0.4065 |
| TAR @ FAR 1% | 0.974 (threshold 0.4058) |
| TAR @ FAR 0.1% | 0.9513 (threshold 0.5197) |
| MTCNN detection rate | 100% of 7,701 photographs |

Then the same 6,000 real pairs were used to grade the constants the code was
already shipping:

| Constant | Was | True-accept rate | Accuracy |
| --- | --- | --- | --- |
| `linkage.SAME_PERSON` | 0.75 | **63.3%** | 0.817 |
| face-match band `low` | 0.7835 | **52.1%** | 0.760 |

The linkage threshold was missing **more than a third of the repeat applicants
it exists to find**. The face-match lower bound was worse: on real pairs it
would have called **nearly half of honest applicants impersonators**.

Neither was an arithmetic mistake. Both were set against corpus pairs whose "two
photographs of one person" are one photograph re-captured twice, which score
0.943 — where two genuinely different photographs of the same person score 0.758
on average, and 0.523 at the 5th percentile. The corpus could not have exposed
this, because the corpus is what caused it.

`linkage.SAME_PERSON` is now **0.5197** — the FAR=0.1% point rather than the
accuracy-optimal one, because a linkage hit accuses somebody of applying twice
under two names, so the false-accept budget should be strict. It costs little:
95.1% of true same-person pairs still link, against 63.3% before.

LFW does not fit the band's *upper* bound, which catches a "selfie" that is a
copy of the printed card portrait. That needs selfie-vs-document pairs and LFW
has no documents, so that bound still comes from the corpus and keeps its
caveat. LFW is also not selfies — it is web photographs of public figures. The
identity question is the same; the capture conditions are not. Treat 0.982 as a
lower bound that is at least measured on real pairs.

### 2. Tamper detection on real documents — the check does not fire at all

MIDV-2020 is 1,000 identity documents across ten types. The identities are
artificial — generated names, numbers and portraits, so the dataset could be
published — but each document was **physically printed, then photographed with a
phone and scanned on a flatbed**. Error Level Analysis reads an image's
compression history, which is a property of capture, not of whose name is on the
card. Those artefacts here are real; in `scripts/idcards.py` we draw them
ourselves.

`make real-docs` builds 500 documents — 250 genuine, 250 tampered, split evenly
across photo and scan — with three attacks that actually happen to identity
documents: `portrait_swap` (another person's photo over the portrait),
`field_splice` (a text line lifted from a different document of the same type)
and `copy_move` (a block copied from elsewhere on the same document).

Two construction rules keep the result meaningful. Every image, both classes,
is rectified to the same size and written by the same function at the same JPEG
quality — otherwise the detector learns the encoder. And **both classes pass the
same admission test** (a locatable portrait plus two compatible text lines), so
the only systematic difference between genuine and tampered is the edit itself.
An earlier version filtered only the tampered class and would have made the two
classes different populations.

| | ID forensics | Tamper (ELA) sub-score |
| --- | --- | --- |
| All 500 | 0.479 | **0.502** |
| Photo (250) | 0.527 | 0.504 |
| Scan (250) | 0.429 | 0.500 |

| Attack | n | Tamper AUC | Localised |
| --- | --- | --- | --- |
| `portrait_swap` | 84 | 0.500 | 0% |
| `field_splice` | 84 | 0.500 | 0% |
| `copy_move` | 82 | 0.506 | 0% |

**These are not weak numbers, they are absent ones.** The tamper sub-score is
exactly 0.0 on **499 of 500 documents** — genuine and tampered alike. An AUC of
0.500 is what a constant produces. On real captured documents this check
contributes nothing, and its 0.30 weight inside `id_forensics` is dead weight.

The report says why, because knowing that a detector scored at chance is much
less useful than knowing whether the evidence was missing or the decision rule
threw it away. `ela_region_diagnostic` compares the robust z-score inside the
rectangle we actually edited against the rest of the document:

| | Value |
| --- | --- |
| Median peak z anywhere on a document | **7.09** |
| Threshold `tamper_score` fires at | **7.5** |
| Mean z inside the edit | −0.213 |
| Mean z outside the edit | +0.088 |
| Fraction where the edit is the *hotter* region | 38.8% |

Two independent failures, and both are informative:

- **The threshold never fires.** `tamper_score` flags pixels above `z > 7.5`.
  On a real document the brightest pixel typically reaches 7.09 — consistently
  just short. That constant was tuned, implicitly, against cards we rendered and
  saved once, which have a quiet compression baseline. Print halftone, sensor
  noise and a phone's own JPEG raise the floor enough that nothing clears the
  bar.
- **The sign is backwards.** Even below the threshold, the edited region is
  *quieter* than its surroundings, not louder. A pasted patch has usually been
  re-compressed, so it is smoother than authentic printed detail and
  re-compresses less. The rule only ever looks for blobs that are hotter than
  baseline.

The per-attack split confirms the mechanism exactly:

| Attack | Mean z, edit minus surroundings | Tamper AUC |
| --- | --- | --- |
| `copy_move` | **+0.349** | 0.506 |
| `field_splice` | −0.308 | 0.500 |
| `portrait_swap` | −0.927 | 0.500 |

`copy_move` pastes uncompressed pixels lifted from the same image, so it is the
one attack that is *hotter* than baseline — and the only one that scores above
chance. `portrait_swap`, whose patch carries the most extra compression, is the
most strongly inverted and the most invisible. The detector is best at the
attack it was least designed for and blind to the one that matters most.

One genuinely good number is in here: **0% of 250 real, untampered documents
were told they had been edited.** That is a real false-positive rate on real
documents, and the synthetic corpus could not produce it. It is also trivially
achieved by a check that never fires, which is precisely the point of reporting
it beside the recall.

The layout check is a different story: it flags **24.4%** of honest documents
for inconsistent glyph heights. Multi-language cards with mixed type sizes are
normal, and that heuristic does not know it.

**What this does not license.** The obvious fix — make the threshold adaptive
and look for anomalies in both directions — is deliberately not applied here.
Every tampered document in this set was made by `build_real_docs.py`, and two of
its three attacks re-compress the patch. A rule tuned to "the edit is quieter"
would fit *that script's* construction, not document fraud, and validating it on
the same generator would be exactly the self-confirming loop this project exists
to avoid. Fixing it needs tamper data not produced by the script that would
score the fix.

### 3. Liveness on recorded video — built, not yet run

`scripts/real_video.py` reads FaceForensics++, Celeb-DF v2 or the DFDC preview,
whichever is extracted, uses each dataset's official test split where one is
published, and carries FF++ compression level through to the report because
detection accuracy varies sharply across c0/c23/c40. `evaluate_real_video.py`
scores each temporal signal separately as well as combined — identity drift,
pose jitter and optical-flow discontinuity are all *motion* signals, and there
is no genuine camera motion anywhere in the synthetic corpus, so they may work
far better on real video or turn out to have been reading the animation
procedure. The whole path is verified end to end against a DFDC-shaped manifest.

FF++ and Celeb-DF are gated behind a signed request form. Until that access
lands, `eval/real_video.json` does not exist and **no liveness number on real
video is claimed anywhere in this README.**

---

## What the evaluation does and does not prove

Three deliberate choices keep the numbers honest:

- **No class-correlated shortcuts.** Genuine and fake assets go through the same
  capture simulation, and EXIF presence is mixed across both classes. Without
  that, a detector learns "PNG means fake" and reports a fraudulent 0.99 AUC.
  The real-document set holds the same line twice over: one encode path for both
  classes, and one admission test applied to both, so the classes cannot differ
  by which documents were eligible.
- **Per-attack recall is reported, worst-first**, so a weak detector cannot hide
  behind a strong one in an aggregate.
- **Tamper detection is split by upload type**, because photo and scan
  populations behave differently and an average would describe neither.

### Known limitations, stated up front

- **Liveness clips are animated from stills.** The video numbers measure
  face-swap separability under identical capture conditions, not end-to-end
  liveness on real recordings. This remains the largest untested surface. The
  ingest and evaluation path for real recorded deepfakes is built and tested
  (`backend/scripts/real_video.py`, `make eval-real-video`) and handles
  FaceForensics++, Celeb-DF v2 and the DFDC preview; FF++ and Celeb-DF are gated
  behind a signed request form, so `eval/real_video.json` does not exist yet and
  no number is claimed for it.
- **Genuine selfie/ID pairs derive from one source photograph per identity**,
  re-captured to approximate a second photo. Their similarity therefore runs far
  higher than real pairs (mean cosine 0.943 here, against 0.758 for two real
  photographs of one person on LFW), and cosine similarity cannot separate a
  copied ID photo from a genuine pair here — duplicate separability AUC is 0.548,
  i.e. chance. The *lower* bound of the band no longer depends on this corpus at
  all: it is fitted on LFW, and doing so revealed that the corpus-fitted value
  would have rejected nearly half of genuine real pairs. The *upper* bound still
  comes from here, is still set above the genuine distribution rather than
  fitted, and that attack is still caught through image provenance instead.
- **The bias audit is a harness, not a conclusion.** It buckets by ITA° — an
  image-derived skin-tone proxy, confounded by lighting, and not self-reported
  demographic data — and 106 packets across six buckets cannot power the result.
  It is reported because deepfake detectors are known to degrade on darker skin
  and measuring that is the minimum bar, not because this run settles anything.
- **The ₹ figures are assumptions.** Fraud loss, merchant LTV, abandonment
  probability and base rate are operator inputs, exposed as sliders on the
  metrics page. A single headline "₹X saved" would be unfalsifiable.
- **The corpus is partly synthetic, and less so than it was.** Selfie faces were
  always real (FFHQ). The face-identity threshold is now fitted on LFW and
  tamper detection is now also measured on real captured documents — see
  [Measured on real data](#measured-on-real-data). What remains generated is the
  liveness video, the ID cards *in this corpus* (the real-document track is
  reported separately rather than merged in), and every attack. It is still a
  lower bound on how much work production deployment is, not a substitute
  for it.
- **These numbers do not include cross-submission linkage.** The evaluation
  harness scores detectors and fusion offline, one packet at a time, with no
  audit log to link against. The live `POST /verify` path adds linkage on top
  and takes `max(fusion, linkage)`, so a linkage hit can move a verdict that
  nothing in this report accounts for. Linkage is measured in the Gauntlet
  instead, which runs the full API path — on the current fixtures it costs no
  fraud recall (9/10 caught) and auto-rejects no genuine merchant.

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
| `GET /metrics/real` | The same detectors measured on real third-party data; says which reports exist rather than treating a missing dataset as a zero. |
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
make test                                  # 65 tests, ~2 s
.venv/bin/python -m pytest backend/tests -q -k verhoeff   # one group
cd frontend && npx tsc --noEmit && npm run build          # dashboard gates
```

The suite covers the deterministic parts — Verhoeff check digits, PAN structural
validation, OCR confusion repair, ELA tamper scoring, spectral features, fusion
arithmetic, policy decisions, the detector base contract, and the real-data
ingest: LFW's fold protocol, MIDV-2020 quad ordering, threshold arithmetic,
tamper placement and localisation scoring. It deliberately
does *not* assert on model outputs: those belong in `eval/metrics.json`, where a
regression shows up as a number rather than a red test. CI
(`.github/workflows/ci.yml`) runs the same suite on CPU torch plus the dashboard
typecheck and production build.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `torch==2.6.0+cu124` won't install | You have no CUDA. Drop the local tag: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu`, then run the two requirements steps below. |
| `ResolutionImpossible` mentioning `facenet-pytorch` and `numpy` | You installed `requirements-nodeps.txt` with dependency resolution on. facenet-pytorch declares `torch<2.3`, `numpy<2` and `Pillow<10.3`; those bounds are stale, not real. Install it second and with `--no-deps` — `make setup` already does. |
| First request takes 60 s+ | Model weights downloading. Set `VERITYNE_WARMUP=1` so the cost is paid at boot, and check `GET /health` for `models`. |
| `cv2.CascadeClassifier` missing | OpenCV 5 removed it and the face-detection fallback needs it. Stay on `opencv-python-headless==4.11.x`. |
| `/metrics` returns empty | `eval/metrics.json` is committed, but a `make clean-data` removes it. Re-run `make evaluate` (or the whole `make pipeline`). |
| `/metrics/real` lists things under `missing` | That dataset was not on disk when the evaluation last ran. `make data-real` fetches LFW and prints the manual steps for MIDV-2020; the video datasets are gated and cannot be scripted. |
| `No supported deepfake video dataset found` | `make eval-real-video` needs FF++, Celeb-DF v2 or the DFDC preview extracted into the directory you point `DATA=` at. Nothing downloads them. |
| Dashboard shows "failed to fetch" | `NEXT_PUBLIC_API_URL` points somewhere the browser can't reach, or the origin isn't in `VERITYNE_CORS`. |
| Gauntlet page is empty | Fixtures aren't loaded: `make gauntlet`. |
| Everything gets rejected as a "reused KYC kit" | The linkage index has accumulated repeat submissions of the same files, which is what re-scoring the corpus during testing looks like. `make clean` drops the database, then `make gauntlet` re-seeds. Note that an *exact* pixel match is a real signal — a near-match no longer rejects on its own. |
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
    midv2020.py           read MIDV-2020: rectify a card out of a photo, carry its annotations
    build_real_docs.py    derive a tamper set from real captured documents
    evaluate_real_docs.py score ID forensics on it, including tamper localisation
    calibrate_face_match_lfw.py  fit the identity threshold on LFW's 6,000 real pairs
    real_video.py         read FaceForensics++ / Celeb-DF / DFDC, whichever is present
    evaluate_real_video.py score liveness on recorded deepfakes
  tests/              65 tests over the deterministic surface
  requirements.txt          resolvable pins
  requirements-nodeps.txt   facenet-pytorch, installed second with --no-deps
frontend/
  app/                Live Verify, Gauntlet, Metrics, Attack Gallery, Review Queue
  components/         DropZone, DetectorPanel, Nav, shared UI primitives
  lib/api.ts          typed API client
eval/                 metrics.json, model_benchmark.json, calibration.json,
                      fusion_training.json, face_match_lfw.json, real_docs.json
                      — committed; these are the evidence
datasets/             generated corpus (git-ignored, rebuild with `make dataset`);
                      lfw/, midv2020/, real_docs/ also git-ignored — research
                      datasets are fetched, not vendored (`make data-real`)
storage/              uploads, heatmaps, model cache, SQLite (git-ignored)
.github/workflows/    CI: backend tests + dashboard typecheck and build
.dockerignore         keeps the 4 GB model cache out of the build context
```

---

## Ethics

**No real citizen's identity document is used anywhere in this project.** Every
card in the synthetic corpus is generated by `backend/scripts/idcards.py` from
fictional names and structurally-valid-but-unissued numbers. The real-document
track uses MIDV-2020, whose documents carry *artificially generated* identities
and portraits precisely so the dataset could be published — what is real about
them is the printing and the capture, which is the part that matters here. Real
faces come from FFHQ and LFW, both public research datasets; synthetic faces are
generated locally.

The research datasets are fetched from their sources, never vendored into this
repository. Celeb-DF in particular is licensed to a named requester, so
redistributing it here would not be ours to do.

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
