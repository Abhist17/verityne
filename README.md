# Verityne

**Deepfake-aware KYC verification for payment platforms.**

[![CI](https://github.com/Abhist17/verityne/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhist17/verityne/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg)](https://www.python.org/)
[![Next.js 14](https://img.shields.io/badge/next.js-14-000000.svg)](https://nextjs.org/)

Six independent detectors, a calibrated fusion layer, and a human-readable
explanation behind every verdict — built on the assumption that the attacker has
Stable Diffusion and DeepFaceLab on their laptop.

Five of them inspect what the applicant uploaded. The sixth inspects how they
filled the form, which is the one input a fraud kit cannot buy.

```
POST /verify  →  { verdict, risk score, top 3 reasons, heatmaps, per-detector breakdown }
```

Held-out ROC-AUC **0.753** · 2.1 s per packet on GPU · every number in this file
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
- [The six detectors](#the-six-detectors)
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

| Selfie detector (measured on our training split, n=195) | ROC-AUC | ms/image |
| --- | --- | --- |
| `dima806/deepfake_vs_real_image_detection` | **0.466** | 7.9 |
| `prithivMLmods/Deep-Fake-Detector-v2-Model` | **0.629** | 8.6 |
| Frequency-domain head, fitted here | **0.969** (train) | 2.5 |

Both pretrained checkpoints are at or below chance. That is not a bug — their
label mappings were verified, and they are printed in
`eval/model_benchmark.json` — it is what happens when a model trained on
StyleGAN and face-swap video frames meets diffusion output. A project that had
shipped the popular checkpoint on reputation would have reported a confident
number for a model that was guessing.

The same habit, turned on this project's own code, found three more — including
one in its own headline number. Every row below is a thing this project believed
until it measured it, and each links to the measurement that changed its mind:

| What was believed | What measuring it showed |
| --- | --- |
| The popular pretrained deepfake checkpoint is a reasonable baseline | **0.47 and 0.63 ROC-AUC** on our own data — at or below chance ([above](#the-honest-version-of-what-this-is)) |
| The face-identity threshold was calibrated | Fitted on this corpus it sits at 0.83, which on LFW's 6,000 real pairs accepts 32% of genuine ones — it **rejects two thirds of real applicants** ([§1](#1-face-identity-on-lfw--and-two-thresholds-that-were-badly-wrong)) |
| Tamper detection works, just weakly | It does not fire at all — the sub-score is **0.0 on 499 of 500 real documents** ([§2](#2-tamper-detection-on-real-documents--the-check-does-not-fire-at-all)) |
| The system scores 0.913 held out | **Half of that was a label the corpus wrote into its own files**, and two of five detectors were making the model worse ([§](#what-the-headline-auc-is-actually-made-of)) |
| The calibrated score was free | Isotonic regression on 195 rows collapsed the held-out scores to **14 distinct values**, and the ties cost **0.02 AUC** ([§](#the-calibrator-was-costing-002-auc)) |
| Linkage catches onboarding rings | Its threshold was fitted for a *pair* and deployed as a *search*: at the shipped scan limit **100% of genuine applicants false-link** to a stranger ([§2b](#2b-the-linkage-threshold-was-answering-the-wrong-question)) |
| The selfie detector detects synthesis | Its frequency head separates **real Indian faces from real FFHQ faces at 0.916** — it learned the prompt list ([§3](#3-the-selfie-detector-is-reading-demography)) |

The last two are the ones that matter. Neither came from a metric: the linkage
bug was found by submitting a genuine packet to the running API and reading the
verdict, and the demographic shortcut is invisible to `eval/metrics.json` by
construction, because every genuine face in that corpus comes from one dataset.

The EXIF leak is the sharpest of the rest, because nothing external caused it and
no aggregate number could show it. It took ablating the fusion layer one detector
at a time — `make ablate`, forty lines — to find that a project built to detect
deepfakes was, on its own evaluation corpus, mostly reading EXIF tags it had
written itself.

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
make ablate       # what each detector is worth; fails if the corpus leaks its labels
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
make calibrate-linkage  # fit the linkage threshold as a search, not a pair
make indian-faces    # is the selfie detector reading demography? Indian faces vs FFHQ
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
                       │  ┌──────────────────────────────────┐│
                       │  │ behavioral biometrics            ││
                       │  │ (keystroke · pointer · locale)   ││
                       │  └──────────────────────────────────┘│
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

## The six detectors

| # | Detector | What it actually looks at |
| --- | --- | --- |
| 1 | **Selfie deepfake** | A pretrained transformer *and* a fitted frequency-domain head. They vote; when they disagree, confidence drops rather than one being picked. Grad-CAM shows which pixels drove the call. |
| 2 | **Liveness video** | Per-frame appearance plus three temporal signals a frame-level model cannot see: identity drift between consecutive frames, head-pose jitter, and optical-flow discontinuity at splice boundaries. |
| 3 | **ID forensics** | OCR with positional confusion repair, then *structural* validation — a PAN's 4th character is a holder-type code and its 5th is the surname initial; Aadhaar carries a Verhoeff check digit. Plus edge-normalised Error Level Analysis, and the printed portrait run through the deepfake classifier. **The ELA component is measured at chance on real captured documents** — see [Measured on real data](#measured-on-real-data); the structural checks are deterministic and unaffected. |
| 4 | **Face match** | 512-d FaceNet embeddings, selfie vs the portrait on the card. Flagged in both directions: too low is impersonation, too high means the "selfie" is a copy of the ID photo. The lower bound is **fitted on LFW's 6,000 real pairs** (`eval/face_match_lfw.json`, 98.2% accuracy); the upper bound still comes from the corpus, because LFW contains no documents. |
| 5 | **Metadata / EXIF** | Deterministic provenance: generator tags, editor software, capture-to-submission age, device/resolution consistency, screen re-capture, and an upscale check that asks whether the file carries the detail its resolution claims. |
| 6 | **Behavioral biometrics** | Not an artifact at all — *how the form was filled*. Keystroke dwell and flight timing and their variance, pointer path straightness, tremor and sampling regularity, paste events into identity fields, field-revisit order, time on form, and device/locale coherence. **No held-out number is claimed for it** — see [Detector 6 has no evaluation number, and why](#detector-6-has-no-evaluation-number-and-why). |

Beyond the six, two cross-cutting signals: **cross-submission linkage** (one
face onboarding under several names is a ring, and the embeddings are already
computed) and **generator fingerprinting** (which model made this fake — free
labels, because we generated the fakes ourselves; 89.2% train accuracy over
`stable_diffusion` / `faceswap` / `real`).

### Detector 6 has no evaluation number, and why

Every other number in this README is reproducible from a file in `eval/`. This
detector has none, and printing one would be worse than printing nothing.

To measure it we would need labelled form-fill telemetry: real Indian merchants
completing a real KYC form, and real fraud kits completing the same one. We have
neither. The only way to manufacture a corpus would be to write a generator for
the human side *and* a generator for the bot side — and then any AUC we reported
would be measuring whether our bot generator differs from our human generator,
which we already know, because we wrote both. This repository has published that
exact mistake twice ([§](#what-the-headline-auc-is-actually-made-of)); doing it a
third time deliberately, on the detector the pitch leans hardest on, is not a
trade we are willing to make.

So three things are true about Detector 6 as shipped, and all three are stated
rather than buried:

1. **Its thresholds are priors, not fits.** Every constant lives in one named
   block at the top of `detectors/behavioral.py` — the ~15 ms floor a finger can
   physically achieve, the dwell-variance level below which a timer is more
   likely than a hand, the round-number sleeps a kit pads with. They are set in
   the direction that costs a false accept rather than a false reject, and
   re-fitting them on real telemetry is a diff to that block alone.
2. **It is not in the trained fusion model.** `config.FUSION_TRAINED_NAMES` is
   deliberately shorter than `DETECTOR_NAMES`. Feeding a logistic regression a
   feature we could only have synthesised would corrupt the one number in this
   README that *is* honest. Detector 6 is combined afterwards as an evidence
   channel — `max()`, with a ceiling — the same way linkage is.
3. **Statistical evidence from it cannot reject anyone.** Low dwell variance, a
   straight pointer path and a fast fill all describe some real person having an
   unusual day: a practised operator on their fourth signup of the morning types
   fast, does not correct, and moves in straight lines. Those hits are capped one
   abstention band below the merchant's reject threshold, so they route to a
   human. Only *categorical* evidence — a flight time below the physical floor,
   or a browser that sets `navigator.webdriver` about itself — may carry a
   rejection, because neither has an innocent explanation.

What the test suite does hold it to is separation on the two cases we can
construct honestly, and the asymmetry between them: `test_behavioral.py` asserts
that no rule fires on any simulated genuine fill, that every simulated kit
outscores every simulated human, and that the false-positive path lands in review
rather than rejection. That is a statement about the wiring, not about field
accuracy, and it is not an AUC.

**The argument for building it anyway** is that it is the only detector here
whose adversary is not on a release cycle. Detectors 1–5 degrade every time a
better generator ships. Defeating Detector 6 needs a rig that reproduces human
motor timing under a form that changes its own field order — not a download. It
is also the cheapest signal in the system: no model, no GPU, no allocation,
sub-millisecond.

---

## Fusion and policy

A **logistic regression** over ten features (each of the five *artifact*
detectors' score *and* its confidence — Detector 6 is combined separately, for
the reason above). XGBoost is fitted alongside for comparison and both are printed to
`eval/fusion_training.json`. On the training split XGBoost is marginally ahead —
0.791 CV AUC against LR's 0.785 — and LR still ships, because the switch rule is
a margin of 0.02 fixed in `train_fusion.py` before either number was known.
Six thousandths of cross-validated AUC on 195 rows is noise, and trading a
coefficient per detector you can argue with for it would be a bad trade.

| Feature | LR coefficient |
| --- | --- |
| `face_match_score` | **+1.15** |
| `metadata_exif_conf` | +0.87 |
| `id_forensics_score` | +0.86 |
| `selfie_deepfake_conf` | +0.48 |
| `metadata_exif_score` | +0.44 |
| `face_match_conf` | +0.31 |
| `liveness_video_score` | +0.23 |
| `selfie_deepfake_score` | +0.13 |
| `id_forensics_conf` | −0.00 |
| `liveness_video_conf` | −0.14 |

Calibration makes the output read as a probability, which is what the policy
thresholds and the cost model both assume it is.

### The calibrator was costing 0.02 AUC

That calibration step was isotonic regression, which is the usual default and is
the wrong choice at this sample size. Isotonic is non-parametric: fitted on 195
training rows it produced a step function with so few distinct levels that the
held-out scores collapsed to **14 distinct values**. Ranking is what AUC
measures, and mass ties destroy ranking — the ties alone cost **0.020 of
held-out AUC, 0.753 down to 0.732**, without a single detector changing.

It also made the score unusable as a dial, which matters more than the AUC. A
packet a hair above a step boundary jumped from 0.44 to 0.92, and `policy.yaml`
cuts that score at fixed thresholds while the cost curve integrates over it.

Platt scaling — two parameters, fitted by a one-feature logistic regression on
the same out-of-fold predictions — is strictly monotonic. It preserves the
ranking exactly, so it leaves AUC identical to the uncalibrated model, and
returns a smooth score. Isotonic is the better choice with thousands of rows;
it is not what this has. See `PlattCalibrator` in `backend/verityne/fusion.py`.

This is the fourth thing on the list at the top of this file, and the cheapest
to have missed: nothing was broken, no test failed, and the number was simply
0.02 lower than the model had earned.

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
held-out split**: 105 packets, 54 fraudulent, 51 genuine. No face that trained
the fusion layer appears in these numbers.

### Headline

| | Value |
| --- | --- |
| Fusion ROC-AUC | **0.753** |
| Mean score, genuine | 0.419 |
| Mean score, fraud | 0.606 |

At the two shipped thresholds:

| Threshold | Precision | Recall | False accept | False reject | Accuracy |
| --- | --- | --- | --- | --- | --- |
| `REVIEW` @ 0.40 | 0.662 | 0.796 | 20.4% | 43.1% | 0.686 |
| `REJECT` @ 0.75 | 0.815 | 0.407 | 59.3% | **9.8%** | 0.648 |

**Those two rows are bad, and the thresholds producing them are stale.** `0.40`
and `0.75` were chosen against the leaked model, whose scores separated the
classes far more widely — mean genuine 0.266 against mean fraud 0.809, a gap of
0.542. On the corrected model the same two means are 0.419 and
0.606, a gap of 0.187. The distribution compressed towards the middle, so
fixed cut points inherited from the old one now sit in the wrong places:
`REJECT` at 0.75 catches only 40.7% of fraud, and `REVIEW` at 0.40 sends 43.1%
of genuine merchants to a human.

They have deliberately not been re-tuned. Picking new thresholds on the same
held-out split that reports them is how a model gets graded on its own answer
sheet, and this README has just finished writing up what that costs. Choosing
them properly needs an operating point argued from fraud loss, merchant lifetime
value and abandonment — which is what `/metrics/cost-curve` and the sliders on
the Metrics page exist to do, and what `policy.yaml` exposes per merchant. The
ranking is what the detectors earn; where to cut it is an operator's decision,
and the honest thing is to show the curve rather than pick a flattering point on
it.

What ranking is worth is 0.753, and that is the number to argue with.

### Per detector

`auc_on_target_attacks` is the number that matters — a detector is not supposed
to catch attacks aimed at a different part of the packet.

| Detector | AUC (all) | AUC (target attacks) | Coverage | Errors |
| --- | --- | --- | --- | --- |
| Metadata / EXIF | 0.744 | **0.902** | 100% | 0 |
| Face match | 0.638 | 0.816 | 100% | 0 |
| Liveness video | 0.555 | 0.477 | 100% | 0 |
| Selfie deepfake | 0.540 | 0.691 | 100% | 0 |
| ID forensics | 0.388 | 0.439 | 100% | 0 |

### What the headline AUC is actually made of

A single fused number does not say which detector earned it, and an ensemble can
be one feature wearing five. `make ablate` refits fusion on the training split
with one detector's columns zeroed, scores the same held-out split, and reports
the drop. The table below is the run that found the defect, kept as
`eval/ablation_leaked_corpus.json`; `eval/ablation.json` is the same measurement
on the corrected corpus.

| Detector muted | Held-out AUC | Change | Share of the model's above-chance AUC |
| --- | --- | --- | --- |
| *(none — full model)* | 0.913 | | |
| Metadata / EXIF | **0.709** | **−0.204** | **49.5%** |
| Face match | 0.858 | −0.055 | 13.3% |
| ID forensics | 0.903 | −0.010 | 2.5% |
| Liveness video | 0.926 | **+0.013** | −3.2% |
| Selfie deepfake | 0.929 | **+0.016** | −3.8% |

Two things fall out of that table, and neither is comfortable.

**Half of the headline rests on one detector — and that detector was reading a
label this project wrote into the file.** `build_dataset.py` chose an EXIF mode
per packet, and the choice was conditioned on the label in a way that left three
modes appearing on fraudulent packets and never on genuine ones:

| Selfie EXIF mode | Genuine | Fraud | P(fraud) |
| --- | --- | --- | --- |
| `fresh` | 79 | 31 | 0.28 |
| `plain` | 34 | 15 | 0.31 |
| `none` | 37 | 52 | 0.58 |
| `stale` | **0** | 26 | **1.00** |
| `edited` | **0** | 16 | **1.00** |
| `generated` | **0** | 10 | **1.00** |

52 of 150 fraudulent packets — 17.3% of the corpus — carried a mode that is
a fraud label in disguise. A detector reading `Software: GIMP 2.10` out of those
files is not detecting fraud; it is reading an answer key, at 100% precision, for
free. That is why `metadata_exif` scores `auc_on_target_attacks` of exactly
1.000 in the table above: a perfect score on real data is a bug report.

The irony is on the record. `pick_exif_mode` carries a docstring explaining that
EXIF is mixed across classes on purpose, "if the corpus made EXIF a perfect class
signal the metadata detector would look superhuman and the fusion model would
learn nothing real" — and the function immediately below it does exactly that.
The guard was written for EXIF *presence*, which is genuinely well mixed
(`none`: 20% of genuine, 24% of fraud). The leak is in EXIF *content*, which
nobody checked.

**Second: the two neural detectors are worth less than nothing.** Muting the
selfie deepfake CNN *raises* held-out AUC by 0.016, and muting the liveness
analyser raises it by 0.013. Fusion had already noticed — the fitted weight on
`selfie_deepfake_score` is −0.044, effectively zero, against +1.671 on
`metadata_exif_score`. The system marketed on deepfake detection was, on this
corpus, a metadata reader with a CNN bolted to the side as ballast.

Both findings come from the same 40-line script, and neither is visible in
`eval/metrics.json`, which reports one aggregate that both defects hide inside.
Two guards now stand where nothing did. `build_dataset.py` refuses to write a
manifest in which any EXIF mode lands on one class only, so the corpus cannot be
built broken; and `ablate_fusion.py` audits the manifest independently and exits
non-zero on the same condition, so it cannot be shipped broken either. One value
is exempt, with its reason recorded in `config.PHYSICALLY_FRAUD_ONLY`: no camera
writes a diffusion model's name into `Software`, so `generated` being fraud-only
is a fact about cameras rather than an artefact of the script. Both guards read
that same allowlist, and a test asserts they agree.

The corpus generator has since been corrected and the numbers above are the
*leaked* ones, kept here because they are the finding. What the system is worth
without the answer key is reported in
[The corrected corpus](#the-corrected-corpus).

### The corrected corpus

`pick_exif_mode` now draws `stale` and `edited` for both classes, at rates that
keep fraud enriched without letting either mode name the class. Everything
downstream was rebuilt on that: a new 300-packet corpus, the spectral head and
generator fingerprint refitted, the identity threshold re-applied from LFW, all
300 packets rescored, fusion refitted, and the held-out report and ablation
regenerated. The ablation fits and calibrates exactly as `train_fusion.py` ships,
so its full-model row equals the headline in `eval/metrics.json` rather than
approximating it.

| Detector muted | Leaked corpus | Corrected corpus |
| --- | --- | --- |
| *(none — full model)* | 0.913 | **0.753** |
| Metadata / EXIF | 0.709 (+49.5%) | 0.603 (+59.2%) |
| Face match | 0.858 (+13.3%) | 0.639 (+45.0%) |
| Selfie deepfake | 0.929 (-3.8%) | 0.726 (+10.6%) |
| Liveness video | 0.926 (-3.2%) | 0.752 (+0.1%) |
| ID forensics | 0.903 (+2.5%) | 0.827 (-29.3%) |

**The headline fell from 0.913 to 0.753.** That gap is what the answer key was
worth, and 0.753 is the number this project stands behind: a measured 0.753
is worth more than a 0.913 that was half bookkeeping. Every threshold, recall
figure and cost curve in this README is recomputed on it.

What is more interesting than the drop is that **removing the leak changed which
detectors matter**, which no aggregate could have shown:

- **The deepfake CNN started working.** Muting `selfie_deepfake` used to *raise*
  held-out AUC by 0.016; it now costs 0.027, 10.6% of the model's
  above-chance performance. The signal was there the whole time, drowned out by
  a feature that was cheating. Fusion's coefficient on `selfie_deepfake_score`
  moved from -0.044 — a model actively discounting it — to +0.133, while the
  weight on `metadata_exif_score` fell from +1.671 to +0.445 and
  `face_match_score` became the largest at +1.147.
- **ID forensics went the other way.** It was worth +2.5% before and now
  costs **0.074 AUC** (-29.3%): the model is measurably better with it
  muted. On the real-document track it also scores at chance and its tamper
  check never fires ([§2](#2-tamper-detection-on-real-documents--the-check-does-not-fire-at-all)).
  Two independent measurements now say the same thing about it.
- **Liveness video remains worth nothing** (+0.1%, three ten-thousandths of
  AUC — the closest to exactly zero any of the five gets), which is expected and
  stated elsewhere: the corpus animates its clips from stills, so there is no
  genuine camera motion for a temporal detector to read.
- **Metadata / EXIF is still the largest single contributor**, and its share of
  above-chance AUC actually *rose*, from 49.5% to 59.2% — because the
  model it is a share of got much smaller. In absolute terms its contribution
  fell from 0.204 to 0.150 AUC. It did not collapse to chance, and it
  should not have: stripped EXIF and forged timestamps are real signals, and the
  detector also carries the image-provenance check — whether a file's actual
  detail matches the resolution it claims — which never depended on the leak.

Those two detectors are reported, not deleted. "We removed the detectors that
did not work" and "we ship six detectors" cannot both be on the same slide, and
a fusion layer that assigns a feature a near-zero coefficient is telling you
something worth printing rather than something worth hiding.

The corpus leak audit now reports clean: no EXIF mode lands on one class only
except `generated`, which is exempt for a stated physical reason and displayed
as such on the Metrics page rather than suppressed.

### Per attack type, worst first

| Attack | n | Caught @ REVIEW | Caught @ REJECT | AUC vs genuine |
| --- | --- | --- | --- | --- |
| `face_swap_liveness` | 3 | 33% | 0% | 0.536 |
| `tampered_document` | 7 | 57% | 29% | 0.599 |
| `reused_id_selfie` | 6 | 67% | 0% | 0.608 |
| `recaptured_screen` | 5 | 80% | 0% | 0.631 |
| `invalid_document` | 6 | 67% | 33% | 0.644 |
| `synthetic_identity` | 7 | 100% | 14% | 0.765 |
| `stale_or_edited_media` | 7 | 100% | 71% | 0.916 |
| `impersonation` | 7 | 86% | 86% | 0.919 |
| `generated_selfie` | 6 | 100% | 100% | 0.997 |

Diffusion-generated selfies remain solved (0.997) — a frequency-domain
head reads upsampling artefacts reliably, and that result survived the corpus
fix intact. Almost nothing else did.

**`face_swap_liveness` is at 0.536, which is chance**, and `reused_id_selfie` at
0.608 is barely above it. Neither is caught at the reject threshold at
all. That is the same conclusion the ablation reaches from the other direction:
the liveness analyser contributes nothing, because the corpus animates its clips
from stills and there is no genuine camera motion in any of them. Two
measurements, one cause, and it is the largest untested surface in the project.

`tampered_document` sits at 0.599 here, on documents we drew ourselves. The
next section stops doing that and measures the same check on documents somebody
else printed, photographed and scanned — where it turns out not to fire at all.

> **On these numbers being different from a previous run.** They are. The whole
> pipeline was rebuilt after the EXIF leak was found — new corpus, refitted
> heads, the LFW threshold re-applied, everything rescored — so every figure in
> this section moved, and the ones that moved most are the ones the leak had
> been propping up. The earlier run is not preserved as a comparison because it
> was measuring a corpus that gave the answer away; the one figure worth keeping
> from it is in [The corrected corpus](#the-corrected-corpus).
>
> An earlier discrepancy-chase, before any of this, turned up a real
> thread-safety bug, which is fixed — see
> [Concurrency](#a-thread-safety-bug-the-real-data-work-uncovered) below.

### Split by capture mode

ELA reads an image's edit history; photographing a card re-encodes the frame and
largely erases it. Averaging photo and scan into one number would describe
neither population.

| Capture mode | n | ID forensics AUC | Tamper-only AUC | Fusion AUC |
| --- | --- | --- | --- | --- |
| Photo | 58 | 0.286 | 0.100 | 0.745 |
| Scan | 47 | 0.513 | 0.484 | 0.804 |

### Latency

| | ms |
| --- | --- |
| p50 | 10,442 |
| p90 | 11,626 |
| p99 | 12,005 |

**This is not per-request latency and should not be read as one.** It is
wall-clock time per packet during `make score`, which runs four packets at once
with each packet's detectors sequential, so every number here includes waiting
for three other packets. It scales with `--workers`, which is the tell that it
measures throughput rather than latency: halving the worker count roughly halves
the per-packet figure without changing a single detector.

The number a caller actually waits for is the API's: stage one runs
concurrently, one request at a time, and a live `POST /verify` against the
corpus measured **2,137 ms** end to end on a CUDA GPU. Per-detector medians from
the batch run: ID forensics 8,773 ms (OCR dominates), liveness 1,217 ms, selfie deepfake 278 ms, face match 49 ms, metadata 41 ms.

### A thread-safety bug the real-data work uncovered

Re-running the pipeline to regenerate these numbers exposed a defect that had
been there the whole time and that no test would have caught.

`scripts/score_corpus.py` scores several packets at once, and `pipeline.py` runs
stage one concurrently for every live `POST /verify` — both by design, because
torch and OpenCV release the GIL. But the models underneath are **`lru_cache`
singletons**: one `DeepfakeClassifier`, one `FaceEmbedder`, one EasyOCR
`Reader`, one MTCNN. The existing lock guarded only their *construction*. Every
forward pass ran unsynchronised on a shared object, and a HuggingFace processor,
a torch module and an EasyOCR reader are none of them safe to call that way.

It failed three different ways, which is why it went unnoticed:

| Symptom | Frequency observed |
| --- | --- |
| Process aborts — `double free or corruption`, `corrupted size vs. prev_size` | 2 of 4 full corpus runs, at both 2 and 4 workers |
| A detector throws, degrades to `status="error"`, and contributes score 0.0 / confidence 0.0 | 1 packet per run |
| Slightly different numeric output for the same input | 3 packets per run |

The silent one is the worst. `Detector.run` catches everything so a broken
detector cannot take down a verdict — correct behaviour in production, but here
it meant a packet quietly entered the fusion *training set* with a zeroed
liveness feature. That is what made `liveness_video_conf` swing between −0.18
and −0.64 across runs whose held-out detector AUCs were identical to the last
digit. A crash announces itself; this corrupted a coefficient and said nothing.

The fix is a lock per model object, held across each forward pass — in
`detectors/models.py`, `utils/ocr.py` and `utils/images.py`. Per model rather
than one global lock on purpose: serialising a single model's forward passes is
what fixes the crash, while OCR, face embedding and the deepfake head still
overlap, which is where the concurrency actually pays. It is not a slowdown — the two
locked 4-worker runs scored the corpus in 811 s and 754 s, against 896 s for the
unlocked 2-worker run that managed to finish. Threads thrashing one CUDA context
were never buying throughput.

Verified afterwards: two full 4-worker runs, no aborts, no error rows, and
**1,500 of 1,500 detector outputs identical** between them — which is also the
first time this pipeline has been shown to be reproducible rather than assumed
to be.

The lesson generalises past this repo. The bug lived in the gap between "the
tests pass" and "the numbers reproduce" — the suite is deterministic and
single-threaded, so it was green throughout. What caught it was regenerating a
result and asking why a coefficient had moved.

### Bias audit

Bucketed by ITA° computed from selfie pixels. **This is a harness, not a
conclusion** — see the limitations below.

| Bucket | n | AUC | False reject | False accept |
| --- | --- | --- | --- | --- |
| brown | 37 | 0.803 | 13.6% | 46.7% |
| dark | 15 | 0.929 | 0.0% | 57.1% |
| intermediate | 11 | 0.700 | 0.0% | 60.0% |
| light | 11 | 0.750 | 0.0% | 85.7% |
| tan | 23 | 0.804 | 16.7% | 58.8% |
| unknown | 1 | — | — | — |
| very_light | 7 | 0.300 | 20.0% | 100.0% |

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
| TAR @ FAR 0.1% | 0.9513 (threshold 0.5198) |
| MTCNN detection rate | 100% of 7,701 photographs |

Then the same 6,000 real pairs were used to grade the constants the code was
already shipping:

| Constant | Value graded | True-accept rate | Accuracy |
| --- | --- | --- | --- |
| `linkage.SAME_PERSON`, as first hard-coded | 0.75 | **63.3%** | 0.817 |
| face-match band `low`, as first fitted here | 0.7835 | **52.1%** | 0.760 |
| face-match band `low`, refitted on the current corpus | 0.8321 | **32.2%** | 0.661 |

The linkage threshold was missing **more than a third of the repeat applicants
it exists to find**. The face-match lower bound was worse: on real pairs the
0.7835 it shipped with would have called **nearly half of honest applicants
impersonators**, and the third row is the same procedure re-run on the current
corpus — it lands at 0.8321, which accepts 32.2% of genuine real pairs and
would turn away **68% of honest merchants**. The defect is not a bad constant
that has since been corrected; it is that fitting this bound on this corpus
produces a wrong answer every time, and a worse one the better the corpus gets
at making a person's two images look alike.

Neither was an arithmetic mistake. Both were set against corpus pairs whose "two
photographs of one person" are one photograph re-captured twice, which score
0.943 — where two genuinely different photographs of the same person score 0.758
on average, and 0.523 at the 5th percentile. The corpus could not have exposed
this, because the corpus is what caused it.

`linkage.SAME_PERSON` was set from this fit to **0.5198** — the FAR=0.1% point
rather than the accuracy-optimal one, because a linkage hit accuses somebody of
applying twice under two names, so the false-accept budget should be strict. On
these pairs that costs little: 95.1% of true same-person pairs still link,
against 63.3% before.

**That reasoning was right about the budget and wrong about the question.** This
is a pairwise fit, and linkage is not a pairwise test — it is a search against
every prior record. The threshold has since been refitted for that, and the
number above is kept here because it is what this section measured;
[§2b](#2b-the-linkage-threshold-was-answering-the-wrong-question) is what
happened when it met a database.

Because that bound can only get worse by being re-fitted here, `calibrate.py` no
longer overwrites it. `make pipeline` used to write a fresh band file and silently
reinstate the corpus value over the LFW one; anyone following the documented
order (`make pipeline`, then `make real`) recovered by accident, and nothing
warned. The corpus still fits the upper bound, which is corpus-derived by design,
and its value for the lower one is recorded beside the better one instead of
replacing it.

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

### 2b. The linkage threshold was answering the wrong question

Found by running the product rather than by reading a report: a **genuine**
corpus packet came back `REJECT` at 0.95, top reason *"this face has already
been submitted under 4 different name(s) — strong indicator of an onboarding
ring."* Every one of the 300 corpus packets carries a distinct identity and a
distinct name. All four links were false, against four unrelated strangers.

The threshold was not too loose by accident. It was **measuring the wrong
thing.** [§1](#1-face-identity-on-lfw--and-two-thresholds-that-were-badly-wrong)
fits `SAME_PERSON` at LFW's FAR=0.1% point — a *verification* question, are these
two photographs the same person, graded on 6,000 pairs. But `find_face_links`
never asks about a pair. It compares one applicant against every prior record,
up to `SCAN_LIMIT = 5000`. A pairwise false-accept rate of `p` applied `N` times
gives a per-applicant false-link probability of `1 − (1 − p)^N`, which grows with
the size of the database while the pair fit stays where it was put.

The official protocol also cannot see this, because 3,000 impostor pairs cannot
resolve a rate below 3.3 × 10⁻⁴ — the shipped 0.0007 was **two pairs**. So
`calibrate_linkage_lfw.py` scores every pair among LFW's 5,749 distinct
identities: **16,522,626 impostor pairs**, resolving to 6 × 10⁻⁸. Reproduce with
`make calibrate-linkage` → `eval/linkage_lfw.json`.

The pairwise rate turned out to be 27× worse than the official protocol's
estimate, and the search rate is what an honest merchant actually meets:

| Prior records scanned | 0.5198 as shipped | 0.8169 as fitted |
| --- | --- | --- |
| 100 | 17.2% | 0.0% |
| 1,000 | 84.8% | 0.2% |
| 5,000 (`SCAN_LIMIT`) | **100.0%** | 1.0% |
| 50,000 | 100.0% | 9.2% |
| *pairwise FAR* | *1.88 × 10⁻³* | *1.94 × 10⁻⁶* |
| *true-accept rate* | *0.9513* | *0.3853* |

**At the shipped threshold and a full scan, every genuine applicant false-links
to a stranger.** Not most — the rounded figure is 100%.

Two changes, and the second matters more than the first:

1. **The threshold is fitted for the search**, from a stated budget: at most 1%
   of honest applicants may pick up a false link over `SCAN_LIMIT` records.
   `--budget` exposes the choice rather than burying it.
2. **A face link can no longer reject anyone by itself.** `pipeline.py` took
   `max(fusion, linkage)`, so one false match overruled all five detectors. A
   similarity search has a false-accept rate that compounds with database size;
   a claim with that error profile belongs in a human's queue, not in an
   automatic rejection. It is now capped one abstention band below the
   merchant's own reject threshold.

   A byte-identical asset is exempt and can still reject alone, because it is not
   a similarity: two files share a SHA-256 or they do not.

**The cost is real and is not hidden.** Holding that budget drops linkage's
true-accept rate from 95.1% to **38.53%** — it now misses roughly three fifths of
genuine repeat applicants. That is the honest price of doing identification with
a face embedder at this scale, and the 9.2% at 50,000 records says the approach
does not stretch much further. A production system needs a stronger embedder or
a blocking key, not a better threshold on this one. What it must not do is keep
a 100% false-link rate because the number that was measured looked reassuring.

### 3. The selfie detector is reading demography

This is the worst thing in this README, and the corpus was structurally
incapable of reporting it.

`build_dataset.py` draws its *synthetic* faces from SD-Turbo with prompts that
name the demographic — "a passport photograph of an indian man", "headshot
portrait of a south asian woman" — and its *genuine* faces from FFHQ, which is
Flickr photographs and is predominantly not South Asian. Every fake face is
Indian by construction; most real ones are not. A detector that learned any part
of "South Asian features" as evidence of synthesis would score beautifully on
this corpus and reject Indian merchants in production, and **no number in
`eval/metrics.json` could tell the two apart**, because every genuine face it
scores comes from FFHQ.

So the check has to come from outside: real photographs of Indian people, scored
beside real FFHQ photographs. Both groups are genuine. Neither is fraudulent.
A detector that separates them is not detecting anything.

Reproduce with `make indian-faces` → `eval/indian_faces.json`.

Two populations photographed differently are not a demographic comparison, and
these two are: the Indian shards are 2268×4032 phone portraits, FFHQ is 512×512
aligned crops. Feed both to a frequency-domain head and it can separate them on
resampling history alone. So both groups are face-detected, cropped at the same
margin and resized to the same pixel size *before* an identical capture
simulation and JPEG encode. The naive whole-frame protocol is reported too,
because the gap between them is evidence.

| Shortcut AUC — real Indian vs real FFHQ | Whole frame | **Controlled** |
| --- | --- | --- |
| Shipped selfie score | 0.704 | **0.793** |
| — the pretrained CNN alone | 0.621 | **0.501** |
| — the frequency head, fitted here | 0.642 | **0.916** |

*0.5 is the only defensible value in this table.*

Controlling the confound did not shrink the effect, it **sharpened** it, which is
the answer to the obvious objection. And it splits the detector cleanly in two:

- **The pretrained CNN is at 0.501 — exactly chance.** The checkpoint this
  README opens by calling a bad baseline is the one component here with no
  demographic signal at all. It was trained on somebody else's data.
- **The frequency-domain head is at 0.916.** That is the component this project
  fitted itself, on this corpus, and reports as its best detector at 0.969 train
  AUC. It is separating real Indian faces from real FFHQ faces almost perfectly.
  It learned the prompt list.

What that costs an honest applicant, at the thresholds the system ships:

| Real faces scored above | Indian | FFHQ |
| --- | --- | --- |
| 0.40 | **79.5%** | 45.5% |
| 0.50 | **65.4%** | 15.3% |
| 0.75 | **12.8%** | 1.2% |

Two thirds of genuine Indian applicants are scored more synthetic than real by a
detector built for a payments platform onboarding in India, against one sixth of
genuine FFHQ ones. At 0.75 the disparity is more than tenfold.

One more thing fell out of it: MTCNN found a face in **84.3%** of the Indian
full-frame photographs against **100%** of FFHQ. That is a second, independent
disparity sitting in front of every downstream detector — a face the detector
cannot find is a packet that gets scored on a whole frame — and it is invisible
to a corpus whose genuine faces are pre-cropped by construction. Cropping first
lifts it to 97.4%, which is why the controlled protocol reports it separately.

**Nothing has been tuned in response.** The corpus needs rebuilding with
demography decoupled from the label — Indian real faces and non-Indian synthetic
ones, both — and the spectral head refitting on it. Adjusting a threshold until
this table looks better would move the disparity somewhere a metric cannot see,
which is the failure this section exists to report. The bias audit
[above](#bias-audit) buckets by ITA° on 105 packets and calls itself a harness
rather than a conclusion; this is the measurement it could not make.

### 4. Liveness on recorded video — built, not yet run

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
  audit log to link against. The live `POST /verify` path adds linkage on top,
  so a linkage hit can move a verdict that nothing in this report accounts for.
  It takes `max(fusion, linkage)`, with the linkage term capped below the reject
  threshold unless it rests on a byte-identical file — a cap that exists because
  the uncapped version rejected a genuine applicant on four false matches
  ([§2b](#2b-the-linkage-threshold-was-answering-the-wrong-question)). Linkage is
  measured in the Gauntlet instead, which runs the full API path — see below.

### The Gauntlet, and a false positive the fixtures were manufacturing

The Gauntlet runs 10 genuine and 10 fraudulent packets from the held-out split
through the full `POST /verify` path — the only place linkage is exercised
end to end. From a clean database (`make clean && make gauntlet`):

| | Result |
| --- | --- |
| Fraud caught (REJECT or REVIEW) | **9 / 10** |
| Genuine auto-rejected | **1 / 10** |
| Genuine passed outright | 1 / 10 |
| Genuine sent to human review | 8 / 10 |
| Mean latency per packet | ~5.3 s (CUDA, 20 packets sequential) |

The scoreboard reports the last two separately rather than as one
"false rejects" figure, because a merchant an analyst clears in a minute and a
merchant turned away are not the same failure, and the aggregate reads as the
second when it is entirely the first.

**These are worse than the numbers this section used to report (10/10 and 5
passing), and the reason is the recalibration, not a regression in the demo.**
Fixing the isotonic calibrator
([§](#the-calibrator-was-costing-002-auc)) moved the whole score distribution:
mean genuine went from 0.394 to 0.419 while mean fraud fell to 0.606, so the
same fixed `0.40` and `0.75` cut points now sit differently on it. Eight of ten
genuine fixtures land in the review band. The thresholds are stale and are
[documented as stale](#headline); they are not being re-tuned against twenty
demo packets.

The one auto-rejected genuine fixture (`REAL-07`, 0.826) is worth naming,
because it is not linkage — its cross-submission lookup returns zero face
matches. It is `selfie_deepfake` scoring a real photograph at 0.745, with the
frequency head at 0.778. That is the same detector, and the same failure mode,
that [§3](#3-the-selfie-detector-is-reading-demography) measures at 0.916 AUC
against real Indian faces. The Gauntlet is showing one instance of it; §3 is the
measurement of how often it happens.

Until recently the fixture set manufactured some of those reviews itself.
Genuine and fraudulent halves were drawn independently from the split, so the
same `identity_index` — the same person — could land on both sides under two
claimed names. Linkage then reported an onboarding ring, which on that
evidence is the correct call: identity reuse across a genuine and a fraudulent
application is a real fraud pattern and the check should keep firing on it.
What was wrong was the scoreboard, which counted the flag as a detector
error. At `--seed 7` four
identities appeared on both sides, and the two whose attack keeps the victim's
real selfie (`tampered_document`, `reused_id_selfie`) cost a genuine fixture a
REVIEW apiece — a 50% review rate of which a fifth was the fixture list
arguing with itself.

Fixing the selection changes which packets are drawn, so the run above is not
the same twenty packets with two flags removed and nothing else about it is
attributable to the fix. What is attributable, and is checked directly: no
genuine fixture carries a linkage reason or an `onboarding_ring` pattern.

That check passed on twenty fixtures and still missed
[§2b](#2b-the-linkage-threshold-was-answering-the-wrong-question), because
twenty records is small enough that a 17%-per-100 false-link rate leaves most
runs clean. The Gauntlet is a smoke test for the API path, not a measurement of
linkage — a search whose error rate grows with database size cannot be graded on
a database of twenty.

`seed_gauntlet.py` now picks the fraudulent half first, then draws the genuine
half only from identities that half did not use, and says so loudly if the
corpus is too small to keep them disjoint.

The reviews that remain are fusion's own output, not linkage, and they are
not being tuned away. They are driven mainly by `metadata_exif` — the corpus
strips EXIF from genuine packets as often as from fake ones, deliberately, so
that no detector can learn "no EXIF means fake". A genuine packet with no camera
metadata is a case the model finds genuinely ambiguous, and moving a threshold
until ten demo fixtures look better would be exactly the self-confirming loop
this project keeps refusing to run.

---

## The dashboard

Next.js 14 App Router, five pages:

| Page | What it does |
| --- | --- |
| **Live Verify** (`/`) | Drag in a selfie, ID and liveness clip; get the verdict, the three reasons, the heatmaps and the per-detector breakdown. Also collects the form-fill telemetry Detector 6 reads, and says on screen that it is doing so. |
| **Gauntlet** (`/gauntlet`) | Runs 10 genuine + 10 fraudulent fixtures over server-sent events, scoring live. |
| **Metrics** (`/metrics`) | The held-out report rendered — ROC, per-attack recall, bias audit, and the cost-of-friction curve with operator-tunable ₹ sliders. |
| **Attack Gallery** (`/attacks`) | Rejected submissions grouped by attack pattern, with the evidence that flagged each. |
| **Review Queue** (`/review`) | Human-in-the-loop: everything that abstained, with accept/reject and an audit note. Tracks how often analysts agree with the model. |

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /verify` | Score one KYC packet (multipart: `selfie`, `liveness_video`, `id_document`, optional `behavioral_token`). |
| `POST /behavioral` | Accept a form-fill telemetry buffer against a token the page minted on load, reduce it to features, and store only those. Posted before the files, because uploads fail and the buffer should not die with them. |
| `GET /behavioral/{token}` | Read back one stored telemetry session with its score and the rules that fired. |
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
make test                                  # 182 tests, ~2 s
.venv/bin/python -m pytest backend/tests -q -k verhoeff   # one group
cd frontend && npx tsc --noEmit && npm run build          # dashboard gates
```

The suite covers the deterministic parts — Verhoeff check digits, PAN structural
validation, OCR confusion repair, ELA tamper scoring, spectral features, fusion
arithmetic, policy decisions, the detector base contract, and the real-data
ingest: LFW's fold protocol, MIDV-2020 quad ordering, threshold arithmetic,
tamper placement and localisation scoring, the Gauntlet's fixture selection and
scoreboard arithmetic, the corpus leak guards, the linkage cap and the search-vs-pair
threshold contract, and the README's own headline numbers against the evidence
files they cite. It deliberately
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
| `double free or corruption` / `corrupted size vs. prev_size` during `make score` | Fixed. The shared model singletons were being called from several threads without a lock; see [the thread-safety section](#a-thread-safety-bug-the-real-data-work-uncovered). If you see it again, `--workers 1` isolates it, but the locks in `detectors/models.py`, `utils/ocr.py` and `utils/images.py` should have settled it. |

---

## Repository layout

```
backend/
  verityne/
    detectors/        the six detectors + model loading, generator fingerprinting
      behavioral.py     detector 6: keystroke, pointer and locale features + rules
    utils/            ELA, spectral, OCR, Verhoeff, Grad-CAM, hashing, provenance
    api/              route modules: verify, behavioral, gauntlet, metrics, ops
    pipeline.py       orchestration (ThreadPool — torch and OpenCV release the GIL,
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
  tests/              182 tests over the deterministic surface
  requirements.txt          resolvable pins
  requirements-nodeps.txt   facenet-pytorch, installed second with --no-deps
frontend/
  app/                Live Verify, Gauntlet, Metrics, Attack Gallery, Review Queue
  components/         DropZone, DetectorPanel, Nav, shared UI primitives
  lib/api.ts          typed API client
  lib/telemetry.ts    detector 6's client half: keystroke, pointer and focus
                      timing. Redacts key identity at source — printable keys
                      are reported as one placeholder character, so the
                      extractor can still count them without the applicant's
                      PAN ever leaving the browser as a keystroke log.
eval/                 metrics.json, model_benchmark.json, calibration.json,
                      fusion_training.json, face_match_lfw.json, real_docs.json,
                      ablation.json, ablation_leaked_corpus.json
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
