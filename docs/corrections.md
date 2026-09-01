[← Verityne](../README.md)

> Nine beliefs this project held, measured, and lost — and the benchmark that started the habit.

# Corrections

9 beliefs this project held, measured, and lost. 4 fixed,
3 still open, 2 designed around.

| # | What measuring it showed | Status | |
| --- | --- | --- | --- |
| 1 | The popular pretrained deepfake checkpoint is at chance on our data | designed around | [§](#the-honest-version-of-what-this-is) |
| 2 | Half the headline AUC was a label we wrote into our own files | fixed | [§](results.md#what-the-headline-auc-is-actually-made-of) |
| 3 | The calibrator was quietly costing 0.02 AUC | fixed | [§](fusion.md#the-calibrator-was-costing-002-auc) |
| 4 | The identity threshold would have rejected two thirds of honest applicants | fixed | [§](real-data.md#1-face-identity-on-lfw--and-two-thresholds-that-were-badly-wrong) |
| 5 | Every genuine applicant false-linked to a stranger | fixed | [§](real-data.md#2b-the-linkage-threshold-was-answering-the-wrong-question) |
| 6 | The frequency head separates real Indian faces from real FFHQ faces | **open** | [§](real-data.md#3-the-selfie-detector-is-reading-demography) |
| 7 | The selfie detector does not work on fakes we did not generate | **open** | [§](real-data.md#measured-on-real-data) |
| 8 | We built the attack that defeats our own behavioral detector | **open** | [§](behavioral.md#the-attack-that-beats-it-which-we-built-ourselves) |
| 9 | The stale thresholds could not be fixed by fitting them properly | designed around | [§](results.md#the-thresholds-and-where-they-came-from) |

The table is not written by hand. `make corrections` assembles
`eval/corrections.json` by resolving each entry's numbers out of the evidence
file it names, and the build fails if a path does not resolve — so a correction
cannot claim a figure no report contains. `/corrections` renders it, and
`GET /metrics/corrections` serves it.

**Only one of these was visible in an aggregate metric.** Three needed data this
project did not generate, one needed submitting a genuine packet to the running
API and reading the verdict, one needed regenerating a result and asking why a
number had moved, and one needed building the attack against ourselves. That
distribution is the argument for why a held-out AUC is not an audit.

---

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
| The face-identity threshold was calibrated | Fitted on this corpus it sits at 0.83, which on LFW's 6,000 real pairs accepts 32% of genuine ones — it **rejects two thirds of real applicants** ([§1](real-data.md#1-face-identity-on-lfw--and-two-thresholds-that-were-badly-wrong)) |
| Tamper detection works, just weakly | It does not fire at all — the sub-score is **0.0 on 499 of 500 real documents** ([§2](real-data.md#2-tamper-detection-on-real-documents--the-check-does-not-fire-at-all)) |
| The system scores 0.913 held out | **Half of that was a label the corpus wrote into its own files**, and two of five detectors were making the model worse ([§](results.md#what-the-headline-auc-is-actually-made-of)) |
| The calibrated score was free | Isotonic regression on 195 rows collapsed the held-out scores to **14 distinct values**, and the ties cost **0.02 AUC** ([§](fusion.md#the-calibrator-was-costing-002-auc)) |
| Linkage catches onboarding rings | Its threshold was fitted for a *pair* and deployed as a *search*: at the shipped scan limit **100% of genuine applicants false-link** to a stranger ([§2b](real-data.md#2b-the-linkage-threshold-was-answering-the-wrong-question)) |
| The selfie detector detects synthesis | Its frequency head separates **real Indian faces from real FFHQ faces at 0.916** — it learned the prompt list ([§3](real-data.md#3-the-selfie-detector-is-reading-demography)) |

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

---

[← Verityne](../README.md) — [Architecture](architecture.md) · **Corrections** · [Results](results.md) · [Measured on real data](real-data.md) · [Detector 6](behavioral.md) · [Fusion & policy](fusion.md) · [What it proves](evaluation.md) · [API & config](api.md)
