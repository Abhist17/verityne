[← Verityne](../README.md)

> The held-out evaluation in full - including the ablation that found half the headline was a leak.

# Results

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
| `REVIEW` @ 0.45 | 0.672 | 0.759 | 24.1% | **39.2%** | 0.686 |
| `REJECT` @ 0.8 | 0.952 | 0.370 | 63.0% | **2.0%** | 0.667 |

### The thresholds, and where they came from

They are **0.45** and **0.8**, and until recently they were 0.40 and
0.75 - inherited from the leaked model, whose scores separated the classes far
more widely, and carried through the corpus fix unchanged.

The reject line is not a judgement call. `GET /metrics/cost-curve`, at the same
default assumptions `policy.yaml` already shipped, prices **every threshold below
0.74 at negative expected net benefit** and puts its argmax at 0.8.
The old 0.75 sat inside that loss-making region, which is not a defensible
default for a system that ships the cost model saying so. Moving takes the
false-reject rate from 9.8% to **2.0%** and
precision from 0.815 to **0.952**, for three points of recall.

The review line the cost curve does not model - it is a capacity decision - so it
comes from the genuine distribution instead. 0.40 sat barely above the genuine
median (0.357) and sent 43% of honest merchants to a human;
0.45 sends 39% and still puts
76% of fraud in front of one.

**The caveat, because this project states them.** Both numbers are read off the
held-out split, so they report themselves: an operating point, not a
generalisation claim. The training split cannot supply one either, and that has
its own measurement - `make thresholds` re-fits properly, out-of-fold,
identity-disjoint, never touching held-out, and the point it picks holds a
2.0% false-positive rate in
cross-validation and **17.6%** on data it
was not chosen on - 8.7× optimistic. It is not fold leakage;
every training packet carries a distinct identity. 195 genuine faces simply
cannot resolve a threshold that survives new faces.

Which is the real answer: **where to cut a ranking is an operator input, not a
constant a corpus this size can supply.** The cost curve and the sliders on the
Metrics page take fraud loss, merchant lifetime value, abandonment and base rate
and return the whole curve; `policy.yaml` carries a different point per merchant,
because a crypto exchange and a gig marketplace do not have the same answer. The
defaults above are a sane place to start, not a recommendation.

The ranking is what the detectors earn - 0.753 - and that is the number to argue
with. Reports: `eval/thresholds.json`, `eval/metrics.json`.

### Per detector

`auc_on_target_attacks` is the number that matters - a detector is not supposed
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
| *(none - full model)* | 0.913 | | |
| Metadata / EXIF | **0.709** | **−0.204** | **49.5%** |
| Face match | 0.858 | −0.055 | 13.3% |
| ID forensics | 0.903 | −0.010 | 2.5% |
| Liveness video | 0.926 | **+0.013** | −3.2% |
| Selfie deepfake | 0.929 | **+0.016** | −3.8% |

Two things fall out of that table, and neither is comfortable.

**Half of the headline rests on one detector - and that detector was reading a
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

52 of 150 fraudulent packets - 17.3% of the corpus - carried a mode that is
a fraud label in disguise. A detector reading `Software: GIMP 2.10` out of those
files is not detecting fraud; it is reading an answer key, at 100% precision, for
free. That is why `metadata_exif` scores `auc_on_target_attacks` of exactly
1.000 in the table above: a perfect score on real data is a bug report.

The irony is on the record. `pick_exif_mode` carries a docstring explaining that
EXIF is mixed across classes on purpose, "if the corpus made EXIF a perfect class
signal the metadata detector would look superhuman and the fusion model would
learn nothing real" - and the function immediately below it does exactly that.
The guard was written for EXIF *presence*, which is genuinely well mixed
(`none`: 20% of genuine, 24% of fraud). The leak is in EXIF *content*, which
nobody checked.

**Second: the two neural detectors are worth less than nothing.** Muting the
selfie deepfake CNN *raises* held-out AUC by 0.016, and muting the liveness
analyser raises it by 0.013. Fusion had already noticed - the fitted weight on
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
| *(none - full model)* | 0.913 | **0.753** |
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
  moved from -0.044 - a model actively discounting it - to +0.133, while the
  weight on `metadata_exif_score` fell from +1.671 to +0.445 and
  `face_match_score` became the largest at +1.147.
- **ID forensics went the other way.** It was worth +2.5% before and now
  costs **0.074 AUC** (-29.3%): the model is measurably better with it
  muted. On the real-document track it also scores at chance and its tamper
  check never fires ([§2](real-data.md#2-tamper-detection-on-real-documents---the-check-does-not-fire-at-all)).
  Two independent measurements now say the same thing about it.
- **Liveness video remains worth nothing** (+0.1%, three ten-thousandths of
  AUC - the closest to exactly zero any of the five gets), which is expected and
  stated elsewhere: the corpus animates its clips from stills, so there is no
  genuine camera motion for a temporal detector to read.
- **Metadata / EXIF is still the largest single contributor**, and its share of
  above-chance AUC actually *rose*, from 49.5% to 59.2% - because the
  model it is a share of got much smaller. In absolute terms its contribution
  fell from 0.204 to 0.150 AUC. It did not collapse to chance, and it
  should not have: stripped EXIF and forged timestamps are real signals, and the
  detector also carries the image-provenance check - whether a file's actual
  detail matches the resolution it claims - which never depended on the leak.

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

Diffusion-generated selfies remain solved (0.997) - a frequency-domain
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
else printed, photographed and scanned - where it turns out not to fire at all.

> **On these numbers being different from a previous run.** They are. The whole
> pipeline was rebuilt after the EXIF leak was found - new corpus, refitted
> heads, the LFW threshold re-applied, everything rescored - so every figure in
> this section moved, and the ones that moved most are the ones the leak had
> been propping up. The earlier run is not preserved as a comparison because it
> was measuring a corpus that gave the answer away; the one figure worth keeping
> from it is in [The corrected corpus](#the-corrected-corpus).
>
> An earlier discrepancy-chase, before any of this, turned up a real
> thread-safety bug, which is fixed - see
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
stage one concurrently for every live `POST /verify` - both by design, because
torch and OpenCV release the GIL. But the models underneath are **`lru_cache`
singletons**: one `DeepfakeClassifier`, one `FaceEmbedder`, one EasyOCR
`Reader`, one MTCNN. The existing lock guarded only their *construction*. Every
forward pass ran unsynchronised on a shared object, and a HuggingFace processor,
a torch module and an EasyOCR reader are none of them safe to call that way.

It failed three different ways, which is why it went unnoticed:

| Symptom | Frequency observed |
| --- | --- |
| Process aborts - `double free or corruption`, `corrupted size vs. prev_size` | 2 of 4 full corpus runs, at both 2 and 4 workers |
| A detector throws, degrades to `status="error"`, and contributes score 0.0 / confidence 0.0 | 1 packet per run |
| Slightly different numeric output for the same input | 3 packets per run |

The silent one is the worst. `Detector.run` catches everything so a broken
detector cannot take down a verdict - correct behaviour in production, but here
it meant a packet quietly entered the fusion *training set* with a zeroed
liveness feature. That is what made `liveness_video_conf` swing between −0.18
and −0.64 across runs whose held-out detector AUCs were identical to the last
digit. A crash announces itself; this corrupted a coefficient and said nothing.

The fix is a lock per model object, held across each forward pass - in
`detectors/models.py`, `utils/ocr.py` and `utils/images.py`. Per model rather
than one global lock on purpose: serialising a single model's forward passes is
what fixes the crash, while OCR, face embedding and the deepfake head still
overlap, which is where the concurrency actually pays. It is not a slowdown - the two
locked 4-worker runs scored the corpus in 811 s and 754 s, against 896 s for the
unlocked 2-worker run that managed to finish. Threads thrashing one CUDA context
were never buying throughput.

Verified afterwards: two full 4-worker runs, no aborts, no error rows, and
**1,500 of 1,500 detector outputs identical** between them - which is also the
first time this pipeline has been shown to be reproducible rather than assumed
to be.

The lesson generalises past this repo. The bug lived in the gap between "the
tests pass" and "the numbers reproduce" - the suite is deterministic and
single-threaded, so it was green throughout. What caught it was regenerating a
result and asking why a coefficient had moved.

### Bias audit

Bucketed by ITA° computed from selfie pixels. **This is a harness, not a
conclusion** - see the limitations below.

| Bucket | n | AUC | False reject | False accept |
| --- | --- | --- | --- | --- |
| brown | 37 | 0.803 | 13.6% | 46.7% |
| dark | 15 | 0.929 | 0.0% | 57.1% |
| intermediate | 11 | 0.700 | 0.0% | 60.0% |
| light | 11 | 0.750 | 0.0% | 85.7% |
| tan | 23 | 0.804 | 16.7% | 58.8% |
| unknown | 1 | - | - | - |
| very_light | 7 | 0.300 | 20.0% | 100.0% |

---

---

[← Verityne](../README.md) - [Architecture](architecture.md) · [Corrections](corrections.md) · **Results** · [Measured on real data](real-data.md) · [Detector 6](behavioral.md) · [Fusion & policy](fusion.md) · [What it proves](evaluation.md) · [API & config](api.md)
