[← Verityne](../README.md)

> The same detectors measured on data this project did not generate. Two of the findings are negative.

# Measured on real data

Everything in [Results](results.md#results) is separability on a corpus we generated. That
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
[above](results.md#bias-audit) buckets by ITA° on 105 packets and calls itself a harness
rather than a conclusion; this is the measurement it could not make.

### 4. The selfie detector on fakes we did not generate

[§3](#3-the-selfie-detector-is-reading-demography) showed the selfie detector
separating two groups that were *both real*. This asks the other half of the
question, and the answer is worse: does it separate fake from real when somebody
else made the fake?

Two third-party tracks, chosen because they fail differently. **DeepFakeFace**
is three generator families over the *same* IMDB-WIKI photographs, so identity,
pose and subject are held fixed — `text2img` (Stable Diffusion from a prompt),
`inpainting` (SD regenerating just the face) and `insight` (an InsightFace
swap). **140k Real and Fake Faces** is StyleGAN against FFHQ, both distributed
at 256×256.

Geometry is not controlled by the pairing and had to be controlled by protocol:
the genuine images are native IMDB-WIKI sizes while every fake is 512×512, so raw
frames are separable on resampling history alone. Both classes are therefore
face-detected, cropped at the same margin and resized to the same pixel size
before an identical capture simulation — the same protocol
[§3](#3-the-selfie-detector-is-reading-demography) uses, and for the same reason.
Reproduce with `make eval-real-faces` → `eval/real_faces.json`.

| Generator family | Whole frame | **Controlled** | n |
| --- | --- | --- | --- |
| `text2img` | 0.428 | **0.394** | 391 |
| `inpainting` | 0.475 | **0.475** | 389 |
| `insight` | 0.479 | **0.512** | 394 |
| `stylegan` (140k) | 0.110 | **0.123** | 397 |

**Every family is at or below chance.** Not weak — at or below the line where a
coin does as well. `stylegan` at 0.123 is *inverted*: the detector
scores those fakes as more genuine than real faces, consistently enough that
flipping its sign would be an improvement.

This is the same detector that scores 0.997 on `generated_selfie` in
[Per attack type](results.md#per-attack-type-worst-first). Both numbers are real. The
corpus figure measures separability against **our own generator's settings** —
one SD-Turbo checkpoint, one prompt list, one scheduler — and the detector
learned those settings, exactly as [§3](#3-the-selfie-detector-is-reading-demography)
found it learning the prompt list's demography. Against four generators it was
not fitted on, there is nothing left.

The false-positive side is measured on the same run, over FFHQ, IMDB-WIKI and
LFW faces pooled — three genuine populations, no fakes:

| Real faces scored above | Rate |
| --- | --- |
| 0.40 | 76.5% |
| 0.50 | 60.2% |
| 0.75 | 15.0% |

At the review threshold the detector calls 76% of genuine
photographs suspicious. That is the same disparity §3 measures, quantified
against a third real population that has nothing to do with demography.

**Why this does not sink the system, and what it does mean.** The fused verdict
does not rest on this detector: the ablation puts it at 10.6% of above-chance
AUC, behind provenance, face match and the document checks, and Detector 6 does
not look at pixels at all. The honest reading is that Verityne's *fusion* is
carrying weight its selfie CNN is not, and that a project marketed on deepfake
detection has to print that sentence rather than the 0.997.

What it forbids is the obvious fix. Fine-tuning the head on DeepFakeFace or the
140k set would raise these numbers and mean nothing — the 140k track is the most
common fine-tuning set for off-the-shelf deepfake checkpoints, so a good score on
it cannot be distinguished from memorisation — the report carries a
`leakage_warning` beside that number for exactly this reason. Fixing this needs a detector whose training set is disjoint
from its evaluation set, and this project does not have one.

### 5. Liveness on recorded video — built, not yet run

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

---

[← Verityne](../README.md) — [Architecture](architecture.md) · [Corrections](corrections.md) · [Results](results.md) · **Measured on real data** · [Detector 6](behavioral.md) · [Fusion & policy](fusion.md) · [What it proves](evaluation.md) · [API & config](api.md)
