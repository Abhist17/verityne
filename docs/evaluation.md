[← Verityne](../README.md)

> What the numbers prove, what they do not, and the limitations stated up front.

# What the evaluation does and does not prove

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
  copied ID photo from a genuine pair here - duplicate separability AUC is 0.548,
  i.e. chance. The *lower* bound of the band no longer depends on this corpus at
  all: it is fitted on LFW, and doing so revealed that the corpus-fitted value
  would have rejected nearly half of genuine real pairs. The *upper* bound still
  comes from here, is still set above the genuine distribution rather than
  fitted, and that attack is still caught through image provenance instead.
- **The bias audit is a harness, not a conclusion.** It buckets by ITA° - an
  image-derived skin-tone proxy, confounded by lighting, and not self-reported
  demographic data - and 106 packets across six buckets cannot power the result.
  It is reported because deepfake detectors are known to degrade on darker skin
  and measuring that is the minimum bar, not because this run settles anything.
- **The ₹ figures are assumptions.** Fraud loss, merchant LTV, abandonment
  probability and base rate are operator inputs, exposed as sliders on the
  metrics page. A single headline "₹X saved" would be unfalsifiable.
- **The corpus is partly synthetic, and less so than it was.** Selfie faces were
  always real (FFHQ). The face-identity threshold is now fitted on LFW and
  tamper detection is now also measured on real captured documents - see
  [Measured on real data](real-data.md#measured-on-real-data). What remains generated is the
  liveness video, the ID cards *in this corpus* (the real-document track is
  reported separately rather than merged in), and every attack. It is still a
  lower bound on how much work production deployment is, not a substitute
  for it.
- **These numbers do not include cross-submission linkage.** The evaluation
  harness scores detectors and fusion offline, one packet at a time, with no
  audit log to link against. The live `POST /verify` path adds linkage on top,
  so a linkage hit can move a verdict that nothing in this report accounts for.
  It takes `max(fusion, linkage)`, with the linkage term capped below the reject
  threshold unless it rests on a byte-identical file - a cap that exists because
  the uncapped version rejected a genuine applicant on four false matches
  ([§2b](real-data.md#2b-the-linkage-threshold-was-answering-the-wrong-question)). Linkage is
  measured in the Gauntlet instead, which runs the full API path - see below.

### The Gauntlet, and a false positive the fixtures were manufacturing

The Gauntlet runs 10 genuine and 10 fraudulent packets from the held-out split
through the full `POST /verify` path - the only place linkage is exercised
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
([§](fusion.md#the-calibrator-was-costing-002-auc)) moved the whole score distribution:
mean genuine went from 0.394 to 0.419 while mean fraud fell to 0.606, so the
same fixed `0.40` and `0.75` cut points now sit differently on it. Eight of ten
genuine fixtures land in the review band. The thresholds are stale and are
[documented as stale](results.md#headline); they are not being re-tuned against twenty
demo packets.

The one auto-rejected genuine fixture (`REAL-07`, 0.826) is worth naming,
because it is not linkage - its cross-submission lookup returns zero face
matches. It is `selfie_deepfake` scoring a real photograph at 0.745, with the
frequency head at 0.778. That is the same detector, and the same failure mode,
that [§3](real-data.md#3-the-selfie-detector-is-reading-demography) measures at 0.916 AUC
against real Indian faces. The Gauntlet is showing one instance of it; §3 is the
measurement of how often it happens.

Until recently the fixture set manufactured some of those reviews itself.
Genuine and fraudulent halves were drawn independently from the split, so the
same `identity_index` - the same person - could land on both sides under two
claimed names. Linkage then reported an onboarding ring, which on that
evidence is the correct call: identity reuse across a genuine and a fraudulent
application is a real fraud pattern and the check should keep firing on it.
What was wrong was the scoreboard, which counted the flag as a detector
error. At `--seed 7` four
identities appeared on both sides, and the two whose attack keeps the victim's
real selfie (`tampered_document`, `reused_id_selfie`) cost a genuine fixture a
REVIEW apiece - a 50% review rate of which a fifth was the fixture list
arguing with itself.

Fixing the selection changes which packets are drawn, so the run above is not
the same twenty packets with two flags removed and nothing else about it is
attributable to the fix. What is attributable, and is checked directly: no
genuine fixture carries a linkage reason or an `onboarding_ring` pattern.

That check passed on twenty fixtures and still missed
[§2b](real-data.md#2b-the-linkage-threshold-was-answering-the-wrong-question), because
twenty records is small enough that a 17%-per-100 false-link rate leaves most
runs clean. The Gauntlet is a smoke test for the API path, not a measurement of
linkage - a search whose error rate grows with database size cannot be graded on
a database of twenty.

`seed_gauntlet.py` now picks the fraudulent half first, then draws the genuine
half only from identities that half did not use, and says so loudly if the
corpus is too small to keep them disjoint.

The reviews that remain are fusion's own output, not linkage, and they are
not being tuned away. They are driven mainly by `metadata_exif` - the corpus
strips EXIF from genuine packets as often as from fake ones, deliberately, so
that no detector can learn "no EXIF means fake". A genuine packet with no camera
metadata is a case the model finds genuinely ambiguous, and moving a threshold
until ten demo fixtures look better would be exactly the self-confirming loop
this project keeps refusing to run.

---

---

[← Verityne](../README.md) - [Architecture](architecture.md) · [Corrections](corrections.md) · [Results](results.md) · [Measured on real data](real-data.md) · [Detector 6](behavioral.md) · [Fusion & policy](fusion.md) · **What it proves** · [API & config](api.md)
