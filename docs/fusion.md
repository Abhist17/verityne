[← Verityne](../README.md)

> How ten detector features become one score, and how a merchant turns that score into a decision.

# Fusion and policy

A **logistic regression** over ten features (each of the five *artifact*
detectors' score *and* its confidence - Detector 6 is combined separately, for
the reason above). XGBoost is fitted alongside for comparison and both are printed to
`eval/fusion_training.json`. On the training split XGBoost is marginally ahead -
0.791 CV AUC against LR's 0.785 - and LR still ships, because the switch rule is
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
measures, and mass ties destroy ranking - the ties alone cost **0.020 of
held-out AUC, 0.753 down to 0.732**, without a single detector changing.

It also made the score unusable as a dial, which matters more than the AUC. A
packet a hair above a step boundary jumped from 0.44 to 0.92, and `policy.yaml`
cuts that score at fixed thresholds while the cost curve integrates over it.

Platt scaling - two parameters, fitted by a one-feature logistic regression on
the same out-of-fold predictions - is strictly monotonic. It preserves the
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
  min_risk_for_reject: 0.8
  min_risk_for_review: 0.45
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

---

[← Verityne](../README.md) - [Architecture](architecture.md) · [Corrections](corrections.md) · [Results](results.md) · [Measured on real data](real-data.md) · [Detector 6](behavioral.md) · **Fusion & policy** · [What it proves](evaluation.md) · [API & config](api.md)
