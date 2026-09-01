"""Fusion layer: detector scores in, one calibrated risk score out.

Deliberately a *logistic regression*, not a boosted forest. With five features
and a few hundred training rows, an XGBoost model buys no measurable AUC over
LR but costs interpretability - and interpretability is the product here. LR
gives us a per-detector weight we can put on screen and defend.

`scripts/train_fusion.py` fits it and writes storage/models/fusion.joblib along
with a calibration curve. Until that file exists we run a documented heuristic
so the system is never dead on arrival.

Not every signal goes through the learned model. The model is only honest about
features it was fitted on, and behavioral telemetry has no labelled corpus here
(see `config.FUSION_TRAINED_NAMES` for why synthesising one would be worse than
useless). Signals in that position are *evidence channels*: they are combined
after the model by taking a maximum, and each is ceilinged so it can force a
human review but not a rejection unless its evidence is categorical rather than
statistical. `behavioral_channel` below and `linkage.linkage_signal` are the two.
"""
from __future__ import annotations

import functools
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DETECTOR_NAMES, MODEL_ROOT, MerchantPolicy
from .schemas import DetectorOutput

log = logging.getLogger("verityne.fusion")

FUSION_PATH = MODEL_ROOT / "fusion.joblib"

#: Prior weights used before a model is trained. They encode the honest belief
#: that deterministic detectors (ID forensics, face match) are more trustworthy
#: than a pretrained deepfake CNN running off-distribution.
HEURISTIC_WEIGHTS: Dict[str, float] = {
    "selfie_deepfake": 1.00,
    "liveness_video": 0.85,
    "id_forensics": 1.25,
    "face_match": 1.20,
    "metadata_exif": 0.70,
    # Weighted above every artifact detector, and the reason is not enthusiasm.
    # Detectors 1-5 are in an arms race with generative models that ship a better
    # checkpoint every few weeks; when one of them is wrong it is because the
    # attacker bought a newer generator. Behavioral evidence does not decay that
    # way - defeating it needs a rig that reproduces human motor timing, not a
    # download. The weight only applies when the detector actually ran, and it is
    # ceilinged downstream, so a heavy weight here cannot reject anyone on its
    # own.
    "behavioral": 1.35,
}


class PlattCalibrator:
    """Maps a raw fusion score onto a probability, by Platt scaling.

    Two parameters, fitted by a one-feature logistic regression on out-of-fold
    predictions. It lives here rather than in the training script because
    `fusion.joblib` pickles it by reference and the API has to be able to load it.

    It replaced isotonic regression, which is the usual default and is wrong at
    this sample size. Isotonic is non-parametric: on 195 training rows it fitted a
    step function with so few distinct levels that the held-out scores collapsed
    to **14 distinct values**, and the ties that created cost 0.020 of held-out
    AUC outright (0.753 to 0.732). It also made the score unusable as a dial - a
    packet a hair above a step boundary jumped from 0.44 to 0.92 - which matters
    because `policy.yaml` cuts that score at fixed thresholds and the cost curve
    integrates over it. Platt scaling is strictly monotonic, so it preserves the
    ranking exactly, leaves AUC identical to the uncalibrated model, and returns
    a smooth score. Isotonic is the better choice with thousands of rows; it is
    not what this has.
    """

    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def fit(cls, scores, labels) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        x = np.asarray(scores, dtype=float).reshape(-1, 1)
        return cls(LogisticRegression(max_iter=1000).fit(x, np.asarray(labels, dtype=int)))

    def predict(self, scores):
        x = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


def feature_map(breakdown: Dict[str, DetectorOutput]) -> Dict[str, float]:
    """Every fusion feature this breakdown can supply, keyed by name.

    The availability flags matter - 'face match scored 0.5 because it could not
    find a face' is a different world from 'face match scored 0.5 confidently',
    and the model needs to be able to tell them apart.
    """
    out: Dict[str, float] = {}
    for n in DETECTOR_NAMES:
        d = breakdown.get(n)
        ok = d is not None and d.status == "ok"
        out[f"{n}_score"] = float(d.score) if ok else 0.0
        out[f"{n}_conf"] = float(d.confidence) if ok else 0.0
    return out


def feature_vector(
    breakdown: Dict[str, DetectorOutput], names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Build the fusion input in a given feature order.

    `names` defaults to the layout a freshly trained model would use. Passing the
    stored model's own `feature_names` is what lets a model fitted before a
    detector existed keep running afterwards: features are looked up by name, so
    adding a sixth detector appends a column the old model simply never asks for,
    instead of shifting every column it does ask for by one.
    """
    fmap = feature_map(breakdown)
    order = list(names) if names else training_feature_names()
    missing = [n for n in order if n not in fmap]
    if missing:
        raise ValueError(f"fusion features unavailable: {missing}")
    return np.asarray([fmap[n] for n in order], dtype=np.float64), order


def training_feature_names() -> List[str]:
    """The layout `scripts/train_fusion.py` fits, in order."""
    from .config import FUSION_TRAINED_NAMES

    return [f"{n}_{suffix}" for n in FUSION_TRAINED_NAMES for suffix in ("score", "conf")]


@functools.lru_cache(maxsize=1)
def _load_model():
    if not FUSION_PATH.exists():
        return None
    try:
        import joblib

        bundle = joblib.load(FUSION_PATH)
        log.info("loaded fusion model %s", bundle.get("kind"))
        return bundle
    except Exception as exc:  # noqa: BLE001
        log.warning("fusion model unreadable: %s", exc)
        return None


def reload_model() -> None:
    _load_model.cache_clear()


def heuristic_score(breakdown: Dict[str, DetectorOutput]) -> float:
    """Confidence-weighted mean, then a max-boost so one loud detector still lands.

    A plain weighted mean lets four quiet detectors bury one screaming detector,
    which is exactly the failure mode a fraudster optimises for. We take the
    weighted mean and pull it toward the strongest confident signal.
    """
    num = den = 0.0
    strongest = 0.0
    for name, w in HEURISTIC_WEIGHTS.items():
        d = breakdown.get(name)
        if d is None or d.status != "ok":
            continue
        eff = w * max(d.confidence, 0.05)
        num += eff * d.score
        den += eff
        if d.confidence >= 0.5:
            strongest = max(strongest, d.score)
    if den == 0.0:
        return 0.5
    mean = num / den
    return float(np.clip(0.65 * mean + 0.35 * strongest, 0.0, 1.0))


def fuse(breakdown: Dict[str, DetectorOutput]) -> Tuple[float, str]:
    """Return (risk score in [0,1], model identifier)."""
    bundle = _load_model()
    if bundle is not None:
        try:
            x, _ = feature_vector(breakdown, list(bundle.get("feature_names") or []) or None)
            p = float(bundle["model"].predict_proba(x[None, :])[0, 1])
            calib = bundle.get("calibrator")
            if calib is not None:
                p = float(np.clip(calib.predict([p])[0], 0.0, 1.0))
            return p, bundle.get("kind", "logreg")
        except Exception as exc:  # noqa: BLE001
            log.warning("fusion model failed, falling back to heuristic: %s", exc)
    return heuristic_score(breakdown), "heuristic"


def decide(score: float, policy: MerchantPolicy, breakdown: Optional[Dict[str, DetectorOutput]] = None) -> Tuple[str, bool]:
    """Map a score to PASS / REVIEW / REJECT under a merchant's policy.

    Two things can force a REVIEW that the raw score alone would not:
      * the score sits inside the abstention band around a threshold - we would
        rather say "I am not sure" than flip a coin on someone's livelihood;
      * a required input was missing or a detector errored, so the score was
        formed on partial evidence.
    """
    band = max(0.0, policy.abstain_band)
    abstained = False

    if score >= policy.min_risk_for_reject:
        verdict = "REJECT"
        if score < policy.min_risk_for_reject + band:
            verdict, abstained = "REVIEW", True
    elif score >= policy.min_risk_for_review:
        verdict = "REVIEW"
    else:
        verdict = "PASS"
        if score > policy.min_risk_for_review - band:
            verdict, abstained = "REVIEW", True

    if breakdown:
        required_missing = []
        if policy.require_liveness:
            d = breakdown.get("liveness_video")
            if d is None or d.status != "ok":
                required_missing.append("liveness video")
        if policy.require_id_document:
            d = breakdown.get("id_forensics")
            if d is None or d.status != "ok":
                required_missing.append("ID document")
        if getattr(policy, "require_behavioral", False):
            d = breakdown.get("behavioral")
            if d is None or d.status != "ok":
                required_missing.append("behavioral telemetry")
        errored = [n for n, d in breakdown.items() if d.status == "error"]
        if (required_missing or errored) and verdict == "PASS":
            verdict, abstained = "REVIEW", True
    return verdict, abstained


def uncorroborated_ceiling(policy: MerchantPolicy) -> float:
    """The highest risk a single un-modelled evidence channel may contribute alone.

    Derived from the merchant's own policy rather than hard-coded, so a merchant
    who moves their reject threshold moves this with it. It sits one abstention
    band below `min_risk_for_reject`: high enough to force a REVIEW under any
    sane review threshold, low enough that `decide` cannot read it as a REJECT.

    Subtracting the band matters - `decide` already turns a score within one band
    *below* the reject line into an abstaining REVIEW, so landing exactly on
    `min_risk_for_reject - epsilon` would work today but only by way of the
    abstention rule. Clearing the band says the intent in the number itself.
    """
    band = max(0.0, policy.abstain_band)
    return max(policy.min_risk_for_review, policy.min_risk_for_reject - band - 1e-6)


def behavioral_channel(
    breakdown: Dict[str, DetectorOutput], policy: MerchantPolicy
) -> Tuple[float, List[str]]:
    """What Detector 6 is allowed to contribute on its own, and why.

    Split the same way linkage splits exact hashes from similarity hits, and for
    the same reason - what the evidence can bear:

      * a **categorical** hit is a statement about physics or self-declaration:
        two keystrokes 9 ms apart, or a browser that sets `navigator.webdriver`.
        Neither has an innocent explanation, so these may carry a rejection.
      * everything else is **statistical**: low dwell variance, a straight
        pointer path, a fast fill. Every one of those describes some real person
        having an unusual day - a practised operator on their fourth merchant
        signup of the morning types fast, does not correct, and moves in
        straight lines. Rejecting on that alone would deny a livelihood over a
        typing style, so it is ceilinged into a human's queue instead.

    Returns (contribution, reasons). Contribution is 0.0 when the detector did
    not run, so a submission with no telemetry is never *helped* by its absence.
    """
    d = breakdown.get("behavioral")
    if d is None or d.status != "ok":
        return 0.0, []

    hits = d.signals.get("rule_hits") or []
    categorical = any(float(h.get("weight", 0.0)) >= 0.90 for h in hits if isinstance(h, dict))

    # Scale by confidence: twelve keystrokes cannot escalate anything, however
    # damning they look, and the detector already reports how much it saw.
    contribution = float(d.score) * float(d.confidence)
    if not categorical:
        contribution = min(contribution, uncorroborated_ceiling(policy))
    return contribution, list(d.reasons[:2])


def detector_weights() -> Dict[str, float]:
    """What the fusion layer currently believes, for the metrics page."""
    bundle = _load_model()
    if bundle is None:
        return {f"{k}_score": v for k, v in HEURISTIC_WEIGHTS.items()}
    try:
        coefs = bundle["model"].named_steps["clf"].coef_[0] if hasattr(bundle["model"], "named_steps") else bundle["model"].coef_[0]
        return {n: round(float(c), 4) for n, c in zip(bundle["feature_names"], coefs)}
    except Exception:
        return {}
