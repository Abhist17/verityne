"""Fusion layer: five detector scores in, one calibrated risk score out.

Deliberately a *logistic regression*, not a boosted forest. With five features
and a few hundred training rows, an XGBoost model buys no measurable AUC over
LR but costs interpretability - and interpretability is the product here. LR
gives us a per-detector weight we can put on screen and defend.

`scripts/train_fusion.py` fits it and writes storage/models/fusion.joblib along
with a calibration curve. Until that file exists we run a documented heuristic
so the system is never dead on arrival.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
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


def feature_vector(breakdown: Dict[str, DetectorOutput]) -> Tuple[np.ndarray, List[str]]:
    """Build the fusion input: each detector's score plus whether it actually ran.

    The availability flags matter - 'face match scored 0.5 because it could not
    find a face' is a different world from 'face match scored 0.5 confidently',
    and the model needs to be able to tell them apart.
    """
    feats: List[float] = []
    names: List[str] = []
    for n in DETECTOR_NAMES:
        d = breakdown.get(n)
        ok = d is not None and d.status == "ok"
        feats.append(float(d.score) if ok else 0.0)
        names.append(f"{n}_score")
        feats.append(float(d.confidence) if ok else 0.0)
        names.append(f"{n}_conf")
    return np.asarray(feats, dtype=np.float64), names


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
            x, names = feature_vector(breakdown)
            expected = bundle.get("feature_names")
            if expected and list(expected) != names:
                raise ValueError("feature layout drift")
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
        errored = [n for n, d in breakdown.items() if d.status == "error"]
        if (required_missing or errored) and verdict == "PASS":
            verdict, abstained = "REVIEW", True
    return verdict, abstained


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
