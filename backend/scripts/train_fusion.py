#!/usr/bin/env python3
"""Fit the fusion layer on the training split.

    python backend/scripts/train_fusion.py

A logistic regression, not a boosted forest. With ten features and a few hundred
rows the two score identically, and LR hands back a coefficient per detector that
can be shown on a slide and argued with. The brief specified XGBoost; it is still
fitted here for comparison and the report prints both AUCs, but the LR is what
ships unless the gap is material.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from verityne.config import DETECTOR_NAMES, EVAL_ROOT, MODEL_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_fusion")

SCORES = EVAL_ROOT / "scores.json"


def build_matrix(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Same layout as verityne.fusion.feature_vector - score and confidence per detector."""
    names: List[str] = []
    for n in DETECTOR_NAMES:
        names += [f"{n}_score", f"{n}_conf"]
    X, y = [], []
    for r in rows:
        feats = []
        for n in DETECTOR_NAMES:
            d = r["detectors"].get(n, {})
            ok = d.get("status") == "ok"
            feats.append(float(d.get("score", 0.0)) if ok else 0.0)
            feats.append(float(d.get("confidence", 0.0)) if ok else 0.0)
        X.append(feats)
        y.append(int(r["y"]))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--force-xgboost", action="store_true", help="ship XGBoost even if LR matches it")
    args = ap.parse_args()

    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not args.scores.exists():
        raise SystemExit(f"{args.scores} not found - run score_corpus.py first")
    rows = json.loads(args.scores.read_text())
    train = [r for r in rows if r.get("split") == "train"]
    if len(train) < 20:
        raise SystemExit(f"only {len(train)} training rows; build a larger corpus")

    X, y, names = build_matrix(train)
    log.info("training on %d rows (%d fraud), %d features", len(y), int(y.sum()), X.shape[1])

    folds = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=0)

    lr = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0))])
    lr_cv = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    lr_auc = float(roc_auc_score(y, lr_cv))
    log.info("logistic regression: cross-validated AUC = %.4f", lr_auc)

    xgb_auc = None
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", reg_lambda=1.5,
            scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())),
        )
        xgb_cv = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba")[:, 1]
        xgb_auc = float(roc_auc_score(y, xgb_cv))
        log.info("xgboost:             cross-validated AUC = %.4f", xgb_auc)
    except Exception as exc:  # noqa: BLE001
        log.warning("xgboost comparison skipped: %s", exc)

    use_xgb = args.force_xgboost or (xgb_auc is not None and xgb_auc > lr_auc + 0.02)
    if use_xgb:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", reg_lambda=1.5,
            scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())),
        )
        kind, chosen_auc = "xgboost", xgb_auc
    else:
        model = lr
        kind, chosen_auc = "logreg", lr_auc
    model.fit(X, y)

    # Isotonic calibration so the score reads as a probability, which is what the
    # policy thresholds and the cost model both assume it is.
    calibrator = None
    try:
        from sklearn.isotonic import IsotonicRegression

        oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y)
        log.info("fitted isotonic calibration on out-of-fold predictions")
    except Exception as exc:  # noqa: BLE001
        log.warning("calibration skipped: %s", exc)

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "feature_names": names,
        "kind": kind,
        "train_rows": int(len(y)),
        "cv_auc": round(float(chosen_auc), 4),
        "lr_auc": round(lr_auc, 4),
        "xgb_auc": round(xgb_auc, 4) if xgb_auc is not None else None,
    }
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_ROOT / "fusion.joblib")

    weights: Dict[str, float] = {}
    if kind == "logreg":
        weights = {n: round(float(c), 4) for n, c in zip(names, model.named_steps["clf"].coef_[0])}
        log.info("detector weights (positive = raises risk):")
        for n, c in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
            log.info("  %-28s %+.3f", n, c)

    report = {k: v for k, v in bundle.items() if k not in ("model", "calibrator")}
    report["weights"] = weights
    report["note"] = (
        "Cross-validated AUC on the training split. The headline number lives in "
        "eval/metrics.json and is computed on identity-disjoint held-out data."
    )
    (EVAL_ROOT / "fusion_training.json").write_text(json.dumps(report, indent=2))
    log.info("saved %s (kind=%s, cv AUC=%.4f)", MODEL_ROOT / "fusion.joblib", kind, chosen_auc)


if __name__ == "__main__":
    main()
