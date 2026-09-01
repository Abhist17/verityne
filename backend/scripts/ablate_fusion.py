#!/usr/bin/env python3
"""What is each detector actually worth to the shipped number?

    python backend/scripts/ablate_fusion.py

`eval/metrics.json` reports one held-out fusion AUC. It does not say which
detector earned it, and a single aggregate can hide a model that is really one
feature wearing five. This script refits fusion on the training split with one
detector's features zeroed out at a time, scores the held-out split with the
same protocol `evaluate.py` uses, and reports the drop.

It also audits the corpus for the failure that makes such a drop meaningless:
a generator setting that only ever fires on one class. If some value of a
categorical - an EXIF mode, a document type, a capture path - appears on
fraudulent packets and never on genuine ones, then a detector that reads it is
not detecting fraud, it is reading a label the corpus wrote into the file. That
is a defect in the data, and it inflates every number computed downstream of it.

Writes eval/ablation.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from verityne.config import (  # noqa: E402
    DATASET_ROOT,
    EVAL_ROOT,
    FUSION_TRAINED_NAMES,
    PHYSICALLY_FRAUD_ONLY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ablate")


def build_matrix(rows: List[Dict], drop: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The same layout `train_fusion.build_matrix` uses, with one detector muted.

    Muting is zeroing both of a detector's columns rather than deleting them, so
    every variant keeps the same feature count and the comparison is between
    models of identical capacity.
    """
    X, y = [], []
    for r in rows:
        feats: List[float] = []
        for n in FUSION_TRAINED_NAMES:
            d = r["detectors"].get(n, {})
            ok = d.get("status") == "ok" and n != drop
            feats.append(float(d.get("score", 0.0)) if ok else 0.0)
            feats.append(float(d.get("confidence", 0.0)) if ok else 0.0)
        X.append(feats)
        y.append(int(r["y"]))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)


def fit_and_score(train: List[Dict], test: List[Dict], drop: Optional[str]) -> Dict:
    """Fit and score exactly the way `train_fusion.py` ships.

    Same pipeline, same hyper-parameters, and the same Platt calibration fitted
    on out-of-fold predictions. Matching it matters: a "full model" row that did
    not equal the headline in `eval/metrics.json` would make every drop below it
    arguable.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = build_matrix(train, drop)
    Xte, yte = build_matrix(test, drop)
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0))])
    model.fit(Xtr, ytr)
    s = model.predict_proba(Xte)[:, 1]
    try:
        from verityne.fusion import PlattCalibrator

        cv = StratifiedKFold(n_splits=max(2, min(5, int(np.bincount(ytr).min()))),
                             shuffle=True, random_state=0)
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        s = PlattCalibrator.fit(oof, ytr).predict(s)
    except Exception as exc:  # noqa: BLE001
        log.warning("calibration skipped for drop=%s: %s", drop, exc)
    return {
        "roc_auc": round(float(roc_auc_score(yte, s)), 4),
        "score_mean_genuine": round(float(s[yte == 0].mean()), 4),
        "score_mean_fraud": round(float(s[yte == 1].mean()), 4),
        "recall_at_0.75": round(float(((s >= 0.75) & (yte == 1)).sum() / max(1, (yte == 1).sum())), 4),
        "false_reject_at_0.75": round(float(((s >= 0.75) & (yte == 0)).sum() / max(1, (yte == 0).sum())), 4),
    }


def leak_audit(manifest: List[Dict]) -> Dict:
    """Categorical settings the generator only ever applies to one class.

    A value with P(fraud | value) of exactly 1.0 or 0.0, on a decent number of
    packets, is a label written into the artefact. Nothing downstream can tell
    the difference between reading it and detecting fraud.
    """
    fields = {
        "selfie_exif_mode": lambda e: (e.get("selfie_exif") or {}).get("exif_mode"),
        "id_exif_mode": lambda e: (e.get("id_exif") or {}).get("exif_mode"),
        "doc_type": lambda e: e.get("doc_type"),
        "capture_mode": lambda e: e.get("capture_mode"),
    }
    out: Dict[str, Dict] = {}
    for field, get in fields.items():
        table: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for e in manifest:
            v = get(e)
            if v is not None:
                table[str(v)][e["label"]] += 1
        rows, leaked = {}, []
        for value, counts in sorted(table.items()):
            real, fake = counts.get("real", 0), counts.get("fake", 0)
            total = real + fake
            p_fraud = round(fake / total, 4) if total else None
            rows[value] = {"genuine": real, "fraud": fake, "p_fraud": p_fraud}
            # One-sided and not a rounding artefact of two or three packets.
            if total >= 10 and p_fraud in (0.0, 1.0):
                # Some values are one-sided because reality is: no camera writes a
                # diffusion tag. Those are reported, so the reader can check the
                # reasoning, but they are not defects and must not fail a build.
                justification = PHYSICALLY_FRAUD_ONLY.get(value)
                leaked.append({"value": value, "p_fraud": p_fraud, "n": total,
                               "class": "fraud" if p_fraud == 1.0 else "genuine",
                               "expected": justification is not None,
                               "justification": justification})
        n_leaked = sum(r["genuine"] + r["fraud"] for v, r in rows.items()
                       if any(l["value"] == v for l in leaked))
        out[field] = {
            "values": rows,
            "one_sided_values": leaked,
            "packets_carrying_a_one_sided_value": n_leaked,
            "share_of_corpus": round(n_leaked / max(1, len(manifest)), 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=EVAL_ROOT / "scores.json")
    ap.add_argument("--manifest", type=Path, default=DATASET_ROOT / "manifest.json")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "ablation.json")
    args = ap.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"{args.scores} not found - run score_corpus.py first")
    rows = json.loads(args.scores.read_text())
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    if not train or not test:
        raise SystemExit("need both a train and a test split in scores.json")
    log.info("ablating on %d train / %d held-out rows", len(train), len(test))

    audit = leak_audit(json.loads(args.manifest.read_text()))

    full = fit_and_score(train, test, drop=None)
    log.info("full model held-out AUC %.4f", full["roc_auc"])

    per_detector: Dict[str, Dict] = {}
    for name in FUSION_TRAINED_NAMES:
        r = fit_and_score(train, test, drop=name)
        r["auc_drop"] = round(full["roc_auc"] - r["roc_auc"], 4)
        r["share_of_headline_auc_above_chance"] = (
            round(r["auc_drop"] / max(1e-9, full["roc_auc"] - 0.5), 4)
        )
        per_detector[name] = r
        # Printed as the change in AUC caused by muting: negative means the model
        # got worse without it, which is a detector doing its job.
        log.info("  without %-18s AUC %.4f  (delta %+.4f, %+.1f%% of above-chance)",
                 name, r["roc_auc"], -r["auc_drop"],
                 100 * r["share_of_headline_auc_above_chance"])

    ranked = sorted(per_detector.items(), key=lambda kv: -kv[1]["auc_drop"])
    report = {
        "protocol": (
            "Fusion refit on the training split with one detector's score and confidence "
            "columns zeroed, then scored on the identity-disjoint held-out split. Same "
            "estimator and calibration as train_fusion.py ships."
        ),
        "n_train": len(train),
        "n_test": len(test),
        "full_model": full,
        "per_detector": per_detector,
        "most_load_bearing": ranked[0][0],
        "corpus_leak_audit": audit,
        "corpus_clean": not any(
            not l["expected"] for d in audit.values() for l in d["one_sided_values"]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)

    defects = [(f, l) for f, d in report["corpus_leak_audit"].items()
               for l in d["one_sided_values"] if not l["expected"]]
    expected = [(f, l) for f, d in report["corpus_leak_audit"].items()
                for l in d["one_sided_values"] if l["expected"]]
    for field, l in expected:
        log.info("one-sided by construction: %s=%s (%d packets) - %s",
                 field, l["value"], l["n"], l["justification"])
    if defects:
        log.error("corpus leak - a label written into the artefact:")
        for field, l in defects:
            log.error("  %s=%s appears on %d packets, all %s", field, l["value"], l["n"], l["class"])
        raise SystemExit(
            f"{len(defects)} corpus value(s) identify the class outright; every AUC above is "
            "inflated by an unknown amount until that is fixed"
        )


if __name__ == "__main__":
    main()
