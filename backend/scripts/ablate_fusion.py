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

from verityne.config import DATASET_ROOT, DETECTOR_NAMES, EVAL_ROOT  # noqa: E402

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
        for n in DETECTOR_NAMES:
            d = r["detectors"].get(n, {})
            ok = d.get("status") == "ok" and n != drop
            feats.append(float(d.get("score", 0.0)) if ok else 0.0)
            feats.append(float(d.get("confidence", 0.0)) if ok else 0.0)
        X.append(feats)
        y.append(int(r["y"]))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)


def fit_and_score(train: List[Dict], test: List[Dict], drop: Optional[str]) -> Dict:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = build_matrix(train, drop)
    Xte, yte = build_matrix(test, drop)
    base = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0))])
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(Xtr, ytr)
    s = model.predict_proba(Xte)[:, 1]
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
                leaked.append({"value": value, "p_fraud": p_fraud, "n": total,
                               "class": "fraud" if p_fraud == 1.0 else "genuine"})
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

    full = fit_and_score(train, test, drop=None)
    log.info("full model held-out AUC %.4f", full["roc_auc"])

    per_detector: Dict[str, Dict] = {}
    for name in DETECTOR_NAMES:
        r = fit_and_score(train, test, drop=name)
        r["auc_drop"] = round(full["roc_auc"] - r["roc_auc"], 4)
        r["share_of_headline_auc_above_chance"] = (
            round(r["auc_drop"] / max(1e-9, full["roc_auc"] - 0.5), 4)
        )
        per_detector[name] = r
        log.info("  without %-18s AUC %.4f  (drop %+.4f)", name, r["roc_auc"], -r["auc_drop"])

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
        "corpus_leak_audit": leak_audit(json.loads(args.manifest.read_text())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)

    leaks = [(f, d) for f, d in report["corpus_leak_audit"].items() if d["one_sided_values"]]
    if leaks:
        log.warning("corpus fields with one-sided values (a label written into the artefact):")
        for field, d in leaks:
            for l in d["one_sided_values"]:
                log.warning("  %s=%s appears on %d packets, all %s", field, l["value"], l["n"], l["class"])


if __name__ == "__main__":
    main()
