#!/usr/bin/env python3
"""Fit the pieces that must be calibrated before the detectors can score fairly.

    python backend/scripts/calibrate.py

Three heads, all fitted on the **training split only**:

  1. face-match decision band - the cosine similarities that separate
     "different person" / "same person" / "literally the same picture". These
     depend on the embedding model and on print quality, so hard-coding them was
     wrong; they are fitted.
  2. spectral head - logistic regression over frequency-domain features. This is
     the detector that generalises across generator families, and it is the
     reason the system does not collapse when the pretrained CNN sees a
     generator it was never trained on.
  3. generator fingerprint - multiclass head over the radial spectrum. Labels are
     free because we generated the fakes ourselves.

Everything lands in storage/models/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from verityne.config import DATASET_ROOT, EVAL_ROOT, MODEL_ROOT  # noqa: E402
from verityne.detectors.models import cosine, face_embedder  # noqa: E402
from verityne.utils.images import crop_face, detect_faces, largest_face, load_rgb  # noqa: E402
from verityne.utils.spectral import profile_vector, spectral_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate")

MANIFEST = DATASET_ROOT / "manifest.json"


# ----------------------------------------------------------------------------------
# 1. Face-match decision band
# ----------------------------------------------------------------------------------

def _best_split(low_group: np.ndarray, high_group: np.ndarray) -> float:
    """Threshold maximising balanced separation between two similarity populations."""
    if low_group.size == 0 or high_group.size == 0:
        return float("nan")
    candidates = np.unique(np.concatenate([low_group, high_group]))
    best_t, best_j = float(np.median(candidates)), -1.0
    for t in candidates:
        j = (high_group >= t).mean() + (low_group < t).mean() - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def preserve_better_low_bound(out: Dict, existing: Dict) -> Dict:
    """Keep an LFW-fitted lower bound rather than regressing it to this corpus.

    The two bounds of the face-match band do not have equally good sources. The
    *upper* one is corpus-derived by design - LFW has no identity documents, so
    there is nothing better to fit it on. The *lower* one is an identity
    threshold, and `calibrate_face_match_lfw.py` fits it on 6,000 pairs of real
    photographs of real people.

    That distinction matters because this corpus cannot fit the lower bound
    honestly: its genuine pairs derive from one source photograph each and sit
    near cosine 0.94, far above what two real photographs of one person score.
    Fitting on them produced a threshold of about 0.83, which on LFW rejects
    close to half of genuine applicants - the defect written up in the README.

    `make pipeline` re-runs this script, and writing a fresh dict here quietly
    reinstated exactly that threshold over the LFW one, with no warning and no
    test failing. Anyone following the documented order (`make pipeline`, then
    `make real`) recovered by accident. So: if a better-sourced low bound is on
    disk, it stays, and the corpus value is recorded beside it rather than
    thrown away.
    """
    if not existing.get("lfw"):
        return out
    merged = {**out}
    merged["corpus_fitted_low"] = out["low"]
    merged["corpus_fitted_on"] = out["fitted_on"]
    merged["high_fitted_on"] = out["fitted_on"]
    for key in ("low", "same_person", "far_1pct_threshold", "far_0.1pct_threshold",
                "low_previously", "lfw", "fitted_on"):
        if key in existing:
            merged[key] = existing[key]
    log.warning(
        "keeping the LFW-fitted low bound %.4f; this corpus would have set %.4f, "
        "which LFW showed rejects nearly half of genuine real pairs",
        merged["low"], out["low"],
    )
    return merged


def calibrate_face_band(entries: List[Dict], force_corpus_low: bool = False) -> Optional[Dict]:
    emb = face_embedder()
    if emb is None:
        log.warning("no face embedder; skipping band calibration")
        return None

    buckets: Dict[str, List[float]] = {"genuine": [], "mismatch": [], "duplicate": []}
    for i, e in enumerate(entries):
        attack = e.get("attack_type")
        if e["label"] == "real":
            bucket = "genuine"
        elif attack in ("impersonation", "generated_selfie"):
            bucket = "mismatch"
        elif attack == "reused_id_selfie":
            bucket = "duplicate"
        else:
            continue
        try:
            selfie = largest_face(load_rgb(e["selfie"]), size=256)
            id_rgb = load_rgb(e["id_document"], max_side=1400)
            boxes = detect_faces(id_rgb, min_size=40)
            if selfie is None or not boxes:
                continue
            id_face = crop_face(id_rgb, boxes[0], margin=0.15, size=224)
            buckets[bucket].append(cosine(emb.embed(selfie), emb.embed(id_face)))
        except Exception as exc:  # noqa: BLE001
            log.debug("pair %s failed: %s", e["id"], exc)
        if (i + 1) % 40 == 0:
            log.info("face pairs: %d/%d", i + 1, len(entries))

    g = np.asarray(buckets["genuine"], dtype=np.float64)
    m = np.asarray(buckets["mismatch"], dtype=np.float64)
    d = np.asarray(buckets["duplicate"], dtype=np.float64)
    if g.size < 5:
        log.warning("only %d genuine pairs; keeping default band", g.size)
        return None

    low = _best_split(m, g) if m.size >= 3 else float(np.percentile(g, 2) - 0.05)
    if not np.isfinite(low):
        low = float(np.percentile(g, 2) - 0.05)

    # The upper bound is deliberately NOT fitted to maximise separation from the
    # duplicate class. On this corpus the two populations overlap almost exactly
    # (genuine pairs derive from one source photograph), so a fitted split would
    # buy duplicate recall by rejecting real merchants. We set it above the
    # genuine distribution instead, accept low recall on that attack from this
    # detector, and catch it through image provenance in the metadata auditor.
    high = float(min(0.9995, np.percentile(g, 99) + 0.005))
    separable = None
    if d.size >= 5:
        overlap_labels = np.concatenate([np.zeros(g.size, int), np.ones(d.size, int)])
        try:
            from sklearn.metrics import roc_auc_score

            separable = round(float(roc_auc_score(overlap_labels, np.concatenate([g, d]))), 4)
        except Exception:
            separable = None

    out = {
        "low": round(float(low), 4),
        "high": round(float(high), 4),
        "fitted_on": f"{g.size} genuine / {m.size} mismatch / {d.size} duplicate pairs (train split)",
        "duplicate_separability_auc": separable,
        "distributions": {
            k: {
                "n": int(np.asarray(v).size),
                "mean": round(float(np.mean(v)), 4) if v else None,
                "p05": round(float(np.percentile(v, 5)), 4) if v else None,
                "p50": round(float(np.percentile(v, 50)), 4) if v else None,
                "p95": round(float(np.percentile(v, 95)), 4) if v else None,
            }
            for k, v in buckets.items()
        },
        "caveat": (
            "Genuine pairs in this corpus derive from a single source photograph per identity, "
            "re-captured to approximate a second photo, so their similarity runs far higher than real "
            "selfie-vs-ID pairs would. Consequence: cosine similarity cannot separate a copied ID photo "
            f"from a genuine pair here (separability AUC {separable}). The upper bound is therefore set "
            "above the genuine distribution rather than fitted, and the reused-photo attack is caught by "
            "the image-provenance check in the metadata auditor instead. Re-fit both on production pairs."
        ),
    }
    band_path = MODEL_ROOT / "face_match_band.json"
    existing = {}
    if band_path.exists():
        try:
            existing = json.loads(band_path.read_text())
        except Exception:  # noqa: BLE001
            existing = {}
    if not force_corpus_low:
        out = preserve_better_low_bound(out, existing)

    band_path.write_text(json.dumps(out, indent=2))
    log.info("face-match band: low=%.3f high=%.3f  %s", out["low"], out["high"], out["fitted_on"])
    for k, v in out["distributions"].items():
        log.info("  %-10s n=%-4s mean=%s p05=%s p95=%s", k, v["n"], v["mean"], v["p05"], v["p95"])
    return out


# ----------------------------------------------------------------------------------
# 2 + 3. Spectral head and generator fingerprint
# ----------------------------------------------------------------------------------

def _selfie_face(entry: Dict):
    try:
        return largest_face(load_rgb(entry["selfie"]), size=256)
    except Exception:
        return None


def fit_spectral_heads(entries: List[Dict]) -> Dict:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feats: List[Dict[str, float]] = []
    profiles: List[np.ndarray] = []
    y: List[int] = []
    gen: List[str] = []

    for i, e in enumerate(entries):
        face = _selfie_face(e)
        if face is None:
            continue
        feats.append(spectral_features(face))
        profiles.append(profile_vector(face))
        # "Is this selfie itself synthetic?" - only these attacks put a generated
        # face in the selfie slot. A tampered document has a real selfie.
        y.append(1 if e.get("attack_type") in ("generated_selfie", "synthetic_identity") else 0)
        gen.append(e.get("generator", "real") or "real")
        if (i + 1) % 40 == 0:
            log.info("spectral: %d/%d", i + 1, len(entries))

    result: Dict[str, object] = {"n": len(feats)}
    if len(feats) < 20 or len(set(y)) < 2:
        log.warning("not enough data for the spectral head (n=%d, classes=%s)", len(feats), set(y))
        return result

    names = sorted(feats[0].keys())
    X = np.asarray([[f[k] for k in names] for f in feats], dtype=np.float64)
    yv = np.asarray(y, dtype=int)

    clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    clf.fit(X, yv)
    # sklearn Pipelines expose feature_names_in_ as a read-only property, so the
    # column order travels alongside the estimator instead of being attached to it.
    joblib.dump({"model": clf, "feature_names": names}, MODEL_ROOT / "spectral_lr.joblib")

    from sklearn.metrics import roc_auc_score

    train_auc = float(roc_auc_score(yv, clf.predict_proba(X)[:, 1]))
    coefs = dict(zip(names, [round(float(c), 4) for c in clf.named_steps["clf"].coef_[0]]))
    log.info("spectral head: n=%d train AUC=%.3f", len(feats), train_auc)
    log.info("  weights: %s", json.dumps(coefs))
    result.update({"spectral_train_auc": round(train_auc, 4), "spectral_weights": coefs, "positives": int(yv.sum())})

    # ---- generator fingerprint -------------------------------------------------
    gen_arr = np.asarray(gen)
    keep = np.isin(gen_arr, [g for g in set(gen) if (gen_arr == g).sum() >= 8])
    if keep.sum() >= 20 and len(set(gen_arr[keep])) >= 2:
        Xg = np.concatenate([np.asarray(profiles)[keep], X[keep]], axis=1)
        fp = Pipeline([("scale", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
        fp.fit(Xg, gen_arr[keep])
        joblib.dump(fp, MODEL_ROOT / "generator_fingerprint.joblib")
        acc = float((fp.predict(Xg) == gen_arr[keep]).mean())
        log.info("generator fingerprint: classes=%s train acc=%.3f", sorted(set(gen_arr[keep])), acc)
        result.update({"fingerprint_classes": sorted(set(gen_arr[keep].tolist())), "fingerprint_train_acc": round(acc, 4)})
    else:
        log.warning("not enough per-generator data for fingerprinting")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-face-band", action="store_true")
    ap.add_argument("--skip-spectral", action="store_true")
    ap.add_argument("--force-corpus-low", action="store_true",
                    help="re-fit the face-band lower bound from this corpus even when an "
                         "LFW-fitted one is on disk (it is worse; see preserve_better_low_bound)")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found - run build_dataset.py first")
    entries = [e for e in json.loads(MANIFEST.read_text()) if e.get("split") == "train"]
    if not entries:
        raise SystemExit("manifest has no train split")
    log.info("calibrating on %d training packets", len(entries))

    report: Dict[str, object] = {"train_packets": len(entries)}
    if not args.skip_face_band:
        report["face_band"] = calibrate_face_band(entries, force_corpus_low=args.force_corpus_low)
    if not args.skip_spectral:
        report["spectral"] = fit_spectral_heads(entries)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "calibration.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s", EVAL_ROOT / "calibration.json")


if __name__ == "__main__":
    main()
