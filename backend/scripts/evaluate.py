#!/usr/bin/env python3
"""Produce the held-out evaluation report.

    python backend/scripts/evaluate.py

Everything here is computed on the **test split**, which is identity-disjoint
from training: no face that trained the fusion layer appears in these numbers.

The report deliberately includes the things that make a system look worse:
per-attack recall (so a weak detector cannot hide behind a strong one),
per-capture-mode tamper performance, a skin-tone bias audit, and the count of
detectors that errored or were skipped.
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

from verityne.config import DETECTOR_LABELS, DETECTOR_NAMES, EVAL_ROOT, MODEL_ROOT, get_policy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")

SCORES = EVAL_ROOT / "scores.json"
REPORT = EVAL_ROOT / "metrics.json"


# ----------------------------------------------------------------------------------
# Skin-tone proxy for the bias audit
# ----------------------------------------------------------------------------------

def individual_typology_angle(face_rgb) -> Optional[float]:
    """ITA°, the standard image-based proxy for skin tone.

    ITA = arctan((L* - 50) / b*) in CIELAB, measured over skin-like pixels.
    It is a *proxy*: it reads pixels under whatever lighting the photo had, and it
    is not self-reported demographic data. It is reported as a proxy everywhere it
    appears, because a bias audit that overstates its own ground truth is worse
    than no audit.
    """
    import cv2

    if face_rgb is None:
        return None
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * 100.0 / 255.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    # Keep plausible skin pixels: positive b*, moderate a*, mid-to-high lightness.
    mask = (b > 5) & (a > 0) & (a < 40) & (L > 15) & (L < 95)
    if mask.sum() < 200:
        return None
    return float(np.degrees(np.arctan2(np.median(L[mask]) - 50.0, np.median(b[mask]))))


ITA_BINS = [
    (55.0, 999.0, "very_light"),
    (41.0, 55.0, "light"),
    (28.0, 41.0, "intermediate"),
    (10.0, 28.0, "tan"),
    (-30.0, 10.0, "brown"),
    (-999.0, -30.0, "dark"),
]


def ita_bucket(ita: Optional[float]) -> str:
    if ita is None:
        return "unknown"
    for lo, hi, name in ITA_BINS:
        if lo <= ita < hi:
            return name
    return "unknown"


# ----------------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------------

def safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(set(y.tolist())) < 2:
        return None
    return round(float(roc_auc_score(y, s)), 4)


def confusion_at(y: np.ndarray, s: np.ndarray, t: float) -> Dict[str, int]:
    pred = (s >= t).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def rates(cm: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
        "false_accept_rate": round(fn / max(1, tp + fn), 4),
        "false_reject_rate": round(fp / max(1, fp + tn), 4),
        "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4),
    }


def roc_points(y: np.ndarray, s: np.ndarray, n: int = 60) -> List[Dict[str, float]]:
    from sklearn.metrics import roc_curve

    if len(set(y.tolist())) < 2:
        return []
    fpr, tpr, thr = roc_curve(y, s)
    idx = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    return [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4),
             "threshold": round(float(thr[i]), 4) if np.isfinite(thr[i]) else 1.0} for i in idx]


def fuse_rows(rows: List[Dict]) -> Tuple[np.ndarray, str]:
    """Score every row through the shipped fusion path, so the report grades what runs."""
    from verityne.fusion import fuse
    from verityne.schemas import DetectorOutput

    out, kind = [], "heuristic"
    for r in rows:
        breakdown = {
            n: DetectorOutput(
                name=n, label=DETECTOR_LABELS.get(n, n),
                score=float(d.get("score", 0.0)), confidence=float(d.get("confidence", 0.0)),
                status=d.get("status", "ok"),
            )
            for n, d in r["detectors"].items()
        }
        s, kind = fuse(breakdown)
        out.append(s)
    return np.asarray(out, dtype=float), kind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bias-audit", action="store_true", default=True)
    ap.add_argument("--no-bias-audit", dest="bias_audit", action="store_false")
    args = ap.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"{args.scores} not found - run score_corpus.py first")
    all_rows = json.loads(args.scores.read_text())
    rows = [r for r in all_rows if r.get("split") == args.split]
    if len(rows) < 10:
        raise SystemExit(f"only {len(rows)} rows in the {args.split} split")

    y = np.asarray([r["y"] for r in rows], dtype=int)
    fused, fusion_kind = fuse_rows(rows)
    policy = get_policy()
    log.info("evaluating %d held-out packets (%d fraud) with fusion=%s", len(rows), int(y.sum()), fusion_kind)

    # ---- per-detector AUC ----------------------------------------------------
    per_detector: Dict[str, Dict] = {}
    for n in DETECTOR_NAMES:
        s = np.asarray([r["detectors"].get(n, {}).get("score", 0.0) for r in rows], dtype=float)
        statuses = [r["detectors"].get(n, {}).get("status", "missing") for r in rows]
        ok_mask = np.asarray([st == "ok" for st in statuses])
        entry: Dict[str, object] = {
            "label": DETECTOR_LABELS[n],
            "auc_all_rows": safe_auc(y, s),
            "coverage": round(float(ok_mask.mean()), 4),
            "errors": int(sum(1 for st in statuses if st == "error")),
            "skipped": int(sum(1 for st in statuses if st == "skipped")),
            "mean_genuine": round(float(s[(y == 0) & ok_mask].mean()), 4) if ((y == 0) & ok_mask).any() else None,
            "mean_fraud": round(float(s[(y == 1) & ok_mask].mean()), 4) if ((y == 1) & ok_mask).any() else None,
            "roc": roc_points(y[ok_mask], s[ok_mask]) if ok_mask.sum() > 4 else [],
        }
        if ok_mask.sum() > 4:
            entry["auc_where_it_ran"] = safe_auc(y[ok_mask], s[ok_mask])
        per_detector[n] = entry

    # A detector is only fairly judged against the attacks it is meant to catch.
    detector_targets = {
        "selfie_deepfake": {"generated_selfie", "synthetic_identity"},
        "liveness_video": {"face_swap_liveness"},
        "id_forensics": {"tampered_document", "invalid_document", "synthetic_identity"},
        "face_match": {"impersonation", "reused_id_selfie", "generated_selfie"},
        "metadata_exif": {"stale_or_edited_media", "recaptured_screen"},
    }
    for n, targets in detector_targets.items():
        mask = np.asarray([(r["y"] == 0) or (r.get("attack_type") in targets) for r in rows])
        if mask.sum() > 6:
            s = np.asarray([r["detectors"].get(n, {}).get("score", 0.0) for r in rows], dtype=float)
            per_detector[n]["auc_on_target_attacks"] = safe_auc(y[mask], s[mask])
            per_detector[n]["target_attacks"] = sorted(targets)

    # ---- fusion --------------------------------------------------------------
    fusion_auc = safe_auc(y, fused)
    cm_reject = confusion_at(y, fused, policy.min_risk_for_reject)
    cm_review = confusion_at(y, fused, policy.min_risk_for_review)

    # ---- per attack type -----------------------------------------------------
    per_attack: Dict[str, Dict] = {}
    genuine_scores = fused[y == 0]
    for attack in sorted({r.get("attack_type") for r in rows if r["y"] == 1 and r.get("attack_type")}):
        mask = np.asarray([r.get("attack_type") == attack for r in rows])
        s = fused[mask]
        per_attack[attack] = {
            "n": int(mask.sum()),
            "mean_score": round(float(s.mean()), 4),
            "caught_at_review": round(float((s >= policy.min_risk_for_review).mean()), 4),
            "caught_at_reject": round(float((s >= policy.min_risk_for_reject).mean()), 4),
            "auc_vs_genuine": safe_auc(
                np.concatenate([np.zeros(len(genuine_scores), int), np.ones(len(s), int)]),
                np.concatenate([genuine_scores, s]),
            ),
        }

    # ---- tamper detection by capture mode ------------------------------------
    manifest_path = Path(__file__).resolve().parents[2] / "datasets" / "manifest.json"
    capture_by_id: Dict[str, str] = {}
    if manifest_path.exists():
        for e in json.loads(manifest_path.read_text()):
            capture_by_id[e["id"]] = e.get("capture_mode", "unknown")
    per_capture: Dict[str, Dict] = {}
    for mode in ("photo", "scan"):
        mask = np.asarray([capture_by_id.get(r["id"]) == mode for r in rows])
        sub = [r for r, m in zip(rows, mask) if m]
        if len(sub) < 6:
            continue
        yy = np.asarray([r["y"] for r in sub], dtype=int)
        ss = np.asarray([r["detectors"].get("id_forensics", {}).get("score", 0.0) for r in sub], dtype=float)
        tamper_mask = np.asarray([(r["y"] == 0) or (r.get("attack_type") == "tampered_document") for r in sub])
        per_capture[mode] = {
            "n": len(sub),
            "id_forensics_auc": safe_auc(yy, ss),
            "tamper_only_auc": safe_auc(yy[tamper_mask], ss[tamper_mask]) if tamper_mask.sum() > 6 else None,
            "fusion_auc": safe_auc(yy, fused[mask]),
        }

    # ---- bias audit ----------------------------------------------------------
    bias: Dict[str, object] = {"method": "ITA° skin-tone proxy computed from selfie pixels", "buckets": {}}
    if args.bias_audit:
        from verityne.utils.images import largest_face, load_rgb

        by_id = {e["id"]: e for e in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
        tones: List[str] = []
        for r in rows:
            e = by_id.get(r["id"])
            face = None
            if e:
                try:
                    face = largest_face(load_rgb(e["selfie"]), size=224)
                except Exception:
                    face = None
            tones.append(ita_bucket(individual_typology_angle(face)))
        for bucket in sorted(set(tones)):
            mask = np.asarray([t == bucket for t in tones])
            if mask.sum() < 5:
                bias["buckets"][bucket] = {"n": int(mask.sum()), "note": "too few samples to report a rate"}
                continue
            yy, ss = y[mask], fused[mask]
            cm = confusion_at(yy, ss, policy.min_risk_for_reject)
            bias["buckets"][bucket] = {
                "n": int(mask.sum()),
                "auc": safe_auc(yy, ss),
                "false_reject_rate": rates(cm)["false_reject_rate"],
                "false_accept_rate": rates(cm)["false_accept_rate"],
                "mean_score_genuine": round(float(ss[yy == 0].mean()), 4) if (yy == 0).any() else None,
            }
        reportable = [b for b in bias["buckets"].values() if b.get("auc") is not None]
        bias["caveat"] = (
            "ITA° is an image-derived proxy for skin tone, not self-reported demographic data, and it is "
            "confounded by lighting. With "
            f"{len(reportable)} bucket(s) above the reporting threshold on {len(rows)} held-out packets, this "
            "audit is a harness and a direction, not a powered conclusion."
        )

    # ---- latency -------------------------------------------------------------
    lat = np.asarray([r["latency_ms"] for r in rows], dtype=float)
    per_det_lat: Dict[str, float] = {}
    for n in DETECTOR_NAMES:
        vals = [r.get("per_detector_latency_ms", {}).get(n) for r in rows]
        vals = [v for v in vals if v]
        if vals:
            per_det_lat[n] = round(float(np.median(vals)), 1)

    report = {
        "split": args.split,
        "n_packets": len(rows),
        "n_fraud": int(y.sum()),
        "n_genuine": int((y == 0).sum()),
        "fusion_model": fusion_kind,
        "fusion": {
            "roc_auc": fusion_auc,
            "roc": roc_points(y, fused),
            "at_reject_threshold": {"threshold": policy.min_risk_for_reject, **cm_reject, **rates(cm_reject)},
            "at_review_threshold": {"threshold": policy.min_risk_for_review, **cm_review, **rates(cm_review)},
            "score_mean_genuine": round(float(fused[y == 0].mean()), 4),
            "score_mean_fraud": round(float(fused[y == 1].mean()), 4),
        },
        "per_detector": per_detector,
        "per_attack_type": per_attack,
        "per_capture_mode": per_capture,
        "bias_audit": bias,
        "latency_ms": {
            "note": "measured in batch mode, detectors run sequentially; the API runs them concurrently",
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p90": round(float(np.percentile(lat, 90)), 1),
            "p99": round(float(np.percentile(lat, 99)), 1),
            "per_detector_median": per_det_lat,
        },
        "raw_scores": {"y_true": y.tolist(), "y_score": [round(float(v), 6) for v in fused],
                       "ids": [r["id"] for r in rows]},
        "limitations": [
            "Liveness clips are animated from stills, so video numbers measure face-swap separability "
            "under identical capture conditions, not end-to-end liveness on real recordings.",
            "Genuine selfie/ID pairs derive from one source photograph per identity, re-captured to "
            "approximate a second photo; real pairs would show lower similarity.",
            "All ID documents are synthetic by design - no real citizen's document is used anywhere.",
            "The bias audit uses an image-derived skin-tone proxy and is under-powered at this corpus size.",
        ],
    }

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))

    log.info("=" * 68)
    log.info("HELD-OUT RESULTS  (n=%d, fraud=%d, fusion=%s)", len(rows), int(y.sum()), fusion_kind)
    log.info("  fusion ROC-AUC              %s", fusion_auc)
    for n, d in per_detector.items():
        log.info("  %-22s AUC %-8s target-AUC %-8s cov %.2f",
                 n, d["auc_all_rows"], d.get("auc_on_target_attacks"), d["coverage"])
    r = rates(cm_reject)
    log.info("  at reject threshold %.2f -> precision %.3f recall %.3f FAR %.3f FRR %.3f",
             policy.min_risk_for_reject, r["precision"], r["recall"], r["false_accept_rate"], r["false_reject_rate"])
    log.info("  per-attack recall @review: %s",
             json.dumps({k: v["caught_at_review"] for k, v in per_attack.items()}))
    log.info("  wrote %s", REPORT)


if __name__ == "__main__":
    main()
