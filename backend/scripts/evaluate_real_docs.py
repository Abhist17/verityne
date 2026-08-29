#!/usr/bin/env python3
"""Score the ID forensics detector on real captured documents.

    python backend/scripts/build_real_docs.py --per-capture 250
    python backend/scripts/evaluate_real_docs.py

This is the real-data counterpart to the ID-forensics rows in
``eval/metrics.json``. Those were measured on cards ``scripts/idcards.py``
rendered; these are measured on documents that were printed, photographed and
scanned by somebody else (see ``scripts/midv2020.py``).

Four things get reported that the synthetic run cannot produce:

1. **False-positive rate on real untampered documents.** How often the detector
   calls an honest card edited, when the only thing that happened to it was a
   printer, a camera and a JPEG encoder. This is the number that decides whether
   the check is deployable, and a synthetic corpus cannot answer it because we
   made the "untampered" images ourselves.

2. **Tamper AUC under real capture noise**, split by photo vs scan, directly
   comparable to the ``Tamper-only AUC`` column already in the README.

3. **Per-attack breakdown.** ``copy_move`` is expected to be much the hardest —
   its pasted pixels share the host document's compression history — and
   reporting the three separately keeps that from hiding inside an average.

4. **Localisation.** We know exactly which rectangle was edited, so we can ask
   whether the detector's top ELA region actually lands on it rather than
   merely producing a high score for some other reason. A detector that scores
   well while pointing at the wrong part of the card is right by accident, and
   nothing in the existing evaluation would notice.

Only the sub-scores that mean something on these documents are reported. The
structural field checks (PAN holder-type codes, the Aadhaar Verhoeff digit) are
built for Indian documents and MIDV-2020 contains none, so they are excluded
rather than scored against documents they were never meant to read.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from verityne.config import DATASET_ROOT, EVAL_ROOT  # noqa: E402
from verityne.detectors.base import SubmissionPayload  # noqa: E402
from verityne.detectors.id_forensics import IDForensicsDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("realdocs.eval")


def auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """ROC-AUC, or None when one class is absent."""
    y = np.asarray(labels)
    if y.size == 0 or len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, np.asarray(scores, dtype=float))), 4)


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / max(1, union))


def hits_truth(regions: Sequence[Dict], truth: Sequence[int], top_k: int = 3) -> Tuple[bool, float]:
    """Does any of the top-k flagged regions overlap the region we actually edited?

    Scored by overlap rather than IoU alone: ELA blobs are ragged and rarely
    match a rectangle tightly, so the fair question is whether the flagged blob
    sits on the edit, not whether it traces its outline.
    """
    best = 0.0
    hit = False
    tx1, ty1, tx2, ty2 = truth
    t_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    for r in list(regions)[:top_k]:
        box = r.get("bbox")
        if not box:
            continue
        best = max(best, iou(box, truth))
        ix1, iy1 = max(box[0], tx1), max(box[1], ty1)
        ix2, iy2 = min(box[2], tx2), min(box[3], ty2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter / t_area >= 0.25:  # a quarter of the edit is inside a flagged blob
            hit = True
    return hit, round(best, 4)


def ela_region_diagnostic(root: Path, records: Sequence[Dict], limit: int = 0) -> Dict:
    """Is there ELA signal at the edit that the detector's threshold discards?

    A detector can fail two ways, and telling them apart matters. Either the
    evidence is not there, or it is there and the decision rule misses it. The
    headline AUC cannot distinguish them, so this looks underneath: for each
    tampered document, the mean robust z-score *inside* the rectangle we edited
    against the mean outside it, on the same edge-normalised ELA map
    ``tamper_score`` thresholds.

    Cheap enough to always run — it needs the ELA map only, not OCR — and it is
    what turns "the check scores at chance" into a statement about why.
    """
    from verityne.utils.ela import edge_normalised_ela
    from verityne.utils.images import load_rgb

    rows = [r for r in records if r.get("label") == "tampered" and r.get("region")]
    if limit:
        rows = rows[:limit]

    by_attack: Dict[str, List[float]] = {}
    inside: List[float] = []
    outside: List[float] = []
    peak: List[float] = []
    for r in rows:
        try:
            rgb = load_rgb(root / r["path"], max_side=1400)
            m = edge_normalised_ela(rgb)
        except Exception:  # noqa: BLE001
            continue
        med = float(np.median(m))
        mad = float(np.median(np.abs(m - med))) + 1e-6
        z = (m - med) / (1.4826 * mad)
        h, w = z.shape
        x1, y1, x2, y2 = (int(v) for v in r["region"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        mask = np.zeros_like(z, dtype=bool)
        mask[y1:y2, x1:x2] = True
        zi, zo = float(z[mask].mean()), float(z[~mask].mean())
        inside.append(zi)
        outside.append(zo)
        peak.append(float(z.max()))
        by_attack.setdefault(r["attack"], []).append(zi - zo)

    if not inside:
        return {"n": 0}

    ins, outs = np.asarray(inside), np.asarray(outside)
    diff = ins - outs
    return {
        "n": len(ins),
        "mean_z_inside_edit": round(float(ins.mean()), 4),
        "mean_z_outside_edit": round(float(outs.mean()), 4),
        "mean_difference": round(float(diff.mean()), 4),
        "fraction_edit_is_hotter": round(float((diff > 0).mean()), 4),
        "median_peak_z_anywhere": round(float(np.median(peak)), 3),
        "firing_threshold": 7.5,
        "by_attack": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_attack.items())},
        "reading": (
            "tamper_score() flags pixels above z=7.5 and reports only blobs that are "
            "HOTTER than the document's baseline. Compare median_peak_z_anywhere against "
            "firing_threshold to see whether anything could fire at all, and the sign of "
            "mean_difference to see whether the edit is hotter or quieter than its "
            "surroundings. A negative mean_difference means the evidence exists but points "
            "the opposite way to what the rule looks for."
        ),
    }


def rate_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    gen, tam = scores[labels == 0], scores[labels == 1]
    return {
        "threshold": thr,
        "recall": round(float((tam >= thr).mean()), 4) if tam.size else None,
        "false_positive_rate": round(float((gen >= thr).mean()), 4) if gen.size else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_ROOT / "real_docs"))
    ap.add_argument("--out", default=str(EVAL_ROOT / "real_docs.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--diagnostic-only", action="store_true",
                    help="recompute only the ELA region diagnostic and merge it into an "
                         "existing report — needs no OCR, so it takes minutes not an hour")
    args = ap.parse_args()

    root = Path(args.data)
    manifest = json.loads((root / "manifest.json").read_text())
    records = manifest["records"]
    if args.limit:
        records = records[: args.limit]

    if args.diagnostic_only:
        out_path = Path(args.out)
        report = json.loads(out_path.read_text())
        log.info("recomputing the ELA region diagnostic over %d records", len(records))
        report["ela_region_diagnostic"] = ela_region_diagnostic(root, records)
        out_path.write_text(json.dumps(report, indent=2))
        log.info("merged into %s", out_path)
        print(json.dumps(report["ela_region_diagnostic"], indent=2))
        return

    log.info("scoring %d documents", len(records))

    det = IDForensicsDetector()
    rows: List[Dict] = []
    t0 = time.time()
    for i, r in enumerate(records):
        payload = SubmissionPayload(
            submission_id=f"realdoc_{i}",
            merchant_id="eval",
            id_doc_path=root / r["path"],
        )
        out = det.run(payload)
        subs = out.signals.get("subscores", {}) if out.signals else {}
        ela = out.signals.get("ela", {}) if out.signals else {}
        row = {
            "path": r["path"],
            "label": r["label"],
            "y": 1 if r["label"] == "tampered" else 0,
            "attack": r.get("attack"),
            "capture": r["capture"],
            "doc_type": r["doc_type"],
            "status": out.status,
            "score": round(float(out.score), 4),
            "tamper_subscore": round(float(subs.get("tamper", 0.0)), 4),
            "photo_subscore": round(float(subs.get("photo", 0.0)), 4),
            "layout_subscore": round(float(subs.get("layout", 0.0)), 4),
            "ela_peak_ratio": ela.get("peak_ratio"),
            "ela_coverage": ela.get("coverage"),
            "n_regions": len(ela.get("regions", []) or []),
            "latency_ms": round(float(out.latency_ms or 0.0), 1),
        }
        if r.get("region") and ela.get("regions"):
            hit, best_iou = hits_truth(ela["regions"], r["region"])
            row["localised"] = hit
            row["best_region_iou"] = best_iou
        elif r.get("region"):
            row["localised"] = False
            row["best_region_iou"] = 0.0
        rows.append(row)
        if (i + 1) % 25 == 0:
            done = i + 1
            rate = done / (time.time() - t0)
            log.info("  %d/%d  %.2f docs/s  eta %.0fs", done, len(records), rate,
                     (len(records) - done) / max(rate, 1e-6))

    ok = [r for r in rows if r["status"] == "ok"]
    y = np.asarray([r["y"] for r in ok])
    overall = np.asarray([r["score"] for r in ok])
    tamper = np.asarray([r["tamper_subscore"] for r in ok])

    def slice_report(sel: Sequence[bool], label: str) -> Optional[Dict]:
        idx = np.asarray(sel, dtype=bool)
        if idx.sum() == 0:
            return None
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            return {"n": int(idx.sum()), "note": f"only one class present in {label}"}
        return {
            "n": int(idx.sum()),
            "n_tampered": int(yy.sum()),
            "detector_auc": auc(overall[idx], yy),
            "tamper_subscore_auc": auc(tamper[idx], yy),
        }

    captures = sorted({r["capture"] for r in ok})
    by_capture = {c: slice_report([r["capture"] == c for r in ok], c) for c in captures}

    # Per attack: each attack class against the full genuine pool, so the
    # comparison is "this attack vs honest documents" rather than one attack
    # against another.
    genuine_mask = y == 0
    by_attack: Dict[str, Dict] = {}
    for atk in sorted({r["attack"] for r in ok if r["attack"]}):
        mask = genuine_mask | np.asarray([r["attack"] == atk for r in ok])
        sub_y = y[mask]
        atk_rows = [r for r in ok if r["attack"] == atk]
        loc = [r for r in atk_rows if "localised" in r]
        by_attack[atk] = {
            "n": len(atk_rows),
            "detector_auc": auc(overall[mask], sub_y),
            "tamper_subscore_auc": auc(tamper[mask], sub_y),
            "mean_tamper_subscore": round(float(np.mean([r["tamper_subscore"] for r in atk_rows])), 4),
            "localised_rate": round(sum(r["localised"] for r in loc) / len(loc), 4) if loc else None,
            "mean_best_iou": round(float(np.mean([r["best_region_iou"] for r in loc])), 4) if loc else None,
        }

    by_doc_type = {}
    for dt in sorted({r["doc_type"] for r in ok}):
        rep = slice_report([r["doc_type"] == dt for r in ok], dt)
        if rep:
            by_doc_type[dt] = rep

    gen_rows = [r for r in ok if r["y"] == 0]
    loc_rows = [r for r in ok if "localised" in r]

    report = {
        "dataset": manifest.get("dataset"),
        "provenance": manifest.get("provenance"),
        "encode_path": manifest.get("encode_path"),
        "n_scored": len(ok),
        "n_errored": len(rows) - len(ok),
        "counts": manifest.get("counts"),
        "headline": {
            "detector_auc": auc(overall, y),
            "tamper_subscore_auc": auc(tamper, y),
            "note": "detector_auc is the full id_forensics score; tamper_subscore_auc isolates "
                    "the Error Level Analysis component, which is the part these documents test.",
        },
        "false_positives_on_real_documents": {
            "n_genuine": len(gen_rows),
            "mean_detector_score": round(float(np.mean([r["score"] for r in gen_rows])), 4),
            "mean_tamper_subscore": round(float(np.mean([r["tamper_subscore"] for r in gen_rows])), 4),
            "flagged_by_ela_reason_rate": round(
                float(np.mean([1.0 if r["tamper_subscore"] > 0.45 and r["n_regions"] else 0.0 for r in gen_rows])), 4),
            "layout_flag_rate": round(float(np.mean([1.0 if r["layout_subscore"] > 0 else 0.0 for r in gen_rows])), 4),
            "note": "flagged_by_ela_reason_rate uses the same >0.45-with-regions test the detector "
                    "uses to emit its 'localised edited region' reason, so it is the rate at which "
                    "an honest document is told it was edited.",
        },
        "operating_points": {
            f"tamper_subscore>={t}": rate_at(tamper, y, t) for t in (0.3, 0.45, 0.6)
        },
        "by_capture": by_capture,
        "by_attack": by_attack,
        "by_doc_type": by_doc_type,
        "localisation": {
            "n_with_ground_truth": len(loc_rows),
            "hit_rate": round(sum(r["localised"] for r in loc_rows) / len(loc_rows), 4) if loc_rows else None,
            "mean_best_iou": round(float(np.mean([r["best_region_iou"] for r in loc_rows])), 4) if loc_rows else None,
            "definition": "a hit means one of the top-3 flagged ELA regions covers at least 25% "
                          "of the rectangle we actually edited",
        },
        "ela_region_diagnostic": ela_region_diagnostic(root, records),
        "excluded_subscores": {
            "fields": "PAN/Aadhaar structural validation — MIDV-2020 contains neither, so the "
                      "field check is not exercised and is excluded rather than scored against "
                      "documents it was not built to read.",
        },
        "latency_ms": {
            "mean": round(float(np.mean([r["latency_ms"] for r in ok])), 1),
            "p50": round(float(np.percentile([r["latency_ms"] for r in ok], 50)), 1),
            "p90": round(float(np.percentile([r["latency_ms"] for r in ok], 90)), 1),
            "note": "OCR dominates; it runs on these documents even though the field check "
                    "cannot use the result.",
        },
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out_path)

    h = report["headline"]
    fp = report["false_positives_on_real_documents"]
    print(f"\nreal documents: {len(ok)} scored ({fp['n_genuine']} genuine)")
    print(f"  id_forensics AUC        {h['detector_auc']}")
    print(f"  tamper (ELA) AUC        {h['tamper_subscore_auc']}")
    print(f"  false 'edited' calls on honest documents: {fp['flagged_by_ela_reason_rate']:.1%}")
    print(f"  localisation hit rate   {report['localisation']['hit_rate']}")
    d = report.get("ela_region_diagnostic") or {}
    if d.get("n"):
        print(f"  ELA at the edit: z inside {d['mean_z_inside_edit']} vs outside "
              f"{d['mean_z_outside_edit']} (peak z {d['median_peak_z_anywhere']}, "
              f"fires at {d['firing_threshold']})")
    for c, rep in by_capture.items():
        if rep and "tamper_subscore_auc" in rep:
            print(f"  {c:6s} n={rep['n']:4d}  detector {rep['detector_auc']}  tamper {rep['tamper_subscore_auc']}")
    for a, rep in by_attack.items():
        print(f"  {a:15s} n={rep['n']:4d}  tamper AUC {rep['tamper_subscore_auc']}  "
              f"localised {rep['localised_rate']}")


if __name__ == "__main__":
    main()
