#!/usr/bin/env python3
"""Score the liveness detector on recorded deepfake video.

    python backend/scripts/evaluate_real_video.py --data datasets/faceforensics
    python backend/scripts/evaluate_real_video.py --data datasets/celebdf --per-class 150

Nothing here downloads anything. FaceForensics++ and Celeb-DF are gated behind a
signed request form and DFDC needs Kaggle credentials; ``scripts/real_video.py``
reads whichever one has been extracted into the directory you point at. Run this
once the data lands.

What this measures that ``eval/metrics.json`` does not
-----------------------------------------------------
The liveness row in the held-out report - 0.604 AUC overall, 0.560 on the
attacks it targets - was measured on clips animated from a still photograph by
``scripts/videos.py``. Its own README entry says so, and calls replacing them the
highest-value upgrade available. Those clips have no camera motion, no encoder
artefacts and no face-swap that anybody outside this repository produced. This
script asks the question the corpus cannot: does the detector separate real
recorded video from face-swaps made by other people's tooling?

The temporal signals are the ones with the most to gain and the most to lose.
Identity drift, pose jitter and optical-flow discontinuity are all *motion*
signals, and there is no genuine camera motion anywhere in the synthetic corpus.
They may work far better here - or turn out to have been reading the animation
procedure. Both outcomes are worth knowing, and both are reported: each temporal
signal is scored on its own alongside the detector's combined score, so a
detector that is right for the wrong reason is visible.

Per-method reporting
--------------------
FaceForensics++ labels which manipulation produced each fake, and they are not
equally hard - NeuralTextures and FaceShifter are consistently harder than
Deepfakes and FaceSwap. Each method is scored against the full pristine pool and
reported separately, worst first, the same way ``eval/metrics.json`` reports per
attack type. An aggregate over five methods of differing difficulty describes
none of them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from real_video import balance, load_clips  # noqa: E402
from verityne.config import DATASET_ROOT, EVAL_ROOT  # noqa: E402
from verityne.detectors.base import SubmissionPayload  # noqa: E402
from verityne.detectors.liveness_video import LivenessVideoDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("realvideo.eval")

#: Per-frame and temporal signals pulled out of the detector's own output, so
#: each can be graded on its own rather than only in combination.
SIGNAL_KEYS = (
    "identity_drift", "pose_jitter", "flow_spike_ratio",
    "frame_mean", "frame_p90", "face_detection_rate",
    "min_consecutive_identity_similarity", "blink_events_proxy",
)


def auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    if y.size == 0 or len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, s)), 4)


def _num(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_ROOT / "faceforensics"),
                    help="directory holding FF++, Celeb-DF v2 or the DFDC preview")
    ap.add_argument("--out", default=str(EVAL_ROOT / "real_video.json"))
    ap.add_argument("--compression", default="c23", choices=("c0", "c23", "c40"),
                    help="FF++ only; detection accuracy varies sharply with this, "
                         "so it is recorded in the report")
    ap.add_argument("--per-class", type=int, default=150,
                    help="cap per class, and per manipulation method within the fake class")
    ap.add_argument("--all-splits", action="store_true",
                    help="score every sequence instead of the dataset's official test split "
                         "(makes the number incomparable to published work)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    clips = load_clips(Path(args.data), compression=args.compression,
                       test_split_only=not args.all_splits)
    if args.per_class:
        clips = balance(clips, args.per_class, seed=args.seed)
    if not clips:
        raise SystemExit("no clips resolved - check --data")

    n_real = sum(1 for c in clips if c.label == 0)
    log.info("scoring %d clips (%d pristine, %d manipulated)", len(clips), n_real, len(clips) - n_real)

    det = LivenessVideoDetector()
    rows: List[Dict] = []
    t0 = time.time()
    for i, clip in enumerate(clips):
        payload = SubmissionPayload(submission_id=f"rv_{i}", merchant_id="eval",
                                    video_path=clip.path)
        out = det.run(payload)
        sig = out.signals or {}
        row = {
            "key": clip.key,
            "dataset": clip.dataset,
            "method": clip.method,
            "compression": clip.compression,
            "split": clip.split,
            "y": clip.label,
            "status": out.status,
            "score": round(float(out.score), 4),
            "confidence": round(float(out.confidence), 4),
            "frames_sampled": sig.get("sampled") or (sig.get("frame_meta") or {}).get("sampled"),
            "latency_ms": round(float(out.latency_ms or 0.0), 1),
        }
        for k in SIGNAL_KEYS:
            row[k] = _num(sig.get(k))
        rows.append(row)
        if (i + 1) % 20 == 0:
            rate = (i + 1) / (time.time() - t0)
            log.info("  %d/%d  %.2f clips/s  eta %.0fs", i + 1, len(clips), rate,
                     (len(clips) - i - 1) / max(rate, 1e-6))

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        raise SystemExit("every clip errored - check that OpenCV can decode this dataset")
    y = np.asarray([r["y"] for r in ok])
    score = np.asarray([r["score"] for r in ok])

    # Each signal on its own, so a good combined score built on a signal that is
    # actually at chance shows up rather than being averaged into respectability.
    per_signal = {
        k: auc([r[k] for r in ok], y) for k in SIGNAL_KEYS
        if not all(np.isnan(r[k]) for r in ok)
    }

    genuine = y == 0
    by_method: Dict[str, Dict] = {}
    for method in sorted({r["method"] for r in ok if r["y"] == 1}):
        mask = genuine | np.asarray([r["method"] == method for r in ok])
        grp = [r for r in ok if r["method"] == method]
        by_method[method] = {
            "n": len(grp),
            "auc_vs_pristine": auc(score[mask], y[mask]),
            "mean_score": round(float(np.mean([r["score"] for r in grp])), 4),
        }
    by_method = dict(sorted(by_method.items(),
                            key=lambda kv: (kv[1]["auc_vs_pristine"] is None,
                                            kv[1]["auc_vs_pristine"] or 0.0)))

    real_scores = [r["score"] for r in ok if r["y"] == 0]
    fake_scores = [r["score"] for r in ok if r["y"] == 1]

    report = {
        "dataset": ok[0]["dataset"],
        "data_root": str(Path(args.data).resolve()),
        "compression": args.compression if ok[0]["dataset"] == "faceforensics" else None,
        "split": "official test split" if not args.all_splits else "all sequences (NOT the published split)",
        "n_scored": len(ok),
        "n_errored": len(rows) - len(ok),
        "n_pristine": int((y == 0).sum()),
        "n_manipulated": int((y == 1).sum()),
        "headline": {
            "detector_auc": auc(score, y),
            "mean_score_pristine": round(float(np.mean(real_scores)), 4) if real_scores else None,
            "mean_score_manipulated": round(float(np.mean(fake_scores)), 4) if fake_scores else None,
        },
        "per_signal_auc": {
            "values": per_signal,
            "reading": "AUC is against label 1 = manipulated. Signals that rise on a fake "
                       "(identity_drift, pose_jitter, flow_spike_ratio, frame_mean, frame_p90) "
                       "should sit above 0.5. Signals that rise on a genuine clip "
                       "(face_detection_rate, min_consecutive_identity_similarity) should sit "
                       "below it - for those, 0.5 is the failure, not the floor.",
        },
        "by_method": by_method,
        "comparison": {
            "synthetic_corpus_liveness_auc": 0.604,
            "synthetic_corpus_on_target_auc": 0.560,
            "note": "the corresponding numbers from eval/metrics.json, measured on clips "
                    "animated from stills by scripts/videos.py. The comparison is the point of "
                    "this run: those clips contain no real camera motion and no third-party "
                    "face-swap, so the temporal signals were never tested against either.",
        },
        "latency_ms": {
            "mean": round(float(np.mean([r["latency_ms"] for r in ok])), 1),
            "p50": round(float(np.percentile([r["latency_ms"] for r in ok], 50)), 1),
            "p90": round(float(np.percentile([r["latency_ms"] for r in ok], 90)), 1),
        },
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out_path)

    print(f"\n{report['dataset']} - {report['n_scored']} clips "
          f"({report['n_pristine']} pristine, {report['n_manipulated']} manipulated)")
    print(f"  liveness AUC {report['headline']['detector_auc']}   "
          f"(synthetic corpus reported {report['comparison']['synthetic_corpus_liveness_auc']})")
    print("  per signal:")
    for k, v in sorted(per_signal.items(), key=lambda kv: kv[1] or 0.0):
        print(f"    {k:22s} {v}")
    print("  per manipulation, worst first:")
    for m, v in by_method.items():
        print(f"    {m:18s} n={v['n']:4d}  AUC {v['auc_vs_pristine']}")


if __name__ == "__main__":
    main()
