#!/usr/bin/env python3
"""Run every detector over the corpus and cache the raw scores.

    python backend/scripts/score_corpus.py            # score everything
    python backend/scripts/score_corpus.py --split test

Writes `eval/scores.json`: one row per packet with each detector's score,
confidence and status, plus timing. Both `train_fusion.py` and `evaluate.py`
read this file, so the expensive part runs once.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from verityne.config import DATASET_ROOT, EVAL_ROOT  # noqa: E402
from verityne.detectors import STAGE_ONE, STAGE_TWO, SubmissionPayload  # noqa: E402
from verityne.detectors.models import warmup  # noqa: E402
from verityne.utils.spectral import profile_vector, spectral_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_corpus")

MANIFEST = DATASET_ROOT / "manifest.json"
SCORES = EVAL_ROOT / "scores.json"


def score_packet(entry: Dict, with_heatmaps: bool = False) -> Dict:
    """Run the full detector suite on one manifest row."""
    import datetime as dt

    submitted = dt.datetime.fromisoformat(entry["submitted_at"]).timestamp()
    payload = SubmissionPayload(
        submission_id=f"eval_{entry['id']}",
        merchant_id="eval",
        selfie_path=Path(entry["selfie"]) if entry.get("selfie") else None,
        video_path=Path(entry["video"]) if entry.get("video") else None,
        id_doc_path=Path(entry["id_document"]) if entry.get("id_document") else None,
        claimed_name=entry.get("claimed_name"),
        claimed_id_number=entry.get("claimed_id_number"),
        claimed_dob=entry.get("claimed_dob"),
        submitted_at=submitted,
    )

    t0 = time.perf_counter()
    outputs = {}
    for det in STAGE_ONE:
        out = det.run(payload)
        outputs[out.name] = out
    for det in STAGE_TWO:
        out = det.run(payload)
        outputs[out.name] = out
    elapsed = (time.perf_counter() - t0) * 1000

    row: Dict[str, object] = {
        "id": entry["id"],
        "label": entry["label"],
        "y": 1 if entry["label"] == "fake" else 0,
        "attack_type": entry.get("attack_type"),
        "generator": entry.get("generator"),
        "split": entry.get("split", "train"),
        "doc_type": entry.get("doc_type"),
        "latency_ms": round(elapsed, 1),
        "detectors": {
            n: {
                "score": round(float(o.score), 6),
                "confidence": round(float(o.confidence), 6),
                "status": o.status,
                "reasons": o.reasons[:3],
            }
            for n, o in outputs.items()
        },
        "per_detector_latency_ms": {n: round(o.latency_ms, 1) for n, o in outputs.items()},
    }

    # Spectral descriptors of the selfie face, used to fit the spectral head and
    # the generator-fingerprinting head. Cached here so those trainers are instant.
    face = payload.cache.get("selfie_face")
    if face is not None:
        row["spectral_features"] = {k: float(v) for k, v in spectral_features(face).items()}
        row["spectral_profile"] = [float(v) for v in profile_vector(face)]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3, help="packets scored concurrently")
    ap.add_argument("--out", type=Path, default=SCORES)
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found - run build_dataset.py first")
    entries: List[Dict] = json.loads(MANIFEST.read_text())
    if args.split != "all":
        entries = [e for e in entries if e.get("split") == args.split]
    if args.limit:
        entries = entries[: args.limit]

    log.info("warming models: %s", warmup())
    log.info("scoring %d packets with %d workers", len(entries), args.workers)

    rows: List[Dict] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(score_packet, entries)):
            rows.append(row)
            if (i + 1) % 20 == 0:
                log.info("scored %d/%d", i + 1, len(entries))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    wall = time.perf_counter() - t0
    log.info("wrote %s (%d rows) in %.1fs (%.2f packets/s)", args.out, len(rows), wall, len(rows) / max(wall, 1e-6))


if __name__ == "__main__":
    main()
