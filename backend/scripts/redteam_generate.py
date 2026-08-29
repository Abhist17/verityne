#!/usr/bin/env python3
"""Generate a held-out pool of never-before-seen attacks.

    python backend/scripts/redteam_generate.py --count 12

Run this **after** training and evaluation. Everything it produces uses fresh
generator seeds and fresh identities, so nothing in the pool appeared in the
training or evaluation sets - which is the whole point: the dashboard's red-team
button demonstrates generalisation, not recall of a memorised set.

Generation happens here, offline, rather than inside the request. A live demo
that depends on a GPU sample completing and on venue wi-fi is a demo that fails
on stage; the button pulls from this pool instead, and the manifest records when
each item was made so the "unseen" claim is checkable.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from build_dataset import build_packet  # noqa: E402
from faces import generate_sd_faces, load_real_faces  # noqa: E402
from verityne.config import DATASET_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("redteam")

POOL = DATASET_ROOT / "redteam"
ATTACKS = [
    "generated_selfie", "synthetic_identity", "tampered_document",
    "face_swap_liveness", "impersonation", "reused_id_selfie",
    "invalid_document", "stale_or_edited_media", "recaptured_screen",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None, help="defaults to a time-derived seed, so every run is new")
    ap.add_argument("--with-video", action="store_true")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(dt.datetime.now().timestamp()) % 10**6
    rng = random.Random(seed)
    POOL.mkdir(parents=True, exist_ok=True)
    log.info("red-team seed %d (fresh generator seeds -> unseen samples)", seed)

    # Offset the real-face sample well past the corpus range so identities differ too.
    real_faces = load_real_faces(max(24, args.count), seed=seed + 991)
    fake_faces = generate_sd_faces(max(12, args.count), seed=seed + 2027, steps=3)

    manifest_path = POOL / "manifest.json"
    existing: List[Dict] = []
    if args.append and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    entries: List[Dict] = []
    original_corpus = None
    try:
        import build_dataset as bd

        original_corpus, bd.CORPUS = bd.CORPUS, POOL
        for i in range(args.count):
            attack = ATTACKS[i % len(ATTACKS)]
            pid = f"rt_{seed}_{i:03d}"
            try:
                entry = build_packet(pid, "fake", attack, real_faces, fake_faces, rng, args.with_video, i)
            except Exception as exc:  # noqa: BLE001
                log.warning("red-team packet %s failed: %s", pid, exc)
                continue
            entry.update({
                "generated_at": generated_at,
                "seed": seed,
                "display_name": f"RT-{attack.replace('_', ' ')}-{i:02d}",
                "unseen": True,
            })
            entries.append(entry)
            log.info("  %s  %s", pid, attack)
    finally:
        if original_corpus is not None:
            import build_dataset as bd

            bd.CORPUS = original_corpus

    manifest_path.write_text(json.dumps(existing + entries, indent=2))
    log.info("wrote %d red-team packets to %s (pool now %d)", len(entries), manifest_path, len(existing) + len(entries))


if __name__ == "__main__":
    main()
