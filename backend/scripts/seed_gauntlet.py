#!/usr/bin/env python3
"""Load the Gauntlet fixtures into the database.

    python backend/scripts/seed_gauntlet.py --real 10 --fake 10

Fixtures are drawn from the **test** split, so nothing the Gauntlet scores was
seen by the fusion model during training. The fake half is spread across attack
types rather than sampled at random, so the demo exercises every detector
instead of accidentally showing ten of the easiest case.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from verityne.config import DATASET_ROOT  # noqa: E402
from verityne.db import Submission, init_db, log_event, session_scope  # noqa: E402
from verityne.pipeline import storage_dir  # noqa: E402
from verityne.utils.artifacts import save_thumbnail  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("seed_gauntlet")


def pick_spread(entries: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """Round-robin across attack types so every detector gets exercised."""
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        by_type[e.get("attack_type") or "genuine"].append(e)
    for v in by_type.values():
        rng.shuffle(v)
    picked: List[Dict] = []
    types = sorted(by_type)
    while len(picked) < n and any(by_type.values()):
        for t in types:
            if by_type[t] and len(picked) < n:
                picked.append(by_type[t].pop())
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=10)
    ap.add_argument("--fake", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--merchant", default="default")
    ap.add_argument("--keep-existing", action="store_true")
    args = ap.parse_args()

    manifest_path = DATASET_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found - run build_dataset.py first")
    entries = [e for e in json.loads(manifest_path.read_text()) if e.get("split") == args.split]
    if not entries:
        raise SystemExit(f"no packets in the {args.split} split")

    rng = random.Random(args.seed)
    reals = [e for e in entries if e["label"] == "real"]
    fakes = [e for e in entries if e["label"] == "fake"]
    rng.shuffle(reals)
    chosen = reals[: args.real] + pick_spread(fakes, args.fake, rng)
    rng.shuffle(chosen)

    init_db()
    with session_scope() as session:
        if not args.keep_existing:
            old = session.query(Submission).filter(Submission.source == "gauntlet").all()
            for s in old:
                session.delete(s)
            session.flush()
            log.info("cleared %d existing fixtures", len(old))

        for i, e in enumerate(chosen):
            display = f"{'FAKE' if e['label'] == 'fake' else 'REAL'}-{i + 1:02d}"
            sub = Submission(
                merchant_id=args.merchant,
                external_ref=e["id"],
                source="gauntlet",
                label=e["label"],
                attack_type=e.get("attack_type"),
                status="PENDING",
                extra={
                    "claimed_name": e.get("claimed_name"),
                    "claimed_id_number": e.get("claimed_id_number"),
                    "claimed_dob": e.get("claimed_dob"),
                    "display_name": display,
                    "doc_type": e.get("doc_type"),
                    "capture_mode": e.get("capture_mode"),
                },
            )
            session.add(sub)
            session.flush()

            dest = storage_dir(sub.id)
            for key, attr in (("selfie", "selfie_path"), ("id_document", "id_doc_path"), ("video", "video_path")):
                src = e.get(key)
                if src and Path(src).exists():
                    target = dest / Path(src).name
                    shutil.copy2(src, target)
                    setattr(sub, attr, str(target))

            thumb = save_thumbnail(sub.selfie_path, sub.id, "selfie") if sub.selfie_path else None
            if thumb:
                sub.extra = {**sub.extra, "thumb_url": thumb}
            log_event(session, "gauntlet_seeded", sub.id, source_packet=e["id"], truth=e["label"])

        log.info("seeded %d fixtures (%d genuine, %d fraudulent) from the %s split",
                 len(chosen), sum(1 for e in chosen if e["label"] == "real"),
                 sum(1 for e in chosen if e["label"] == "fake"), args.split)
        for e in chosen:
            log.info("  %-6s %-24s %s", e["label"], e.get("attack_type") or "genuine", e["id"])


if __name__ == "__main__":
    main()
