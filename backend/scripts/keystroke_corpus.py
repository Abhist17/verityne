#!/usr/bin/env python3
"""The genuine half of Detector 6's corpus: real people, really typing.

Two third-party datasets, neither of them ours, neither of them synthetic:

  * **Aalto 136M** (`userinterfaces.aalto.fi/136Mkeystrokes`) - 168,595 people
    typing sentences in a browser, with press and release timestamps per key.
    This is the training source. It is a browser, which matters: the timing
    granularity, the event ordering and the rollover behaviour are the ones a
    KYC form would actually see, not a lab keylogger's.
  * **CMU / Killourhy-Maxion** (`cs.cmu.edu/~keystroke`) - 51 subjects typing one
    fixed password 400 times each. Held back entirely, and used only as a
    *cross-corpus* test: different people, different task, different decade,
    different capture rig. A model that holds up there is reading typing rhythm
    rather than the Aalto collection apparatus.

**Sessions are built to match what the detector sees at inference.** One Aalto
sentence is about 40 keystrokes; a KYC form is about 110. Training on 40-key
sessions and serving on 110-key ones would shift every variance statistic in the
feature vector, so consecutive sections from the same participant are
concatenated until the session is form-sized. The gap between two sections is a
real between-field pause and is kept as one.

Nothing is vendored: both files are fetched to `datasets/keystrokes/`, which is
git-ignored, exactly as LFW and MIDV-2020 are.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

log = logging.getLogger("keystroke_corpus")

AALTO_ZIP = REPO / "datasets/keystrokes/Keystrokes.zip"
CMU_CSV = REPO / "datasets/keystrokes/cmu_strong_password.csv"
AALTO_URL = "https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip"
CMU_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"

#: Keystrokes per synthesised session, chosen to match a filled KYC form rather
#: than a single sentence. `bot_telemetry.py` produces 111 keys for its five
#: fields, so the two halves of the corpus are the same size by construction and
#: the model cannot separate them on length alone.
SESSION_KEYS = 110
BACKSPACE_KEYCODE = 8


def _to_float(x: str) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None  # NaN check
    except (TypeError, ValueError):
        return None


def iter_aalto_participants(zip_path: Path, limit: int, seed: int = 0) -> Iterator[Tuple[str, List[dict]]]:
    """Yield (participant_id, keystroke rows) straight out of the zip.

    Streamed rather than extracted: the archive holds 168,595 files and unpacking
    it costs several gigabytes for data we read exactly once.
    """
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("_keystrokes.txt")]
        # Sampled, not truncated. The archive is ordered by participant id, which
        # correlates with recruitment date and therefore with the population;
        # taking the first N would quietly select a cohort.
        random.Random(seed).shuffle(names)
        for name in names[:limit]:
            try:
                with z.open(name) as fh:
                    text = io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")
                    rows = list(csv.DictReader(text, delimiter="\t"))
            except Exception:  # noqa: BLE001 - a handful of files are truncated
                continue
            if rows:
                yield Path(name).stem.split("_")[0], rows


def aalto_sessions(zip_path: Path, participants: int, seed: int = 0) -> Iterator[dict]:
    """Form-sized typing sessions, one participant never spanning two sessions."""
    for pid, rows in iter_aalto_participants(zip_path, participants, seed):
        by_section: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            by_section[r.get("TEST_SECTION_ID", "")].append(r)

        dwells: List[float] = []
        flights: List[float] = []
        n_keys = backspaces = printable = 0
        first_press: Optional[float] = None
        last_release: Optional[float] = None
        prev_release: Optional[float] = None

        for _, section in sorted(by_section.items()):
            events = []
            for r in section:
                p, q = _to_float(r.get("PRESS_TIME", "")), _to_float(r.get("RELEASE_TIME", ""))
                if p is None or q is None:
                    continue
                events.append((p, q, r.get("KEYCODE", ""), r.get("LETTER", "")))
            events.sort(key=lambda e: e[0])

            for press, release, keycode, letter in events:
                n_keys += 1
                if first_press is None:
                    first_press = press
                last_release = max(last_release or release, release)
                dwells.append(release - press)
                if prev_release is not None:
                    # Negative values are real: the next key goes down before the
                    # previous comes up. See utils/keystroke.py on why they stay.
                    flights.append(press - prev_release)
                prev_release = release
                if keycode == str(BACKSPACE_KEYCODE):
                    backspaces += 1
                elif len(str(letter)) == 1:
                    printable += 1

            if n_keys >= SESSION_KEYS:
                span = ((last_release or 0) - (first_press or 0)) / 1000.0
                yield {
                    "subject": f"aalto::{pid}", "label": 0, "source": "aalto",
                    "dwells": dwells, "flights": flights, "n_keys": n_keys,
                    "backspaces": backspaces, "printable": printable, "span_s": span,
                }
                dwells, flights = [], []
                n_keys = backspaces = printable = 0
                first_press = last_release = prev_release = None


def cmu_sessions(csv_path: Path) -> Iterator[dict]:
    """51 subjects x 400 repetitions of one fixed password.

    The columns are seconds: `H.x` is the hold (dwell) of key x, `UD.x.y` is the
    up-to-down latency between x and y, which is flight by another name. `DD.*`
    is down-to-down and is skipped, being the sum of the other two.

    One repetition is 11 keystrokes, far short of a form, so repetitions are
    grouped into form-sized sessions the same way Aalto sections are.
    """
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        hold = [c for c in cols if c.startswith("H.")]
        ud = [c for c in cols if c.startswith("UD.")]

        buf: Dict[str, dict] = defaultdict(lambda: {"d": [], "f": [], "n": 0})
        for row in reader:
            subj = row["subject"]
            b = buf[subj]
            b["d"].extend(v * 1000.0 for v in (_to_float(row[c]) for c in hold) if v is not None)
            b["f"].extend(v * 1000.0 for v in (_to_float(row[c]) for c in ud) if v is not None)
            b["n"] += len(hold)
            if b["n"] >= SESSION_KEYS:
                yield {
                    "subject": f"cmu::{subj}", "label": 0, "source": "cmu",
                    "dwells": b["d"], "flights": b["f"], "n_keys": b["n"],
                    "backspaces": 0, "printable": b["n"], "span_s": None,
                }
                buf[subj] = {"d": [], "f": [], "n": 0}


def write_jsonl(rows: Iterator[dict], out: Path, cap: Optional[int] = None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            n += 1
            if cap and n >= cap:
                break
            if n % 2000 == 0:
                log.info("  %d sessions", n)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participants", type=int, default=6000,
                    help="Aalto participants to sample (168,595 available)")
    ap.add_argument("--cap", type=int, default=20000, help="max Aalto sessions to write")
    ap.add_argument("--out-dir", type=Path, default=REPO / "datasets/keystrokes")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for path, url in ((AALTO_ZIP, AALTO_URL), (CMU_CSV, CMU_URL)):
        if not path.exists():
            raise SystemExit(f"{path} missing - fetch it from {url}")

    log.info("Aalto: sampling %d participants", args.participants)
    n_aalto = write_jsonl(
        aalto_sessions(AALTO_ZIP, args.participants), args.out_dir / "human_aalto.jsonl", args.cap
    )
    log.info("wrote %d Aalto sessions", n_aalto)

    n_cmu = write_jsonl(cmu_sessions(CMU_CSV), args.out_dir / "human_cmu.jsonl")
    log.info("wrote %d CMU sessions (held back for cross-corpus test)", n_cmu)


if __name__ == "__main__":
    main()
