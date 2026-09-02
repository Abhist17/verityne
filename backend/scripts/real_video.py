"""Ingest for real deepfake video datasets: FaceForensics++, Celeb-DF v2, DFDC preview.

The README names this as the single highest-value upgrade to the corpus, and it
is right. Every liveness number in ``eval/metrics.json`` was measured on clips
``scripts/videos.py`` animated from a still photograph. That measures whether a
face-swap is separable *under identical capture conditions*, which is a real
question but not the question a liveness detector is deployed to answer. It has
never seen a recorded video: no camera shake, no rolling shutter, no autofocus
hunt, no compression from a real encoder, and no face-swap produced by anybody
but us.

This module does not download anything - FaceForensics++ and Celeb-DF are both
gated behind a signed request form, and DFDC needs Kaggle credentials. It reads
whichever of them is present on disk, so the evaluation path is ready before
the data is.

Layouts recognised
------------------
**FaceForensics++** - the useful one, because it labels *which* manipulation
produced each fake, and the methods differ enormously in difficulty::

    <root>/original_sequences/youtube/<c0|c23|c40>/videos/*.mp4
    <root>/manipulated_sequences/<method>/<c0|c23|c40>/videos/*.mp4
    <root>/splits/test.json                       (official split, used if present)

**Celeb-DF v2** - harder fakes, no method labels::

    <root>/Celeb-real/*.mp4  <root>/YouTube-real/*.mp4  <root>/Celeb-synthesis/*.mp4
    <root>/List_of_testing_videos.txt             (official test list, used if present)

**DFDC preview** - available without a form::

    <root>/dataset.json  with {filename: {"label": "fake"|"real", ...}}

Two things this module is careful about
---------------------------------------
*Official splits.* Where a dataset publishes one, it is used. Reporting a number
on the whole of FaceForensics++ when everyone else reports on its test split
would make our figure look better than published work for no reason other than
picking a different denominator.

*Compression.* FF++ ships each video at three compression levels and detection
accuracy falls sharply from c0 to c40. A number quoted without its compression
level is not comparable to anything, so the level is carried through into the
report rather than averaged away.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

log = logging.getLogger("verityne.realvideo")

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
COMPRESSIONS = ("c0", "c23", "c40")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


@dataclass(frozen=True)
class Clip:
    path: Path
    label: int  # 1 = manipulated, 0 = pristine
    method: str  # 'real', or the manipulation name
    dataset: str
    compression: Optional[str] = None
    split: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.method}/{self.path.stem}"


# ---------------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------------

def detect_layout(root: Path) -> Optional[str]:
    """Which dataset is sitting in `root`, if any."""
    if (root / "original_sequences").is_dir() or (root / "manipulated_sequences").is_dir():
        return "faceforensics"
    if (root / "Celeb-synthesis").is_dir() or (root / "Celeb-real").is_dir():
        return "celebdf"
    if (root / "dataset.json").is_file():
        return "dfdc"
    return None


def _videos(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


# ---------------------------------------------------------------------------------
# FaceForensics++
# ---------------------------------------------------------------------------------

def _ffpp_test_ids(root: Path) -> Optional[set]:
    """The official test split, as a set of sequence ids.

    ``splits/test.json`` is a list of ``[target, source]`` id pairs. A pristine
    clip is named by one id; a manipulated one by ``target_source``. Both forms
    are returned so either can be matched by stem.
    """
    path = root / "splits" / "test.json"
    if not path.is_file():
        return None
    try:
        pairs = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    ids = set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        ids.update({a, b, f"{a}_{b}", f"{b}_{a}"})
    return ids or None


def load_faceforensics(root: Path, compression: str = "c23",
                       methods: Sequence[str] = FFPP_METHODS,
                       test_split_only: bool = True) -> List[Clip]:
    clips: List[Clip] = []
    test_ids = _ffpp_test_ids(root) if test_split_only else None
    if test_split_only and test_ids is None:
        log.warning("splits/test.json not found - scoring every sequence, which is NOT the "
                    "split published numbers are quoted on")

    def keep(p: Path) -> bool:
        return test_ids is None or p.stem in test_ids

    real_dir = root / "original_sequences" / "youtube" / compression / "videos"
    if not real_dir.is_dir():  # some mirrors omit the 'youtube' level
        real_dir = root / "original_sequences" / compression / "videos"
    for p in _videos(real_dir):
        if keep(p):
            clips.append(Clip(p, 0, "real", "faceforensics", compression,
                              "test" if test_ids else None))

    for method in methods:
        mdir = root / "manipulated_sequences" / method / compression / "videos"
        for p in _videos(mdir):
            if keep(p):
                clips.append(Clip(p, 1, method, "faceforensics", compression,
                                  "test" if test_ids else None))

    log.info("FaceForensics++ %s: %d pristine + %d manipulated%s", compression,
             sum(1 for c in clips if c.label == 0), sum(1 for c in clips if c.label == 1),
             " (official test split)" if test_ids else "")
    return clips


# ---------------------------------------------------------------------------------
# Celeb-DF v2
# ---------------------------------------------------------------------------------

def load_celebdf(root: Path, test_split_only: bool = True) -> List[Clip]:
    listing = root / "List_of_testing_videos.txt"
    if test_split_only and listing.is_file():
        clips: List[Clip] = []
        for line in listing.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            # The list is "<label> <relative path>", where 1 marks a *real* video.
            flag, rel = parts
            path = root / rel
            if not path.is_file():
                continue
            is_real = flag == "1"
            clips.append(Clip(path, 0 if is_real else 1,
                              "real" if is_real else "celeb_synthesis",
                              "celebdf", None, "test"))
        log.info("Celeb-DF v2 official test list: %d pristine + %d manipulated",
                 sum(1 for c in clips if c.label == 0), sum(1 for c in clips if c.label == 1))
        if clips:
            return clips
        log.warning("test list present but no files resolved - falling back to directories")

    clips = []
    for sub, label, method in (("Celeb-real", 0, "real"), ("YouTube-real", 0, "real"),
                               ("Celeb-synthesis", 1, "celeb_synthesis")):
        for p in _videos(root / sub):
            clips.append(Clip(p, label, method, "celebdf"))
    log.info("Celeb-DF v2 (all): %d pristine + %d manipulated",
             sum(1 for c in clips if c.label == 0), sum(1 for c in clips if c.label == 1))
    return clips


# ---------------------------------------------------------------------------------
# DFDC preview
# ---------------------------------------------------------------------------------

def load_dfdc(root: Path) -> List[Clip]:
    meta = json.loads((root / "dataset.json").read_text())
    clips: List[Clip] = []
    for name, info in meta.items():
        path = root / name
        if not path.is_file():
            hits = list(root.rglob(name))
            if not hits:
                continue
            path = hits[0]
        label = 1 if str(info.get("label", "")).lower() == "fake" else 0
        clips.append(Clip(path, label, "dfdc_fake" if label else "real", "dfdc",
                          None, info.get("set")))
    log.info("DFDC preview: %d pristine + %d manipulated",
             sum(1 for c in clips if c.label == 0), sum(1 for c in clips if c.label == 1))
    return clips


# ---------------------------------------------------------------------------------

def load_clips(root: Path, compression: str = "c23", test_split_only: bool = True) -> List[Clip]:
    """Load whichever supported dataset is present at `root`."""
    root = Path(root)
    layout = detect_layout(root)
    if layout is None:
        raise FileNotFoundError(
            f"No supported deepfake video dataset found under {root}.\n"
            "Expected one of:\n"
            "  FaceForensics++  <root>/original_sequences/... and manipulated_sequences/...\n"
            "  Celeb-DF v2      <root>/Celeb-real/, Celeb-synthesis/\n"
            "  DFDC preview     <root>/dataset.json\n"
            "FF++ and Celeb-DF are gated: request access from their maintainers, then "
            "extract into this directory. Nothing here downloads them."
        )
    if layout == "faceforensics":
        return load_faceforensics(root, compression, test_split_only=test_split_only)
    if layout == "celebdf":
        return load_celebdf(root, test_split_only=test_split_only)
    return load_dfdc(root)


def balance(clips: Sequence[Clip], per_class: int, seed: int = 11) -> List[Clip]:
    """Cap each class, and each manipulation method within the fake class.

    FF++ ships one pristine video for every five manipulated ones, so an
    unbalanced sample would let whichever method has the most clips dominate the
    aggregate. Capping per method keeps the headline honest and keeps the
    per-method rows comparable to each other.
    """
    rng = random.Random(seed)
    by_method: Dict[str, List[Clip]] = {}
    for c in clips:
        by_method.setdefault(c.method, []).append(c)

    reals = by_method.pop("real", [])
    rng.shuffle(reals)
    out = reals[:per_class]

    if by_method:
        share = max(1, per_class // len(by_method))
        for method, group in sorted(by_method.items()):
            rng.shuffle(group)
            out.extend(group[:share])
    return out
