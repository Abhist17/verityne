#!/usr/bin/env python3
"""Access to OpenRL/DeepFakeFace - real photographs and three ways of faking them.

    https://huggingface.co/datasets/OpenRL/DeepFakeFace

Four archives, 30,000 images each, and the reason this dataset is worth the
trouble: **they share filenames**. `wiki/61/19242061_1982-08-16_2009.jpg` is a
real photograph from IMDB-WIKI, and the file of that same name in each of the
other three archives is a fake *derived from that photograph*:

  * ``text2img``   - Stable Diffusion, generated from a text prompt. Whole-image
                     synthesis; nothing of the original pixel history survives.
  * ``inpainting`` - Stable Diffusion inpainting. The face is regenerated, the
                     rest of the photograph is the original.
  * ``insight``    - InsightFace swap. A different face composited into the
                     original photograph.

That pairing controls the thing a random real-vs-fake scrape cannot: identity,
pose, framing and photographic subject are held fixed, because each fake was
derived from the real image beside it.

**It does not control geometry, and assuming it does would be the same mistake
`evaluate_indian_faces` was written to catch.** The genuine images are IMDB-WIKI
originals at their native sizes - 400x711, 368x455, 280x532 - while all three
fake splits were emitted at a uniform 512x512. A detector handed the raw frames
can separate the classes on resampling history alone, without looking at a face,
exactly as the naive protocol there scored 0.704 on two groups that were both
real. So the face-crop control is not optional here either: both classes must be
detected, cropped and resized to a common size before scoring, and only that
number describes the detector rather than the dataset's encoder settings.

What the pairing does buy is that for ``inpainting`` and ``insight`` the
untouched regions of the frame carry the original photograph's provenance, so
the manipulation is the difference between the pair rather than the whole image.

The archives are ~1 GB each and we want a few hundred images from each, so
members are pulled individually over HTTP ranges - see `remote_zip`.
"""
from __future__ import annotations

import io
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from remote_zip import RemoteZip

log = logging.getLogger("deepfakeface")

BASE = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main"

#: split name -> archive. `real` is the genuine class every fake split is paired
#: against; the other three are distinct generator families, and the whole point
#: of this evaluation is that they are never pooled into one number.
SPLITS: Dict[str, str] = {
    "real": "wiki.zip",
    "text2img": "text2img.zip",
    "inpainting": "inpainting.zip",
    "insight": "insight.zip",
}

GENERATOR_FAMILY = {
    "text2img": "Stable Diffusion, text-to-image (whole-image synthesis)",
    "inpainting": "Stable Diffusion inpainting (face regenerated in a real photograph)",
    "insight": "InsightFace swap (a different face composited into a real photograph)",
}

#: Directory each archive uses internally, which is not the split name: the
#: genuine images live under `wiki/` because they come from IMDB-WIKI.
MEMBER_PREFIX: Dict[str, str] = {
    "real": "wiki",
    "text2img": "text2img",
    "inpainting": "inpainting",
    "insight": "insight",
}

_zips: Dict[str, RemoteZip] = {}


def _archive(split: str) -> RemoteZip:
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}; have {sorted(SPLITS)}")
    if split not in _zips:
        _zips[split] = RemoteZip(f"{BASE}/{SPLITS[split]}")
    return _zips[split]


def paired_keys(n: int, seed: int = 0) -> List[str]:
    """Sample `n` image keys that exist in every split.

    A key is the path with its split prefix removed - ``61/19242061_...jpg`` -
    so the same key addresses the real photograph and each of its three fakes.

    Sampled from the `real` archive's directory and *verified against every
    other archive* rather than assumed: the four archives are documented as
    parallel, and if that ever stops being true this should fail loudly here
    rather than quietly score mismatched pairs.
    """
    listing = {s: {k.split("/", 1)[1] for k in _archive(s).namelist() if "/" in k} for s in SPLITS}
    common = set.intersection(*listing.values())
    missing = len(listing["real"]) - len(common)
    if missing:
        log.warning("%d of %d real images have no counterpart in every fake split",
                    missing, len(listing["real"]))
    if not common:
        raise RuntimeError("the four DeepFakeFace archives share no filenames")

    rng = random.Random(seed)
    keys = sorted(common)
    rng.shuffle(keys)
    return keys[: min(n, len(keys))]


def load_images(split: str, keys: List[str], cache_root: Optional[Path] = None,
                workers: int = 8) -> Dict[str, Image.Image]:
    """Fetch `keys` from one split, caching the raw bytes under `cache_root`.

    Cached so that re-running the evaluation - the normal case while reading its
    output - does not re-fetch. Returns a dict rather than a list because a key
    that fails to decode is skipped with a warning rather than ending the run,
    and pairing must survive those gaps.

    Each member costs two round trips (local header, then payload), so the run
    is latency-bound rather than bandwidth-bound and threads help roughly
    linearly. The central directory is fetched once before the pool starts, so
    the workers only ever read `RemoteZip` state that is already populated.
    """
    zf = _archive(split)
    zf.namelist()  # populate the directory before any thread touches it
    prefix = MEMBER_PREFIX[split]

    def fetch(key: str) -> Optional[bytes]:
        cached = (cache_root / split / key) if cache_root else None
        if cached is not None and cached.exists():
            return cached.read_bytes()
        try:
            blob = zf.read(f"{prefix}/{key}")
        except Exception as exc:  # noqa: BLE001 - one bad member must not end the run
            log.warning("%s/%s: %s", split, key, exc)
            return None
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(blob)
        return blob

    out: Dict[str, Image.Image] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, blob in zip(keys, pool.map(fetch, keys)):
            done += 1
            if blob is None:
                continue
            try:
                img = Image.open(io.BytesIO(blob))
                img.load()
                out[key] = img.convert("RGB")
            except Exception as exc:  # noqa: BLE001
                log.warning("%s/%s: undecodable (%s)", split, key, exc)
            if done % 100 == 0:
                log.info("  %s: fetched %d/%d", split, done, len(keys))
    return out


def load_paired(keys: List[str], cache_root: Optional[Path] = None,
                workers: int = 8) -> Dict[str, List[Image.Image]]:
    """Every split for the same keys, dropping any key that failed anywhere.

    Returns lists aligned by position across splits, so index `i` is one
    photograph and its three fakes. Dropping a key everywhere when it fails
    anywhere is what keeps that guarantee true.
    """
    per_split = {s: load_images(s, keys, cache_root, workers) for s in SPLITS}

    usable = [k for k in keys if all(k in per_split[s] for s in SPLITS)]
    dropped = len(keys) - len(usable)
    if dropped:
        log.info("dropped %d/%d keys that did not load in every split", dropped, len(keys))
    return {s: [per_split[s][k] for k in usable] for s in SPLITS}
