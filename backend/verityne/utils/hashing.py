"""Asset hashing. Catches the same 'KYC kit' image resubmitted under a new name.

Two hashes, because they support two different claims:

* ``content_hash`` is a SHA-256 over the decoded pixels. It is exact. Two images
  share one only if they are literally the same picture, so it is what backs the
  statement "the exact same image file was used". Stripping EXIF or changing the
  container does not move it; re-encoding does.
* ``phash`` is perceptual and tolerant, which makes it useful for spotting a
  near-duplicate and useless for asserting identity. Synthetic ID cards drawn
  from one template collide at a Hamming distance of 2 despite belonging to
  different people, so a phash hit is a hint to look closer, never proof.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image

from .images import to_pil


def content_hash(rgb: np.ndarray) -> str:
    """Exact SHA-256 of the decoded pixels. Same picture in, same digest out."""
    arr = np.ascontiguousarray(rgb)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def phash(rgb: np.ndarray) -> str:
    return str(imagehash.phash(to_pil(rgb), hash_size=8))


def phash_file(path: str | Path) -> Optional[str]:
    try:
        return str(imagehash.phash(Image.open(str(path)), hash_size=8))
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    """Distance in bits between two hex perceptual hashes (0 = identical)."""
    try:
        # imagehash subtraction yields a numpy integer; callers store this in JSON.
        return int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))
    except Exception:
        return 64
