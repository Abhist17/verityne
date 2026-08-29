"""Perceptual hashing. Catches the same 'KYC kit' asset resubmitted under a new name."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image

from .images import to_pil


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
