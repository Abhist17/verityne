"""Where generated heatmaps and thumbnails live, and how the API addresses them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..config import HEATMAP_DIR, UPLOAD_DIR

HEATMAP_URL_PREFIX = "/static/heatmaps"
UPLOAD_URL_PREFIX = "/static/uploads"


def save_heatmap(rgb: np.ndarray, submission_id: str, tag: str) -> str:
    """Persist an overlay image and return the URL the dashboard should load."""
    out = HEATMAP_DIR / f"{submission_id}_{tag}.png"
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out, "PNG", optimize=True)
    return f"{HEATMAP_URL_PREFIX}/{out.name}"


def save_thumbnail(src: str | Path, submission_id: str, tag: str, size: int = 320) -> str | None:
    try:
        img = Image.open(str(src)).convert("RGB")
    except Exception:
        return None
    img.thumbnail((size, size), Image.LANCZOS)
    out = HEATMAP_DIR / f"{submission_id}_{tag}_thumb.jpg"
    img.save(out, "JPEG", quality=82)
    return f"{HEATMAP_URL_PREFIX}/{out.name}"


def upload_url(path: str | Path) -> str | None:
    p = Path(path)
    try:
        rel = p.resolve().relative_to(UPLOAD_DIR.resolve())
    except Exception:
        return None
    return f"{UPLOAD_URL_PREFIX}/{rel.as_posix()}"
