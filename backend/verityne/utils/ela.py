"""Error Level Analysis: find regions whose JPEG compression history differs from the rest.

A photo saved once compresses uniformly. A region that was pasted in (a swapped
name, a replaced photo) has been through a different number of compression
generations, so it re-compresses at a measurably different error level.
"""
from __future__ import annotations

import io
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from .images import to_pil


def ela_map(rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """Absolute difference between the image and its re-compression, as a float map."""
    buf = io.BytesIO()
    to_pil(rgb).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    diff = np.abs(rgb.astype(np.float32) - recompressed)
    return diff.max(axis=2)


def multi_quality_ela(rgb: np.ndarray, qualities: Tuple[int, ...] = (75, 85, 95)) -> np.ndarray:
    """Average ELA across several qualities. Single-quality ELA is noisy; this is steadier."""
    maps = [ela_map(rgb, q) for q in qualities]
    stacked = np.stack(maps, axis=0).mean(axis=0)
    return cv2.GaussianBlur(stacked, (0, 0), sigmaX=2.0)


def edge_normalised_ela(rgb: np.ndarray) -> np.ndarray:
    """ELA with the contribution of ordinary sharp edges divided out.

    Raw ELA lights up wherever there is high-contrast detail, because edges are
    exactly what JPEG re-quantises hardest. On an ID card that means every glyph
    and every guilloche line glows, and the map is useless. Dividing by local
    gradient energy leaves only the error that the image content does not
    explain - which is what a splice actually looks like.
    """
    raw = multi_quality_ela(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), sigmaX=3.0)
    grad = grad / (np.percentile(grad, 99) + 1e-6)
    return raw / (1.0 + 3.0 * np.clip(grad, 0.0, 1.5))


def tamper_score(rgb: np.ndarray) -> Dict[str, object]:
    """Score how likely this image contains a spliced/edited region.

    We look for *localised* high-error blobs, not overall error: a uniformly
    noisy image is just a low-quality scan, whereas a bright compact blob on a
    quiet background is a splice.
    """
    m = edge_normalised_ela(rgb)
    if m.size == 0:
        return {"score": 0.0, "map": m, "regions": [], "peak_ratio": 0.0, "coverage": 0.0}

    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med))) + 1e-6
    # Robust z-score: how many MADs above the typical error level is each pixel?
    z = (m - med) / (1.4826 * mad)
    hot = (z > 7.5).astype(np.uint8)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    coverage = float(hot.mean())
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    h, w = m.shape
    min_area = max(120, int(0.0015 * h * w))
    regions = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        regions.append(
            {
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "area_frac": round(area / (h * w), 5),
                "mean_z": round(float(z[labels == i].mean()), 2),
            }
        )
    regions.sort(key=lambda r: r["area_frac"], reverse=True)
    regions = regions[:6]

    peak_ratio = float(np.percentile(m, 99.5) / (med + 1e-6))

    # Score rises with how much area is anomalous, but saturates: a fully "hot"
    # image is a compression artefact, not a splice, so we damp very high coverage.
    blob_area = sum(r["area_frac"] for r in regions)
    localisation = blob_area / (coverage + 1e-6) if coverage > 0 else 0.0
    raw = 0.0
    if regions:
        raw = min(1.0, blob_area * 26.0) * min(1.0, localisation)
        raw = raw * min(1.0, peak_ratio / 6.0)
    if coverage > 0.25:  # whole-image noise, not a localised edit
        raw *= 0.3
    return {
        "score": float(np.clip(raw, 0.0, 1.0)),
        "map": m,
        "regions": regions,
        "peak_ratio": round(peak_ratio, 2),
        "coverage": round(coverage, 5),
    }
