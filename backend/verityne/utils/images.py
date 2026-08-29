"""Image loading, face detection and crop helpers shared by every detector."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


def load_rgb(path: str | Path, max_side: int = 1024) -> np.ndarray:
    """Load an image as RGB uint8, downscaled so no side exceeds `max_side`."""
    img = Image.open(str(path))
    img = _apply_exif_orientation(img).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def _haar():
    """Haar cascade, or None. OpenCV 5 dropped `CascadeClassifier` from the default
    build, so this must never be assumed present."""
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    try:
        clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if clf.empty() else clf
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _mtcnn():
    """MTCNN if facenet-pytorch is importable; None otherwise (we fall back to Haar)."""
    try:
        import torch
        from facenet_pytorch import MTCNN

        from ..config import resolve_device

        return MTCNN(keep_all=True, device=resolve_device(), post_process=False)
    except Exception:
        return None


def detect_faces(rgb: np.ndarray, min_size: int = 24) -> List[BBox]:
    """Return face boxes, largest first. MTCNN when available, Haar as the fallback."""
    boxes: List[BBox] = []
    m = _mtcnn()
    if m is not None:
        try:
            det, probs = m.detect(to_pil(rgb))
            if det is not None:
                for b, p in zip(det, probs):
                    if p is not None and p >= 0.90:
                        x1, y1, x2, y2 = [int(v) for v in b]
                        boxes.append((max(0, x1), max(0, y1), x2, y2))
        except Exception:
            boxes = []
    cascade = _haar()
    if not boxes and cascade is not None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # ID-card photos are small and low contrast, so equalise before cascading.
        gray = cv2.equalizeHist(gray)
        for sf, nn in ((1.1, 5), (1.05, 3), (1.3, 2)):
            found = cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=nn, minSize=(min_size, min_size))
            if len(found):
                boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in found]
                break
    boxes = [b for b in boxes if (b[2] - b[0]) >= min_size and (b[3] - b[1]) >= min_size]
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def crop_face(rgb: np.ndarray, box: BBox, margin: float = 0.25, size: int = 224) -> np.ndarray:
    """Crop a face with margin and resize to a square tile."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        face = rgb
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)


def largest_face(rgb: np.ndarray, size: int = 224, margin: float = 0.25) -> Optional[np.ndarray]:
    boxes = detect_faces(rgb)
    if not boxes:
        return None
    return crop_face(rgb, boxes[0], margin=margin, size=size)


def overlay_heatmap(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] saliency map over an image using the JET colormap."""
    h, w = rgb.shape[:2]
    heat = np.nan_to_num(heat.astype(np.float32))
    rng = float(heat.max() - heat.min())
    heat = (heat - heat.min()) / rng if rng > 1e-8 else np.zeros_like(heat)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * rgb.astype(np.float32) + alpha * colored.astype(np.float32), 0, 255).astype(np.uint8)


def is_probably_screenshot(rgb: np.ndarray) -> Tuple[bool, str]:
    """Screen recapture leaves flat uniform borders and common device aspect ratios."""
    h, w = rgb.shape[:2]
    ratio = round(w / h, 3)
    known = {0.462: "1080x2340 phone screen", 0.5: "2:1 phone screen", 0.562: "16:9 screen", 1.778: "16:9 landscape"}
    border = 12
    edges = np.concatenate(
        [rgb[:border].reshape(-1, 3), rgb[-border:].reshape(-1, 3), rgb[:, :border].reshape(-1, 3), rgb[:, -border:].reshape(-1, 3)]
    )
    flat_border = float(edges.std()) < 6.0
    for r, tag in known.items():
        if abs(ratio - r) < 0.01 and flat_border:
            return True, f"uniform border with {tag} aspect ratio"
    if flat_border:
        return True, "uniform flat border consistent with a screen capture"
    return False, ""


def upscale_evidence(rgb: np.ndarray) -> dict:
    """How much real detail does this image have for its stated size?

    A photo straight off a sensor carries high-frequency detail all the way to
    its nominal resolution: halve it and re-enlarge it and you lose a lot. An
    image that was already upscaled - a 200px ID-card portrait blown up to pass
    as a 960px selfie, or a screen re-capture - has no such detail to lose, so
    the round-trip residual collapses.

    This is provenance, not content: it fires on "this file is not what it claims
    to be" regardless of who is in it, which makes it one of the few signals in
    the system that transfers cleanly to generators we have never seen.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    if min(h, w) < 64:
        return {"residual": None, "effective_scale": None, "upscaled": False}

    spread = float(gray.std()) + 1e-6
    small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    residual = float(np.abs(gray - back).mean() / spread)

    # Where does the radial spectrum hit the noise floor? The ratio of that
    # radius to Nyquist estimates the image's real resolution.
    from .spectral import azimuthal_average

    prof = azimuthal_average(rgb, n_bins=32)
    floor = float(np.median(prof[-6:]))
    above = np.where(prof > floor + 0.35)[0]
    effective_scale = float((above[-1] + 1) / len(prof)) if above.size else 1.0

    return {
        "residual": round(residual, 5),
        "effective_scale": round(effective_scale, 4),
        "nominal_px": int(max(h, w)),
        "effective_px": int(max(h, w) * effective_scale),
    }
