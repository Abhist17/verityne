"""Liveness-clip synthesis for the evaluation corpus.

An honest caveat, stated here and repeated in the README and the metrics page:
we do not have a corpus of genuine recorded liveness videos, so both the
"genuine" and the "attacked" clips in this project are animated from stills
through the *same* motion, blink and sensor pipeline. The only difference
between the two classes is whether a face swap is composited in per frame.

That means the video numbers measure one specific thing - can we separate a
swapped face from an unswapped one under identical capture conditions - and
they should not be read as end-to-end liveness accuracy on real recordings.
Replacing this module with recorded clips is the single highest-value upgrade
to the eval set.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

from faces import swap_face


def _animate_frame(base: np.ndarray, t: float, rng: random.Random, amplitude: float = 1.0) -> np.ndarray:
    """One frame of a head-turn: yaw as horizontal shear + rotation + breathing scale."""
    h, w = base.shape[:2]
    yaw = math.sin(t * 2.0 * math.pi * 0.32) * 12.0 * amplitude
    roll = math.sin(t * 2.0 * math.pi * 0.21 + 1.1) * 3.5 * amplitude
    scale = 1.0 + 0.012 * math.sin(t * 2.0 * math.pi * 0.5)

    m = cv2.getRotationMatrix2D((w / 2, h / 2), roll, scale)
    m[0, 2] += yaw * 1.6
    m[1, 2] += math.sin(t * 2 * math.pi * 0.27) * 4.0 * amplitude
    frame = cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    shear = np.float32([[1, 0.035 * math.sin(t * 2 * math.pi * 0.32) * amplitude, 0], [0, 1, 0]])
    return cv2.warpAffine(frame, shear, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _apply_blink(frame: np.ndarray, closed: float) -> np.ndarray:
    """Darken and compress the eye band to fake an eyelid closing."""
    if closed <= 0.01:
        return frame
    h, w = frame.shape[:2]
    y0, y1 = int(h * 0.36), int(h * 0.50)
    band = frame[y0:y1].astype(np.float32)
    band *= (1.0 - 0.55 * closed)
    frame = frame.copy()
    frame[y0:y1] = np.clip(band, 0, 255).astype(np.uint8)
    return frame


def make_liveness_video(
    face: Image.Image,
    out_path: str | Path,
    swap_source: Optional[Image.Image] = None,
    seed: int = 0,
    fps: int = 15,
    duration_s: float = 4.0,
    size: int = 384,
) -> Dict:
    """Render a liveness clip. With `swap_source`, a face swap is composited per frame."""
    rng = random.Random(seed)
    base = cv2.resize(np.asarray(face.convert("RGB")), (size, size), interpolation=cv2.INTER_AREA)
    n_frames = int(fps * duration_s)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))

    # Blinks every 4-6 seconds, the human baseline the detector cites.
    blink_period = rng.uniform(4.0, 6.0)
    blink_offset = rng.uniform(0.0, blink_period)
    swapped_frames = 0

    for i in range(n_frames):
        t = i / fps
        frame = _animate_frame(base, t, rng)

        phase = ((t + blink_offset) % blink_period) / blink_period
        closed = max(0.0, 1.0 - abs(phase - 0.02) / 0.035) if phase < 0.06 else 0.0
        frame = _apply_blink(frame, closed)

        if swap_source is not None:
            # Per-frame jitter is what a real swap pipeline produces: the aligner
            # re-estimates landmarks every frame and lands slightly differently.
            jx, jy = rng.randint(-4, 4), rng.randint(-4, 4)
            blend = rng.uniform(0.82, 0.96)
            frame = np.asarray(
                swap_face(Image.fromarray(frame), swap_source, blend=blend, jitter=(jx, jy)), dtype=np.uint8
            )
            swapped_frames += 1

        # Identical sensor simulation for both classes.
        frame = frame.astype(np.float32)
        frame += np.random.normal(0, rng.uniform(1.5, 3.5), frame.shape).astype(np.float32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        # A per-frame JPEG pass, the way a phone encoder would.
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(72, 90)])
        if ok:
            frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    return {
        "path": str(out_path),
        "frames": n_frames,
        "fps": fps,
        "duration_s": duration_s,
        "face_swapped": swap_source is not None,
        "swapped_frames": swapped_frames,
        "synthesis_note": "animated from a still; see videos.py docstring for the eval caveat",
    }
