"""Live face match: a webcam frame against the portrait on a document.

This is the verification question on its own, with nothing else attached. The
full pipeline answers "should this merchant be onboarded", which needs a liveness
clip, provenance checks and a fusion pass. Here the only question is whether the
person in front of the camera is the person on the card, answered fast enough to
run against a live video feed.

Two deliberate choices:

* **Nothing is written to disk.** Frames arrive, are decoded in memory, embedded,
  and dropped. A webcam preview streams many frames per session and almost all of
  them are noise; persisting them would build a face database nobody asked for.
  The full ``POST /verify`` path is the one that stores evidence, because there a
  verdict has to be auditable.
* **The threshold is reported with its provenance.** The caller is told which
  number was used and where it came from, so a match near the boundary can be
  read for what it is rather than as a bare yes.
"""
from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from ..config import MAX_UPLOAD_BYTES
from ..detectors.face_match import decision_band
from ..detectors.models import cosine, face_embedder
from ..utils.images import largest_face
from .deps import ALLOWED_IMAGE, require_api_key

router = APIRouter()


class FaceMatchResult(BaseModel):
    similarity: Optional[float] = Field(None, description="Cosine similarity, -1..1. Null if a face was missing.")
    match: Optional[bool] = Field(None, description="Whether similarity clears the identity threshold.")
    threshold: float
    threshold_source: str = Field(description="Where the threshold was fitted, so the number can be argued with.")
    selfie_face_found: bool
    reference_face_found: bool
    margin: Optional[float] = Field(None, description="similarity - threshold. Negative means below the line.")
    confidence: str = Field(description="strong | borderline | weak — how far the score sits from the threshold.")
    detail: str
    latency_ms: float


async def _read_image(upload: UploadFile, label: str) -> np.ndarray:
    from pathlib import Path

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_IMAGE:
        raise HTTPException(415, f"{label} must be one of {sorted(ALLOWED_IMAGE)}, got '{suffix}'")
    raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    await upload.close()
    if not raw:
        raise HTTPException(400, f"{label} is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{label} exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, f"{label} is not a readable image")
    # Cap the long side: a webcam frame arrives far larger than the embedder needs,
    # and downscaling first is most of the latency budget.
    w, h = img.size
    scale = 1024 / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


@router.post("/face-match/live", response_model=FaceMatchResult,
             summary="Match a live webcam frame against a document portrait")
async def face_match_live(
    selfie: UploadFile = File(..., description="A single webcam frame"),
    reference: UploadFile = File(..., description="The ID document, or any reference photo"),
    _: str = Depends(require_api_key),
) -> FaceMatchResult:
    t0 = time.perf_counter()
    embedder = face_embedder()
    if embedder is None:
        raise HTTPException(503, "Face embedder is not loaded")

    selfie_rgb = await _read_image(selfie, "selfie")
    ref_rgb = await _read_image(reference, "reference")

    low, high, source = decision_band()

    selfie_face = largest_face(selfie_rgb)
    ref_face = largest_face(ref_rgb)
    elapsed = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731

    if selfie_face is None or ref_face is None:
        missing = " and ".join(
            [n for n, f in (("the camera frame", selfie_face), ("the reference image", ref_face)) if f is None]
        )
        return FaceMatchResult(
            similarity=None, match=None, threshold=low, threshold_source=source,
            selfie_face_found=selfie_face is not None, reference_face_found=ref_face is not None,
            margin=None, confidence="weak",
            detail=f"No face found in {missing}. Move closer, improve the lighting, or hold the card flat.",
            latency_ms=elapsed(),
        )

    sim = cosine(embedder.embed(selfie_face), embedder.embed(ref_face))
    margin = sim - low
    # A score sitting on the threshold is exactly the case a human should see,
    # so it is reported as borderline rather than rounded into a yes or a no.
    confidence = "strong" if abs(margin) >= 0.10 else ("borderline" if abs(margin) >= 0.03 else "weak")

    if sim >= high:
        detail = (
            f"Similarity {sim:.3f} is above {high:.3f} — too close for two separate photographs. "
            "This is what a selfie copied from the ID portrait looks like, not a live capture."
        )
    elif sim >= low:
        detail = f"Same person: similarity {sim:.3f} clears the {low:.3f} identity threshold."
    else:
        detail = f"Different person: similarity {sim:.3f} is below the {low:.3f} identity threshold."

    return FaceMatchResult(
        similarity=round(sim, 4), match=bool(low <= sim < high), threshold=round(low, 4),
        threshold_source=source, selfie_face_found=True, reference_face_found=True,
        margin=round(margin, 4), confidence=confidence, detail=detail, latency_ms=elapsed(),
    )


@router.get("/face-match/threshold", summary="The identity threshold in force, and where it came from")
def face_match_threshold() -> dict:
    low, high, source = decision_band()
    return {"low": round(low, 4), "high": round(high, 4), "fitted_on": source}
