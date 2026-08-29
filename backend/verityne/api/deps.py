"""Shared FastAPI dependencies: auth, upload handling, response shaping."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import Depends, Header, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..config import API_KEY, MAX_UPLOAD_BYTES
from ..db import get_db

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_VIDEO = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    """Single-tenant demo auth. Production would resolve a per-merchant key here."""
    if x_api_key != API_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")
    return x_api_key


def db_session(session: Session = Depends(get_db)) -> Session:
    return session


async def save_upload(upload: Optional[UploadFile], dest_dir: Path, stem: str, kind: str = "image") -> Optional[Path]:
    """Stream an upload to disk with an extension allow-list and a size ceiling."""
    if upload is None or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix.lower()
    allowed = ALLOWED_IMAGE if kind == "image" else ALLOWED_VIDEO
    if suffix not in allowed:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{kind} must be one of {sorted(allowed)}, got '{suffix or 'no extension'}'",
        )
    dest = dest_dir / f"{stem}{suffix}"
    written = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"{kind} exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit",
                )
            fh.write(chunk)
    await upload.close()
    if written == 0:
        dest.unlink(missing_ok=True)
        return None
    return dest
