"""MIDV-2020 ingest: real identity documents, really printed, really captured.

What this dataset is, stated precisely
--------------------------------------
MIDV-2020 is 1,000 identity documents across ten types (Albanian, Azeri,
Spanish, Estonian, Finnish, Greek, Latvian, Russian, Serbian and Slovak ID
cards and passports). The *identities* are artificial — the names, numbers and
portraits were generated so the dataset could be published without exposing
anybody's real documents. The *documents* are not: each one was physically
printed and then photographed with a phone and scanned on a flatbed.

That distinction is the whole reason this is worth ingesting. The signal our ID
forensics detector reads — JPEG error levels, print halftone, sensor noise,
perspective, lighting, the compression history of a region — is a property of
capture, not of whose name is on the card. Those artefacts here are genuine,
whereas in ``scripts/idcards.py`` they are drawn by us. A detector that has only
ever seen cards we rendered has never been asked the real question.

What it therefore lets us measure that the synthetic corpus cannot:

  * the **false-positive rate of tamper detection on real, untampered
    documents** — how often ELA flags an honest card that merely went through a
    printer, a phone camera and a JPEG encoder;
  * **tamper AUC under real capture noise**, split by photo vs scan, which is
    where the corpus is weakest (0.526 overall, and the README's honest
    headline).

What it does not give us: Indian PAN or Aadhaar, so the structural field checks
(Verhoeff, PAN holder-type codes) are not exercised here and are excluded from
the real-document report rather than being scored against documents they were
never meant to read.

Annotations
-----------
Each document type ships a VIA-format JSON with two regions per image:

  ``doc_quad``  four corners of the card within the frame — used to rectify the
                document out of the photo, the way an upload pipeline would;
  ``face``      the portrait's bounding box on the card.

Both are in original-image pixel coordinates.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("verityne.midv")

#: The ten document types, as directory names.
DOC_TYPES = [
    "alb_id", "aze_passport", "esp_id", "est_id", "fin_id",
    "grc_passport", "lva_passport", "rus_internalpassport", "srb_passport", "svk_id",
]

#: Human-readable names, for report tables.
DOC_LABELS = {
    "alb_id": "Albania — ID card",
    "aze_passport": "Azerbaijan — passport",
    "esp_id": "Spain — ID card",
    "est_id": "Estonia — ID card",
    "fin_id": "Finland — ID card",
    "grc_passport": "Greece — passport",
    "lva_passport": "Latvia — passport",
    "rus_internalpassport": "Russia — internal passport",
    "srb_passport": "Serbia — passport",
    "svk_id": "Slovakia — ID card",
}

CAPTURES = ("scan", "photo")


@dataclass(frozen=True)
class Document:
    """One captured identity document, with its annotation."""

    doc_type: str
    capture: str
    path: Path
    face_box: Optional[Tuple[int, int, int, int]]  # x1, y1, x2, y2 in source pixels
    doc_quad: Optional[np.ndarray]  # (4, 2) float32, source pixels

    @property
    def key(self) -> str:
        return f"{self.doc_type}/{self.capture}/{self.path.stem}"


# ---------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------

def _regions(entry: Dict) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    face: Optional[Tuple[int, int, int, int]] = None
    quad: Optional[np.ndarray] = None
    for region in entry.get("regions", []):
        name = region.get("region_attributes", {}).get("field_name")
        shape = region.get("shape_attributes", {})
        if name == "face" and shape.get("name") == "rect":
            x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
            face = (int(x), int(y), int(x + w), int(y + h))
        elif name == "doc_quad":
            xs, ys = shape.get("all_points_x"), shape.get("all_points_y")
            if xs and ys and len(xs) >= 4:
                quad = np.asarray(list(zip(xs[:4], ys[:4])), dtype=np.float32)
    return face, quad


def load_documents(root: Path, capture: str, doc_types: Sequence[str] = DOC_TYPES) -> List[Document]:
    """All annotated documents for one capture mode ('scan' or 'photo')."""
    base = Path(root) / capture
    if not base.is_dir():
        raise FileNotFoundError(
            f"{base} not found. Extract the MIDV-2020 tarballs first:\n"
            f"  mkdir -p {root}/scan {root}/photo\n"
            f"  tar -xf {root}/scan_upright.tar -C {root}/scan\n"
            f"  tar -xf {root}/photo.tar -C {root}/photo"
        )

    out: List[Document] = []
    for doc_type in doc_types:
        ann_path = base / "annotations" / f"{doc_type}.json"
        img_dir = base / "images" / doc_type
        if not ann_path.exists() or not img_dir.is_dir():
            log.warning("skipping %s/%s — annotation or images missing", capture, doc_type)
            continue
        meta = json.loads(ann_path.read_text()).get("_via_img_metadata", {})
        for entry in meta.values():
            path = img_dir / entry["filename"]
            if not path.exists():
                continue
            face, quad = _regions(entry)
            out.append(Document(doc_type=doc_type, capture=capture, path=path,
                                face_box=face, doc_quad=quad))
    out.sort(key=lambda d: d.key)
    log.info("%s: %d documents across %d types", capture, len(out), len({d.doc_type for d in out}))
    return out


# ---------------------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------------------

def _order_quad(quad: np.ndarray) -> np.ndarray:
    """Order four corners as top-left, top-right, bottom-right, bottom-left."""
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).ravel()
    return np.asarray([
        quad[np.argmin(s)],  # top-left has the smallest x+y
        quad[np.argmin(d)],  # top-right has the smallest y-x
        quad[np.argmax(s)],  # bottom-right has the largest x+y
        quad[np.argmax(d)],  # bottom-left has the largest y-x
    ], dtype=np.float32)


def rectify(doc: Document, rgb: np.ndarray, long_side: int = 1024) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Warp the card out of the frame, carrying the face box along with it.

    This is what an upload pipeline does before anything else, and doing it here
    means the detector sees a document rather than a document on a desk. The
    perspective transform is applied to the face box too, so the portrait stays
    locatable after the warp.

    Returns (rectified_rgb, face_box_in_rectified_coords).
    """
    if doc.doc_quad is None:
        return rgb, doc.face_box

    src = _order_quad(doc.doc_quad)
    w = float(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = float(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    if w < 8 or h < 8:
        return rgb, doc.face_box

    scale = long_side / max(w, h)
    out_w, out_h = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    dst = np.asarray([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(rgb, m, (out_w, out_h), flags=cv2.INTER_AREA)

    face = None
    if doc.face_box is not None:
        x1, y1, x2, y2 = doc.face_box
        corners = np.asarray([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
        moved = cv2.perspectiveTransform(corners, m)[0]
        nx1, ny1 = moved.min(axis=0)
        nx2, ny2 = moved.max(axis=0)
        face = (
            int(max(0, nx1)), int(max(0, ny1)),
            int(min(out_w, nx2)), int(min(out_h, ny2)),
        )
        if face[2] - face[0] < 8 or face[3] - face[1] < 8:
            face = None
    return warped, face


def read_rgb(path: Path, max_side: int = 2600) -> Tuple[np.ndarray, float]:
    """Read a source capture as RGB uint8, bounded so a 4032px photo stays workable.

    Returns (rgb, scale) — the caller needs the scale to move the annotations,
    which are recorded in original-image pixels.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8), scale
    return np.asarray(img, dtype=np.uint8), 1.0


def load_rectified(doc: Document, long_side: int = 1024, max_side: int = 2600):
    """Read a document and return it rectified, with the face box moved with it."""
    rgb, scale = read_rgb(doc.path, max_side=max_side)
    if scale != 1.0:
        # Annotations are in original pixels; move them onto the resized image.
        scaled_quad = doc.doc_quad * scale if doc.doc_quad is not None else None
        scaled_face = (
            tuple(int(v * scale) for v in doc.face_box) if doc.face_box is not None else None
        )
        doc = Document(doc.doc_type, doc.capture, doc.path, scaled_face, scaled_quad)
    return rectify(doc, rgb, long_side=long_side)


def iter_rectified(docs: Sequence[Document], long_side: int = 1024) -> Iterator[Tuple[Document, np.ndarray, Optional[Tuple[int, int, int, int]]]]:
    for doc in docs:
        try:
            warped, face = load_rectified(doc, long_side=long_side)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read %s: %s", doc.key, exc)
            continue
        yield doc, warped, face
