#!/usr/bin/env python3
"""Build a tamper-detection evaluation set from real captured documents.

    python backend/scripts/build_real_docs.py --per-capture 250

Why
---
``eval/metrics.json`` reports ID forensics at 0.526 AUC, and tamper-only at
0.664 on photos and 0.565 on scans. Those numbers were measured on cards this
repo drew itself in ``scripts/idcards.py``. The detector's core signal - Error
Level Analysis - reads an image's *compression history*, and a card we render
and save once has a compression history we manufactured. Whether the check works
on a document that was actually printed, actually photographed and actually
JPEG'd by a phone is a question that corpus cannot answer.

MIDV-2020 can. See ``scripts/midv2020.py`` for exactly what is and is not real
about it: the identities are artificial, the printing and the capture are not,
and it is the capture that ELA reads.

What this builds
----------------
For each capture mode (scan, photo), an equal number of genuine and tampered
documents, all rectified out of their frames and all written through **one
identical encode path**. That last part is not a detail. If tampered files were
saved at a different JPEG quality, or resized differently, or written by a
different code path than genuine ones, the detector would learn the encoder and
report a beautiful AUC that means nothing. Every image here - genuine and
tampered alike - is rectified to the same long side and saved at the same
quality by the same function. The only difference between the classes is the
edit itself.

The three tampers are the ones that actually happen to identity documents:

``portrait_swap``   Another person's portrait pasted over the photo. The single
                    most common document fraud, and the one with the largest
                    compression-history discontinuity.
``field_splice``    A text field lifted from a different document of the same
                    type and pasted over this one's - a changed name, number or
                    date sourced from another card.
``copy_move``       A text block copied from *elsewhere on the same document*
                    and pasted over another. Deliberately the hard case: the
                    pasted pixels share the host's compression and print
                    history, so the error-level discontinuity is far weaker.

Attacks are assigned round-robin so each is equally represented, and the choice
is seeded, so a re-run reproduces the same set.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from midv2020 import CAPTURES, DOC_TYPES, Document, load_documents, load_rectified  # noqa: E402
from verityne.config import DATASET_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("realdocs")

ATTACKS = ("portrait_swap", "field_splice", "copy_move")

#: Every image, both classes, is written by `save_jpeg` at this quality and this
#: long side. Keeping it a module constant makes it hard to accidentally diverge.
LONG_SIDE = 1024
JPEG_QUALITY = 92


def save_jpeg(rgb: np.ndarray, path: Path) -> None:
    """The single encode path. Genuine and tampered images both go through here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "JPEG", quality=JPEG_QUALITY, subsampling=1)


# ---------------------------------------------------------------------------------
# Finding text to tamper with
# ---------------------------------------------------------------------------------

def text_boxes(rgb: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
    """Locate text-line rectangles on a card, largest first.

    A morphological gradient followed by a wide horizontal close merges glyphs
    into lines. This does not need to be a good text detector - it only needs to
    hand back rectangles that contain printed text, so the splice lands
    somewhere a forger would actually edit rather than on blank laminate.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))

    h, w = gray.shape
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for c in contours:
        x, y, bw_, bh_ = cv2.boundingRect(c)
        if bw_ < 0.06 * w or bh_ < 0.02 * h or bh_ > 0.18 * h or bw_ > 0.75 * w:
            continue
        if bw_ / max(1, bh_) < 1.6:  # want lines, not blobs
            continue
        box = (x, y, x + bw_, y + bh_)
        if face_box is not None and _overlaps(box, face_box):
            continue
        ink = _ink_ratio(gray[y : y + bh_, x : x + bw_])
        # Printed data fields are dark ink on a light substrate. Chips,
        # holograms and guilloche backgrounds are mid-tone and low-contrast, and
        # a "tampered" card whose edit lands on a decorative swirl is not the
        # attack this is meant to represent.
        if not 0.04 <= ink <= 0.45:
            continue
        scored.append((ink * bw_ * bh_, box))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in scored]


def _ink_ratio(gray_patch: np.ndarray) -> float:
    """Fraction of the patch that is distinctly darker than its own background."""
    if gray_patch.size < 32:
        return 0.0
    p = gray_patch.astype(np.float32)
    bg = float(np.percentile(p, 80))
    spread = bg - float(np.percentile(p, 5))
    if spread < 25:  # nothing that reads as ink on paper
        return 0.0
    return float((p < bg - 0.45 * spread).mean())


def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _shape(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return box[2] - box[0], box[3] - box[1]


def pick_compatible(dst_boxes: Sequence[Tuple[int, int, int, int]],
                    src_boxes: Sequence[Tuple[int, int, int, int]],
                    rng: random.Random,
                    exclude_identical: bool = False):
    """Choose a (destination, source) text-line pair of comparable shape.

    Without this the splice stretches a short line across a long one and the
    result is an illegible smear - visually obvious, and obvious for a reason
    that has nothing to do with compression history. A forger pastes a field
    that fits. Constraining the pair to a similar aspect ratio and height keeps
    the text legible, so what the detector has to find is the error-level seam
    rather than a blur.
    """
    cands: List[Tuple[float, Tuple, Tuple]] = []
    for dst in dst_boxes:
        dw, dh = _shape(dst)
        if dw < 8 or dh < 8:
            continue
        for src in src_boxes:
            if exclude_identical and src == dst:
                continue
            sw, sh = _shape(src)
            if sw < 8 or sh < 8:
                continue
            if _overlaps(src, dst) and exclude_identical:
                continue
            aspect = abs(np.log((sw / sh) / (dw / dh)))
            height = abs(np.log(sh / dh))
            if aspect > 0.25 or height > 0.35:
                continue
            cands.append((aspect + height, dst, src))
    if not cands:
        return None
    # Sample from the better-matched half rather than always taking the best,
    # so the set does not collapse onto one field per document type.
    cands.sort(key=lambda t: t[0])
    _, dst, src = rng.choice(cands[: max(1, len(cands) // 2)])
    return dst, src


def _paste(dst: np.ndarray, patch: np.ndarray, box: Tuple[int, int, int, int], feather: int = 2) -> None:
    """Paste `patch` into `box` with a soft edge, in place.

    The feather matters: a hard rectangular seam is a giveaway that no real
    forger would leave and that would make the task artificially easy.
    """
    x1, y1, x2, y2 = box
    tw, th = x2 - x1, y2 - y1
    if tw < 4 or th < 4:
        return
    patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    mask = np.ones((th, tw), dtype=np.float32)
    f = max(1, min(feather, th // 3, tw // 3))
    mask[:f, :] *= np.linspace(0, 1, f)[:, None]
    mask[-f:, :] *= np.linspace(1, 0, f)[:, None]
    mask[:, :f] *= np.linspace(0, 1, f)[None, :]
    mask[:, -f:] *= np.linspace(1, 0, f)[None, :]
    mask = mask[:, :, None]

    region = dst[y1:y2, x1:x2].astype(np.float32)
    dst[y1:y2, x1:x2] = np.clip(region * (1 - mask) + patch.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def _match_tone(patch: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shift the patch's per-channel mean/std onto the region it replaces.

    A forger with any competence colour-matches. Without this the attack would
    be detectable by a colour histogram rather than by compression forensics,
    which is not the thing being measured.
    """
    p = patch.astype(np.float32)
    for c in range(3):
        ps, pm = p[:, :, c].std() + 1e-6, p[:, :, c].mean()
        ts, tm = target[:, :, c].std(), target[:, :, c].mean()
        p[:, :, c] = (p[:, :, c] - pm) * (ts / ps) + tm
    return np.clip(p, 0, 255).astype(np.uint8)


def _recompress(rgb: np.ndarray, quality: int) -> np.ndarray:
    """One extra JPEG generation - what a patch picked up before being pasted."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------------
# The tampers
# ---------------------------------------------------------------------------------

def portrait_swap(rgb: np.ndarray, face_box, donor_rgb: np.ndarray, donor_face, rng: random.Random) -> Optional[Dict]:
    """Paste a different person's portrait over this document's photo."""
    if face_box is None or donor_face is None:
        return None
    dx1, dy1, dx2, dy2 = donor_face
    patch = donor_rgb[dy1:dy2, dx1:dx2]
    if patch.size == 0:
        return None
    x1, y1, x2, y2 = face_box
    target = rgb[y1:y2, x1:x2]
    if target.size == 0:
        return None
    patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    patch = _match_tone(patch, target.astype(np.float32))
    patch = _recompress(patch, rng.choice([70, 80, 88]))
    _paste(rgb, patch, face_box, feather=3)
    return {"attack": "portrait_swap", "region": list(face_box)}


def field_splice(rgb: np.ndarray, boxes, donor_rgb: np.ndarray, donor_boxes, rng: random.Random) -> Optional[Dict]:
    """Paste a text line taken from a different document of the same type."""
    if not boxes or not donor_boxes:
        return None
    pair = pick_compatible(boxes, donor_boxes, rng)
    if pair is None:
        return None
    box, dbox = pair
    dx1, dy1, dx2, dy2 = dbox
    patch = donor_rgb[dy1:dy2, dx1:dx2]
    x1, y1, x2, y2 = box
    target = rgb[y1:y2, x1:x2]
    if patch.size == 0 or target.size == 0:
        return None
    patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    patch = _match_tone(patch, target.astype(np.float32))
    patch = _recompress(patch, rng.choice([72, 82, 90]))
    _paste(rgb, patch, box, feather=2)
    return {"attack": "field_splice", "region": list(box)}


def copy_move(rgb: np.ndarray, boxes, rng: random.Random) -> Optional[Dict]:
    """Copy a text block from elsewhere on this same document over another.

    The hard case on purpose: source and destination share one print run, one
    camera and one compression history, so the only discontinuity is the seam.
    """
    if len(boxes) < 2:
        return None
    pair = pick_compatible(boxes, boxes, rng, exclude_identical=True)
    if pair is None:
        return None
    dst, src = pair
    sx1, sy1, sx2, sy2 = src
    patch = rgb[sy1:sy2, sx1:sx2].copy()
    x1, y1, x2, y2 = dst
    target = rgb[y1:y2, x1:x2]
    if patch.size == 0 or target.size == 0:
        return None
    patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    patch = _match_tone(patch, target.astype(np.float32))
    _paste(rgb, patch, dst, feather=2)
    return {"attack": "copy_move", "region": list(dst), "source_region": list(src)}


# ---------------------------------------------------------------------------------

def admissible(face, boxes, rng: random.Random) -> bool:
    """Could any of the three attacks be placed on this document?

    Both classes are filtered by this, and that symmetry is the point. An
    earlier version of this script applied the test only where it was
    structurally necessary - a "tampered" document was dropped when no attack
    would fit, while genuine documents were taken as they came. That silently
    made the two classes different populations: the tampered class would have
    been drawn only from documents with a findable portrait and two compatible
    text lines, and any property that correlates with (say) crisper text would
    then separate the classes without a single edit being involved. Retention
    was 83% overall and 66% on photos, so it was not a rounding error.

    Applying it to both classes costs some documents and buys the guarantee that
    the only systematic difference between genuine and tampered here is the
    tamper.
    """
    if face is None:
        return False
    if len(boxes) < 2:
        return False
    return pick_compatible(boxes, boxes, rng, exclude_identical=True) is not None


def build(root: Path, out_dir: Path, per_capture: int, seed: int,
          doc_types: Sequence[str], captures: Sequence[str]) -> Tuple[List[Dict], Dict]:
    records: List[Dict] = []
    admission: Dict[str, Dict[str, int]] = {}

    for capture in captures:
        rng = random.Random(seed)
        docs = load_documents(root, capture, doc_types)
        rng.shuffle(docs)

        want = per_capture // 2
        # Donors come from the tail of the shuffled list, so a spliced field is
        # never lifted from a card that also appears in either class.
        split = int(len(docs) * 0.75)
        candidates, donor_docs = docs[:split], docs[split:] or docs[:split]
        by_type: Dict[str, List[Document]] = {}
        for d in donor_docs:
            by_type.setdefault(d.doc_type, []).append(d)

        donor_cache: Dict[str, tuple] = {}

        def donor_rect(d: Document):
            if d.key not in donor_cache:
                donor_cache[d.key] = load_rectified(d, long_side=LONG_SIDE)
            return donor_cache[d.key]

        genuine_n = 0
        tampered_n = 0
        attack_i = 0
        rejected = 0
        considered = 0
        fell_back = 0

        for doc in candidates:
            if genuine_n >= want and tampered_n >= want:
                break
            try:
                rgb, face = load_rectified(doc, long_side=LONG_SIDE)
            except Exception as exc:  # noqa: BLE001
                log.warning("skip %s: %s", doc.key, exc)
                continue
            considered += 1
            boxes = text_boxes(rgb, face)
            if not admissible(face, boxes, rng):
                rejected += 1
                continue

            # Alternate so both classes are drawn from one stream of admitted
            # documents rather than from two independently filtered ones.
            make_tampered = tampered_n < want and (genuine_n >= want or tampered_n <= genuine_n)

            if not make_tampered:
                rel = Path(capture) / "genuine" / f"{doc.doc_type}_{doc.path.stem}.jpg"
                save_jpeg(rgb, out_dir / rel)
                records.append({
                    "path": str(rel), "label": "genuine", "attack": None,
                    "doc_type": doc.doc_type, "capture": capture, "source": doc.key,
                    "face_box": list(face) if face else None,
                })
                genuine_n += 1
                continue

            attack = ATTACKS[attack_i % len(ATTACKS)]
            work = rgb.copy()
            info: Optional[Dict] = None
            if attack == "copy_move":
                info = copy_move(work, boxes, rng)
            else:
                pool = by_type.get(doc.doc_type) or donor_docs
                # A donor may not offer a compatible field; try a few before
                # giving up, so admission stays the thing that decides the set.
                for _ in range(3):
                    donor = rng.choice(pool)
                    try:
                        d_rgb, d_face = donor_rect(donor)
                    except Exception:  # noqa: BLE001
                        continue
                    if attack == "portrait_swap":
                        info = portrait_swap(work, face, d_rgb, d_face, rng)
                    else:
                        info = field_splice(work, boxes, d_rgb, text_boxes(d_rgb, d_face), rng)
                    if info is not None:
                        info["donor"] = donor.key
                        break
                    work = rgb.copy()  # a partial paste must not leak into the retry

            if info is None:
                # Admitted but this particular attack would not place. Fall back
                # to writing it as genuine rather than discarding it, so the
                # admitted population still lands wholly in one class or the
                # other and nothing is filtered away asymmetrically.
                fell_back += 1
                if genuine_n < want:
                    rel = Path(capture) / "genuine" / f"{doc.doc_type}_{doc.path.stem}.jpg"
                    save_jpeg(rgb, out_dir / rel)
                    records.append({
                        "path": str(rel), "label": "genuine", "attack": None,
                        "doc_type": doc.doc_type, "capture": capture, "source": doc.key,
                        "face_box": list(face) if face else None,
                    })
                    genuine_n += 1
                continue

            rel = Path(capture) / "tampered" / f"{doc.doc_type}_{doc.path.stem}_{attack}.jpg"
            save_jpeg(work, out_dir / rel)
            records.append({
                "path": str(rel), "label": "tampered", "doc_type": doc.doc_type,
                "capture": capture, "source": doc.key,
                "face_box": list(face) if face else None, **info,
            })
            tampered_n += 1
            attack_i += 1

        admission[capture] = {
            "considered": considered,
            "admitted": considered - rejected,
            "rejected": rejected,
            "genuine": genuine_n,
            "tampered": tampered_n,
            "assigned_tampered_but_attack_would_not_place": fell_back,
        }
        log.info("%s: %d genuine + %d tampered; %d/%d documents failed admission "
                 "(the same test applied to both classes), %d fell back to genuine",
                 capture, genuine_n, tampered_n, rejected, considered, fell_back)

    return records, admission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midv-root", default=str(DATASET_ROOT / "midv2020"))
    ap.add_argument("--out", default=str(DATASET_ROOT / "real_docs"))
    ap.add_argument("--per-capture", type=int, default=250,
                    help="documents per capture mode, split evenly genuine/tampered")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--captures", nargs="+", default=list(CAPTURES), choices=list(CAPTURES))
    ap.add_argument("--doc-types", nargs="+", default=DOC_TYPES, choices=DOC_TYPES)
    args = ap.parse_args()

    out_dir = Path(args.out)
    records, admission = build(Path(args.midv_root), out_dir, args.per_capture, args.seed,
                               args.doc_types, args.captures)

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "dataset": "MIDV-2020 derived tamper-detection set",
        "provenance": (
            "Documents from MIDV-2020: artificial identities on real document templates, "
            "physically printed, then photographed and scanned. The identity data is synthetic; "
            "the printing, capture and compression artefacts that ELA reads are real."
        ),
        "encode_path": {
            "long_side": LONG_SIDE, "jpeg_quality": JPEG_QUALITY,
            "note": "genuine and tampered images are written by the same function at the same "
                    "settings, so no encoding difference is correlated with the label",
        },
        "attacks": {
            "portrait_swap": "another person's portrait pasted over the document photo",
            "field_splice": "a text line lifted from a different document of the same type",
            "copy_move": "a text block copied from elsewhere on the same document - shares the "
                         "host's compression history, so the hardest of the three",
        },
        "seed": args.seed,
        "class_balance": {
            "admission_test": "has a locatable portrait and at least two mutually compatible "
                              "text lines - applied identically to both classes, so the only "
                              "systematic difference between genuine and tampered is the edit",
            "per_capture": admission,
        },
        "counts": _counts(records),
        "records": records,
    }, indent=2))
    log.info("wrote %d records to %s", len(records), manifest)
    for k, v in _counts(records).items():
        log.info("  %-28s %d", k, v)


def _counts(records: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {"total": len(records)}
    for r in records:
        out[r["label"]] = out.get(r["label"], 0) + 1
        out[f"{r['capture']}/{r['label']}"] = out.get(f"{r['capture']}/{r['label']}", 0) + 1
        if r.get("attack"):
            out[f"attack/{r['attack']}"] = out.get(f"attack/{r['attack']}", 0) + 1
    return out


if __name__ == "__main__":
    main()
