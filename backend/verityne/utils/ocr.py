"""Pluggable OCR.

Backends are tried in order: pytesseract (fast, needs the tesseract binary),
easyocr (pure pip, heavier), then a layout fallback that reads our own synthetic
card templates by region. The fallback means the ID detector degrades to
"no text extracted" rather than failing, and the demo never dies on a missing
system package.
"""
from __future__ import annotations

import functools
import re
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@functools.lru_cache(maxsize=1)
def available_backend() -> str:
    if shutil.which("tesseract"):
        try:
            import pytesseract  # noqa: F401

            return "tesseract"
        except Exception:
            pass
    try:
        import easyocr  # noqa: F401

        return "easyocr"
    except Exception:
        pass
    return "none"


@functools.lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def preprocess_for_ocr(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 55, 55)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)


def extract_text(rgb: np.ndarray) -> Tuple[str, List[Dict]]:
    """Return (full_text, word_boxes). Empty when no backend is installed."""
    backend = available_backend()
    if backend == "tesseract":
        import pytesseract

        proc = preprocess_for_ocr(rgb)
        data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config="--psm 6")
        words = []
        for i, txt in enumerate(data["text"]):
            if txt.strip() and int(data["conf"][i]) > 0:
                words.append(
                    {
                        "text": txt.strip(),
                        "conf": float(data["conf"][i]) / 100.0,
                        # Boxes come from the 2x upscale, so halve them back.
                        "bbox": [data["left"][i] // 2, data["top"][i] // 2,
                                 (data["left"][i] + data["width"][i]) // 2,
                                 (data["top"][i] + data["height"][i]) // 2],
                    }
                )
        return " ".join(w["text"] for w in words), words
    if backend == "easyocr":
        res = _easyocr_reader().readtext(rgb)
        words = []
        for box, txt, conf in res:
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            words.append({"text": txt.strip(), "conf": float(conf), "bbox": [min(xs), min(ys), max(xs), max(ys)]})
        return " ".join(w["text"] for w in words), words
    return "", []


PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
AADHAAR_RE = re.compile(r"\b([2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4})\b")
DOB_RE = re.compile(r"\b([0-3]?\d[/\-.][0-1]?\d[/\-.](?:19|20)\d{2})\b")

#: OCR routinely confuses these glyph pairs. Which way to resolve the ambiguity
#: depends on whether the position is meant to hold a letter or a digit, so we
#: normalise positionally rather than globally.
TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                          "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"})
TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})

TOKEN_RE = re.compile(r"[A-Z0-9]+")


def normalise_pan(token: str) -> str:
    """Apply PAN's known shape (5 letters, 4 digits, 1 letter) to fix OCR confusions."""
    if len(token) != 10:
        return token
    head = token[:5].translate(TO_ALPHA)
    mid = token[5:9].translate(TO_DIGIT)
    tail = token[9].translate(TO_ALPHA)
    return head + mid + tail


def find_pan(text: str) -> Optional[str]:
    up = text.upper()
    direct = PAN_RE.search(up)
    if direct:
        return direct.group(1)
    # Fall back to shape-guided repair. Slide a 10-character window across every
    # alphanumeric run: OCR splits and merges tokens unpredictably, so a
    # non-overlapping scan of the whole string misses the number more often than not.
    for token in TOKEN_RE.findall(up):
        # A PAN is printed as its own token; scanning long words invites a
        # coincidental match once OCR confusions are forced onto the shape.
        if not 10 <= len(token) <= 14:
            continue
        for i in range(len(token) - 9):
            fixed = normalise_pan(token[i: i + 10])
            if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", fixed):
                return fixed
    return None


def find_aadhaar(text: str) -> Optional[str]:
    up = text.upper()
    direct = AADHAAR_RE.search(up)
    if direct:
        return re.sub(r"\s", "", direct.group(1))
    # Aadhaar is printed in 4-4-4 groups; repair each group as digits.
    for m in re.finditer(r"\b([A-Z0-9]{4})\s([A-Z0-9]{4})\s([A-Z0-9]{4})\b", up):
        joined = "".join(g.translate(TO_DIGIT) for g in m.groups())
        if joined.isdigit() and joined[0] not in "01":
            return joined
    return None


def find_fields(text: str) -> Dict[str, Optional[str]]:
    """Extract the fields we can validate. Returns None per field when absent."""
    pan = find_pan(text)
    aadhaar = None if pan else find_aadhaar(text)
    dob = DOB_RE.search(text.upper())
    return {"pan": pan, "aadhaar": aadhaar, "dob": dob.group(1) if dob else None}
