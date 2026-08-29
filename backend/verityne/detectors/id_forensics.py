"""Detector 3 - ID Document Forensics.

Three sub-checks, all of which are cheap and deterministic enough to trust:

  1. OCR + structural validation. A PAN is not just 5 letters/4 digits/1 letter:
     the 4th character encodes holder type and the 5th is the first letter of the
     surname. Open-source PAN generators get these wrong constantly. Aadhaar
     carries a Verhoeff check digit.
  2. Error Level Analysis. A field that was pasted over a real template has a
     different compression history from its surroundings and lights up.
  3. The photo on the card is itself run through the deepfake classifier - a
     fully synthetic identity has a synthetic face printed on its ID.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..schemas import DetectorOutput
from ..utils.artifacts import save_heatmap
from ..utils.ela import tamper_score
from ..utils.images import detect_faces, crop_face, load_rgb, overlay_heatmap
from ..utils.ocr import available_backend, extract_text, find_fields
from ..utils.verhoeff import validate_aadhaar
from ..utils.spectral import spectral_features
from .base import Detector, SubmissionPayload, clamp
from .models import deepfake_classifier
from .selfie_deepfake import heuristic_spectral_score

# 4th character of a PAN: the holder category. Anything else is structurally invalid.
PAN_HOLDER_TYPES = set("ABCFGHJLPT")
PAN_HOLDER_MEANING = {
    "P": "Individual", "C": "Company", "H": "Hindu Undivided Family", "F": "Firm",
    "A": "Association of Persons", "T": "Trust", "B": "Body of Individuals",
    "L": "Local Authority", "J": "Artificial Juridical Person", "G": "Government",
}


def validate_pan(pan: str, claimed_name: Optional[str] = None) -> Dict[str, object]:
    out: Dict[str, object] = {"value": pan, "format_ok": False, "holder_type_ok": False,
                              "surname_initial_ok": None, "issues": []}
    if not pan or not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan):
        out["issues"].append("PAN does not match the AAAAA9999A structure")
        return out
    out["format_ok"] = True
    holder = pan[3]
    if holder in PAN_HOLDER_TYPES:
        out["holder_type_ok"] = True
        out["holder_type"] = PAN_HOLDER_MEANING.get(holder, holder)
    else:
        out["issues"].append(f"PAN 4th character '{holder}' is not a valid holder-type code")
    if claimed_name:
        parts = [p for p in re.split(r"\s+", claimed_name.strip()) if p]
        if parts:
            surname_initial = parts[-1][0].upper()
            out["surname_initial_ok"] = pan[4] == surname_initial
            if not out["surname_initial_ok"]:
                out["issues"].append(
                    f"PAN 5th character '{pan[4]}' should be the surname initial '{surname_initial}'"
                )
    return out


def template_consistency(rgb: np.ndarray, words: List[Dict]) -> Dict[str, object]:
    """Cheap layout/typography sanity checks that catch generator output.

    Real cards have consistent text height and baseline alignment inside a field
    block. Generated cards drawn with a default font often have uniform-to-the-pixel
    glyph heights, oddly perfect alignment, or wildly mixed heights where fields
    were pasted at different scales.
    """
    if len(words) < 4:
        return {"checked": False}
    heights = np.array([w["bbox"][3] - w["bbox"][1] for w in words if w["bbox"][3] > w["bbox"][1]], dtype=np.float32)
    if heights.size < 4:
        return {"checked": False}
    cv_height = float(heights.std() / (heights.mean() + 1e-6))
    lefts = np.array([w["bbox"][0] for w in words], dtype=np.float32)
    # How many distinct left-edge columns the text snaps to.
    col_spread = float(np.std(np.round(lefts / 8.0)))
    mean_conf = float(np.mean([w.get("conf", 0.0) for w in words]))
    return {
        "checked": True, "word_count": len(words),
        "glyph_height_cv": round(cv_height, 4),
        "left_edge_spread": round(col_spread, 3),
        "mean_ocr_confidence": round(mean_conf, 3),
    }


class IDForensicsDetector(Detector):
    name = "id_forensics"
    heavy = True

    def applicable(self, payload: SubmissionPayload) -> bool:
        return payload.id_doc_path is not None

    def _run(self, payload: SubmissionPayload) -> DetectorOutput:
        rgb = load_rgb(payload.id_doc_path, max_side=1400)
        payload.cache["id_rgb"] = rgb
        reasons: List[str] = []
        signals: Dict[str, object] = {}
        parts: Dict[str, float] = {}

        # ---- 1. OCR + structural validation ------------------------------------
        signals["ocr_backend"] = available_backend()
        text, words = extract_text(rgb)
        signals["ocr_chars"] = len(text)
        fields = find_fields(text)
        signals["extracted_fields"] = fields

        field_score = 0.0
        doc_type = "unknown"
        if fields.get("pan"):
            doc_type = "PAN"
            pan_check = validate_pan(fields["pan"], payload.claimed_name)
            signals["pan_validation"] = pan_check
            issues = pan_check["issues"]
            if issues:
                field_score = 0.85 if not pan_check["holder_type_ok"] else 0.6
                reasons.extend(issues)
            else:
                reasons.append(f"PAN {fields['pan']} passes structural validation ({pan_check.get('holder_type')})")
        elif fields.get("aadhaar"):
            doc_type = "Aadhaar"
            ok = validate_aadhaar(fields["aadhaar"])
            signals["aadhaar_verhoeff_valid"] = ok
            if not ok:
                field_score = 0.9
                reasons.append("Aadhaar number fails the Verhoeff check digit - the number is not a real UIDAI number")
            else:
                reasons.append("Aadhaar number passes the Verhoeff check digit")
        else:
            if signals["ocr_backend"] == "none":
                reasons.append("No OCR backend installed - document text could not be validated (install tesseract-ocr)")
                signals["ocr_degraded"] = True
            elif len(text) < 12:
                field_score = 0.45
                reasons.append("No readable ID number could be extracted from the document image")
        signals["doc_type"] = doc_type
        parts["fields"] = field_score

        # ---- 2. Error Level Analysis --------------------------------------------
        ela = tamper_score(rgb)
        parts["tamper"] = float(ela["score"])
        signals["ela"] = {"score": round(float(ela["score"]), 4), "peak_ratio": ela["peak_ratio"],
                          "coverage": ela["coverage"], "regions": ela["regions"]}
        if ela["score"] > 0.45 and ela["regions"]:
            biggest = ela["regions"][0]
            where = self._describe_region(biggest["bbox"], rgb.shape)
            reasons.append(
                f"Compression analysis shows {len(ela['regions'])} localised edited region(s); the largest sits "
                f"{where} and is {biggest['mean_z']:.0f} deviations above the document's baseline error level"
            )

        # ---- 3. Is the printed portrait itself synthetic? -------------------------
        photo_fake = 0.0
        boxes = detect_faces(rgb, min_size=40)
        if boxes:
            id_face = crop_face(rgb, boxes[0], margin=0.15, size=224)
            payload.cache["id_face"] = id_face
            payload.cache["id_face_box"] = boxes[0]
            clf = deepfake_classifier()
            photo_fake = clf.predict(id_face) if clf is not None else heuristic_spectral_score(spectral_features(id_face))
            signals["id_photo_p_fake"] = round(float(photo_fake), 4)
            if photo_fake > 0.7:
                reasons.append(f"The portrait printed on the document itself scores {photo_fake:.0%} synthetic")
        else:
            signals["id_photo_found"] = False
            reasons.append("No portrait photo could be located on the document")
            photo_fake = 0.35
        parts["photo"] = float(photo_fake)

        # ---- 4. Typography / layout ----------------------------------------------
        tmpl = template_consistency(rgb, words)
        signals["template"] = tmpl
        layout_score = 0.0
        if tmpl.get("checked"):
            cvh = tmpl["glyph_height_cv"]
            if cvh > 0.55:
                layout_score = 0.55
                reasons.append(f"Text on the card is rendered at inconsistent sizes (height CV {cvh:.2f}) - fields likely pasted separately")
            elif cvh < 0.03:
                layout_score = 0.35
                reasons.append("Every glyph on the card is pixel-identical in height, which is typical of programmatically generated cards rather than a scan")
        parts["layout"] = layout_score

        score = clamp(0.34 * parts["fields"] + 0.30 * parts["tamper"] + 0.24 * parts["photo"] + 0.12 * parts["layout"])
        confidence = 0.8 if signals["ocr_backend"] != "none" else 0.5
        if not boxes:
            confidence *= 0.85
        signals["subscores"] = {k: round(v, 3) for k, v in parts.items()}

        if not reasons:
            reasons.append("Document structure, compression profile and printed portrait all look consistent")

        heatmap_url = None
        try:
            heatmap_url = save_heatmap(overlay_heatmap(rgb, ela["map"], alpha=0.5), payload.submission_id, "id_document")
        except Exception:
            pass

        return DetectorOutput(
            name=self.name, label=self.label, score=score, confidence=confidence,
            reasons=reasons, signals=signals, heatmap_url=heatmap_url,
        )

    @staticmethod
    def _describe_region(bbox, shape) -> str:
        h, w = shape[:2]
        cx, cy = (bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h
        vert = "upper" if cy < 0.38 else ("lower" if cy > 0.62 else "middle")
        horiz = "left" if cx < 0.38 else ("right" if cx > 0.62 else "centre")
        return f"in the {vert}-{horiz} of the card"
