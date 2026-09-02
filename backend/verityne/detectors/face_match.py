"""Detector 4 - Face Match, selfie vs the portrait on the ID.

Flagged in both directions:

  * too low  -> the person holding the phone is not the person on the document
                (impersonation, or a bought selfie paired with a stolen ID);
  * too high -> the "selfie" is a copy of the ID portrait rather than a fresh
                capture. A genuine selfie of the same person against a printed,
                low-resolution card portrait lands well short of 1.0, so a
                near-perfect match is evidence of copying, not of identity.
"""
from __future__ import annotations

import functools
import json
from typing import Dict, List


from ..config import MODEL_ROOT

from ..schemas import DetectorOutput
from ..utils.hashing import hamming, phash
from ..utils.images import crop_face, detect_faces, largest_face, load_rgb
from .base import Detector, SubmissionPayload, clamp
from .models import cosine, face_embedder

#: Fallback band, used only until a calibration file exists.
#:
#: The two bounds are fitted by different scripts on different data, because they
#: are different questions. ``low`` - is this the same person - is fitted by
#: ``scripts/calibrate_face_match_lfw.py`` on LFW's 6,000 real photo-vs-photo
#: pairs, where the pipeline scores 98.2% ± 0.4%. ``high`` - is this "selfie"
#: just a copy of the printed card portrait - needs selfie-vs-document pairs,
#: which LFW has none of, so it still comes from ``scripts/calibrate.py`` on the
#: corpus and carries that corpus's caveat.
#:
#: The default below is LFW's, and the previous corpus-fitted 0.7835 is why this
#: matters: on real pairs that value accepted only 52% of genuine same-person
#: matches, i.e. it would have called nearly half of honest applicants
#: impersonators. It was not a wrong arithmetic - it was fitted on genuine pairs
#: that all derive from a single source photograph, which is not what a selfie
#: and a card portrait look like.
DEFAULT_LOW, DEFAULT_HIGH = 0.4065, 0.9878
BAND_PATH = MODEL_ROOT / "face_match_band.json"


@functools.lru_cache(maxsize=1)
def decision_band() -> tuple[float, float, str]:
    """(low, high, provenance). Falls back to the documented defaults."""
    if BAND_PATH.exists():
        try:
            data = json.loads(BAND_PATH.read_text())
            return float(data["low"]), float(data["high"]), data.get("fitted_on", "calibration file")
        except Exception:
            pass
    return DEFAULT_LOW, DEFAULT_HIGH, "uncalibrated defaults"


def reload_band() -> None:
    decision_band.cache_clear()


class FaceMatchDetector(Detector):
    name = "face_match"
    heavy = True

    def applicable(self, payload: SubmissionPayload) -> bool:
        return payload.selfie_path is not None and payload.id_doc_path is not None

    def _run(self, payload: SubmissionPayload) -> DetectorOutput:
        reasons: List[str] = []
        signals: Dict[str, object] = {}

        selfie_face = payload.cache.get("selfie_face")
        if selfie_face is None:
            selfie_rgb = payload.cache.get("selfie_rgb")
            if selfie_rgb is None:
                selfie_rgb = load_rgb(payload.selfie_path)
                payload.cache["selfie_rgb"] = selfie_rgb
            selfie_face = largest_face(selfie_rgb, size=256)
            payload.cache["selfie_face"] = selfie_face

        id_face = payload.cache.get("id_face")
        if id_face is None:
            id_rgb = payload.cache.get("id_rgb")
            if id_rgb is None:
                id_rgb = load_rgb(payload.id_doc_path, max_side=1400)
                payload.cache["id_rgb"] = id_rgb
            boxes = detect_faces(id_rgb, min_size=40)
            if boxes:
                id_face = crop_face(id_rgb, boxes[0], margin=0.15, size=224)
                payload.cache["id_face"] = id_face

        signals["selfie_face_found"] = selfie_face is not None
        signals["id_face_found"] = id_face is not None

        if selfie_face is None or id_face is None:
            missing = "selfie" if selfie_face is None else "ID document"
            return DetectorOutput(
                name=self.name, label=self.label, score=0.5, confidence=0.2,
                reasons=[f"Could not locate a usable face in the {missing} - identity match could not be computed"],
                signals=signals,
            )

        emb = face_embedder()
        if emb is None:
            return DetectorOutput(
                name=self.name, label=self.label, score=0.0, confidence=0.0, status="error",
                detail="face embedding model unavailable", signals=signals,
            )

        v_selfie, v_id = emb.embed(selfie_face), emb.embed(id_face)
        sim = cosine(v_selfie, v_id)
        genuine_low, genuine_high, provenance = decision_band()
        signals["cosine_similarity"] = round(sim, 4)
        signals["decision_band"] = {"low": genuine_low, "high": genuine_high, "source": provenance}

        # Keep the selfie embedding for cross-submission linkage.
        payload.cache["selfie_embedding"] = v_selfie.tolist()
        payload.cache["id_embedding"] = v_id.tolist()

        # Pixel-level duplicate check backs up the "too high" branch: a true copy
        # is close in perceptual-hash space too, whereas a real re-photograph is not.
        ph_selfie, ph_id = phash(selfie_face), phash(id_face)
        ph_dist = hamming(ph_selfie, ph_id)
        signals["phash_distance"] = ph_dist

        if sim < genuine_low:
            score = clamp(0.55 + (genuine_low - sim) * 1.4)
            reasons.append(
                f"The face in the selfie does not match the portrait on the ID (similarity {sim:.2f}, "
                f"below the {genuine_low:.2f} identity threshold) - possible impersonation"
            )
            direction = "mismatch"
        elif sim > genuine_high:
            score = clamp(0.55 + (sim - genuine_high) * 12.0)
            direction = "duplicate"
            reasons.append(
                f"Selfie and ID portrait are near-identical (similarity {sim:.2f}). A live selfie never matches a "
                "printed card portrait this closely - the selfie appears to be a copy of the ID photo"
            )
            if ph_dist <= 8:
                score = clamp(score + 0.2)
                reasons.append(f"Perceptual hashes are {ph_dist} bits apart, confirming the two images are the same picture")
        else:
            # Confidence in a genuine match tapers toward the band edges.
            mid = (genuine_low + genuine_high) / 2
            score = clamp(abs(sim - mid) / max(1e-6, genuine_high - genuine_low) * 0.5)
            direction = "match"
            reasons.append(f"Selfie matches the ID portrait at a plausible similarity for a live capture ({sim:.2f})")

        signals["direction"] = direction
        confidence = 0.85 if direction != "match" else 0.7
        return DetectorOutput(
            name=self.name, label=self.label, score=score, confidence=confidence,
            reasons=reasons, signals=signals,
        )
