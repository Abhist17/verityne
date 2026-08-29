"""Detector 1 - Selfie Deepfake Detector.

Two independent opinions, deliberately:

  * a pretrained transformer/CNN classifier, which is strong on the generator
    families it saw in training and weak off-distribution;
  * a frequency-domain score, which keys on the upsampling grid that virtually
    every generative model leaves behind and therefore holds up on generators
    the CNN has never seen.

We report both and fuse them by agreement. When they disagree we lower
confidence rather than picking a winner - the fusion layer is told how much to
trust this detector, and honest uncertainty beats a confident coin flip.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from ..schemas import DetectorOutput
from ..utils.artifacts import save_heatmap
from ..utils.images import is_probably_screenshot, largest_face, load_rgb, overlay_heatmap
from ..utils.spectral import spectral_features
from .base import Detector, SubmissionPayload, clamp
from .models import deepfake_classifier, spectral_scorer


def heuristic_spectral_score(feats: Dict[str, float]) -> float:
    """Fallback when no trained spectral head exists yet.

    Hand-set weights on the two features that separate best in practice: the
    half-Nyquist grid peak (upsampling) and radial ripple (checkerboarding).
    """
    grid = clamp((feats.get("grid_peak", 1.0) - 1.15) / 1.6)
    ripple = clamp((feats.get("ripple", 0.0) - 0.30) / 0.35)
    smooth = clamp((260.0 - feats.get("laplacian_var", 260.0)) / 240.0)
    return clamp(0.45 * grid + 0.35 * ripple + 0.20 * smooth)


def spectral_probability(rgb: np.ndarray) -> tuple[float, Dict[str, float]]:
    """P(selfie is synthetic) from frequency-domain features.

    Uses the head fitted by scripts/calibrate.py when it exists; otherwise falls
    back to the documented hand-set weights so the detector still runs.
    """
    feats = spectral_features(rgb)
    bundle = spectral_scorer()
    if bundle is not None:
        try:
            model, names = bundle["model"], bundle["feature_names"]
            x = np.array([[feats[k] for k in names]], dtype=np.float64)
            return float(model.predict_proba(x)[0, 1]), feats
        except Exception:
            pass
    return heuristic_spectral_score(feats), feats


class SelfieDeepfakeDetector(Detector):
    name = "selfie_deepfake"
    heavy = True

    def applicable(self, payload: SubmissionPayload) -> bool:
        return payload.selfie_path is not None

    def _run(self, payload: SubmissionPayload) -> DetectorOutput:
        rgb = load_rgb(payload.selfie_path)
        payload.cache["selfie_rgb"] = rgb

        face = largest_face(rgb, size=256)
        payload.cache["selfie_face"] = face
        target = face if face is not None else rgb
        no_face = face is None

        reasons: List[str] = []
        signals: Dict[str, object] = {"face_detected": not no_face}

        clf = deepfake_classifier()
        cnn_p: float | None = None
        if clf is not None:
            cnn_p = clf.predict(target)
            signals["cnn_model"] = clf.repo_id
            signals["cnn_p_fake"] = round(cnn_p, 4)

        spec_p, feats = spectral_probability(target)
        signals["spectral_p_fake"] = round(spec_p, 4)
        signals["spectral_features"] = {k: round(v, 4) for k, v in feats.items()}

        if cnn_p is None:
            score = spec_p
            confidence = 0.45
            signals["fusion_mode"] = "spectral_only"
        else:
            agreement = 1.0 - abs(cnn_p - spec_p)
            # Weight the CNN higher, but let the spectral head pull the score when
            # it is confident and the CNN is ambivalent (the off-distribution case).
            score = clamp(0.62 * cnn_p + 0.38 * spec_p)
            confidence = clamp(0.45 + 0.5 * agreement)
            signals["agreement"] = round(agreement, 3)
            signals["fusion_mode"] = "cnn+spectral"
            if agreement < 0.5:
                reasons.append(
                    f"Model disagreement: neural detector {cnn_p:.0%} vs frequency analysis {spec_p:.0%} "
                    "- routing to human review is safer than either verdict alone"
                )

        if no_face:
            # No face means the classifier scored a scene, not a person. Say so.
            confidence *= 0.5
            reasons.append("No face could be located in the selfie - the image may be cropped, blurred or not a portrait")

        shot, why = is_probably_screenshot(rgb)
        if shot:
            score = clamp(max(score, 0.55))
            reasons.append(f"Selfie appears to be a re-captured screen image ({why})")
            signals["screenshot_suspected"] = True

        if cnn_p is not None and cnn_p > 0.7:
            reasons.append(f"Neural deepfake detector flags synthetic-face artefacts at {cnn_p:.0%} confidence")
        if spec_p > 0.65:
            reasons.append(
                f"Frequency analysis shows a periodic upsampling signature typical of generated imagery "
                f"(grid peak {feats.get('grid_peak', 0):.2f}x background)"
            )
        if score < 0.25 and not reasons:
            reasons.append("Selfie shows natural sensor noise and no generative artefacts")

        heatmap_url = self._heatmap(payload, rgb, target, no_face)

        return DetectorOutput(
            name=self.name,
            label=self.label,
            score=score,
            confidence=confidence,
            reasons=reasons,
            signals=signals,
            heatmap_url=heatmap_url,
        )

    def _heatmap(self, payload: SubmissionPayload, full_rgb, target, no_face: bool) -> str | None:
        """Grad-CAM over the region the classifier actually looked at."""
        clf = deepfake_classifier()
        cam = clf.saliency(target) if clf is not None else None
        if cam is None:
            # Without a network to hook, show where high-frequency residual energy sits.
            gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
            cam = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0))
            cam = cv2.resize(cam, (32, 32), interpolation=cv2.INTER_AREA)
        try:
            overlay = overlay_heatmap(target, cam)
            return save_heatmap(overlay, payload.submission_id, "selfie")
        except Exception:
            return None
