"""Generator fingerprinting: not just 'fake', but *which* generator made it.

Different architectures leave different radial spectra - GAN transposed
convolutions ring at different radii from diffusion decoders. We fit a small
multiclass head on the radial profile of fakes whose generator we know, because
we generated them ourselves and therefore have free labels.

Output is threat intelligence, not a verdict: it never changes the risk score,
it tells the platform which toolchain is currently attacking them.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.spectral import profile_vector, spectral_features
from .models import fingerprint_model

GENERATOR_LABELS = {
    "stable_diffusion": "Stable Diffusion (latent diffusion)",
    "sdxl": "SDXL / SDXL-Turbo",
    "stylegan": "StyleGAN family",
    "faceswap": "Face-swap compositing (DeepFaceLab/Roop family)",
    "gan_other": "Other GAN",
    "real": "No generator - camera capture",
}


def identify(rgb: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
    """Return (generator_key, confidence, per-class probabilities)."""
    model = fingerprint_model()
    if model is None or rgb is None:
        return None, 0.0, {}
    try:
        prof = profile_vector(rgb)
        feats = spectral_features(rgb)
        x = np.concatenate([prof, np.array([feats[k] for k in sorted(feats)], dtype=np.float32)])[None, :]
        probs = model.predict_proba(x)[0]
        classes = list(model.classes_)
        best = int(np.argmax(probs))
        return classes[best], float(probs[best]), {c: round(float(p), 4) for c, p in zip(classes, probs)}
    except Exception:
        return None, 0.0, {}


def describe(key: Optional[str]) -> Optional[str]:
    return GENERATOR_LABELS.get(key or "", None)
