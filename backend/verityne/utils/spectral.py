"""Frequency-domain features.

Generative models build images by repeated upsampling, which stamps a periodic
grid into the spectrum. That artefact survives resizing and mild JPEG, and it
generalises across generator families far better than a CNN trained on one of
them - which is exactly why it is here as a second opinion on the CNN.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def _gray_float(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def azimuthal_average(rgb: np.ndarray, n_bins: int = 64, size: int = 256) -> np.ndarray:
    """1-D radial power spectrum. Real photos fall off smoothly; synthetic images ripple."""
    g = _gray_float(rgb, size)
    # Hann window kills edge-wrap energy that would otherwise dominate the spectrum.
    win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(g * win))
    power = np.log1p(np.abs(f) ** 2)

    cy, cx = size // 2, size // 2
    y, x = np.indices((size, size))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = size // 2
    bins = np.clip((r / r_max * n_bins).astype(int), 0, n_bins - 1)
    prof = np.bincount(bins.ravel(), weights=power.ravel(), minlength=n_bins)
    counts = np.bincount(bins.ravel(), minlength=n_bins)
    prof = prof / np.maximum(counts, 1)
    # Normalise out global brightness/contrast so the *shape* is what we compare.
    prof = prof - prof.mean()
    denom = prof.std()
    return (prof / denom if denom > 1e-8 else prof).astype(np.float32)


def resample_residual(gray: np.ndarray) -> float:
    """Detail lost by a halve-and-restore round trip, normalised by contrast.

    High on a real sensor capture (there is genuine detail to destroy), near zero
    on anything that was already upscaled - an ID-card portrait blown up to pass
    as a selfie, or a photo of a screen.
    """
    h, w = gray.shape
    if min(h, w) < 64:
        return 0.0
    spread = float(gray.std()) + 1e-6
    small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return float(np.abs(gray - back).mean() / spread)


def effective_scale(prof: np.ndarray) -> float:
    """Fraction of the nominal resolution that carries real detail."""
    floor = float(np.median(prof[-6:]))
    above = np.where(prof > floor + 0.35)[0]
    return float((above[-1] + 1) / len(prof)) if above.size else 1.0


def spectral_features(rgb: np.ndarray) -> Dict[str, float]:
    """Compact, interpretable spectral descriptors used by the fingerprinting head.

    Includes two provenance features (`resample_residual`, `effective_scale`)
    that describe whether the file is the resolution it claims to be. They turned
    out to carry more signal than the pretrained CNN on this corpus, which is
    exactly why they are measured rather than assumed.
    """
    prof = azimuthal_average(rgb)
    n = len(prof)
    high = prof[int(n * 0.6):]
    mid = prof[int(n * 0.25): int(n * 0.6)]

    # Upsampling grids show up as periodic ripple in the radial profile.
    d = np.diff(prof)
    sign_flips = float(np.mean(np.diff(np.sign(d)) != 0)) if len(d) > 2 else 0.0

    g = _gray_float(rgb)
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    size = mag.shape[0]
    cy = cx = size // 2
    # Energy at exactly the half-Nyquist points, where 2x upsampling leaves peaks.
    quarter = size // 4
    peak_pts = [mag[cy, cx + quarter], mag[cy, cx - quarter], mag[cy + quarter, cx], mag[cy - quarter, cx]]
    ring = mag[cy - quarter - 3: cy - quarter + 4, cx - quarter - 3: cx - quarter + 4]
    grid_peak = float(np.mean(peak_pts) / (np.median(ring) + 1e-8))

    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray_full, cv2.CV_32F)
    return {
        "resample_residual": resample_residual(gray_full),
        "effective_scale": effective_scale(prof),
        "hf_energy": float(high.mean()),
        "mf_energy": float(mid.mean()),
        "hf_slope": float(np.polyfit(np.arange(len(high)), high, 1)[0]) if len(high) > 2 else 0.0,
        "ripple": sign_flips,
        "grid_peak": grid_peak,
        "spectral_std": float(prof.std()),
        "laplacian_var": float(lap.var()),
        "hf_over_mf": float(high.mean() - mid.mean()),
    }


def profile_vector(rgb: np.ndarray, n_bins: int = 48) -> np.ndarray:
    """The full radial profile, for the generator-fingerprinting classifier."""
    return azimuthal_average(rgb, n_bins=n_bins)
