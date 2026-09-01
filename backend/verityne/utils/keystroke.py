"""Keystroke rhythm features, computed once for both training and inference.

This module exists so that the feature vector a model is *fitted* on and the
feature vector it is *served* is produced by the same function. That skew is the
classic way a tabular model rots silently: the training script computes a
coefficient of variation over the whole session, the request path computes it
over a truncated buffer, nothing raises, and the model quietly reads a
distribution it never saw.

The inputs are deliberately primitive - two sequences of milliseconds - because
the sources are wildly different. A real keystroke corpus arrives as press and
release timestamps in a TSV; a browser buffer arrives as `{key, down, up}`
objects; a headless Chromium arrives through the same collector as the browser.
All three reduce to dwell and flight before they get here, so all three are
measured identically.

**Flight can be negative, and that is the point.** Human typists roll over: the
next key goes down before the previous one comes up. Automation that types
key-by-key cannot produce a negative flight at all, and tooling that does so
through the devtools protocol produces it only if someone deliberately modelled
it. Clamping negatives to zero - the obvious defensive move - would throw away
one of the strongest human signals in the data, so they are kept and counted.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

#: Gaps longer than this are the applicant reading, tabbing, or answering the
#: door - they describe attention, not motor control, so they are excluded from
#: the rhythm statistics and counted separately as pauses.
PAUSE_MS = 2000.0
#: Below this a "dwell" is a stuck-key artifact or a clock glitch rather than a
#: press. Kept generous; real presses run from about 20 ms upward.
MIN_DWELL_MS = 1.0
MAX_DWELL_MS = 2000.0

#: Features grounded in motor physiology: the shape of the dwell and flight
#: distributions, how often the hand rolls over from one key to the next, whether
#: the two co-vary, and how finely quantised the timings are. These describe a
#: hand, so they are the ones with any right to transfer between one population
#: and another - and the cross-corpus test in `train_behavioral.py` is what holds
#: that claim to account rather than asserting it.
CORE_FEATURES = [
    "dwell_mean", "dwell_std", "dwell_cv", "dwell_median",
    "dwell_p10", "dwell_p90", "dwell_min", "dwell_max", "dwell_iqr",
    "flight_mean", "flight_std", "flight_cv", "flight_median",
    "flight_p10", "flight_p90", "flight_min", "flight_max", "flight_iqr",
    "rollover_rate", "dwell_flight_corr",
    "dwell_unique_ratio", "flight_unique_ratio",
]

#: Features that describe the *task* rather than the hand. Every one of them was
#: measured carrying real predictive weight and real generalisation risk:
#:
#:   * `backspace_rate` - Aalto participants make typos, our automation never
#:     does, and the CMU corpus contains only clean entries. A model leaning on
#:     it flags every human who happens to type accurately.
#:   * `pause_rate`     - an artefact of Aalto's sentence-by-sentence protocol.
#:   * `typing_speed_cps` - depends on what is being typed, which differs between
#:     every corpus here and a real KYC form.
#:
#: They are kept computable, and kept out of the default fit. The ablation in
#: `eval/behavioral.json` reports exactly what they buy held-out and what they
#: cost on an unseen population, so the choice is evidenced rather than asserted.
CONTEXT_FEATURES = ["pause_rate", "typing_speed_cps", "backspace_rate"]

#: `n_keys` is deliberately in neither list. Session length is a property of how
#: this corpus was chopped - human sessions are accumulated to a target length,
#: automated ones fill a fixed five-field form - so a model given it scores well
#: by reading our own preprocessing back. It was the single highest-gain feature
#: on the first fit (0.435), which is exactly what a label leaking into the
#: features looks like, and this repository has shipped that bug twice already.

#: The ordered feature names a model is fitted on. Order is part of the contract
#: and is stored alongside the model, so adding a feature cannot silently shift
#: the columns of an already-trained one.
FEATURE_NAMES = CORE_FEATURES + CONTEXT_FEATURES

_ALL_COMPUTED = [
    "n_keys",
    "dwell_mean", "dwell_std", "dwell_cv", "dwell_median",
    "dwell_p10", "dwell_p90", "dwell_min", "dwell_max", "dwell_iqr",
    "flight_mean", "flight_std", "flight_cv", "flight_median",
    "flight_p10", "flight_p90", "flight_min", "flight_max", "flight_iqr",
    "rollover_rate", "pause_rate",
    "dwell_flight_corr",
    "dwell_unique_ratio", "flight_unique_ratio",
    "typing_speed_cps", "backspace_rate",
]


def _spread(a: np.ndarray, prefix: str) -> Dict[str, float]:
    if a.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in
                ("mean", "std", "cv", "median", "p10", "p90", "min", "max", "iqr")}
    mean = float(a.mean())
    std = float(a.std())
    q10, q25, q50, q75, q90 = (float(x) for x in np.percentile(a, [10, 25, 50, 75, 90]))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": float(std / mean) if abs(mean) > 1e-9 else 0.0,
        f"{prefix}_median": q50,
        f"{prefix}_p10": q10,
        f"{prefix}_p90": q90,
        f"{prefix}_min": float(a.min()),
        f"{prefix}_max": float(a.max()),
        f"{prefix}_iqr": q75 - q25,
    }


def _unique_ratio(a: np.ndarray) -> float:
    """How many distinct values the timings actually take, as a share of samples.

    A hand produces a near-continuous distribution, so this sits close to 1.0. A
    timer produces the same few numbers over and over, and quantised tooling -
    anything driving the devtools protocol on a fixed tick - collapses it hard.
    It catches the uniform case that a coefficient of variation catches, and also
    the case a CV misses: several distinct values, each repeated exactly.
    """
    if a.size == 0:
        return 0.0
    return float(np.unique(np.round(a, 1)).size) / float(a.size)


def keystroke_features(
    dwells: Sequence[float],
    flights: Sequence[float],
    *,
    n_keys: Optional[int] = None,
    backspaces: int = 0,
    span_s: Optional[float] = None,
    printable: Optional[int] = None,
) -> Dict[str, float]:
    """Reduce one typing session to the fitted feature row.

    `dwells` and `flights` are milliseconds. Both are filtered here rather than
    by the caller, so a browser buffer and a research corpus get the same
    treatment. Returns every name in FEATURE_NAMES, always, so a caller can rely
    on the shape without checking.
    """
    d = np.asarray([x for x in dwells if x is not None], dtype=float)
    d = d[np.isfinite(d)]
    d = d[(d >= MIN_DWELL_MS) & (d <= MAX_DWELL_MS)]

    f_all = np.asarray([x for x in flights if x is not None], dtype=float)
    f_all = f_all[np.isfinite(f_all)]
    # Split attention from motor control: long gaps are pauses, and are reported
    # as a rate rather than allowed to dominate the mean and the variance.
    pauses = f_all[f_all > PAUSE_MS]
    fl = f_all[f_all <= PAUSE_MS]

    out: Dict[str, float] = {}
    out.update(_spread(d, "dwell"))
    out.update(_spread(fl, "flight"))

    out["n_keys"] = float(n_keys if n_keys is not None else d.size)
    out["rollover_rate"] = float((fl < 0).sum()) / float(fl.size) if fl.size else 0.0
    out["pause_rate"] = float(pauses.size) / float(f_all.size) if f_all.size else 0.0
    out["dwell_unique_ratio"] = _unique_ratio(d)
    out["flight_unique_ratio"] = _unique_ratio(fl)

    # Do people hold keys longer when they are typing slower? In a hand, yes -
    # dwell and flight co-vary with fatigue and with the difficulty of the
    # bigram. Independently sampled synthetic timings show no such relationship.
    n = min(d.size, fl.size)
    corr = 0.0
    if n >= 4:
        a, b = d[:n], fl[:n]
        if a.std() > 1e-9 and b.std() > 1e-9:
            corr = float(np.corrcoef(a, b)[0, 1])
    out["dwell_flight_corr"] = 0.0 if not np.isfinite(corr) else corr

    total_keys = float(n_keys if n_keys is not None else d.size)
    chars = float(printable if printable is not None else total_keys)
    if span_s is None:
        span_s = float((d.sum() + f_all.clip(min=0).sum()) / 1000.0)
    out["typing_speed_cps"] = chars / span_s if span_s and span_s > 0.5 else 0.0
    out["backspace_rate"] = (backspaces / total_keys) if total_keys else 0.0

    return {k: float(out.get(k, 0.0)) for k in _ALL_COMPUTED}


def feature_row(feats: Dict[str, float], names: Optional[Sequence[str]] = None) -> np.ndarray:
    """A float array in a named order. The one place column ordering is decided.

    `names` defaults to FEATURE_NAMES; a model stores the order it was fitted on
    and passes it back here, so a feature added later appends a column an older
    model never asks for instead of shifting the ones it does.
    """
    order = names if names is not None else FEATURE_NAMES
    return np.asarray([float(feats.get(k, 0.0)) for k in order], dtype=np.float64)
