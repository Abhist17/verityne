"""Make detector output safe to persist and serialise.

Detectors compute with numpy, so their signal dicts are full of `np.float32`,
`np.int64` and `np.bool_`. Those are not JSON-serialisable, and the failure only
surfaces at the INSERT - the detector runs fine, the verdict is computed fine,
and then the whole request dies at the database boundary. Sanitising once, at
the point where every detector's output funnels through `Detector.run`, removes
a whole class of "works until it doesn't" bugs.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars, arrays and containers to plain Python."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        # JSON has no NaN or Infinity; both would produce invalid output.
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # 0-d numpy / torch scalars
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)
