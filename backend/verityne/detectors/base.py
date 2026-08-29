"""Detector contract.

Every detector is a small object with a `run` method that takes the submission
payload and returns a DetectorOutput. Detectors must never raise: a failure
degrades to status='error' with score 0 and confidence 0 so the fusion layer can
route around it instead of the whole request dying.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import DETECTOR_LABELS
from ..schemas import DetectorOutput
from ..utils.jsonsafe import to_jsonable


@dataclass
class SubmissionPayload:
    """Everything a detector may look at, plus a scratch space they share."""

    submission_id: str
    merchant_id: str = "default"
    selfie_path: Optional[Path] = None
    video_path: Optional[Path] = None
    id_doc_path: Optional[Path] = None
    claimed_name: Optional[str] = None
    claimed_id_number: Optional[str] = None
    claimed_dob: Optional[str] = None
    submitted_at: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    # Shared cache so we crop/detect a face once and reuse it across detectors.
    cache: Dict[str, Any] = field(default_factory=dict)


class Detector:
    name: str = "base"
    #: Detectors that only need cheap CPU work run inline; heavy ones go to the process pool.
    heavy: bool = False

    @property
    def label(self) -> str:
        return DETECTOR_LABELS.get(self.name, self.name)

    def applicable(self, payload: SubmissionPayload) -> bool:
        return True

    def _run(self, payload: SubmissionPayload) -> DetectorOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, payload: SubmissionPayload) -> DetectorOutput:
        started = time.perf_counter()
        if not self.applicable(payload):
            return DetectorOutput(
                name=self.name,
                label=self.label,
                score=0.0,
                confidence=0.0,
                status="skipped",
                detail="Required input not supplied",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        try:
            out = self._run(payload)
        except Exception as exc:  # noqa: BLE001 - a broken detector must not break the verdict
            return DetectorOutput(
                name=self.name,
                label=self.label,
                score=0.0,
                confidence=0.0,
                status="error",
                detail=f"{type(exc).__name__}: {exc}",
                signals={"traceback": traceback.format_exc(limit=3)},
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        out.latency_ms = (time.perf_counter() - started) * 1000
        out.name, out.label = self.name, self.label
        out.score = float(min(1.0, max(0.0, out.score)))
        out.confidence = float(min(1.0, max(0.0, out.confidence)))
        # Detectors compute in numpy; strip numpy types here so the verdict can be
        # both persisted and serialised without every detector having to remember.
        out.signals = to_jsonable(out.signals)
        out.reasons = [str(r) for r in out.reasons]
        return out


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, x)))
