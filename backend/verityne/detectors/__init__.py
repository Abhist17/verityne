"""Detector registry. Order here is the order shown in the dashboard."""
from __future__ import annotations

from typing import List

from .base import Detector, SubmissionPayload
from .face_match import FaceMatchDetector
from .id_forensics import IDForensicsDetector
from .liveness_video import LivenessVideoDetector
from .metadata_exif import MetadataExifDetector
from .selfie_deepfake import SelfieDeepfakeDetector

#: Stage one detectors are mutually independent and run concurrently. Between
#: them they populate the shared cache (decoded images, face crops), which is why
#: face matching - which needs both crops - waits for stage two rather than
#: decoding and re-detecting the same faces a second time.
STAGE_ONE: List[Detector] = [
    SelfieDeepfakeDetector(),
    IDForensicsDetector(),
    LivenessVideoDetector(),
    MetadataExifDetector(),
]
STAGE_TWO: List[Detector] = [FaceMatchDetector()]

ALL_DETECTORS: List[Detector] = STAGE_ONE + STAGE_TWO

__all__ = ["Detector", "SubmissionPayload", "ALL_DETECTORS", "STAGE_ONE", "STAGE_TWO"]
