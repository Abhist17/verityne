"""Central configuration. Paths, thresholds, and per-merchant policy loading."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
STORAGE_ROOT = Path(os.getenv("VERITYNE_STORAGE", REPO_ROOT / "storage"))
DATASET_ROOT = Path(os.getenv("VERITYNE_DATASETS", REPO_ROOT / "datasets"))
EVAL_ROOT = Path(os.getenv("VERITYNE_EVAL", REPO_ROOT / "eval"))
MODEL_ROOT = STORAGE_ROOT / "models"

UPLOAD_DIR = STORAGE_ROOT / "uploads"
HEATMAP_DIR = STORAGE_ROOT / "heatmaps"

for _d in (STORAGE_ROOT, UPLOAD_DIR, HEATMAP_DIR, MODEL_ROOT, DATASET_ROOT, EVAL_ROOT):
    _d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("VERITYNE_DB", f"sqlite:///{STORAGE_ROOT / 'verityne.db'}")
API_KEY = os.getenv("VERITYNE_API_KEY", "verityne-demo-key")
SLACK_WEBHOOK_URL = os.getenv("VERITYNE_SLACK_WEBHOOK", "")
DEVICE = os.getenv("VERITYNE_DEVICE", "auto")

MAX_UPLOAD_BYTES = int(os.getenv("VERITYNE_MAX_UPLOAD_MB", "40")) * 1024 * 1024
MAX_VIDEO_FRAMES = int(os.getenv("VERITYNE_MAX_VIDEO_FRAMES", "24"))
VIDEO_FRAME_STRIDE = int(os.getenv("VERITYNE_FRAME_STRIDE", "5"))

#: EXIF modes a genuine capture can physically never carry, with the reason.
#: `build_dataset.audit_exif_balance` and `scripts/ablate_fusion.leak_audit` both
#: consult this, so "allowed to be one-sided" has exactly one definition. A camera
#: cannot stamp a diffusion model's name into `Software`, so that mode being
#: fraud-only is a fact about cameras. Nothing else belongs here: every other mode
#: describes something that happens to honest people constantly, and the last time
#: one of them was fraud-only it was worth half the fused model's AUC
#: (`eval/ablation.json`).
PHYSICALLY_FRAUD_ONLY = {
    "generated": "no camera writes a diffusion model's name into the Software tag",
}

DETECTOR_NAMES = [
    "selfie_deepfake",
    "liveness_video",
    "id_forensics",
    "face_match",
    "metadata_exif",
    "behavioral",
]

#: The detectors the *learned* fusion model was fitted on. This is deliberately
#: not the same list as DETECTOR_NAMES. Behavioral telemetry has no labelled
#: corpus here - the only way to put it in the training matrix would be to
#: synthesise both the human and the bot side of it, which teaches the model our
#: own assumptions and nothing else. This repository has already paid for that
#: mistake twice (see eval/ablation.json), so behavioral is combined downstream
#: as its own evidence channel instead, with a ceiling. When real telemetry
#: exists, move the name into this list and retrain.
FUSION_TRAINED_NAMES = [
    "selfie_deepfake",
    "liveness_video",
    "id_forensics",
    "face_match",
    "metadata_exif",
]

# Human-facing labels used by the dashboard and the NL explainer.
DETECTOR_LABELS = {
    "selfie_deepfake": "Selfie Deepfake Detector",
    "liveness_video": "Liveness Video Analyzer",
    "id_forensics": "ID Document Forensics",
    "face_match": "Face Match (Selfie vs ID)",
    "metadata_exif": "Metadata / EXIF Auditor",
    "behavioral": "Behavioral Biometrics",
}


class MerchantPolicy(BaseModel):
    """One merchant's risk tolerance. Loaded from policy.yaml, overridable per request."""

    merchant_id: str = "default"
    min_risk_for_reject: float = 0.75
    min_risk_for_review: float = 0.40
    require_liveness: bool = True
    require_id_document: bool = True
    #: Off by default: the API accepts packets from server-to-server integrations
    #: that never rendered a form, and those must not all land in review. A
    #: merchant whose onboarding *is* the hosted form turns this on, and any
    #: submission arriving without telemetry is then routed to a human rather
    #: than scored as though the absence were innocent.
    require_behavioral: bool = False
    abstain_band: float = Field(
        0.05, description="Half-width around a threshold where we route to human review instead of deciding."
    )
    webhook_min_score: float = 0.90
    audit_retention_days: int = 365
    # Business inputs for the cost-of-friction model. Merchant-tunable on purpose:
    # these are assumptions, not measurements, and the dashboard exposes them as sliders.
    avg_fraud_loss_inr: float = 85000.0
    merchant_ltv_inr: float = 42000.0
    false_reject_abandon_prob: float = 0.35

    @property
    def review_band(self) -> tuple[float, float]:
        return (self.min_risk_for_review, self.min_risk_for_reject)


DEFAULT_POLICY_PATH = Path(__file__).parent / "policy.yaml"


@lru_cache(maxsize=1)
def _load_policy_file() -> Dict[str, Any]:
    if not DEFAULT_POLICY_PATH.exists():
        return {}
    with DEFAULT_POLICY_PATH.open() as fh:
        return yaml.safe_load(fh) or {}


def get_policy(merchant_id: str = "default") -> MerchantPolicy:
    """Resolve a merchant's policy: file defaults, overlaid with any per-merchant block."""
    raw = _load_policy_file()
    base = dict(raw.get("default", {}))
    base.update(raw.get("merchants", {}).get(merchant_id, {}))
    base["merchant_id"] = merchant_id
    return MerchantPolicy(**base)


def reload_policy() -> None:
    _load_policy_file.cache_clear()


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
