"""Public API contracts."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

VerdictLiteral = Literal["PASS", "REVIEW", "REJECT"]


class DetectorOutput(BaseModel):
    name: str
    label: str
    score: float = Field(0.0, ge=0.0, le=1.0, description="0 = looks genuine, 1 = looks fraudulent")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="How much weight this detector's own score deserves")
    status: Literal["ok", "skipped", "error"] = "ok"
    reasons: List[str] = []
    signals: Dict[str, Any] = {}
    heatmap_url: Optional[str] = None
    latency_ms: float = 0.0
    detail: Optional[str] = None


class VerifyResponse(BaseModel):
    submission_id: str
    merchant_id: str
    verdict: VerdictLiteral
    final_score: float
    abstained: bool = False
    top_reasons: List[str] = []
    explanation: str = ""
    attack_pattern: Optional[str] = None
    generator_guess: Optional[str] = None
    detector_breakdown: Dict[str, DetectorOutput] = {}
    heatmaps: Dict[str, str] = {}
    latency_ms: float = 0.0
    fusion_model: str = "heuristic"
    policy: Dict[str, Any] = {}
    created_at: Optional[dt.datetime] = None


class BatchVerifyRequest(BaseModel):
    submission_ids: List[str] = Field(default_factory=list, description="Existing submission ids to re-score")
    merchant_id: Optional[str] = None
    since: Optional[dt.datetime] = Field(None, description="Alternatively, re-score everything created after this")
    limit: int = Field(500, ge=1, le=5000)
    only_previously_passed: bool = Field(True, description="The retroactive-sweep case: look for fakes we let through")


class BatchVerifyItem(BaseModel):
    submission_id: str
    previous_verdict: Optional[str] = None
    previous_score: Optional[float] = None
    new_verdict: VerdictLiteral
    new_score: float
    newly_flagged: bool
    top_reasons: List[str] = []


class BatchVerifyResponse(BaseModel):
    scanned: int
    newly_flagged: int
    items: List[BatchVerifyItem]
    fusion_model: str
    elapsed_ms: float


class GauntletResult(BaseModel):
    submission_id: str
    name: str
    truth: Literal["real", "fake"]
    attack_type: Optional[str] = None
    verdict: VerdictLiteral
    score: float
    correct: bool
    top_reasons: List[str] = []
    latency_ms: float = 0.0
    thumb_url: Optional[str] = None


class GauntletSummary(BaseModel):
    total: int
    fakes_caught: int
    fakes_total: int
    reals_passed: int
    reals_reviewed: int
    reals_rejected: int
    reals_total: int
    detection_rate: float
    false_accept_rate: float
    false_reject_rate: float
    accuracy: float
    mean_latency_ms: float
    results: List[GauntletResult]


class LinkageMatch(BaseModel):
    submission_id: str
    merchant_id: str
    similarity: float
    claimed_name: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    verdict: Optional[str] = None
