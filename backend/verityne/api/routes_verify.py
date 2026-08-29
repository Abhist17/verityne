"""/verify, /batch-verify and submission browsing."""
from __future__ import annotations

import datetime as dt
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..config import get_policy
from ..db import Submission, Verdict, log_event
from ..pipeline import run_pipeline, storage_dir
from ..schemas import BatchVerifyItem, BatchVerifyRequest, BatchVerifyResponse, VerifyResponse
from .deps import db_session, require_api_key, save_upload

router = APIRouter()


@router.post("/verify", response_model=VerifyResponse, summary="Score one KYC packet")
async def verify(
    selfie: Optional[UploadFile] = File(None, description="Selfie still"),
    liveness_video: Optional[UploadFile] = File(None, description="Short liveness clip"),
    id_document: Optional[UploadFile] = File(None, description="Government ID image"),
    merchant_id: str = Form("default"),
    external_ref: Optional[str] = Form(None),
    claimed_name: Optional[str] = Form(None),
    claimed_id_number: Optional[str] = Form(None),
    claimed_dob: Optional[str] = Form(None),
    session: Session = Depends(db_session),
    _: str = Depends(require_api_key),
) -> VerifyResponse:
    if selfie is None and id_document is None and liveness_video is None:
        raise HTTPException(400, "At least one of selfie, liveness_video or id_document is required")

    submission = Submission(
        merchant_id=merchant_id,
        external_ref=external_ref,
        status="PROCESSING",
        source="api",
        extra={"claimed_name": claimed_name, "claimed_id_number": claimed_id_number, "claimed_dob": claimed_dob},
    )
    session.add(submission)
    session.flush()  # assigns the id we key storage on

    d = storage_dir(submission.id)
    submission.selfie_path = str(await save_upload(selfie, d, "selfie", "image") or "") or None
    submission.id_doc_path = str(await save_upload(id_document, d, "id_document", "image") or "") or None
    submission.video_path = str(await save_upload(liveness_video, d, "liveness", "video") or "") or None
    log_event(session, "submission_received", submission.id, merchant_id=merchant_id)
    session.commit()

    return await run_pipeline(session, submission, policy=get_policy(merchant_id))


@router.post("/batch-verify", response_model=BatchVerifyResponse, summary="Retroactively re-score stored submissions")
async def batch_verify(
    req: BatchVerifyRequest,
    session: Session = Depends(db_session),
    _: str = Depends(require_api_key),
) -> BatchVerifyResponse:
    """Sweep history with the current model.

    The use case: a new attack family is discovered on Tuesday, and the platform
    needs to know which of the last 90 days of *approvals* would fail today.
    """
    t0 = time.perf_counter()
    q = select(Submission)
    if req.submission_ids:
        q = q.where(Submission.id.in_(req.submission_ids))
    else:
        if req.merchant_id:
            q = q.where(Submission.merchant_id == req.merchant_id)
        if req.since:
            q = q.where(Submission.created_at >= req.since)
    subs = session.execute(q.order_by(desc(Submission.created_at)).limit(req.limit)).scalars().all()

    items: List[BatchVerifyItem] = []
    flagged = 0
    model_kind = "heuristic"
    for sub in subs:
        prev = session.execute(select(Verdict).where(Verdict.submission_id == sub.id)).scalars().first()
        prev_verdict = prev.verdict if prev else None
        prev_score = prev.final_score if prev else None
        if req.only_previously_passed and prev_verdict is not None and prev_verdict != "PASS":
            continue

        resp = await run_pipeline(session, sub, persist=True, fire_webhook=False)
        model_kind = resp.fusion_model
        newly = resp.verdict != "PASS" and (prev_verdict is None or prev_verdict == "PASS")
        flagged += int(newly)
        items.append(
            BatchVerifyItem(
                submission_id=sub.id,
                previous_verdict=prev_verdict,
                previous_score=round(prev_score, 4) if prev_score is not None else None,
                new_verdict=resp.verdict,
                new_score=resp.final_score,
                newly_flagged=newly,
                top_reasons=resp.top_reasons,
            )
        )
    log_event(session, "batch_sweep", None, scanned=len(items), newly_flagged=flagged)
    session.commit()
    return BatchVerifyResponse(
        scanned=len(items), newly_flagged=flagged, items=items,
        fusion_model=model_kind, elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@router.get("/submissions", summary="Browse the audit log")
def list_submissions(
    limit: int = 50,
    offset: int = 0,
    merchant_id: Optional[str] = None,
    verdict: Optional[str] = None,
    attack_pattern: Optional[str] = None,
    source: Optional[str] = None,
    session: Session = Depends(db_session),
):
    q = select(Submission, Verdict).join(Verdict, Verdict.submission_id == Submission.id, isouter=True)
    if merchant_id:
        q = q.where(Submission.merchant_id == merchant_id)
    if verdict:
        q = q.where(Verdict.verdict == verdict)
    if attack_pattern:
        q = q.where(Verdict.attack_pattern == attack_pattern)
    if source:
        q = q.where(Submission.source == source)
    rows = session.execute(q.order_by(desc(Submission.created_at)).limit(limit).offset(offset)).all()
    return {
        "count": len(rows),
        "items": [
            {
                "submission_id": s.id,
                "merchant_id": s.merchant_id,
                "created_at": s.created_at,
                "status": s.status,
                "source": s.source,
                "label": s.label,
                "attack_type": s.attack_type,
                "claimed_name": (s.extra or {}).get("claimed_name"),
                "verdict": v.verdict if v else None,
                "score": round(v.final_score, 4) if v else None,
                "attack_pattern": v.attack_pattern if v else None,
                "generator_guess": v.generator_guess if v else None,
                "top_reasons": v.reasons if v else [],
                "latency_ms": round(v.latency_ms, 1) if v else None,
                "heatmaps": v.heatmaps if v else {},
            }
            for s, v in rows
        ],
    }


@router.get("/submissions/{submission_id}", summary="Full record for one submission")
def get_submission(submission_id: str, session: Session = Depends(db_session)):
    sub = session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(404, "submission not found")
    v = session.execute(select(Verdict).where(Verdict.submission_id == submission_id)).scalars().first()
    from ..utils.artifacts import upload_url

    return {
        "submission_id": sub.id,
        "merchant_id": sub.merchant_id,
        "created_at": sub.created_at,
        "status": sub.status,
        "source": sub.source,
        "label": sub.label,
        "attack_type": sub.attack_type,
        "extra": sub.extra,
        "assets": {
            "selfie": upload_url(sub.selfie_path) if sub.selfie_path else None,
            "id_document": upload_url(sub.id_doc_path) if sub.id_doc_path else None,
            "liveness_video": upload_url(sub.video_path) if sub.video_path else None,
        },
        "verdict": None if v is None else {
            "verdict": v.verdict,
            "final_score": round(v.final_score, 4),
            "abstained": v.abstained,
            "top_reasons": v.reasons,
            "explanation": v.explanation,
            "attack_pattern": v.attack_pattern,
            "generator_guess": v.generator_guess,
            "detector_breakdown": v.detector_breakdown,
            "heatmaps": v.heatmaps,
            "latency_ms": round(v.latency_ms, 1),
            "fusion_model": v.fusion_model,
            "policy_snapshot": v.policy_snapshot,
            "created_at": v.created_at,
        },
    }


@router.post("/submissions/{submission_id}/rescore", response_model=VerifyResponse)
async def rescore(submission_id: str, session: Session = Depends(db_session), _: str = Depends(require_api_key)):
    sub = session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(404, "submission not found")
    return await run_pipeline(session, sub, fire_webhook=False)
