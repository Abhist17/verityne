"""/behavioral - telemetry ingest for Detector 6.

Separate from `/verify` because of *when* it arrives. The applicant finishes
typing and hits submit; the files then take seconds to upload over a phone
connection. Making the browser hold its telemetry buffer until the multipart
upload is assembled would mean losing it whenever the upload fails, so the
buffer is posted first against a token the page minted on load, and `/verify`
binds the token to the submission afterwards.

The endpoint reduces on write. Raw keystroke timings are biometric data with no
retention story and nothing downstream reads them, so `extract_features` runs
here and only the feature row is stored.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from ..db import BehavioralSession, log_event
from ..detectors.behavioral import extract_features, score_features
from ..schemas import BehavioralAck, BehavioralEnvelope
from ..utils.jsonsafe import to_jsonable
from .deps import db_session, require_api_key

router = APIRouter()


@router.post("/behavioral", response_model=BehavioralAck, summary="Post a form-fill telemetry buffer")
async def ingest_behavioral(
    envelope: BehavioralEnvelope,
    user_agent: Optional[str] = Header(default=None),
    session: Session = Depends(db_session),
    _: str = Depends(require_api_key),
) -> BehavioralAck:
    events = envelope.model_dump()
    # The user agent the *browser* sent is worth more than the one the page
    # reported about itself, which a script can set to anything. Prefer it.
    env = dict(events.get("env") or {})
    if user_agent:
        env["user_agent"] = user_agent
    events["env"] = env

    features = to_jsonable(extract_features(events))
    score, confidence, reasons, _hits = score_features(features)

    # Re-posting a token replaces its row. A page that retries a failed submit
    # must not create a second, half-length session that looks more robotic than
    # the real one.
    row = session.query(BehavioralSession).filter(BehavioralSession.token == envelope.token).one_or_none()
    if row is None:
        row = BehavioralSession(token=envelope.token)
        session.add(row)
    row.merchant_id = envelope.merchant_id
    row.features = features
    row.event_count = int(features.get("event_count") or 0)
    row.user_agent = (user_agent or "")[:500] or None

    log_event(
        session, "behavioral_received", None,
        token=envelope.token, merchant_id=envelope.merchant_id,
        events=row.event_count, preview_score=round(score, 4),
    )
    session.commit()

    return BehavioralAck(
        token=envelope.token,
        event_count=row.event_count,
        features_extracted=len(features),
        preview_score=round(score, 4),
        preview_confidence=round(confidence, 4),
        preview_reasons=reasons[:3],
    )


@router.get("/behavioral/{token}", summary="Read back one stored telemetry session")
async def read_behavioral(
    token: str,
    session: Session = Depends(db_session),
    _: str = Depends(require_api_key),
):
    row = session.query(BehavioralSession).filter(BehavioralSession.token == token).one_or_none()
    if row is None:
        raise HTTPException(404, "No telemetry stored under that token")
    score, confidence, reasons, hits = score_features(row.features or {})
    return {
        "token": row.token,
        "merchant_id": row.merchant_id,
        "submission_id": row.submission_id,
        "event_count": row.event_count,
        "features": row.features,
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "reasons": reasons,
        "rule_hits": hits,
        "created_at": row.created_at,
    }
