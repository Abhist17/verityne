"""Ops surfaces: attack-pattern gallery, human review queue, red-team button, policy."""
from __future__ import annotations

import datetime as dt
import random
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..config import DATASET_ROOT, get_policy, reload_policy
from ..db import AssetHash, AuditEvent, Submission, Verdict, log_event
from ..explain import PATTERN_LABELS
from ..pipeline import run_pipeline, storage_dir
from ..schemas import VerifyResponse
from ..utils.artifacts import save_thumbnail, upload_url
from .deps import db_session, require_api_key

router = APIRouter()


@router.get("/attacks", summary="Rejected submissions grouped by attack pattern")
def attack_gallery(hours: int = 24, limit_per_group: int = 8, session: Session = Depends(db_session)):
    """What is attacking the platform right now, grouped so waves are visible."""
    since = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=hours)
    rows = session.execute(
        select(Submission, Verdict)
        .join(Verdict, Verdict.submission_id == Submission.id)
        .where(Verdict.created_at >= since, Verdict.verdict != "PASS")
        .order_by(desc(Verdict.created_at))
    ).all()

    groups: Dict[str, dict] = {}
    for s, v in rows:
        key = v.attack_pattern or "unclassified"
        g = groups.setdefault(
            key,
            {"pattern": key, "label": PATTERN_LABELS.get(key, key.replace("_", " ").title()),
             "count": 0, "mean_score": 0.0, "generators": {}, "merchants": set(), "samples": []},
        )
        g["count"] += 1
        g["mean_score"] += v.final_score
        g["merchants"].add(s.merchant_id)
        if v.generator_guess:
            g["generators"][v.generator_guess] = g["generators"].get(v.generator_guess, 0) + 1
        if len(g["samples"]) < limit_per_group:
            g["samples"].append(
                {
                    "submission_id": s.id,
                    "merchant_id": s.merchant_id,
                    "score": round(v.final_score, 4),
                    "verdict": v.verdict,
                    "created_at": v.created_at,
                    "top_reason": (v.reasons or [None])[0],
                    "thumb_url": (s.extra or {}).get("thumb_url") or (v.heatmaps or {}).get("selfie"),
                    "heatmaps": v.heatmaps,
                }
            )
    out = []
    for g in groups.values():
        g["mean_score"] = round(g["mean_score"] / g["count"], 4)
        g["merchants"] = sorted(g["merchants"])
        out.append(g)
    out.sort(key=lambda g: g["count"], reverse=True)
    return {"window_hours": hours, "total_flagged": sum(g["count"] for g in out), "groups": out}


@router.get("/review-queue", summary="Submissions awaiting a human decision")
def review_queue(limit: int = 50, session: Session = Depends(db_session)):
    """Everything the model refused to decide, with the evidence pre-surfaced."""
    decided = {
        e.submission_id
        for e in session.execute(select(AuditEvent).where(AuditEvent.event == "analyst_decision")).scalars().all()
    }
    rows = session.execute(
        select(Submission, Verdict)
        .join(Verdict, Verdict.submission_id == Submission.id)
        .where(Verdict.verdict == "REVIEW")
        .order_by(desc(Verdict.final_score))
        .limit(limit * 2)
    ).all()
    # Which submissions share a selfie file, and under how many names.
    #
    # Four of the seeded images are deliberately reused across two to five
    # different claimed identities - that IS the fraud, and it is what the
    # threat graph draws as a ring. But the queue listed them as unrelated
    # cards showing the same face, so the repetition read as broken demo data
    # rather than as the single strongest signal on the page. One pass over the
    # hashes, so the list can say what the graph already knew.
    ring: dict[str, dict] = {}
    by_hash: dict[str, list] = {}
    sid_to_hash: dict[str, str] = {}
    for h in session.execute(select(AssetHash).where(AssetHash.kind == "selfie")).scalars().all():
        by_hash.setdefault(h.content_hash, []).append(h.submission_id)
        sid_to_hash[h.submission_id] = h.content_hash
    names = {
        sub.id: (sub.extra or {}).get("claimed_name")
        for sub in session.execute(select(Submission)).scalars().all()
    }
    for sid, sha in sid_to_hash.items():
        siblings = [o for o in by_hash.get(sha, []) if o != sid]
        if not siblings:
            continue
        other_names = sorted({names.get(o) for o in siblings if names.get(o)})
        ring[sid] = {"shared_selfie_with": len(siblings), "other_names": other_names}

    items = []
    for s, v in rows:
        if s.id in decided:
            continue
        items.append(
            {
                "submission_id": s.id,
                "merchant_id": s.merchant_id,
                "claimed_name": (s.extra or {}).get("claimed_name"),
                "created_at": s.created_at,
                "score": round(v.final_score, 4),
                "abstained": v.abstained,
                "attack_pattern": v.attack_pattern,
                "pattern_label": PATTERN_LABELS.get(v.attack_pattern or "", None),
                "top_reasons": v.reasons,
                "explanation": v.explanation,
                "heatmaps": v.heatmaps,
                "assets": {
                    "selfie": upload_url(s.selfie_path) if s.selfie_path else None,
                    "id_document": upload_url(s.id_doc_path) if s.id_doc_path else None,
                },
                "detector_summary": {
                    k: {"score": round(d.get("score", 0), 3), "status": d.get("status"), "label": d.get("label")}
                    for k, d in (v.detector_breakdown or {}).items()
                },
                # None when this face appears once, which is most of the queue.
                "linkage": ring.get(s.id),
            }
        )
        if len(items) >= limit:
            break
    pending = len(items)
    return {"pending": pending, "items": items}


class AnalystDecision(BaseModel):
    decision: str  # approve | reject | escalate
    analyst: str = "ops"
    note: Optional[str] = None


@router.post("/review/{submission_id}/decision", summary="Record a human decision on a reviewed case")
def record_decision(
    submission_id: str,
    body: AnalystDecision,
    session: Session = Depends(db_session),
    _: str = Depends(require_api_key),
):
    if body.decision not in {"approve", "reject", "escalate"}:
        raise HTTPException(400, "decision must be approve, reject or escalate")
    sub = session.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(404, "submission not found")
    v = session.execute(select(Verdict).where(Verdict.submission_id == submission_id)).scalars().first()
    log_event(
        session, "analyst_decision", submission_id, actor=body.analyst,
        decision=body.decision, note=body.note,
        model_verdict=v.verdict if v else None, model_score=round(v.final_score, 4) if v else None,
    )
    sub.status = f"HUMAN_{body.decision.upper()}"
    session.commit()
    # Agreement between the model and the analyst is the label source for the next retrain.
    return {"ok": True, "submission_id": submission_id, "decision": body.decision,
            "model_verdict": v.verdict if v else None}


@router.get("/review/agreement", summary="How often analysts agree with the model")
def review_agreement(session: Session = Depends(db_session)):
    events = session.execute(select(AuditEvent).where(AuditEvent.event == "analyst_decision")).scalars().all()
    if not events:
        return {"decisions": 0}
    agree = sum(
        1 for e in events
        if (e.payload.get("decision") == "reject" and e.payload.get("model_score", 0) >= 0.5)
        or (e.payload.get("decision") == "approve" and e.payload.get("model_score", 1) < 0.5)
    )
    return {
        "decisions": len(events),
        "agreement_rate": round(agree / len(events), 4),
        "breakdown": {d: sum(1 for e in events if e.payload.get("decision") == d) for d in ("approve", "reject", "escalate")},
    }


class RedTeamRequest(BaseModel):
    merchant_id: str = "default"
    count: int = 1


@router.post("/redteam/generate", summary="Score a never-before-seen synthetic attack")
async def redteam(
    body: RedTeamRequest = Body(default=RedTeamRequest()),
    session: Session = Depends(db_session),
    _: str = Depends(require_api_key),
) -> List[VerifyResponse]:
    """Pull an unseen fake from the held-out red-team pool and score it live.

    The pool is generated by `scripts/redteam_generate.py` *after* the fusion
    model is trained, so nothing here was in the training or evaluation set.
    We deliberately do not run diffusion inside the request: a demo that depends
    on a 20-second GPU sample and venue wi-fi is a demo that fails on stage.
    """
    pool = DATASET_ROOT / "redteam"
    manifest = pool / "manifest.json"
    if not manifest.exists():
        raise HTTPException(409, "No red-team pool. Run `python backend/scripts/redteam_generate.py` first.")
    import json

    entries = json.loads(manifest.read_text())
    used = {
        e.external_ref for e in session.execute(
            select(Submission).where(Submission.source == "redteam")
        ).scalars().all()
    }
    unused = [e for e in entries if e["id"] not in used] or entries
    picked = random.sample(unused, k=min(body.count, len(unused)))

    responses: List[VerifyResponse] = []
    for entry in picked:
        sub = Submission(
            merchant_id=body.merchant_id, external_ref=entry["id"], source="redteam",
            label="fake", attack_type=entry.get("attack_type"),
            extra={"claimed_name": entry.get("claimed_name"), "display_name": entry.get("display_name"),
                   "generated_at": entry.get("generated_at"), "generator": entry.get("generator")},
        )
        session.add(sub)
        session.flush()
        d = storage_dir(sub.id)
        import shutil

        for key, attr in (("selfie", "selfie_path"), ("id_document", "id_doc_path"), ("video", "video_path")):
            src = entry.get(key)
            if src and Path(src).exists():
                dst = d / Path(src).name
                shutil.copy2(src, dst)
                setattr(sub, attr, str(dst))
        thumb = save_thumbnail(sub.selfie_path, sub.id, "selfie") if sub.selfie_path else None
        if thumb:
            sub.extra = {**(sub.extra or {}), "thumb_url": thumb}
        session.commit()
        responses.append(await run_pipeline(session, sub, fire_webhook=False))
    return responses


@router.get("/policy", summary="Effective policy for a merchant")
def policy(merchant_id: str = "default"):
    return get_policy(merchant_id).model_dump()


@router.post("/admin/reload-policy", summary="Hot-reload policy.yaml")
def do_reload_policy(_: str = Depends(require_api_key)):
    reload_policy()
    from .. import fusion

    fusion.reload_model()
    return {"ok": True, "policy": get_policy().model_dump()}


@router.get("/audit", summary="Raw audit event stream")
def audit(limit: int = 100, submission_id: Optional[str] = None, session: Session = Depends(db_session)):
    q = select(AuditEvent).order_by(desc(AuditEvent.created_at)).limit(limit)
    if submission_id:
        q = q.where(AuditEvent.submission_id == submission_id)
    rows = session.execute(q).scalars().all()
    return {
        "count": len(rows),
        "events": [
            {"id": e.id, "submission_id": e.submission_id, "actor": e.actor, "event": e.event,
             "payload": e.payload, "created_at": e.created_at}
            for e in rows
        ],
    }
