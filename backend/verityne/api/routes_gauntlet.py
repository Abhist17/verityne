"""The Gauntlet: run the pre-loaded real/fake fixture set and score ourselves live."""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_policy
from ..db import SessionLocal, Submission, Verdict
from ..pipeline import run_pipeline
from ..schemas import GauntletResult, GauntletSummary
from ..utils.artifacts import save_thumbnail
from .deps import db_session

router = APIRouter()


def _fixtures(session: Session) -> List[Submission]:
    return (
        session.execute(
            select(Submission).where(Submission.source == "gauntlet").order_by(Submission.created_at.asc())
        )
        .scalars()
        .all()
    )


@router.get("/gauntlet", summary="List the loaded gauntlet fixtures")
def gauntlet_manifest(session: Session = Depends(db_session)):
    subs = _fixtures(session)
    items = []
    for s in subs:
        v = session.execute(select(Verdict).where(Verdict.submission_id == s.id)).scalars().first()
        items.append(
            {
                "submission_id": s.id,
                "name": (s.extra or {}).get("display_name", s.external_ref or s.id[:8]),
                "truth": s.label,
                "attack_type": s.attack_type,
                "thumb_url": (s.extra or {}).get("thumb_url"),
                "last_verdict": v.verdict if v else None,
                "last_score": round(v.final_score, 4) if v else None,
            }
        )
    return {
        "count": len(items),
        "real": sum(1 for i in items if i["truth"] == "real"),
        "fake": sum(1 for i in items if i["truth"] == "fake"),
        "items": items,
    }


def _summarise(results: List[GauntletResult]) -> GauntletSummary:
    fakes = [r for r in results if r.truth == "fake"]
    reals = [r for r in results if r.truth == "real"]
    caught = sum(1 for r in fakes if r.verdict in ("REJECT", "REVIEW"))
    passed = sum(1 for r in reals if r.verdict == "PASS")
    # A genuine merchant sent to review and one auto-rejected are both friction,
    # but they are not the same failure: the first costs an analyst a few minutes,
    # the second loses the merchant outright. false_reject_rate counts them
    # together because both are packets that did not pass on their own; these two
    # say which kind actually happened, so the headline cannot read as "blocked"
    # when nothing was blocked.
    reviewed = sum(1 for r in reals if r.verdict == "REVIEW")
    rejected = sum(1 for r in reals if r.verdict == "REJECT")
    return GauntletSummary(
        total=len(results),
        fakes_caught=caught,
        fakes_total=len(fakes),
        reals_passed=passed,
        reals_reviewed=reviewed,
        reals_rejected=rejected,
        reals_total=len(reals),
        detection_rate=round(caught / len(fakes), 4) if fakes else 0.0,
        false_accept_rate=round((len(fakes) - caught) / len(fakes), 4) if fakes else 0.0,
        false_reject_rate=round((len(reals) - passed) / len(reals), 4) if reals else 0.0,
        accuracy=round(sum(1 for r in results if r.correct) / len(results), 4) if results else 0.0,
        mean_latency_ms=round(sum(r.latency_ms for r in results) / len(results), 1) if results else 0.0,
        results=results,
    )


async def _score_one(session: Session, sub: Submission) -> GauntletResult:
    resp = await run_pipeline(session, sub, policy=get_policy(sub.merchant_id), fire_webhook=False)
    # For a fake, either REJECT or REVIEW counts as caught - REVIEW means a human
    # sees it, which is the outcome the platform actually needs.
    correct = (resp.verdict in ("REJECT", "REVIEW")) if sub.label == "fake" else (resp.verdict == "PASS")
    return GauntletResult(
        submission_id=sub.id,
        name=(sub.extra or {}).get("display_name", sub.external_ref or sub.id[:8]),
        truth=sub.label or "real",
        attack_type=sub.attack_type,
        verdict=resp.verdict,
        score=resp.final_score,
        correct=correct,
        top_reasons=resp.top_reasons,
        latency_ms=resp.latency_ms,
        thumb_url=(sub.extra or {}).get("thumb_url"),
    )


@router.post("/gauntlet/run", response_model=GauntletSummary, summary="Run the whole gauntlet, return the scoreboard")
async def run_gauntlet(session: Session = Depends(db_session)):
    results = [await _score_one(session, s) for s in _fixtures(session)]
    return _summarise(results)


@router.get("/gauntlet/stream", summary="Run the gauntlet, streaming each result as it lands")
async def stream_gauntlet():
    """Server-sent events so the dashboard scoreboard fills in live."""

    async def gen() -> AsyncIterator[str]:
        session = SessionLocal()
        try:
            subs = _fixtures(session)
            yield f"event: start\ndata: {json.dumps({'total': len(subs)})}\n\n"
            results: List[GauntletResult] = []
            t0 = time.perf_counter()
            for i, sub in enumerate(subs):
                try:
                    r = await _score_one(session, sub)
                except Exception as exc:  # noqa: BLE001 - one bad fixture must not kill the run
                    yield f"event: error\ndata: {json.dumps({'index': i, 'error': str(exc)})}\n\n"
                    continue
                results.append(r)
                payload = {"index": i, "result": json.loads(r.model_dump_json()),
                           "running": json.loads(_summarise(results).model_dump_json(exclude={'results'}))}
                yield f"event: result\ndata: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)  # let the event loop flush to the client
            summary = _summarise(results)
            summary_json = json.loads(summary.model_dump_json())
            summary_json["wall_clock_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            yield f"event: done\ndata: {json.dumps(summary_json)}\n\n"
        finally:
            session.close()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
