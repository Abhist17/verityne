"""The orchestrator: files in, verdict out.

Concurrency note. The detectors are CPU/GPU-bound, so `asyncio.gather` over
coroutines would buy nothing - the event loop would run them one at a time.
They are dispatched to a ThreadPoolExecutor instead, which *is* real parallelism
here because torch, OpenCV and Pillow all release the GIL around their heavy
kernels. Threads also let all five detectors share one decoded copy of each
image through `payload.cache`, which a process pool could not do without paying
to re-decode and re-load models per worker.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from . import fusion, linkage
from .config import MerchantPolicy, UPLOAD_DIR, get_policy
from .db import AssetHash, FaceEmbedding, Submission, Verdict, log_event
from .detectors import STAGE_ONE, STAGE_TWO, SubmissionPayload
from .detectors.fingerprint import describe as describe_generator
from .detectors.fingerprint import identify as identify_generator
from .explain import classify_attack, narrate, rank_reasons
from .schemas import DetectorOutput, VerifyResponse
from .utils.artifacts import save_thumbnail
from .utils.hashing import phash
from .utils.jsonsafe import to_jsonable

log = logging.getLogger("verityne.pipeline")

#: One shared pool. Sized for the detector fan-out, not the core count - the
#: heavy work is inside torch, which does its own intra-op threading.
_EXECUTOR = ThreadPoolExecutor(max_workers=6, thread_name_prefix="detector")


def storage_dir(submission_id: str) -> Path:
    d = UPLOAD_DIR / submission_id
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _run_stage(detectors, payload: SubmissionPayload) -> Dict[str, DetectorOutput]:
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(_EXECUTOR, d.run, payload) for d in detectors]
    results = await asyncio.gather(*tasks)
    return {r.name: r for r in results}


def _persist_biometrics(session: Session, submission_id: str, merchant_id: str,
                        payload: SubmissionPayload) -> Dict[str, str]:
    """Store embeddings and perceptual hashes so future submissions can be linked back.

    Idempotent: re-scoring a submission replaces its biometrics rather than adding
    a second copy, so repeated gauntlet runs do not inflate the linkage index.
    """
    session.query(FaceEmbedding).filter(FaceEmbedding.submission_id == submission_id).delete()
    session.query(AssetHash).filter(AssetHash.submission_id == submission_id).delete()
    session.flush()

    hashes: Dict[str, str] = {}
    for kind, cache_key in (("selfie", "selfie_face"), ("id_photo", "id_face")):
        vec_key = "selfie_embedding" if kind == "selfie" else "id_embedding"
        vec = payload.cache.get(vec_key)
        if vec:
            session.add(
                FaceEmbedding(
                    submission_id=submission_id, merchant_id=merchant_id, kind=kind,
                    claimed_name=payload.claimed_name, vector=list(vec),
                )
            )
    for kind, cache_key in (("selfie", "selfie_rgb"), ("id_document", "id_rgb")):
        rgb = payload.cache.get(cache_key)
        if rgb is not None:
            try:
                h = phash(rgb)
                hashes[kind] = h
                session.add(AssetHash(submission_id=submission_id, kind=kind, phash=h))
            except Exception:
                pass
    return hashes


async def run_pipeline(
    session: Session,
    submission: Submission,
    policy: Optional[MerchantPolicy] = None,
    persist: bool = True,
    fire_webhook: bool = True,
) -> VerifyResponse:
    """Score one submission end to end."""
    t0 = time.perf_counter()
    policy = policy or get_policy(submission.merchant_id)

    payload = SubmissionPayload(
        submission_id=submission.id,
        merchant_id=submission.merchant_id,
        selfie_path=Path(submission.selfie_path) if submission.selfie_path else None,
        video_path=Path(submission.video_path) if submission.video_path else None,
        id_doc_path=Path(submission.id_doc_path) if submission.id_doc_path else None,
        claimed_name=(submission.extra or {}).get("claimed_name"),
        claimed_id_number=(submission.extra or {}).get("claimed_id_number"),
        claimed_dob=(submission.extra or {}).get("claimed_dob"),
        submitted_at=submission.created_at.timestamp() if submission.created_at else time.time(),
        extra=dict(submission.extra or {}),
    )

    breakdown = await _run_stage(STAGE_ONE, payload)
    breakdown.update(await _run_stage(STAGE_TWO, payload))

    # ---- linkage + fingerprinting: threat intel, computed from what the detectors cached
    hashes = _persist_biometrics(session, submission.id, submission.merchant_id, payload) if persist else {}
    link_info: Dict[str, list] = {"face_links": [], "asset_links": []}
    try:
        link_info["face_links"] = linkage.find_face_links(
            session, submission.id, payload.cache.get("selfie_embedding") or [], payload.claimed_name
        )
        link_info["asset_links"] = linkage.find_asset_links(session, submission.id, hashes)
    except Exception as exc:  # noqa: BLE001
        log.warning("linkage failed: %s", exc)
    link_score, link_reasons = linkage.linkage_signal(link_info["face_links"], link_info["asset_links"])

    gen_key, gen_conf, gen_probs = (None, 0.0, {})
    face = payload.cache.get("selfie_face")
    if face is not None:
        gen_key, gen_conf, gen_probs = identify_generator(face)
    generator_guess = describe_generator(gen_key) if gen_key and gen_key != "real" and gen_conf >= 0.5 else None

    # ---- fuse ------------------------------------------------------------------
    base_score, model_kind = fusion.fuse(breakdown)
    # Linkage is a hard, non-statistical fact about history; it can only raise risk.
    final_score = max(base_score, link_score) if link_score > 0 else base_score
    verdict_label, abstained = fusion.decide(final_score, policy, breakdown)

    attack_pattern = classify_attack(breakdown, link_info)
    top_reasons = rank_reasons(breakdown, extra=[(link_score * 1.2, r) for r in link_reasons])
    explanation = narrate(
        verdict_label, final_score, breakdown, attack_pattern,
        generator=generator_guess, linkage=link_info, abstained=abstained,
    )

    heatmaps = {n: d.heatmap_url for n, d in breakdown.items() if d.heatmap_url}
    latency_ms = (time.perf_counter() - t0) * 1000

    if persist:
        submission.status = "COMPLETE"
        existing = session.query(Verdict).filter(Verdict.submission_id == submission.id).one_or_none()
        if existing is not None:
            session.delete(existing)
            session.flush()
        session.add(
            Verdict(
                submission_id=submission.id,
                final_score=final_score,
                verdict=verdict_label,
                abstained=abstained,
                reasons=top_reasons,
                detector_breakdown=to_jsonable({n: d.model_dump() for n, d in breakdown.items()}),
                heatmaps=heatmaps,
                explanation=explanation,
                attack_pattern=attack_pattern,
                generator_guess=generator_guess,
                latency_ms=latency_ms,
                fusion_model=model_kind,
                policy_snapshot=to_jsonable(policy.model_dump()),
            )
        )
        log_event(
            session, "verdict", submission.id,
            verdict=verdict_label, score=round(final_score, 4),
            attack_pattern=attack_pattern, latency_ms=round(latency_ms, 1),
            linkage={"face": len(link_info["face_links"]), "asset": len(link_info["asset_links"])},
        )
        session.commit()

    response = VerifyResponse(
        submission_id=submission.id,
        merchant_id=submission.merchant_id,
        verdict=verdict_label,
        final_score=round(final_score, 4),
        abstained=abstained,
        top_reasons=top_reasons,
        explanation=explanation,
        attack_pattern=attack_pattern,
        generator_guess=generator_guess,
        detector_breakdown=breakdown,
        heatmaps=heatmaps,
        latency_ms=round(latency_ms, 2),
        fusion_model=model_kind,
        policy={
            "merchant_id": policy.merchant_id,
            "min_risk_for_review": policy.min_risk_for_review,
            "min_risk_for_reject": policy.min_risk_for_reject,
            "abstain_band": policy.abstain_band,
        },
        created_at=submission.created_at,
    )
    response.detector_breakdown = breakdown
    # Attach linkage + fingerprint detail without widening the top-level schema.
    response.policy["linkage"] = {
        "face_links": link_info["face_links"][:5],
        "asset_links": link_info["asset_links"][:5],
        "linkage_score": round(link_score, 4),
    }
    response.policy["generator_probabilities"] = gen_probs

    if fire_webhook and verdict_label == "REJECT" and final_score >= policy.webhook_min_score:
        asyncio.create_task(_notify(response))
    return response


async def _notify(resp: VerifyResponse) -> None:
    """Best-effort Slack ping. Never allowed to affect the verdict or the response."""
    from .config import SLACK_WEBHOOK_URL

    if not SLACK_WEBHOOK_URL:
        return
    try:
        import httpx

        reasons = "\n".join(f"• {r}" for r in resp.top_reasons)
        payload = {
            "text": f"*Verityne · high-confidence fraud blocked*",
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": "🚨 KYC submission rejected"}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*Merchant*\n{resp.merchant_id}"},
                    {"type": "mrkdwn", "text": f"*Risk score*\n{resp.final_score:.2f}"},
                    {"type": "mrkdwn", "text": f"*Attack pattern*\n{resp.attack_pattern}"},
                    {"type": "mrkdwn", "text": f"*Submission*\n`{resp.submission_id}`"},
                ]},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Why*\n{reasons}"}},
            ],
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(SLACK_WEBHOOK_URL, json=payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("slack webhook failed: %s", exc)


def make_thumbnails(submission: Submission) -> Dict[str, str]:
    out = {}
    for tag, path in (("selfie", submission.selfie_path), ("id_document", submission.id_doc_path)):
        if path:
            url = save_thumbnail(path, submission.id, tag)
            if url:
                out[tag] = url
    return out
