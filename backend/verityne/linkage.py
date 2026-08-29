"""Cross-submission linkage: one face wearing many names.

Fraud rings do not onboard once. They buy a kit and reuse the face across
merchants under different identities. Because Detector 4 already computes a
512-d embedding for every selfie, catching this costs one matrix multiply
against the audit log.

Two lookups:
  * face linkage  - the same person under a different claimed name / merchant;
  * asset linkage - the literal same image file, via perceptual hash.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import AssetHash, FaceEmbedding, Submission, Verdict
from .utils.hashing import hamming

#: Above this cosine similarity two selfies are the same person.
SAME_PERSON = 0.75
#: Perceptual-hash distance at or below which two files are the same picture.
SAME_IMAGE_BITS = 8
#: Cap on how much history we scan, so linkage stays inside the latency budget.
SCAN_LIMIT = 5000


def _fetch_embeddings(session: Session, exclude_submission: str, kind: str = "selfie") -> Tuple[np.ndarray, List[FaceEmbedding]]:
    rows = (
        session.execute(
            select(FaceEmbedding)
            .where(FaceEmbedding.kind == kind, FaceEmbedding.submission_id != exclude_submission)
            .order_by(FaceEmbedding.created_at.desc())
            .limit(SCAN_LIMIT)
        )
        .scalars()
        .all()
    )
    rows = [r for r in rows if r.vector]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    mat = np.asarray([r.vector for r in rows], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-9), rows


def find_face_links(
    session: Session, submission_id: str, vector: List[float], claimed_name: Optional[str] = None
) -> List[Dict]:
    """Prior submissions whose selfie is the same person as this one."""
    if not vector:
        return []
    mat, rows = _fetch_embeddings(session, submission_id)
    if mat.size == 0:
        return []
    q = np.asarray(vector, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    if q.shape[0] != mat.shape[1]:
        return []
    sims = mat @ q

    matches: List[Dict] = []
    for idx in np.argsort(-sims)[:25]:
        sim = float(sims[idx])
        if sim < SAME_PERSON:
            break
        row = rows[int(idx)]
        sub = session.get(Submission, row.submission_id)
        verdict = session.execute(
            select(Verdict).where(Verdict.submission_id == row.submission_id)
        ).scalars().first()
        name_differs = bool(claimed_name and row.claimed_name and _normalise(claimed_name) != _normalise(row.claimed_name))
        matches.append(
            {
                "submission_id": row.submission_id,
                "merchant_id": row.merchant_id,
                "similarity": round(sim, 4),
                "claimed_name": row.claimed_name,
                "name_differs": name_differs,
                "created_at": sub.created_at.isoformat() if sub and sub.created_at else None,
                "verdict": verdict.verdict if verdict else None,
            }
        )
    return matches


def find_asset_links(session: Session, submission_id: str, hashes: Dict[str, str]) -> List[Dict]:
    """Prior submissions that reused the exact same image file."""
    if not hashes:
        return []
    rows = (
        session.execute(
            select(AssetHash)
            .where(AssetHash.submission_id != submission_id)
            .order_by(AssetHash.created_at.desc())
            .limit(SCAN_LIMIT)
        )
        .scalars()
        .all()
    )
    out: List[Dict] = []
    for kind, h in hashes.items():
        if not h:
            continue
        for row in rows:
            if row.kind != kind:
                continue
            dist = hamming(h, row.phash)
            if dist <= SAME_IMAGE_BITS:
                out.append({"submission_id": row.submission_id, "kind": kind, "hamming": dist})
    # Keep the closest match per prior submission.
    best: Dict[str, Dict] = {}
    for m in out:
        prev = best.get(m["submission_id"])
        if prev is None or m["hamming"] < prev["hamming"]:
            best[m["submission_id"]] = m
    return sorted(best.values(), key=lambda m: m["hamming"])[:15]


def linkage_signal(face_links: List[Dict], asset_links: List[Dict]) -> Tuple[float, List[str]]:
    """Turn linkage hits into a risk contribution and human-readable reasons."""
    reasons: List[str] = []
    score = 0.0

    distinct_names = {m["claimed_name"] for m in face_links if m.get("claimed_name")}
    conflicting = [m for m in face_links if m.get("name_differs")]
    if conflicting:
        merchants = {m["merchant_id"] for m in conflicting}
        score = max(score, min(0.95, 0.55 + 0.12 * len(conflicting)))
        reasons.append(
            f"This face has already been submitted under {len(distinct_names)} different name(s) across "
            f"{len(merchants)} merchant account(s) - strong indicator of an onboarding ring"
        )
    elif face_links:
        score = max(score, 0.30)
        reasons.append(f"This face matches {len(face_links)} earlier submission(s) under the same name (possible duplicate application)")

    if asset_links:
        score = max(score, min(0.9, 0.6 + 0.1 * len(asset_links)))
        reasons.append(
            f"The exact same image file was used in {len(asset_links)} earlier submission(s) - "
            "consistent with a purchased, pre-made KYC kit"
        )
    return score, reasons


def _normalise(name: str) -> str:
    return " ".join(name.lower().split())
