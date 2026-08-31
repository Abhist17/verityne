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

import functools
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import MODEL_ROOT
from .db import AssetHash, FaceEmbedding, Submission, Verdict
from .utils.hashing import hamming

#: Above this cosine similarity two selfies are the same person.
#:
#: **This is a search threshold, not a pair threshold, and the difference is the
#: whole story.** ``find_face_links`` compares one applicant against every prior
#: record, up to ``SCAN_LIMIT``. A pairwise false-accept rate of ``p`` applied
#: ``N`` times gives a per-applicant false-link probability of ``1-(1-p)^N``.
#:
#: This constant used to be 0.5198, taken from the FAR=0.1% operating point of
#: LFW's official 10-fold protocol — a *verification* fit, graded on 6,000 pairs.
#: Deployed as a search it meant a genuine applicant false-linking to a stranger
#: with probability 6.8% against 100 prior records and 97.0% against 5,000. That
#: 0.1% was itself two impostor pairs out of the official 3,000, the resolution
#: floor of that set.
#:
#: It is now fitted by ``scripts/calibrate_linkage_lfw.py``, which scores every
#: pair among LFW's distinct identities — millions of impostor pairs rather than
#: 3,000 — and picks the point holding a stated per-applicant false-link rate
#: over ``SCAN_LIMIT`` records. Report: ``eval/linkage_lfw.json``.
#:
#: Before either fit it was 0.75, hard-coded, which linked only 63.3% of true
#: same-person pairs. The corpus could not have revealed that — its genuine pairs
#: derive from one source photograph each, so their similarity runs far above
#: what two real photographs of one person score (LFW same-person mean 0.758).
#: The fitted value, duplicated here so a checkout with no calibration file still
#: behaves. Chosen for a 1% per-applicant false-link rate over SCAN_LIMIT records.
DEFAULT_SAME_PERSON = 0.8169

#: Written by ``calibrate_linkage_lfw.py --apply``; absent until it has been run.
THRESHOLD_PATH = MODEL_ROOT / "linkage_threshold.json"


@functools.lru_cache(maxsize=1)
def same_person_threshold() -> Tuple[float, str]:
    """(threshold, provenance). Falls back to the documented default."""
    if THRESHOLD_PATH.exists():
        try:
            data = json.loads(THRESHOLD_PATH.read_text())
            return float(data["same_person"]), data.get("fitted_on", "calibration file")
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_SAME_PERSON, "uncalibrated default"


def reload_threshold() -> None:
    same_person_threshold.cache_clear()


#: Module-level convenience for callers that want the value without the
#: provenance (tests, the calibration scripts grading what ships). The search
#: path in `find_face_links` reads `same_person_threshold()` live instead, so a
#: re-fit does not need a restart.
SAME_PERSON = same_person_threshold()[0]
#: Perceptual-hash distance at or below which two files *look* alike. This is a
#: similarity bound, not an identity one: ID cards drawn from a shared template
#: land inside it while belonging to different people, so a hit here is a hint,
#: never proof. Exactness is what ``content_hash`` is for.
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

    # Read live rather than through the module constant, so that applying a new
    # calibration (or POST /admin/reload-policy) takes effect without a restart.
    threshold = same_person_threshold()[0]
    matches: List[Dict] = []
    for idx in np.argsort(-sims)[:25]:
        sim = float(sims[idx])
        if sim < threshold:
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


def find_asset_links(session: Session, submission_id: str, hashes: Dict[str, Dict[str, str]]) -> List[Dict]:
    """Prior submissions sharing an asset with this one.

    Every match is labelled ``exact`` or ``near``. An exact match means the
    decoded pixels are byte-identical - the same picture, whatever the file
    around it was doing. A near match only means two images look alike to a
    64-bit perceptual hash, which templated documents do by construction.
    Callers must not treat the two as interchangeable.
    """
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
    for kind, digests in hashes.items():
        ph, ch = digests.get("phash"), digests.get("content_hash")
        for row in rows:
            if row.kind != kind:
                continue
            if ch and row.content_hash and row.content_hash == ch:
                out.append({"submission_id": row.submission_id, "kind": kind, "hamming": 0, "match": "exact"})
                continue
            if not ph:
                continue
            dist = hamming(ph, row.phash)
            if dist <= SAME_IMAGE_BITS:
                out.append({"submission_id": row.submission_id, "kind": kind, "hamming": dist, "match": "near"})
    # Keep the strongest match per prior submission: exact beats near, then closest.
    best: Dict[str, Dict] = {}
    for m in out:
        prev = best.get(m["submission_id"])
        if prev is None or (m["match"] == "exact" and prev["match"] == "near") or (
            m["match"] == prev["match"] and m["hamming"] < prev["hamming"]
        ):
            best[m["submission_id"]] = m
    return sorted(best.values(), key=lambda m: (m["match"] != "exact", m["hamming"]))[:15]


def linkage_signal(
    face_links: List[Dict], asset_links: List[Dict]
) -> Tuple[float, List[str], bool]:
    """Turn linkage hits into a risk contribution, reasons, and whether it is exact.

    The third return value is what separates a *measured* claim from a
    *statistical* one, and the caller needs it to know how far to trust the score.

    A face match is a similarity threshold applied against every prior record, so
    it carries a false-accept rate that compounds with database size - see
    ``SAME_PERSON``. A byte-identical file does not: two submissions either share
    a SHA-256 or they do not, and the only false positives are collisions.

    So ``exact`` is True only when a byte-identical asset was found. Everything
    else - face similarity, perceptual-hash near-matches - is a probabilistic
    hit that must not carry a rejection on its own.
    """
    reasons: List[str] = []
    score = 0.0
    exact_match = False

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

    exact = [m for m in asset_links if m.get("match") == "exact"]
    near = [m for m in asset_links if m.get("match") != "exact"]
    if exact:
        # Byte-identical pixels across identities is the KYC-kit signature, and
        # the one asset claim strong enough to carry a rejection on its own.
        exact_match = True
        score = max(score, min(0.9, 0.6 + 0.1 * len(exact)))
        reasons.append(
            f"The exact same image file was used in {len(exact)} earlier submission(s) - "
            "consistent with a purchased, pre-made KYC kit"
        )
    elif near:
        # Looks alike, is not provably the same. Worth an analyst's attention;
        # not worth overruling five detectors that all read the packet as clean.
        score = max(score, 0.35)
        reasons.append(
            f"A visually near-identical image appeared in {len(near)} earlier submission(s) - "
            "could be a reused asset, or two documents sharing a template"
        )
    return score, reasons, exact_match


def _normalise(name: str) -> str:
    return " ".join(name.lower().split())
