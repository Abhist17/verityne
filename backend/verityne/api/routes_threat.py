"""/threat/graph - the fraud-ring view the Threat Intelligence page draws.

The rest of the dashboard answers "is *this* submission fraudulent". This
answers a question no single submission can: which submissions are the same
operation. Nodes are submissions, edges are linkage claims already computed by
`linkage.py`, and a connected component is a ring.

**Two edge kinds, and the page draws them differently on purpose.** A shared
SHA-256 is a fact - two files are byte-identical or they are not. A face match is
a similarity search whose false-accept rate compounds with the size of the
database it runs against, which is why `linkage.SAME_PERSON` is fitted as a
search threshold rather than a verification one. A ring built only from face
edges is an inference and is labelled as one; a ring containing an exact asset
edge is evidence, and `has_exact_asset_reuse` is what says which is which.

The whole graph is computed per request from the audit log rather than
maintained incrementally. At demo and pilot volumes that is a few thousand
embeddings and one matrix multiply; at production volume this becomes a job and
a table, which is noted here rather than pretended away.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import AssetHash, FaceEmbedding, Submission, Verdict
from ..linkage import same_person_threshold
from ..utils.artifacts import HEATMAP_URL_PREFIX
from .deps import db_session, require_api_key

router = APIRouter()
log = logging.getLogger("verityne.threat")

#: Ceiling on submissions pulled into one graph. The pairwise comparison is
#: O(n^2) in embeddings; past this the page is unreadable long before the query
#: is slow, so the limit is about the render, not the database.
MAX_NODES = 400


def _components(node_ids: List[str], edges: List[Tuple[str, str]]) -> Dict[str, int]:
    """Connected components by union-find. Ring index per node, singletons excluded."""
    parent = {n: n for n in node_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups: Dict[str, List[str]] = {}
    for n in node_ids:
        groups.setdefault(find(n), []).append(n)

    ring_of: Dict[str, int] = {}
    idx = 0
    for members in groups.values():
        if len(members) < 2:
            continue  # an isolated submission is not a ring
        for m in members:
            ring_of[m] = idx
        idx += 1
    return ring_of


@router.get("/threat/graph", summary="Fraud rings: submissions linked by face or by file")
async def threat_graph(
    hours: int = Query(168, ge=1, le=8760),
    session: Session = Depends(db_session),
    _: str = Depends(require_api_key),
):
    cutoff = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=hours)
    threshold, provenance = same_person_threshold()

    rows: List[Tuple[Submission, Optional[Verdict]]] = (
        session.query(Submission, Verdict)
        .outerjoin(Verdict, Verdict.submission_id == Submission.id)
        .filter(Submission.created_at >= cutoff)
        .order_by(Submission.created_at.desc())
        .limit(MAX_NODES)
        .all()
    )
    if not rows:
        return {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "window_hours": hours,
            "threshold": {"same_person": round(threshold, 4), "fitted_on": provenance},
            "nodes": [], "edges": [], "rings": [],
        }

    by_id = {s.id: (s, v) for s, v in rows}
    ids = list(by_id)

    nodes = []
    for sid in ids:
        s, v = by_id[sid]
        nodes.append({
            "id": sid,
            "merchant_id": s.merchant_id,
            "claimed_name": (s.extra or {}).get("claimed_name"),
            "verdict": v.verdict if v else "REVIEW",
            "score": round(float(v.final_score), 4) if v else 0.0,
            "created_at": (s.created_at.isoformat() if s.created_at else None),
            "generator_guess": v.generator_guess if v else None,
            # Same resolution order the attack gallery uses: whatever the
            # pipeline stored, else the thumbnail path it writes by convention.
            "thumb_url": (
                (s.extra or {}).get("thumb_url")
                or ((v.heatmaps or {}).get("selfie") if v else None)
                or f"{HEATMAP_URL_PREFIX}/{sid}_selfie_thumb.jpg"
            ),
            "ring": None,
        })

    edges: List[dict] = []

    # ---- face edges: one matrix multiply over the window's selfie embeddings
    embs = (
        session.query(FaceEmbedding)
        .filter(FaceEmbedding.submission_id.in_(ids), FaceEmbedding.kind == "selfie")
        .all()
    )
    usable = [e for e in embs if e.vector]
    if len(usable) >= 2:
        M = np.asarray([e.vector for e in usable], dtype=np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = (M / norms) @ (M / norms).T
        names = [(e.submission_id, (e.claimed_name or "").strip().lower()) for e in usable]
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                sim = float(sims[i, j])
                if sim < threshold:
                    continue
                (sa, na), (sb, nb) = names[i], names[j]
                if sa == sb:
                    continue
                edges.append({
                    "source": sa, "target": sb, "kind": "face",
                    "weight": round(sim, 4),
                    # The claim that turns a match into a ring: the same face
                    # onboarding under two different names.
                    "name_differs": bool(na and nb and na != nb),
                })

    # ---- asset edges: exact SHA-256 is a fact; a shared perceptual hash is not
    hashes = session.query(AssetHash).filter(AssetHash.submission_id.in_(ids)).all()
    by_content: Dict[str, List[str]] = {}
    by_phash: Dict[str, List[str]] = {}
    for h in hashes:
        if h.content_hash:
            by_content.setdefault(h.content_hash, []).append(h.submission_id)
        if h.phash:
            by_phash.setdefault(h.phash, []).append(h.submission_id)

    def _pairs(groups: Dict[str, List[str]], kind: str, weight: float) -> None:
        for members in groups.values():
            uniq = sorted(set(members))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    edges.append({"source": uniq[i], "target": uniq[j], "kind": kind, "weight": weight})

    _pairs(by_content, "asset_exact", 1.0)
    # A perceptual near-match is drawn, but never upgraded to an exact claim: two
    # documents sharing a government template hash alike and are not a ring.
    near = {k: v for k, v in by_phash.items() if k not in {h.phash for h in hashes if h.content_hash and len(by_content.get(h.content_hash, [])) > 1}}
    _pairs(near, "asset_near", 0.9)

    # De-duplicate, keeping the strongest claim for any pair.
    rank = {"asset_exact": 3, "face": 2, "asset_near": 1}
    best: Dict[Tuple[str, str], dict] = {}
    for e in edges:
        key = tuple(sorted((e["source"], e["target"])))
        cur = best.get(key)
        if cur is None or rank[e["kind"]] > rank[cur["kind"]]:
            best[key] = e
    edges = list(best.values())

    ring_of = _components(ids, [(e["source"], e["target"]) for e in edges])
    for n in nodes:
        n["ring"] = ring_of.get(n["id"])

    rings: List[dict] = []
    for ring_id in sorted(set(ring_of.values())):
        members = [n for n in nodes if n["ring"] == ring_id]
        member_ids = {m["id"] for m in members}
        ring_edges = [e for e in edges if e["source"] in member_ids and e["target"] in member_ids]
        names = {(m["claimed_name"] or "").strip().lower() for m in members if m["claimed_name"]}
        rings.append({
            "id": ring_id,
            "size": len(members),
            "merchants": sorted({m["merchant_id"] for m in members}),
            "distinct_names": len(names),
            "max_score": round(max((m["score"] for m in members), default=0.0), 4),
            "has_exact_asset_reuse": any(e["kind"] == "asset_exact" for e in ring_edges),
        })
    rings.sort(key=lambda r: (-r["size"], -r["max_score"]))

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "window_hours": hours,
        "threshold": {"same_person": round(threshold, 4), "fitted_on": provenance},
        "nodes": nodes,
        "edges": edges,
        "rings": rings,
    }
