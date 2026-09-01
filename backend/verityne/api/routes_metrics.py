"""Metrics: the honest-numbers surface.

Everything served here is either (a) read straight from the held-out evaluation
report written by `scripts/evaluate.py`, or (b) computed live from the audit log.
Nothing is hard-coded, and the cost model's assumptions are exposed as inputs
rather than baked into a headline number.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import EVAL_ROOT, get_policy
from ..db import Submission, Verdict
from ..fusion import detector_weights
from .deps import db_session

router = APIRouter()

REPORT_PATH = EVAL_ROOT / "metrics.json"
ABLATION_PATH = EVAL_ROOT / "ablation.json"

#: Reports measured on data this project did not generate. Each is optional:
#: the face-match one needs LFW, the document one needs MIDV-2020, and the video
#: one needs a dataset that is gated behind a signed request form. The endpoint
#: reports which are present rather than pretending a missing one is a zero.
REAL_DATA_REPORTS = {
    "face_match": (EVAL_ROOT / "face_match_lfw.json",
                   "LFW — 6,000 pairs of real photographs of real people"),
    "id_documents": (EVAL_ROOT / "real_docs.json",
                     "MIDV-2020 — identity documents physically printed, photographed and scanned"),
    "behavioral": (EVAL_ROOT / "behavioral.json",
                   "Aalto 136M keystrokes + CMU Killourhy-Maxion, against real browser automation"),
    "selfie_faces": (EVAL_ROOT / "real_faces.json",
                     "DeepFakeFace (SD text2img, SD inpainting, InsightFace swap) and "
                     "140k Real-and-Fake (StyleGAN) — fakes this project did not generate"),
    "liveness_video": (EVAL_ROOT / "real_video.json",
                       "recorded deepfake video (FaceForensics++ / Celeb-DF / DFDC)"),
}


def _load_json(path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_report() -> Optional[dict]:
    return _load_json(REPORT_PATH)


@router.get("/metrics", summary="Held-out evaluation report + live operational stats")
def metrics(session: Session = Depends(db_session)):
    report = _load_report()
    live = _live_stats(session)
    return {
        "evaluation": report,
        "evaluation_available": report is not None,
        "fusion_weights": detector_weights(),
        "live": live,
        "note": (
            "Evaluation numbers come from a held-out split that the fusion model never saw during "
            "training. Live stats are computed from the audit log of this instance."
        ),
    }


@router.get("/metrics/corrections", summary="Every belief this project measured and lost")
def corrections():
    """The audit trail, generated from the evidence files each entry cites.

    Served as its own endpoint rather than folded into `/metrics` because it is
    not a metric: it is the record of what the metrics used to say and why they
    were wrong. `scripts/build_corrections.py` resolves every number out of the
    report it names, so nothing here can claim a figure no evidence file holds.
    """
    report = _load_json(EVAL_ROOT / "corrections.json")
    if report is None:
        raise HTTPException(
            404,
            "No corrections file on disk. Build it with "
            "`python backend/scripts/build_corrections.py`.",
        )
    return report


@router.get("/metrics/behavioral", summary="Detector 6: what the keystroke model was measured at")
def behavioral_metrics():
    """The Detector 6 evaluation report, or an explicit absence.

    404 rather than an empty shape: this file only exists once someone has built
    the corpus and run the fit, and a page that silently rendered zeros for it
    would be claiming a measurement nobody made.
    """
    report = _load_json(EVAL_ROOT / "behavioral.json")
    if report is None:
        raise HTTPException(
            404,
            "No behavioral evaluation on disk. Build the corpus "
            "(`make behavioral-corpus`) and fit the model (`make behavioral-train`).",
        )
    return report


@router.get("/metrics/ablation", summary="What each detector is worth, and whether the corpus leaks its labels")
def metrics_ablation():
    """The headline AUC, taken apart.

    `/metrics` reports one fused number. It cannot tell you whether five
    detectors earned it or one did, and the difference decides whether the
    system degrades gracefully when a signal goes missing or falls over. This
    serves `eval/ablation.json`: fusion refit with each detector muted in turn,
    scored on the same held-out split.

    It also carries the corpus leak audit, because an ablation is only
    meaningful if the features are measuring the packet rather than reading a
    label the generator wrote into it. That check is the reason this endpoint
    exists — it found `metadata_exif` reading an answer key worth half the
    model's above-chance AUC.
    """
    data = _load_json(ABLATION_PATH)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"{ABLATION_PATH.name} not found — run `make ablate` (or `make pipeline`)",
        )
    one_sided = [
        {"field": field, **entry}
        for field, audit in (data.get("corpus_leak_audit") or {}).items()
        for entry in audit.get("one_sided_values", [])
    ]
    # A value one-sided for a stated physical reason is reported, not counted
    # against the corpus. Everything else is a label written into the artefact.
    # Reports predating the distinction carry no `expected` key; treat those as
    # defects rather than silently clearing them.
    leaks = [l for l in one_sided if not l.get("expected")]
    expected = [l for l in one_sided if l.get("expected")]
    ranked = sorted(
        ((name, d) for name, d in (data.get("per_detector") or {}).items()),
        key=lambda kv: -kv[1].get("auc_drop", 0),
    )
    return {
        **data,
        "ranked": [{"detector": n, **d} for n, d in ranked],
        "leaks": leaks,
        "one_sided_but_expected": expected,
        "corpus_clean": not leaks,
        "note": (
            "auc_drop is how much held-out AUC the fused model loses when that detector is "
            "muted. A negative drop means the detector was making the model worse. "
            "leaks lists corpus values that appear on one class only without a physical "
            "reason to — where any is present, every number downstream of it is inflated by "
            "an unknown amount. one_sided_but_expected lists the ones that are one-sided "
            "because reality is, each with its justification."
        ),
    }


@router.get("/metrics/real", summary="Detector results measured on real, third-party data")
def metrics_real(full: bool = Query(False, description="include the per-item rows, not just the summary")):
    """The counterpart to `/metrics`, measured on data this project did not generate.

    `/metrics` reports the held-out split of a corpus we built: real faces, but
    documents we rendered and liveness clips we animated from stills. That
    measures separability, and it is a lower bound on the work of deploying
    this, not a substitute for it.

    This endpoint reports the same detectors on data from elsewhere — real
    photographs of real people, and identity documents that somebody else
    printed, photographed and scanned. Each report is independent and any of
    them may be absent, so `available` says which actually ran rather than
    letting a missing file read as a result.
    """
    out: Dict[str, object] = {"available": [], "missing": [], "reports": {}}
    for name, (path, source) in REAL_DATA_REPORTS.items():
        data = _load_json(path)
        if data is None:
            out["missing"].append({"name": name, "source": source, "expected_at": path.name})
            continue
        if not full:
            data = {k: v for k, v in data.items() if k != "rows"}
        out["available"].append(name)
        out["reports"][name] = {"source": source, **data}
    out["note"] = (
        "Absence here means the dataset was not on disk when the evaluation last ran — "
        "not that the detector scored zero. See the README section 'Measured on real data'."
    )
    return out


def _live_stats(session: Session) -> Dict:
    total = session.scalar(select(func.count()).select_from(Verdict)) or 0
    if total == 0:
        return {"total_verdicts": 0}

    by_verdict = dict(session.execute(select(Verdict.verdict, func.count()).group_by(Verdict.verdict)).all())
    by_pattern = dict(
        session.execute(
            select(Verdict.attack_pattern, func.count()).group_by(Verdict.attack_pattern).order_by(func.count().desc())
        ).all()
    )
    lat = [r for (r,) in session.execute(select(Verdict.latency_ms)).all() if r]
    lat_arr = np.asarray(lat, dtype=np.float64) if lat else np.zeros(1)
    hist, edges = np.histogram(lat_arr, bins=12)

    scores = [r for (r,) in session.execute(select(Verdict.final_score)).all()]
    return {
        "total_verdicts": total,
        "by_verdict": by_verdict,
        "by_attack_pattern": by_pattern,
        "abstention_rate": round(
            (session.scalar(select(func.count()).select_from(Verdict).where(Verdict.abstained.is_(True))) or 0) / total, 4
        ),
        "latency_ms": {
            "p50": round(float(np.percentile(lat_arr, 50)), 1),
            "p90": round(float(np.percentile(lat_arr, 90)), 1),
            "p99": round(float(np.percentile(lat_arr, 99)), 1),
            "mean": round(float(lat_arr.mean()), 1),
            "max": round(float(lat_arr.max()), 1),
            "histogram": {"counts": hist.tolist(), "edges": [round(float(e), 1) for e in edges]},
        },
        "score_distribution": {
            "deciles": [round(float(np.percentile(scores, p)), 4) for p in range(0, 101, 10)] if scores else [],
        },
    }


@router.get("/metrics/cost-curve", summary="Fraud prevented vs friction lost, across every threshold")
def cost_curve(
    avg_fraud_loss_inr: float = Query(85000, ge=0, description="Average loss when one fake merchant gets through"),
    merchant_ltv_inr: float = Query(42000, ge=0, description="Lifetime value of a legitimate merchant"),
    abandon_prob: float = Query(0.35, ge=0, le=1, description="Probability a falsely-rejected merchant walks away"),
    fraud_base_rate: float = Query(0.03, ge=0, le=1, description="Share of real onboarding traffic that is fraudulent"),
    per_1000: int = Query(1000, ge=1),
):
    """The tradeoff curve, computed from held-out scores rather than asserted.

    Three of these inputs are business assumptions, not measurements. They are
    query parameters precisely so a reviewer can move them and watch the optimum
    move - a single headline "₹X saved" number would be unfalsifiable.
    """
    report = _load_report()
    if not report or "raw_scores" not in report:
        raise HTTPException(
            409,
            "No evaluation report yet. Run `python backend/scripts/evaluate.py` to generate eval/metrics.json.",
        )
    y = np.asarray(report["raw_scores"]["y_true"], dtype=int)
    s = np.asarray(report["raw_scores"]["y_score"], dtype=float)
    if y.size == 0:
        raise HTTPException(409, "Evaluation report contains no scores")

    n_fraud, n_legit = max(1, int(y.sum())), max(1, int((1 - y).sum()))
    curve: List[Dict] = []
    for t in np.linspace(0.02, 0.98, 49):
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())

        far = fn / n_fraud   # fraud that slipped through
        frr = fp / n_legit   # legitimate merchants blocked

        fraud_volume = per_1000 * fraud_base_rate
        legit_volume = per_1000 * (1 - fraud_base_rate)
        prevented = fraud_volume * (1 - far) * avg_fraud_loss_inr
        leaked = fraud_volume * far * avg_fraud_loss_inr
        friction = legit_volume * frr * merchant_ltv_inr * abandon_prob
        curve.append(
            {
                "threshold": round(float(t), 3),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "false_accept_rate": round(far, 4),
                "false_reject_rate": round(frr, 4),
                "precision": round(tp / max(1, tp + fp), 4),
                "recall": round(tp / max(1, tp + fn), 4),
                "fraud_prevented_inr": round(prevented, 0),
                "fraud_leaked_inr": round(leaked, 0),
                "friction_cost_inr": round(friction, 0),
                "net_benefit_inr": round(prevented - friction, 0),
            }
        )
    best = max(curve, key=lambda c: c["net_benefit_inr"])
    return {
        "assumptions": {
            "avg_fraud_loss_inr": avg_fraud_loss_inr,
            "merchant_ltv_inr": merchant_ltv_inr,
            "abandon_prob": abandon_prob,
            "fraud_base_rate": fraud_base_rate,
            "per_onboardings": per_1000,
            "caveat": "These four inputs are assumptions supplied by the operator, not measurements by Verityne.",
        },
        "curve": curve,
        "optimal": best,
        "eval_set_size": int(y.size),
    }


@router.get("/metrics/thresholds", summary="Where each merchant policy currently sits on the curve")
def thresholds(merchant_id: str = "default"):
    p = get_policy(merchant_id)
    return {
        "merchant_id": p.merchant_id,
        "min_risk_for_review": p.min_risk_for_review,
        "min_risk_for_reject": p.min_risk_for_reject,
        "abstain_band": p.abstain_band,
        "require_liveness": p.require_liveness,
        "require_id_document": p.require_id_document,
        "cost_inputs": {
            "avg_fraud_loss_inr": p.avg_fraud_loss_inr,
            "merchant_ltv_inr": p.merchant_ltv_inr,
            "false_reject_abandon_prob": p.false_reject_abandon_prob,
        },
    }
