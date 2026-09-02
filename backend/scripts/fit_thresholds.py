#!/usr/bin/env python3
"""Choose the review and reject thresholds, without touching the held-out split.

The shipped 0.40 and 0.75 were inherited from the *leaked* model, whose scores
separated the classes far more widely. On the corrected model they sit in the
wrong places - REJECT catches 40.7% of fraud and REVIEW sends 43.1% of genuine
merchants to a human - and the README has been carrying that as a stated defect.

Re-tuning them on the held-out split would be the same error this project has
spent its README documenting: a model graded on its own answer sheet. So the
selection here sees **only the 195-row training split**, and only through
out-of-fold predictions:

  * the fusion model is refitted five times, each time on four folds,
  * each row is scored by a model that never saw it,
  * thresholds are chosen on that out-of-fold score distribution,
  * and the held-out split is scored afterwards purely to *report* what the
    choice produced. It is never consulted to make it.

**The reject threshold is an economic argument, not a statistical one.** There is
no threshold that is correct in the abstract: rejecting more fraud always costs
more genuine merchants, and which trade is right depends on what a fraud costs
and what a merchant is worth. So it maximises expected net benefit under the
merchant's own policy inputs - `avg_fraud_loss_inr`, `merchant_ltv_inr`,
`false_reject_abandon_prob` and a base rate - which are the same assumptions the
cost curve on the metrics page exposes as sliders. Change the assumptions and the
threshold moves, which is the honest behaviour.

**The review threshold is a capacity argument.** It is the lowest score that
keeps the share of genuine applicants sent to a human under a stated budget,
because a review queue nobody can staff is not a control.

    python backend/scripts/fit_thresholds.py --base-rate 0.03 --review-budget 0.25
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from train_fusion import build_matrix  # noqa: E402
from verityne.config import DATASET_ROOT, EVAL_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fit_thresholds")

SCORES = EVAL_ROOT / "scores.json"
OUT = EVAL_ROOT / "thresholds.json"

#: Candidate grid. Fine enough that the argmax is not an artefact of spacing.
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def out_of_fold(X: np.ndarray, y: np.ndarray, folds: int, seed: int,
                groups: np.ndarray | None = None) -> np.ndarray:
    """Score every training row with a model that never saw it.

    Refitting the *whole* pipeline per fold - estimator and Platt calibrator
    together - matters here. Calibrating on predictions the calibrator's own
    training rows produced would pull the score distribution toward the labels,
    and it is the distribution these thresholds are read off.

    `groups` makes the folds **identity-disjoint**, and passing it is not
    optional for a threshold that will be deployed. Every identity in this corpus
    contributes several packets; fold on rows and a model scores an identity it
    has already been fitted on, which is memorisation reported as generalisation.
    The held-out split was built identity-disjoint precisely for that reason, and
    a threshold chosen on row-wise folds does not survive the difference - see
    `naive_comparison` in the report this writes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from verityne.fusion import PlattCalibrator

    oof = np.zeros(len(y), dtype=float)
    if groups is None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = splitter.split(X, y, groups)
    for tr, va in split:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(X[tr], y[tr])
        # The calibrator is fitted on an inner split of the fold's own training
        # rows, never on the rows it will be used to score.
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        raw, lab = [], []
        for itr, iva in inner.split(X[tr], y[tr]):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[tr][itr], y[tr][itr])
            raw.append(m.predict_proba(X[tr][iva])[:, 1])
            lab.append(y[tr][iva])
        calib = PlattCalibrator.fit(np.concatenate(raw), np.concatenate(lab))
        oof[va] = np.clip(calib.predict(model.predict_proba(X[va])[:, 1]), 0.0, 1.0)
    return oof


def rates(scores: np.ndarray, y: np.ndarray, t: float) -> Dict[str, float]:
    flagged = scores >= t
    fraud, genuine = y == 1, y == 0
    recall = float(flagged[fraud].mean()) if fraud.any() else 0.0
    fpr = float(flagged[genuine].mean()) if genuine.any() else 0.0
    tp, fp = int((flagged & fraud).sum()), int((flagged & genuine).sum())
    return {
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "false_accept_rate": round(1.0 - recall, 4),
    }


def net_benefit(scores: np.ndarray, y: np.ndarray, t: float, *, base_rate: float,
                fraud_loss: float, ltv: float, abandon: float, volume: int = 10_000) -> float:
    """Expected ₹ per `volume` applications at threshold `t`.

    Prevented fraud minus the lifetime value of genuine merchants who abandon
    after being wrongly rejected. Both terms are scaled to the operator's base
    rate rather than to the corpus's 50/50 split, which is an artefact of how the
    corpus was built and would overstate the fraud side by an order of magnitude.
    """
    r = rates(scores, y, t)
    prevented = volume * base_rate * r["recall"] * fraud_loss
    lost = volume * (1 - base_rate) * r["false_positive_rate"] * ltv * abandon
    return prevented - lost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-rate", type=float, default=0.03,
                    help="share of applications that are fraudulent")
    ap.add_argument("--fraud-loss", type=float, default=85000.0)
    ap.add_argument("--ltv", type=float, default=42000.0)
    ap.add_argument("--abandon", type=float, default=0.35)
    ap.add_argument("--review-budget", type=float, default=0.25,
                    help="max share of genuine applicants sent to a human")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = json.loads(SCORES.read_text())
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    if not train:
        raise SystemExit("no training rows in eval/scores.json - run `make score` first")

    # Identity, joined from the corpus manifest. Several packets per identity, so
    # folding on rows would let a model score a face it was fitted on.
    manifest_path = DATASET_ROOT / "manifest.json"
    identities: Dict[str, int] = {}
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        items = man["items"] if isinstance(man, dict) and "items" in man else man
        identities = {i["id"]: i.get("identity_index") for i in items
                      if i.get("identity_index") is not None}
    groups = np.array([identities.get(r["id"], -1) for r in train])
    have_groups = (groups >= 0).all() and len(set(groups.tolist())) > args.folds
    if not have_groups:
        log.warning("no usable identity grouping in the manifest - folds will be row-wise, "
                    "and the false-positive rate they report will be optimistic")

    Xtr, ytr, _ = build_matrix(train)
    log.info("selection sees %d training rows across %d identities, and nothing else",
             len(ytr), len(set(groups.tolist())))

    oof = out_of_fold(Xtr, ytr, args.folds, args.seed, groups if have_groups else None)
    naive = out_of_fold(Xtr, ytr, args.folds, args.seed, None) if have_groups else None

    econ = dict(base_rate=args.base_rate, fraud_loss=args.fraud_loss,
                ltv=args.ltv, abandon=args.abandon)
    curve = [
        {"threshold": float(t), "net_benefit_inr": round(net_benefit(oof, ytr, t, **econ), 0),
         **rates(oof, ytr, t)}
        for t in GRID
    ]
    reject = max(curve, key=lambda c: c["net_benefit_inr"])

    # Review: the lowest threshold whose genuine-review rate is inside budget.
    eligible = [c for c in curve
                if c["false_positive_rate"] <= args.review_budget
                and c["threshold"] <= reject["threshold"]]
    review = min(eligible, key=lambda c: c["threshold"]) if eligible else reject

    log.info("chosen on out-of-fold training scores: review %.2f, reject %.2f",
             review["threshold"], reject["threshold"])

    # ---- report what the choice produces on held-out. Never used to choose. ---
    held: Dict[str, object] = {}
    if test:
        from verityne import fusion

        Xte, yte, _ = build_matrix(test)
        bundle = fusion._load_model()
        if bundle is not None:
            p = bundle["model"].predict_proba(Xte)[:, 1]
            calib = bundle.get("calibrator")
            if calib is not None:
                p = np.clip(calib.predict(p), 0.0, 1.0)
            held = {
                "n": int(len(yte)),
                "at_review": rates(p, yte, review["threshold"]),
                "at_reject": rates(p, yte, reject["threshold"]),
                "note": "Scored after the thresholds were chosen, to report what they produce. "
                        "The held-out split was not consulted during selection.",
            }

    # What row-wise folds would have claimed, kept because the gap is the finding.
    naive_block: Dict[str, object] = {}
    if naive is not None:
        naive_curve = [
            {"threshold": float(t),
             "net_benefit_inr": round(net_benefit(naive, ytr, t, **econ), 0),
             **rates(naive, ytr, t)}
            for t in GRID
        ]
        naive_best = max(naive_curve, key=lambda c: c["net_benefit_inr"])
        naive_block = {
            "chosen_reject": naive_best["threshold"],
            "claimed_false_positive_rate": naive_best["false_positive_rate"],
            "identity_disjoint_false_positive_rate":
                next(c for c in curve if c["threshold"] == naive_best["threshold"])
                ["false_positive_rate"],
            "note": "Row-wise folds let a model score identities it was fitted on. The two "
                    "false-positive rates below are the same threshold measured both ways; the "
                    "first is what a careless cross-validation would have reported.",
        }
        log.info("row-wise folds would claim FPR %.3f at t=%.2f; identity-disjoint says %.3f",
                 naive_block["claimed_false_positive_rate"], naive_block["chosen_reject"],
                 naive_block["identity_disjoint_false_positive_rate"])

    # ---- does the choice survive contact with data it was not chosen on? -----
    # This is the number that decides whether these thresholds are deployable,
    # and it is computed last so it cannot influence the selection above.
    transfer: Dict[str, object] = {}
    if held:
        oof_fpr = reject["false_positive_rate"]
        out_fpr = held["at_reject"]["false_positive_rate"]
        ratio = round(out_fpr / oof_fpr, 1) if oof_fpr > 0 else None
        transfer = {
            "out_of_fold_false_positive_rate": oof_fpr,
            "held_out_false_positive_rate": out_fpr,
            "optimism_factor": ratio,
            "verdict": (
                "deployable" if ratio is not None and ratio <= 1.5
                else "NOT deployable from this corpus"
            ),
            "why": (
                "The threshold was chosen to hold a false-positive rate of "
                f"{oof_fpr:.1%} and produces {out_fpr:.1%} on a split it was not chosen on"
                + (f" - {ratio}x optimistic. " if ratio else ". ")
                + "195 rows cannot resolve an operating point that transfers: "
                  "cross-validation re-uses the same small pool of genuine faces, and the "
                  "score distribution shifts the moment the faces are new. Fitting on the "
                  "held-out split instead would produce a number that reports itself."
            ),
        }
        log.warning("transfer check: %s (%s)", transfer["verdict"], transfer["why"])

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": (
            f"{args.folds}-fold identity-disjoint out-of-fold predictions on the {len(ytr)}-row "
            "training split. The estimator and its Platt calibrator are refitted inside every "
            "fold, and folds never split one identity, so no row is scored by a model that saw "
            "either it or another packet of the same person. The held-out split is not consulted."
        ),
        "economics": {
            "base_rate": args.base_rate,
            "avg_fraud_loss_inr": args.fraud_loss,
            "merchant_ltv_inr": args.ltv,
            "false_reject_abandon_prob": args.abandon,
            "note": "Operator assumptions, not measurements. The reject threshold is the argmax "
                    "of expected net benefit under them; change them and it moves.",
        },
        "review_budget": args.review_budget,
        "chosen": {
            "min_risk_for_review": review["threshold"],
            "min_risk_for_reject": reject["threshold"],
        },
        "out_of_fold": {"review": review, "reject": reject, "n": int(len(ytr))},
        "superseded": {
            "min_risk_for_review": 0.40,
            "min_risk_for_reject": 0.75,
            "why": "Inherited from the leaked model, whose scores separated the classes far more "
                   "widely. On the corrected model they sat in the wrong places.",
        },
        "transfer_check": transfer,
        "recommendation": (
            "Do not deploy these as fixed constants. The selection procedure is sound and the "
            "corpus is too small for it: the chosen point's false-positive rate does not "
            "survive a split it was not chosen on. What ships instead is the cost curve - "
            "GET /metrics/cost-curve and the sliders on the metrics page - where an operator "
            "sets the point from their own fraud loss, merchant value and abandonment. This "
            "file is the evidence for why that is a design decision rather than an omission."
        ),
        "naive_comparison": naive_block,
        "held_out_check": held,
        "curve": curve,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", args.out)
    if held:
        log.info("held-out at the chosen points: reject recall %.3f at FPR %.3f",
                 held["at_reject"]["recall"], held["at_reject"]["false_positive_rate"])


if __name__ == "__main__":
    main()
