#!/usr/bin/env python3
"""The linkage threshold is a search threshold, and was calibrated as a pair one.

    python backend/scripts/calibrate_linkage_lfw.py --apply

`calibrate_face_match_lfw.py` fits the threshold that answers *are these two
photographs the same person* - a verification question, graded on LFW's official
6,000 pairs. `linkage.SAME_PERSON` was taken from that fit, at the FAR=0.1%
operating point.

But linkage does not ask a pair question. `find_face_links` compares one
applicant against every prior submission, up to `SCAN_LIMIT` of them. A pairwise
false-accept rate of `p` applied `N` times gives a per-applicant false-link
probability of ``1 - (1 - p)^N``, and that is the number an honest merchant
experiences:

    p = 0.0007 (the shipped point)   N =   100  ->   6.8%
                                     N = 1,000  ->  50.4%
                                     N = 5,000  ->  97.0%

At the shipped scan limit, virtually every genuine applicant false-links to
somebody. And 0.0007 is itself two impostor pairs out of the official 3,000 -
the resolution floor of that set, which cannot measure a rate any finer.

So this script measures the far tail properly. It takes one photograph per LFW
identity - 5,749 distinct people - and scores *every* pair among them. Those are
all impostor pairs by construction, which yields ~16.5 million of them and
resolves the false-accept rate to around 1e-7 instead of 3e-4.

The threshold is then chosen from a stated budget: a per-applicant false-link
rate `q` against a database of `N` records needs a pairwise FAR of
``1 - (1 - q)^(1/N)``. True-accept rate at the chosen point is reported from
LFW's official same-person pairs, so the recall cost is measured rather than
assumed.

Writes eval/linkage_lfw.json; `--apply` moves the threshold the API loads.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from calibrate_face_match_lfw import EmbeddingCache, ensure_dataset, parse_pairs  # noqa: E402
from verityne.config import EVAL_ROOT, MODEL_ROOT  # noqa: E402
from verityne.detectors.models import face_embedder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("linkage_cal")

#: Database sizes to report the per-applicant false-link rate at. 5,000 is
#: `linkage.SCAN_LIMIT`; the rest bracket it so the growth is visible.
REPORT_N = (20, 100, 500, 1_000, 5_000, 50_000)

#: The budget the shipped threshold is chosen for: at most this share of honest
#: applicants may pick up a false link, against a database of `SCAN_LIMIT`.
DEFAULT_BUDGET = 0.01

#: The value this repository shipped before linkage was calibrated as a search:
#: LFW's FAR=0.1% *verification* point, from `calibrate_face_match_lfw.py`.
#:
#: Graded on every run and named explicitly rather than read from the running
#: code, for the same reason the sibling script pins its own superseded
#: constants: once the threshold moves, `as_shipped` grades the *new* value and
#: the README's before/after table would otherwise have nothing behind it. Read
#: live, this row would also silently grade whatever happened to be in the file
#: mid-migration, which is exactly how a report ends up describing a number that
#: never shipped.
SUPERSEDED = 0.5198


def one_photo_per_identity(images: Path, limit: int = 0) -> List[Path]:
    """One photograph of each distinct person, sorted for reproducibility.

    One per identity, not all 13,233 photographs: two photographs of the same
    person are a *genuine* pair, and mixing them into the impostor set would
    contaminate exactly the tail this script exists to measure.
    """
    people = sorted(p for p in images.iterdir() if p.is_dir())
    if limit:
        people = people[:: max(1, len(people) // limit)][:limit]
    out = []
    for person in people:
        shots = sorted(person.glob("*.jpg"))
        if shots:
            out.append(shots[0])
    return out


def far_for_budget(q: float, n: int) -> float:
    """Pairwise FAR that holds a per-applicant false-link rate of `q` over `n` records."""
    return 1.0 - (1.0 - q) ** (1.0 / n)


def threshold_at_far(impostor: np.ndarray, target_far: float) -> Dict[str, object]:
    """Smallest threshold whose impostor acceptance is at or below `target_far`.

    Rounded *up* to four places, then re-measured, for the same reason
    `rate_at_far` does it: the value is applied with `>=` after being copied into
    the code, so rounding down after measuring would hand back a threshold that
    misses the rate printed beside it.
    """
    budget = target_far * impostor.size
    resolved = budget >= 1
    srt = np.sort(impostor)[::-1]
    thr = float(srt[0]) + 1e-6 if not resolved else float(srt[int(budget) - 1])
    thr = float(np.ceil(thr * 10_000) / 10_000)
    return {
        "threshold": round(thr, 4),
        "far": float((impostor >= thr).mean()),
        "impostor_pairs_accepted": int((impostor >= thr).sum()),
        "resolved_by_this_sample": bool(resolved),
    }


def per_applicant(far: float) -> Dict[str, float]:
    return {str(n): round(1.0 - (1.0 - far) ** n, 6) for n in REPORT_N}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-home", default="datasets/lfw/lfw_home")
    ap.add_argument("--identities", type=int, default=0,
                    help="debug: evenly spaced subsample of identities (0 = all)")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                    help="per-applicant false-link rate the shipped threshold is chosen for")
    ap.add_argument("--scan-limit", type=int, default=0,
                    help="database size the budget is held over (0 = linkage.SCAN_LIMIT)")
    ap.add_argument("--apply", action="store_true",
                    help="write the fitted threshold into the file linkage.py loads")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "linkage_lfw.json")
    args = ap.parse_args()

    from verityne.linkage import SAME_PERSON, SCAN_LIMIT

    scan_limit = args.scan_limit or SCAN_LIMIT
    images, pairs_path = ensure_dataset(Path(args.data_home))

    embedder = face_embedder()
    if embedder is None:
        raise SystemExit("face embedder unavailable — cannot calibrate")
    embed = EmbeddingCache(embedder)

    # ---- the impostor tail: every pair among distinct people -------------------
    shots = one_photo_per_identity(images, args.identities)
    log.info("embedding %d distinct identities", len(shots))
    t0 = time.time()
    vecs, kept = [], []
    for i, p in enumerate(shots):
        v = embed(p)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float32)
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
        kept.append(p)
        if (i + 1) % 500 == 0:
            log.info("  %d/%d (%.0fs)", i + 1, len(shots), time.time() - t0)
    V = np.stack(vecs)
    log.info("embedded %d identities in %.0fs", len(V), time.time() - t0)

    S = V @ V.T
    impostor = S[np.triu_indices(len(V), 1)].astype(np.float32)
    log.info("%s impostor pairs; FAR resolution %.2e", f"{impostor.size:,}", 1.0 / impostor.size)

    # ---- the recall cost: LFW's official same-person pairs ---------------------
    paths, labels, _ = parse_pairs(pairs_path, images)
    gen: List[float] = []
    for (pa, pb), lab in zip(paths, labels):
        if lab != 1:
            continue
        va, vb = embed(pa), embed(pb)
        if va is None or vb is None:
            continue
        a = np.asarray(va, dtype=np.float32); b = np.asarray(vb, dtype=np.float32)
        gen.append(float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)))
    genuine = np.asarray(gen, dtype=np.float32)
    log.info("%d genuine pairs for the recall side", genuine.size)

    def grade(thr: float) -> Dict[str, object]:
        far = float((impostor >= thr).mean())
        return {
            "threshold": round(float(thr), 4),
            "pairwise_far": far,
            "impostor_pairs_accepted": int((impostor >= thr).sum()),
            "tar": round(float((genuine >= thr).mean()), 4),
            "per_applicant_false_link_rate": per_applicant(far),
        }

    target = far_for_budget(args.budget, scan_limit)
    fitted = threshold_at_far(impostor, target)
    chosen = grade(fitted["threshold"])

    report: Dict[str, object] = {
        "what_this_measures": (
            "The operating point for a one-against-many search, not a pairwise test. "
            "linkage.find_face_links compares each applicant against up to SCAN_LIMIT "
            "prior records, so the rate that matters is the per-applicant probability of "
            "at least one false link, not the pairwise false-accept rate."
        ),
        "method": (
            f"One photograph per LFW identity ({len(V)} distinct people), every pair among "
            f"them scored ({impostor.size:,} impostor pairs, all impostor by construction). "
            "True-accept rate comes from LFW's official same-person pairs."
        ),
        "far_resolution": {
            "this_sample": 1.0 / impostor.size,
            "official_lfw_pairs": 1.0 / 3000,
            "note": (
                "The official protocol has 3,000 impostor pairs, so it cannot measure a "
                "false-accept rate below 3.3e-4. The shipped 0.0007 was two of those pairs."
            ),
        },
        "budget": {
            "per_applicant_false_link_rate": args.budget,
            "over_database_of": scan_limit,
            "required_pairwise_far": target,
        },
        "fitted": {**chosen, "sample_resolves_this_far": fitted["resolved_by_this_sample"]},
        "as_shipped": grade(SAME_PERSON),
        "superseded": {
            **grade(SUPERSEDED),
            "note": (
                "LFW's FAR=0.1% verification point, which linkage used before it was "
                "calibrated as a search. Kept so the README's before/after table has "
                "evidence behind it after the constant moved."
            ),
        },
        "impostor_distribution": {
            "n": int(impostor.size),
            "mean": round(float(impostor.mean()), 4),
            "p99": round(float(np.percentile(impostor, 99)), 4),
            "p99.99": round(float(np.percentile(impostor, 99.99)), 4),
            "max": round(float(impostor.max()), 4),
        },
        "genuine_distribution": {
            "n": int(genuine.size),
            "mean": round(float(genuine.mean()), 4),
            "p05": round(float(np.percentile(genuine, 5)), 4),
            "p50": round(float(np.percentile(genuine, 50)), 4),
        },
        "caveat": (
            "LFW photographs are web images of public figures, not phone selfies. The "
            "identification question is the same; the capture conditions are not. This is a "
            "lower bound on the threshold a production queue needs, measured on real faces."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)

    if args.apply:
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        path = MODEL_ROOT / "linkage_threshold.json"
        path.write_text(json.dumps({
            "same_person": chosen["threshold"],
            "previously": SAME_PERSON,
            "superseded_verification_point": SUPERSEDED,
            "fitted_on": (
                f"LFW, {len(V)} distinct identities, {impostor.size:,} impostor pairs; "
                f"chosen for a {args.budget:.1%} per-applicant false-link rate over "
                f"{scan_limit:,} records"
            ),
            "pairwise_far": chosen["pairwise_far"],
            "tar": chosen["tar"],
            "per_applicant_false_link_rate": chosen["per_applicant_false_link_rate"],
        }, indent=2))
        log.info("applied fitted threshold to %s", path)
    else:
        log.info("report only; re-run with --apply to move the API's threshold")

    print(f"\nimpostor pairs: {impostor.size:,}   (official protocol: 3,000)")
    print(f"budget: {args.budget:.1%} of applicants may false-link over {scan_limit:,} records")
    print(f"  -> required pairwise FAR {target:.2e}\n")
    for title, block in (("superseded", report["superseded"]),
                         ("as shipped", report["as_shipped"]),
                         ("fitted", report["fitted"])):
        b = block  # type: ignore[assignment]
        print(f"{title:>11}: threshold {b['threshold']:.4f}  pairwise FAR {b['pairwise_far']:.2e}  "
              f"TAR {b['tar']:.4f}")
        rates = b["per_applicant_false_link_rate"]
        print(f"             per-applicant false-link rate: " +
              "  ".join(f"N={k}: {v:.1%}" for k, v in rates.items()))


if __name__ == "__main__":
    main()
