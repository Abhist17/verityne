#!/usr/bin/env python3
"""Fit the face-identity threshold on LFW — real photos of real people.

    python backend/scripts/calibrate_face_match_lfw.py

Why this script exists
----------------------
Two numbers in this system decide whether two faces are the same person:

  * ``linkage.SAME_PERSON`` — used to link one face across submissions;
  * the ``low`` end of the face-match band — below it, the selfie is not the
    person on the document.

Both were hard-coded. Worse, the corpus could not fit them honestly: its
"genuine" selfie/ID pairs derive from a single source photograph per identity,
re-captured, so their similarity runs far above what two real photos of one
person score. A threshold fitted on that would be tuned to an artefact.

LFW is the standard answer: 13,233 photographs of 5,749 real people, with a
published 10-fold protocol of 6,000 pairs — 3,000 same-person, 3,000 different.
Every pair is two genuinely different photographs, which is exactly the question
being asked. It also makes the result comparable to published work instead of
self-reported.

Protocol
--------
``pairs.txt`` ships the fold assignment, so this reads it rather than
re-deriving one: ten folds of 300 matched and 300 mismatched pairs. For each
fold the threshold is chosen on the other nine and evaluated on the held-out
one, so no fold's threshold is scored on the data that chose it. The reported
accuracy is the mean across folds, with its standard deviation — the form LFW
results are published in.

Images are read at full funneled resolution and go through ``largest_face`` and
``FaceEmbedder`` — the same MTCNN crop and the same FaceNet weights the API
uses — so the fitted number transfers to the running system rather than
describing a parallel pipeline. Embeddings are cached per image because the
6,000 pairs reference far fewer than 12,000 distinct photographs.

What this does and does not fit
-------------------------------
Fits the *identity* question — are these two faces the same person — on
photo-vs-photo pairs. That is precisely the linkage question, and a defensible
lower bound for face match.

It does not fit the band's ``high`` end. That guards against a selfie that is a
copy of the printed ID portrait, which needs selfie-vs-document pairs; LFW has
no documents in it. That bound stays where the corpus calibration put it, and
the limitation is recorded in the output.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from verityne.config import EVAL_ROOT, MODEL_ROOT  # noqa: E402
from verityne.detectors.models import cosine, face_embedder  # noqa: E402
from verityne.utils.images import largest_face, load_rgb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lfw")

FOLDS = 10
PAIRS_URL = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
FUNNELED_URL = "https://ndownloader.figshare.com/files/5976015"


# ---------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------

def ensure_dataset(home: Path) -> Tuple[Path, Path]:
    """Return (image_root, pairs.txt), downloading and unpacking if needed."""
    home.mkdir(parents=True, exist_ok=True)
    images = home / "lfw_funneled"
    pairs = home / "pairs.txt"

    if not pairs.exists():
        log.info("downloading pairs.txt")
        urllib.request.urlretrieve(PAIRS_URL, pairs)

    if not images.is_dir():
        tgz = home / "lfw-funneled.tgz"
        if not tgz.exists():
            log.info("downloading lfw-funneled.tgz (~233 MB)")
            urllib.request.urlretrieve(FUNNELED_URL, tgz)
        log.info("unpacking %s", tgz)
        import tarfile

        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(home)
    return images, pairs


def parse_pairs(pairs_path: Path, images: Path) -> Tuple[List[Tuple[Path, Path]], np.ndarray, np.ndarray]:
    """Parse the official pairs file into (paths, label, fold).

    Format: a header of ``<folds> <pairs_per_class>``, then per fold
    ``pairs_per_class`` matched lines ``name n1 n2`` followed by the same number
    of mismatched lines ``name1 n1 name2 n2``. The fold index is positional,
    which is why it is read from the file rather than derived from the row
    order after any filtering.
    """
    lines = [ln.strip() for ln in pairs_path.read_text().splitlines() if ln.strip()]
    n_folds, per_class = (int(x) for x in lines[0].split())

    def photo(name: str, idx: str) -> Path:
        return images / name / f"{name}_{int(idx):04d}.jpg"

    paths: List[Tuple[Path, Path]] = []
    labels: List[int] = []
    folds: List[int] = []

    cursor = 1
    for fold in range(n_folds):
        for _ in range(per_class):  # same person
            name, a, b = lines[cursor].split("\t") if "\t" in lines[cursor] else lines[cursor].split()
            paths.append((photo(name, a), photo(name, b)))
            labels.append(1)
            folds.append(fold)
            cursor += 1
        for _ in range(per_class):  # different people
            parts = lines[cursor].split("\t") if "\t" in lines[cursor] else lines[cursor].split()
            n1, a, n2, b = parts
            paths.append((photo(n1, a), photo(n2, b)))
            labels.append(0)
            folds.append(fold)
            cursor += 1

    return paths, np.asarray(labels, dtype=int), np.asarray(folds, dtype=int)


# ---------------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------------

class EmbeddingCache:
    """Embed each photograph once — the 6,000 pairs reuse images heavily."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._cache: Dict[Path, Optional[np.ndarray]] = {}
        self.detected = 0
        self.total = 0

    def __call__(self, path: Path) -> Optional[np.ndarray]:
        if path in self._cache:
            return self._cache[path]
        self.total += 1
        vec: Optional[np.ndarray] = None
        try:
            rgb = load_rgb(path, max_side=1024)
            face = largest_face(rgb, size=256)
            if face is None:
                # LFW is funneled: the subject is centred by construction, so a
                # failed detection is a detector miss, not a missing face.
                # Falling back to a centre crop keeps a detector failure from
                # silently becoming a favourable filter on the evaluation set.
                h, w = rgb.shape[:2]
                s = int(min(h, w) * 0.62)
                y, x = (h - s) // 2, (w - s) // 2
                face = rgb[y : y + s, x : x + s]
            else:
                self.detected += 1
            vec = self.embedder.embed(face)
        except Exception as exc:  # noqa: BLE001
            log.warning("embedding failed for %s: %s", path.name, exc)
        self._cache[path] = vec
        return vec


# ---------------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------------

def best_threshold(sims: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Threshold maximising accuracy, searched over the observed similarities."""
    order = np.unique(sims)
    if order.size == 0:
        return 0.5, 0.0
    cands = (order[:-1] + order[1:]) / 2.0 if order.size > 1 else order
    accs = ((sims[None, :] >= cands[:, None]) == (labels[None, :] == 1)).mean(axis=1)
    i = int(np.argmax(accs))
    return float(cands[i]), float(accs[i])


def rate_at_far(sims: np.ndarray, labels: np.ndarray, target_far: float) -> Dict[str, float]:
    """Threshold that holds impostor acceptance at `target_far`, and the TAR there.

    This is the operating point a KYC system actually runs at: accepting an
    impostor is the expensive error, so the threshold is set by the false-accept
    budget rather than by accuracy.
    """
    imp = np.sort(sims[labels == 0])[::-1]
    gen = sims[labels == 1]
    if imp.size == 0 or gen.size == 0:
        return {"threshold": float("nan"), "tar": float("nan"), "far": target_far}
    budget = target_far * imp.size
    if budget < 1:
        thr = float(imp[0]) + 1e-6  # cannot afford a single impostor
    else:
        thr = float(imp[int(round(budget)) - 1])
    # Report the rates at the *rounded* threshold, not the exact one. This value
    # gets copied into linkage.py and the band file and applied there with `>=`,
    # so rounding it down by a fraction after measuring would quietly hand back
    # a threshold that does not deliver the FAR printed beside it.
    thr = float(np.ceil(thr * 10_000) / 10_000)
    return {
        "threshold": round(thr, 4),
        "tar": round(float((gen >= thr).mean()), 4),
        "far": round(float((imp >= thr).mean()), 4),
    }


def score_threshold(sims: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    """What a given threshold actually does on real pairs: TAR, FAR, accuracy."""
    gen, imp = sims[labels == 1], sims[labels == 0]
    return {
        "threshold": round(float(thr), 4),
        "tar": round(float((gen >= thr).mean()), 4),
        "far": round(float((imp >= thr).mean()), 4),
        "accuracy": round(float(((sims >= thr) == (labels == 1)).mean()), 4),
    }


#: The values this repository shipped before it was calibrated on real pairs.
#:
#: Graded on every run so the claim in the README - that these were missing a
#: third of true links, and would have called half of honest applicants
#: impostors - keeps evidence behind it. Once the constants are corrected,
#: ``as_shipped`` grades the *new* ones and the old numbers would otherwise have
#: nothing in the committed reports to support them.
SUPERSEDED = {
    "linkage.SAME_PERSON": 0.75,
    "face_match_band.low": 0.7835,
}


def superseded(sims: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Grade the pre-calibration constants, for the record."""
    out: Dict[str, object] = {
        name: score_threshold(sims, labels, thr) for name, thr in SUPERSEDED.items()
    }
    out["note"] = (
        "What this repo used before calibrating on LFW. linkage.SAME_PERSON was a "
        "hard-coded guess; face_match_band.low was fitted on corpus pairs that derive "
        "from one source photograph per identity and so score far higher than two real "
        "photographs of one person. Kept here so the README's before/after table has "
        "evidence behind it after the constants moved."
    )
    return out


def as_shipped(sims: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Grade the constants the code currently ships against this real-pair set.

    The point of calibrating on real data is not only to produce a better
    number, it is to show what the old one was doing. Both values are read from
    the running code rather than pasted in, so this block cannot drift out of
    date once the thresholds move.
    """
    out: Dict[str, object] = {}
    try:
        from verityne.linkage import SAME_PERSON

        out["linkage.SAME_PERSON"] = score_threshold(sims, labels, SAME_PERSON)
    except Exception:  # noqa: BLE001
        pass
    try:
        from verityne.detectors.face_match import decision_band

        low, high, provenance = decision_band()
        out["face_match_band.low"] = {**score_threshold(sims, labels, low), "source": provenance}
    except Exception:  # noqa: BLE001
        pass
    return out


def _dist(sims: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(sims.size),
        "mean": round(float(sims.mean()), 4),
        "p05": round(float(np.percentile(sims, 5)), 4),
        "p50": round(float(np.percentile(sims, 50)), 4),
        "p95": round(float(np.percentile(sims, 95)), 4),
    }


# ---------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-home", default="datasets/lfw/lfw_home")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: score an evenly spaced subsample of N pairs. Spaced, not a "
                         "prefix, because pairs.txt is ordered same-then-different within each "
                         "fold — a prefix would be one class from one fold.")
    ap.add_argument("--apply", action="store_true",
                    help="write the fitted threshold into the band the API loads")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    images, pairs_path = ensure_dataset(Path(args.data_home))
    paths, labels, folds = parse_pairs(pairs_path, images)
    if args.limit and args.limit < len(paths):
        pick = np.linspace(0, len(paths) - 1, args.limit).astype(int)
        paths = [paths[i] for i in pick]
        labels, folds = labels[pick], folds[pick]
    n = len(paths)
    log.info("%d pairs (%d same, %d different) across %d folds",
             n, int(labels.sum()), int((1 - labels).sum()), len(set(folds.tolist())))

    embedder = face_embedder()
    if embedder is None:
        raise SystemExit("face embedder unavailable — cannot calibrate")
    embed = EmbeddingCache(embedder)

    sims = np.zeros(n, dtype=np.float32)
    ok = np.ones(n, dtype=bool)
    t0 = time.time()
    for i, (pa, pb) in enumerate(paths):
        va, vb = embed(pa), embed(pb)
        if va is None or vb is None:
            ok[i] = False
            continue
        sims[i] = cosine(va, vb)
        if (i + 1) % 500 == 0:
            log.info("  %d/%d  %.1f pairs/s  (%d distinct photos embedded)",
                     i + 1, n, (i + 1) / (time.time() - t0), embed.total)

    sims, labels, folds = sims[ok], labels[ok], folds[ok]
    det_rate = embed.detected / max(1, embed.total)
    log.info("scored %d/%d pairs in %.0fs; MTCNN found a face in %.1f%% of %d distinct photos",
             len(sims), n, time.time() - t0, 100.0 * det_rate, embed.total)

    # ---- strict 10-fold: threshold from nine folds, scored on the tenth ----
    accs: List[float] = []
    thrs: List[float] = []
    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        thr, _ = best_threshold(sims[tr], labels[tr])
        accs.append(float(((sims[te] >= thr) == (labels[te] == 1)).mean()))
        thrs.append(thr)

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    mean_thr = float(np.mean(thrs))
    auc = float(roc_auc_score(labels, sims))
    far1 = rate_at_far(sims, labels, 0.01)
    far01 = rate_at_far(sims, labels, 0.001)

    report = {
        "dataset": "LFW (deep-funneled), official 10-fold pair protocol from pairs.txt",
        "source": "13,233 photographs of 5,749 real people; every pair is two distinct photographs",
        "embedder": "facenet-pytorch InceptionResnetV1 (vggface2) — the same model and MTCNN crop the API uses",
        "n_pairs_scored": int(len(sims)),
        "n_pairs_total": int(n),
        "n_photos_embedded": int(embed.total),
        "face_detection_rate": round(det_rate, 4),
        "roc_auc": round(auc, 4),
        "accuracy_10fold": {
            "mean": round(mean_acc, 4),
            "std": round(std_acc, 4),
            "per_fold": [round(a, 4) for a in accs],
            "protocol": "threshold chosen on nine folds, scored on the held-out tenth",
        },
        "threshold": {
            "mean_over_folds": round(mean_thr, 4),
            "per_fold": [round(t, 4) for t in thrs],
        },
        "operating_points": {"far_1pct": far1, "far_0.1pct": far01},
        "as_shipped": as_shipped(sims, labels),
        "superseded": superseded(sims, labels),
        "similarity": {
            "same_person": _dist(sims[labels == 1]),
            "different_person": _dist(sims[labels == 0]),
        },
        "fits": [
            "linkage.SAME_PERSON — is this the same person as an earlier submission",
            "the low end of the face-match band — is the selfie the person on the document",
        ],
        "does_not_fit": (
            "The high end of the band, which catches a selfie copied from the printed ID "
            "portrait. That needs selfie-vs-document pairs and LFW contains no documents, so "
            "that bound still comes from the corpus calibration."
        ),
        "caveat": (
            "LFW photographs are web images of public figures, not phone selfies against a "
            "printed card. The identity question is the same one; the capture conditions are "
            "not. Treat this as a lower bound that is at least measured on real pairs."
        ),
    }

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    out = EVAL_ROOT / "face_match_lfw.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out)

    if args.apply:
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        band_path = MODEL_ROOT / "face_match_band.json"
        band = json.loads(band_path.read_text()) if band_path.exists() else {}
        previous_low = band.get("low")
        band["low"] = round(mean_thr, 4)
        band["same_person"] = round(mean_thr, 4)
        band["far_1pct_threshold"] = far1["threshold"]
        band["far_0.1pct_threshold"] = far01["threshold"]
        band.setdefault("high", 0.9878)
        # `fitted_on` is the one string the API surfaces in its signals, so it has
        # to describe both bounds — they now come from different data, and a
        # provenance line that mentions only the better one would be misleading.
        band["fitted_on"] = (
            f"low: LFW 10-fold, {len(sims)} real pairs, accuracy {mean_acc:.4f}±{std_acc:.4f}; "
            f"high: corpus calibration (see high_fitted_on)"
        )
        band.setdefault("high_fitted_on", band.get("corpus_fitted_on")
                        or "corpus train split; not re-fitted on real data — LFW has no documents")
        band["low_previously"] = previous_low
        band["lfw"] = {"accuracy": round(mean_acc, 4), "std": round(std_acc, 4),
                       "roc_auc": round(auc, 4), "n_pairs": int(len(sims)),
                       "far_1pct": far1, "far_0.1pct": far01,
                       "as_shipped_before": report["as_shipped"]}
        band_path.write_text(json.dumps(band, indent=2))
        log.info("applied fitted low bound to %s", band_path)
    else:
        log.info("report only; re-run with --apply to move the API's threshold")

    print(f"\nLFW accuracy {mean_acc:.4f} +/- {std_acc:.4f}   ROC-AUC {auc:.4f}")
    print(f"fitted identity threshold {mean_thr:.4f}")
    print(f"at FAR=1%:   threshold {far1['threshold']}  TAR {far1['tar']}")
    print(f"at FAR=0.1%: threshold {far01['threshold']}  TAR {far01['tar']}")
    for title, block in (("currently shipped", report["as_shipped"]),
                         ("superseded (pre-calibration)", report["superseded"])):
        rows = {k: v for k, v in block.items() if isinstance(v, dict)}
        if not rows:
            continue
        print(f"\nwhat the {title} constants do on these real pairs:")
        for name, r in rows.items():
            print(f"  {name:24s} thr {r['threshold']:.4f}  TAR {r['tar']:.4f}  "
                  f"FAR {r['far']:.4f}  acc {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
