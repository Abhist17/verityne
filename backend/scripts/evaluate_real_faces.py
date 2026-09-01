#!/usr/bin/env python3
"""Does the selfie detector work on fakes this project did not generate?

    python backend/scripts/evaluate_real_faces.py --n 400

`eval/metrics.json` reports the detector on a corpus we built. That measures
separability on our own generator settings, and the checkpoint was *selected* on
that same corpus by `benchmark_models.py`, so the number carries a selection
effect on top of everything else. Nothing in it answers the only question that
matters in production: does this catch a face somebody else generated?

This scores the shipped detector on third-party data, and reports it **per
generator family, never pooled**. Pooling is what lets a detector that aces one
family and is blind to another report a respectable average - the failure mode
`benchmark_models.py` was written to avoid at selection time and which nothing
has re-checked since. Detector 6's evaluation reports its worst unseen strategy
rather than its mean for the same reason; this is that discipline applied to the
vision side.

Two tracks, chosen because they fail differently:

  * **DeepFakeFace** - three generator families over the *same* photographs, so
    identity, pose and subject are held fixed. The genuine images are native
    IMDB-WIKI sizes and every fake is 512x512, so the raw frames are separable
    on resampling history alone; the face protocol exists to remove that.
  * **140k Real and Fake Faces** - StyleGAN against FFHQ, both distributed at
    256x256. Geometry-matched by construction, so it is the cleaner comparison,
    and it is also the likeliest thing an off-the-shelf ViT deepfake checkpoint
    was fine-tuned on. A great score here beside a poor one elsewhere is a
    leakage signature, not a capability.

Both protocols are reported for both tracks, as `evaluate_indian_faces` does:

  * ``frame`` - the naive protocol, whole source image into the shared capture
    path. Reported because it is what a careless evaluation would report.
  * ``face``  - both classes face-detected, cropped at the same margin and
    resized to the same pixel size before the identical capture simulation.
    **This is the number that describes the detector.**

Writes eval/real_faces.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import deepfakeface as dff  # noqa: E402
from evaluate_indian_faces import (  # noqa: E402
    CNN_WEIGHT,
    SPECTRAL_WEIGHT,
    geometry,
    normalise_face,
    score_group,
)
from verityne.config import DATASET_ROOT, EVAL_ROOT, MODEL_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("real_faces")

#: The policy's review and reject thresholds, plus the midpoint. Sharing these
#: with the rest of the project is the point - a recall quoted at an operating
#: point nothing ships at is a number about nothing.
OPERATING_POINTS = (0.4, 0.5, 0.75)

#: Face crop geometry for the controlled protocol, matched to
#: `evaluate_indian_faces` so the two reports are comparable.
CROP_MARGIN, CROP_SIZE = 0.4, 512

STYLEGAN_REPO = "JamieWithofs/Deepfake-and-real-images"
STYLEGAN_FILE = "data/test-00000-of-00001.parquet"
LFW_ROOT = DATASET_ROOT / "lfw" / "lfw_home" / "lfw_funneled"


# ------------------------------------------------------------------ loading
def load_stylegan(n: int, seed: int = 0) -> Tuple[List[Image.Image], List[Image.Image]]:
    """(real FFHQ, fake StyleGAN) from the 140k test split, balanced at `n` each."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(STYLEGAN_REPO, STYLEGAN_FILE, repo_type="dataset")
    pf = pq.ParquetFile(path)

    import io as _io

    real: List[Image.Image] = []
    fake: List[Image.Image] = []
    for batch in pf.iter_batches(batch_size=512, columns=["image", "label"]):
        d = batch.to_pydict()
        for rec, label in zip(d["image"], d["label"]):
            bucket = real if label == 1 else fake  # ClassLabel: 0=Fake, 1=Real
            if len(bucket) >= n:
                continue
            try:
                img = Image.open(_io.BytesIO(rec["bytes"]))
                img.load()
                bucket.append(img.convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                log.warning("stylegan row undecodable: %s", exc)
        if len(real) >= n and len(fake) >= n:
            break
    log.info("stylegan track: %d real, %d fake", len(real), len(fake))
    return real, fake


def load_lfw(n: int, seed: int = 0) -> List[Image.Image]:
    """A third real population, already on disk from the face-match calibration.

    LFW is web-scraped news photography: low resolution, heavily recompressed,
    nothing like FFHQ's curated crops or IMDB-WIKI's portraits. If the detector's
    behaviour on genuine faces is a property of the detector rather than of one
    dataset's encoder, it should hold here too.
    """
    if not LFW_ROOT.is_dir():
        log.warning("LFW not found at %s - skipping the third real population", LFW_ROOT)
        return []
    paths = sorted(LFW_ROOT.rglob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(paths)
    out: List[Image.Image] = []
    for p in paths[: n * 2]:
        if len(out) >= n:
            break
        try:
            img = Image.open(p)
            img.load()
            out.append(img.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: %s", p.name, exc)
    log.info("lfw: %d images", len(out))
    return out


# ------------------------------------------------------------------ scoring
def cropped(images: List[Image.Image]) -> Tuple[List[Image.Image], List[int], float]:
    """Face-detect and normalise a group; returns (crops, native widths, hit rate).

    Images with no detectable face are dropped rather than passed through
    uncropped: a mix of cropped and uncropped images inside one group would put
    the very artefact this protocol removes back into it.
    """
    crops: List[Image.Image] = []
    natives: List[int] = []
    for img in images:
        got = normalise_face(img, size=CROP_SIZE, margin=CROP_MARGIN)
        if got is None:
            continue
        crop, native = got
        crops.append(crop)
        natives.append(native)
    rate = len(crops) / max(1, len(images))
    return crops, natives, rate


def group_scores(rows: List[Dict], group: str) -> np.ndarray:
    return np.array([r["score"] for r in rows if r["group"] == group], dtype=float)


def family_report(real: np.ndarray, fake: np.ndarray, roc_auc_score) -> Dict:
    """Everything one generator family is worth against its real control."""
    if real.size == 0 or fake.size == 0:
        return {"n_real": int(real.size), "n_fake": int(fake.size), "auc": None}
    y = np.concatenate([np.zeros(real.size), np.ones(fake.size)])
    s = np.concatenate([real, fake])
    return {
        "n_real": int(real.size),
        "n_fake": int(fake.size),
        "auc": round(float(roc_auc_score(y, s)), 4),
        "mean_real": round(float(real.mean()), 4),
        "mean_fake": round(float(fake.mean()), 4),
        "recall_at": {str(t): round(float(np.mean(fake >= t)), 4) for t in OPERATING_POINTS},
        "real_fpr_at": {str(t): round(float(np.mean(real >= t)), 4) for t in OPERATING_POINTS},
    }


def run_protocol(groups: Dict[str, List[Image.Image]], protocol: str, tmp: Path,
                 seed: int) -> List[Dict]:
    """Score every group under one protocol, through the one shared capture path."""
    rows: List[Dict] = []
    for name, imgs in groups.items():
        if not imgs:
            continue
        log.info("[%s] scoring %s (n=%d)", protocol, name, len(imgs))
        rows += score_group(imgs, name, protocol, tmp, seed)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="images per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="parallel range fetches")
    ap.add_argument("--skip-stylegan", action="store_true",
                    help="skip the 140k track (avoids a 115 MB download)")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "real_faces.json")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    cache = DATASET_ROOT / "deepfakeface"
    tmp = MODEL_ROOT.parent / "tmp_collector" / "real_faces"
    tmp.mkdir(parents=True, exist_ok=True)

    active = (MODEL_ROOT / "active_detector.txt")
    report: Dict[str, object] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "detector": {
            "active_checkpoint": active.read_text().strip() if active.exists() else None,
            "cnn_weight": CNN_WEIGHT,
            "spectral_weight": SPECTRAL_WEIGHT,
            "note": "The shipped blend, so this measures what /verify runs, not one half of it.",
        },
        "what_this_measures": (
            "The selfie detector on fakes this project did not generate, reported per "
            "generator family. The corpus in eval/metrics.json cannot see this: its fakes "
            "come from our own generator settings, and benchmark_models.py selected this "
            "checkpoint on that same corpus."
        ),
        "protocols": {
            "frame": "Naive: whole source image into the shared capture path. Confounded "
                     "wherever the classes differ in resolution; reported because it is what "
                     "a careless evaluation would report.",
            "face": "Controlled: both classes face-detected, cropped at margin "
                    f"{CROP_MARGIN} and resized to {CROP_SIZE}px BEFORE the identical capture "
                    "simulation. This is the number that describes the detector.",
        },
        "operating_points": list(OPERATING_POINTS),
        "tracks": {},
    }
    tracks: Dict[str, Dict] = report["tracks"]  # type: ignore[assignment]

    # ------------------------------------------------------- DeepFakeFace
    log.info("fetching DeepFakeFace (%d paired keys)", args.n)
    keys = dff.paired_keys(args.n, seed=args.seed)
    paired = dff.load_paired(keys, cache, workers=args.workers)

    frame_groups = dict(paired)
    face_groups: Dict[str, List[Image.Image]] = {}
    detect_rate: Dict[str, float] = {}
    natives: Dict[str, List[int]] = {}
    for name, imgs in paired.items():
        crops, nat, rate = cropped(imgs)
        face_groups[name], detect_rate[name], natives[name] = crops, round(rate, 4), nat
        log.info("face crop %s: %d/%d (%.1f%%)", name, len(crops), len(imgs), rate * 100)

    dff_rows = (run_protocol(frame_groups, "frame", tmp, args.seed)
                + run_protocol(face_groups, "face", tmp, args.seed))

    dff_track: Dict[str, object] = {
        "source": "OpenRL/DeepFakeFace - IMDB-WIKI photographs and three fakes of each",
        "paired": True,
        "pairing_note": (
            "Every fake is derived from the real image it is scored against, so identity, "
            "pose and subject are held fixed. Geometry is NOT held fixed: the genuine images "
            "are native IMDB-WIKI sizes and every fake is 512x512, which is why the frame "
            "protocol here is not a measurement of the detector."
        ),
        "families": dff.GENERATOR_FAMILY,
        "face_detection_rate": detect_rate,
        "geometry": {n: geometry(imgs, natives.get(n, [])) for n, imgs in paired.items()},
        "protocols": {},
    }
    for protocol in ("frame", "face"):
        rows = [r for r in dff_rows if r["protocol"] == protocol]
        real = group_scores(rows, "real")
        dff_track["protocols"][protocol] = {  # type: ignore[index]
            "per_family": {
                fam: family_report(real, group_scores(rows, fam), roc_auc_score)
                for fam in dff.GENERATOR_FAMILY
            },
        }
    tracks["deepfakeface"] = dff_track

    # -------------------------------------------------------- 140k StyleGAN
    # This track is fetched over the network, and it runs *after* the expensive
    # DeepFakeFace scoring. An unguarded failure here would therefore throw away
    # a completed measurement to report a download error, so it is caught and
    # recorded as an absent track instead - the same treatment a missing dataset
    # gets everywhere else in this project.
    if not args.skip_stylegan:
        log.info("fetching the 140k StyleGAN track")
        try:
            sg_real, sg_fake = load_stylegan(args.n, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("stylegan track unavailable, continuing without it: %s", exc)
            sg_real, sg_fake = [], []
        if not sg_real or not sg_fake:
            tracks["stylegan_140k"] = {
                "source": f"{STYLEGAN_REPO} - not retrieved",
                "unavailable": True,
                "why": (
                    "The 140k track could not be fetched on this run. It is absent rather "
                    "than zero; every other track in this report stands on its own."
                ),
            }
            sg_real, sg_fake = [], []
        sg_frame = {"real": sg_real, "stylegan": sg_fake}
        sg_face: Dict[str, List[Image.Image]] = {}
        sg_rate: Dict[str, float] = {}
        for name, imgs in sg_frame.items():
            crops, _nat, rate = cropped(imgs)
            sg_face[name], sg_rate[name] = crops, round(rate, 4)

    if not args.skip_stylegan and sg_real and sg_fake:
        sg_rows = (run_protocol(sg_frame, "frame", tmp, args.seed)
                   + run_protocol(sg_face, "face", tmp, args.seed))
        sg_track: Dict[str, object] = {
            "source": f"{STYLEGAN_REPO} - FFHQ photographs against StyleGAN faces, test split",
            "paired": False,
            "pairing_note": (
                "Not paired, but both classes ship at 256x256, so this track is "
                "geometry-matched by construction and the frame protocol is closer to "
                "honest here than on DeepFakeFace."
            ),
            "leakage_warning": (
                "FFHQ-plus-StyleGAN is the most common fine-tuning set for off-the-shelf ViT "
                "deepfake checkpoints, and this one names no training data. A high score here "
                "cannot be distinguished from memorisation; read it beside the other families."
            ),
            "face_detection_rate": sg_rate,
            "protocols": {},
        }
        for protocol in ("frame", "face"):
            rows = [r for r in sg_rows if r["protocol"] == protocol]
            sg_track["protocols"][protocol] = {  # type: ignore[index]
                "per_family": {
                    "stylegan": family_report(group_scores(rows, "real"),
                                              group_scores(rows, "stylegan"), roc_auc_score)
                },
            }
        tracks["stylegan_140k"] = sg_track

    # ------------------------------------------------- real populations only
    lfw = load_lfw(args.n, seed=args.seed)
    if lfw:
        lfw_face, _n, lfw_rate = cropped(lfw)
        lfw_rows = (run_protocol({"lfw": lfw}, "frame", tmp, args.seed)
                    + run_protocol({"lfw": lfw_face}, "face", tmp, args.seed))
        tracks["real_populations"] = {
            "source": "LFW funneled - web-scraped news photography, already on disk",
            "why": (
                "A third genuine population, photographed nothing like FFHQ or IMDB-WIKI. "
                "The false-positive rate here is what an honest applicant pays, and it is "
                "the number a merchant feels."
            ),
            "face_detection_rate": {"lfw": round(lfw_rate, 4)},
            "false_positive_rate": {
                protocol: {
                    str(t): round(float(np.mean(group_scores(
                        [r for r in lfw_rows if r["protocol"] == protocol], "lfw") >= t)), 4)
                    for t in OPERATING_POINTS
                }
                for protocol in ("frame", "face")
            },
        }

    # ------------------------------------------------------------- headline
    face_aucs: Dict[str, Optional[float]] = {}
    for track in tracks.values():
        for fam, rep in (track.get("protocols", {}).get("face", {}) or {}).get("per_family", {}).items():
            face_aucs[fam] = rep.get("auc")
    scored = {k: v for k, v in face_aucs.items() if v is not None}
    if scored:
        worst = min(scored, key=lambda k: scored[k])
        best = max(scored, key=lambda k: scored[k])
        report["headline"] = {
            "protocol": "face",
            "per_family_auc": scored,
            "worst_family": worst,
            "worst_family_auc": scored[worst],
            "best_family": best,
            "best_family_auc": scored[best],
            "spread": round(scored[best] - scored[worst], 4),
            "note": (
                "The worst family is the deployment number: an attacker picks the generator, "
                "not the defender. The spread is how much a pooled average would have hidden."
            ),
        }

    report["rows"] = dff_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)

    head = report.get("headline")
    if isinstance(head, dict):
        for fam, auc in sorted(head["per_family_auc"].items(), key=lambda kv: kv[1]):
            log.info("  [face] %-12s AUC %.4f", fam, auc)
        log.info("worst family: %s at %.4f", head["worst_family"], head["worst_family_auc"])


if __name__ == "__main__":
    main()
