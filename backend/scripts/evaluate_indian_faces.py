#!/usr/bin/env python3
"""Does the selfie detector treat Indian faces differently from FFHQ faces?

    python backend/scripts/evaluate_indian_faces.py --n 300

This exists because of an asymmetry in how the corpus was built, which nothing
in `eval/metrics.json` can see. Its synthetic faces are generated from prompts
that name the demographic - "a passport photograph of an indian man", "headshot
portrait of a south asian woman" - while its genuine faces come from FFHQ, which
is Flickr photographs and is predominantly not South Asian.

If the detector picked up any part of that as evidence, it would score well on
the corpus and reject Indian merchants in production. For a payments platform
onboarding in India that is not a subtle failure mode, it is the whole product
failing on its actual users, and the corpus is structurally incapable of
reporting it.

The measurement: real photographs of Indian people from a third-party dataset,
scored beside real FFHQ photographs. Both classes are genuine, so a detector
that separates them at all is reading something other than synthesis.

**Two populations photographed differently are not a demographic comparison.**
The Indian shards are full-frame phone portraits, around 2268x4032; FFHQ is
512x512 aligned face crops. Feed both to a frequency-domain head and it can
separate them on resampling history alone, without ever looking at a face. So
this script runs the comparison twice:

  * ``frame`` - the naive protocol, whole source image into the shared capture
    path. This is what the corpus itself would do, and it is reported precisely
    because it is confounded.
  * ``face``  - the controlled protocol. Both populations are face-detected,
    cropped at the same margin and resized to the same pixel size *before* the
    identical capture simulation, so the two groups enter the detector as the
    same kind of image and the faces are what is left to differ.

The gap between them is the point. Whatever ``frame`` reports and ``face`` does
not was never demography.

Each protocol yields two numbers:

  * The **shortcut AUC**: how well the detector's fake-score separates
    real-Indian from real-FFHQ. Both groups are genuine, so the only honest
    answer is 0.5. Anything above it is the detector reading something that is
    not synthesis.
  * The **false-positive rate on real Indian faces** at the same operating
    points used elsewhere - what fraction of honest Indian applicants this
    detector calls synthetic.

Writes eval/indian_faces.json.
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

from build_dataset import simulate_selfie_capture, write_image  # noqa: E402
from faces import load_real_faces  # noqa: E402
from indian_faces import REPO, ensure_dataset, iter_faces  # noqa: E402
from verityne.config import DATASET_ROOT, EVAL_ROOT  # noqa: E402
from verityne.detectors.models import deepfake_classifier  # noqa: E402
from verityne.detectors.selfie_deepfake import spectral_probability  # noqa: E402
from verityne.utils.images import crop_face, detect_faces, largest_face, load_rgb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("indian_eval")

#: The blend `SelfieDeepfakeDetector` ships, copied so this measures the shipped
#: score rather than one of its halves.
CNN_WEIGHT, SPECTRAL_WEIGHT = 0.62, 0.38


def normalise_face(img: Image.Image, size: int, margin: float,
                   detect_max_side: int = 1024) -> Optional[Tuple[Image.Image, int]]:
    """Crop to the face at a fixed margin and pixel size.

    Detection runs on a downscaled copy - MTCNN on a 4032px frame is slow and no
    more accurate - but the crop is taken from the *original* pixels and resized
    once. Detecting and cropping on the downscaled copy would upsample a small
    box back to `size`, which is the very artefact this control exists to remove.

    Returns the crop and the native (pre-resize) width of the box, so the report
    can show how much resampling each group actually took.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    scale = min(1.0, detect_max_side / max(h, w))
    small = rgb if scale >= 1.0 else np.asarray(
        img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS), dtype=np.uint8
    )
    boxes = detect_faces(small)
    if not boxes:
        return None
    x1, y1, x2, y2 = boxes[0]
    if scale < 1.0:  # map the box back onto the original pixels
        x1, y1, x2, y2 = (int(v / scale) for v in (x1, y1, x2, y2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    native = int((x2 - x1) * (1 + 2 * margin))
    return Image.fromarray(crop_face(rgb, (x1, y1, x2, y2), margin=margin, size=size)), native


def prepare(img: Image.Image, rng: random.Random, tmp: Path, idx: int, tag: str) -> np.ndarray:
    """One capture path for both groups.

    Both populations are resized, put through the same capture simulation and
    written at the same JPEG quality by the same function. Skip this and the
    comparison measures two encoders rather than two sets of faces - the same
    rule the real-document track holds itself to.
    """
    sim = simulate_selfie_capture(img, rng)
    path = tmp / f"{tag}_{idx:04d}.jpg"
    write_image(sim, path, "plain", rng, dt.datetime.now())
    return load_rgb(path)


def score_group(images: List[Image.Image], tag: str, protocol: str, tmp: Path, seed: int) -> List[Dict]:
    """Score one group under one protocol.

    Both RNGs are reseeded per group, not once per run. `simulate_selfie_capture`
    draws its sensor noise from the *global* numpy RNG, so without this the two
    groups get different noise and successive runs disagree - two n=25 runs moved
    the headline by 0.05 AUC before this was pinned. Reseeding here makes the run
    reproducible and, more importantly, makes the noise identical between the
    groups being compared, so it cannot be part of what separates them.
    """
    clf = deepfake_classifier()
    rng = random.Random(seed)
    np.random.seed(seed)
    out: List[Dict] = []
    for i, img in enumerate(images):
        rgb = prepare(img, rng, tmp, i, f"{protocol}_{tag}")
        face = largest_face(rgb, size=256)
        target = face if face is not None else rgb
        spec_p, _ = spectral_probability(target)
        cnn_p = float(clf.predict(target)) if clf is not None else None
        combined = (float(np.clip(CNN_WEIGHT * cnn_p + SPECTRAL_WEIGHT * spec_p, 0, 1))
                    if cnn_p is not None else spec_p)
        out.append({"group": tag, "protocol": protocol, "face_detected": face is not None,
                    "cnn_p_fake": None if cnn_p is None else round(cnn_p, 4),
                    "spectral_p_fake": round(spec_p, 4), "score": round(combined, 4)})
        if (i + 1) % 50 == 0:
            log.info("  [%s] scored %d/%d %s", protocol, i + 1, len(images), tag)
    return out


def summarise(rows: List[Dict], key: str) -> Dict:
    v = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
    if v.size == 0:
        return {}
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4), "std": round(float(v.std()), 4),
            "p90": round(float(np.percentile(v, 90)), 4)}


def geometry(images: List[Image.Image], natives: List[int]) -> Dict:
    """What the detector was actually handed, before anything normalised it."""
    wh = np.array([im.size for im in images], dtype=float)
    g: Dict[str, object] = {
        "median_source_px": [int(np.median(wh[:, 0])), int(np.median(wh[:, 1]))],
        "median_megapixels": round(float(np.median(wh[:, 0] * wh[:, 1]) / 1e6), 2),
    }
    if natives:
        g["median_native_face_crop_px"] = int(np.median(natives))
    return g


def protocol_report(rows: List[Dict], roc_auc_score) -> Dict:
    """Everything that can be said about one protocol's rows."""
    ind = [r for r in rows if r["group"] == "indian"]
    ffh = [r for r in rows if r["group"] == "ffhq"]
    y = np.array([1] * len(ind) + [0] * len(ffh))  # 1 = Indian, purely as a group label

    shortcut: Dict[str, float] = {}
    for key in ("score", "cnn_p_fake", "spectral_p_fake"):
        vals = [r[key] for r in ind + ffh if r[key] is not None]
        if len(vals) == len(rows):
            shortcut[key] = round(float(roc_auc_score(y, np.array(vals, dtype=float))), 4)

    fp = {
        str(t): {
            "indian": round(float(np.mean([r["score"] >= t for r in ind])), 4),
            "ffhq": round(float(np.mean([r["score"] >= t for r in ffh])), 4),
        }
        for t in (0.4, 0.5, 0.75)
    }
    return {
        "n_per_group": {"indian": len(ind), "ffhq": len(ffh)},
        "face_detection_rate": {
            "indian": round(sum(r["face_detected"] for r in ind) / max(1, len(ind)), 4),
            "ffhq": round(sum(r["face_detected"] for r in ffh) / max(1, len(ffh)), 4),
        },
        "shortcut_auc": shortcut,
        "scores": {
            "indian": {k: summarise(ind, k) for k in ("score", "cnn_p_fake", "spectral_p_fake")},
            "ffhq": {k: summarise(ffh, k) for k in ("score", "cnn_p_fake", "spectral_p_fake")},
        },
        "fraction_scored_above": fp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="faces per group")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--face-size", type=int, default=512,
                    help="common pixel size for the controlled protocol (FFHQ's native size)")
    ap.add_argument("--margin", type=float, default=0.4, help="crop margin around the face box")
    ap.add_argument("--data", type=Path, default=DATASET_ROOT / "indian_faces")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "indian_faces.json")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    shards = ensure_dataset(args.data, args.shards)
    indian = list(iter_faces(shards, limit=args.n))
    if len(indian) < 20:
        raise SystemExit(f"only {len(indian)} Indian faces available; need more shards")
    log.info("loaded %d real Indian faces", len(indian))

    ffhq = load_real_faces(len(indian), seed=args.seed)
    log.info("loaded %d real FFHQ faces", len(ffhq))

    tmp = args.data / "_scored"
    tmp.mkdir(parents=True, exist_ok=True)

    # Controlled protocol: both groups cropped to the face at one pixel size.
    norm: Dict[str, List[Image.Image]] = {}
    natives: Dict[str, List[int]] = {}
    for tag, imgs in (("indian", indian), ("ffhq", ffhq)):
        keep, px = [], []
        for im in imgs:
            got = normalise_face(im, args.face_size, args.margin)
            if got is not None:
                keep.append(got[0])
                px.append(got[1])
        norm[tag], natives[tag] = keep, px
        log.info("normalised %d/%d %s faces (median native crop %d px)",
                 len(keep), len(imgs), tag, int(np.median(px)) if px else 0)

    frame_rows = (score_group(indian, "indian", "frame", tmp, args.seed)
                  + score_group(ffhq, "ffhq", "frame", tmp, args.seed))
    face_rows = (score_group(norm["indian"], "indian", "face", tmp, args.seed)
                 + score_group(norm["ffhq"], "ffhq", "face", tmp, args.seed))

    frame = protocol_report(frame_rows, roc_auc_score)
    face = protocol_report(face_rows, roc_auc_score)

    report: Dict[str, object] = {
        "what_this_measures": (
            "Both groups are REAL photographs of real people. Neither is fraudulent. A "
            "detector that separates them is reading something that is not synthesis, and "
            "the only honest shortcut AUC is 0.5."
        ),
        "source_indian": f"{REPO} - phone portraits of Indian people",
        "source_control": "FFHQ (bitmind/ffhq-256), the genuine faces the corpus itself uses",
        "protocols": {
            "frame": (
                "Naive: the whole source image into the shared capture path. The two "
                "populations are not photographed alike, so this measures capture geometry "
                "as much as anything about the faces. Reported because it is what the "
                "corpus itself does."
            ),
            "face": (
                f"Controlled: both groups face-detected, cropped at margin {args.margin} and "
                f"resized to {args.face_size}px BEFORE the identical capture simulation, so "
                "the faces are what is left to differ. This is the defensible number."
            ),
        },
        "shared_capture_path": (
            "Identical capture simulation and JPEG encode for both groups, same face crop, "
            f"same CNN and spectral head, same {CNN_WEIGHT}/{SPECTRAL_WEIGHT} blend the "
            "detector ships."
        ),
        "source_geometry": {
            "indian": geometry(indian, natives["indian"]),
            "ffhq": geometry(ffhq, natives["ffhq"]),
            "note": (
                "The confound, stated in numbers: the two populations arrive at very "
                "different resolutions and framings. The controlled protocol exists to "
                "remove this; the residual is that FFHQ's crop is mildly upsampled to the "
                "common size while the Indian crop is downsampled to it."
            ),
        },
        "frame": frame,
        "face": face,
        "shortcut_auc_note": (
            "AUC of the detector's own fake-score at telling a real Indian face from a real "
            "FFHQ face. 0.5 is the only defensible value; above 0.5 means Indian faces are "
            "scored as more synthetic, below means less."
        ),
    }

    f_auc = frame["shortcut_auc"].get("score")
    c_auc = face["shortcut_auc"].get("score")
    if f_auc is not None and c_auc is not None:
        report["confound_share"] = {
            "frame_shortcut_auc": f_auc,
            "face_shortcut_auc": c_auc,
            "explained_by_capture_geometry": round(
                (abs(f_auc - 0.5) - abs(c_auc - 0.5)) / max(1e-9, abs(f_auc - 0.5)), 4
            ),
            "note": (
                "Share of the naive protocol's above-chance separation that disappears once "
                "both groups are cropped and sized alike. What is left is the part that "
                "survives a controlled comparison."
            ),
        }
    report["rows"] = frame_rows + face_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)
    for name, rep in (("frame", frame), ("face", face)):
        log.info("[%s] shortcut AUC (0.5 = no group signal): %s", name, rep["shortcut_auc"])
        for t, d in rep["fraction_scored_above"].items():
            log.info("  [%s] scored >= %s: indian %.1f%%  ffhq %.1f%%",
                     name, t, d["indian"] * 100, d["ffhq"] * 100)


if __name__ == "__main__":
    main()
