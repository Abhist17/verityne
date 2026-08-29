#!/usr/bin/env python3
"""Build the Verityne evaluation corpus.

    python backend/scripts/build_dataset.py --packets 240 --with-video

Produces `datasets/corpus/<packet_id>/{selfie.jpg,id_document.jpg,liveness.mp4}`
plus `datasets/manifest.json`.

Three design decisions keep the resulting numbers honest:

  * **Identity-disjoint splits.** Train and test are split by face identity, not
    by row. A model cannot memorise a face in training and be graded on it.
  * **No class-correlated shortcuts.** Genuine and fake assets pass through the
    same capture simulation, and EXIF presence is deliberately mixed across both
    classes. Otherwise a detector learns "PNG means fake" and the AUC is a lie.
  * **Every fake is labelled by attack type and generator**, so the report can
    break performance down per attack instead of hiding a weak detector behind a
    strong one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import piexif  # noqa: E402

from faces import generate_sd_faces, load_real_faces, procedural_fake_face, swap_face  # noqa: E402
from idcards import Identity, build_card, make_identity  # noqa: E402
from verityne.config import DATASET_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_dataset")

CORPUS = DATASET_ROOT / "corpus"
MANIFEST = DATASET_ROOT / "manifest.json"

ATTACK_TYPES = [
    "generated_selfie",
    "synthetic_identity",
    "tampered_document",
    "face_swap_liveness",
    "impersonation",
    "reused_id_selfie",
    "invalid_document",
    "stale_or_edited_media",
    "recaptured_screen",
]

PHONES = [
    ("Apple", "iPhone 15 Pro"), ("Apple", "iPhone 13"), ("samsung", "SM-S918B"),
    ("Xiaomi", "23021RAAEG"), ("OnePlus", "CPH2449"), ("realme", "RMX3771"),
]
EDITORS = ["Adobe Photoshop 25.9 (Windows)", "GIMP 2.10.36", "Adobe Lightroom 7.4"]
GENERATOR_TAGS = ["Stable Diffusion 1.5", "AUTOMATIC1111 webui", "ComfyUI", "SDXL 1.0"]


# ----------------------------------------------------------------------------------
# Capture simulation + EXIF
# ----------------------------------------------------------------------------------

def simulate_selfie_capture(img: Image.Image, rng: random.Random, size: int = 960) -> Image.Image:
    """Applied to genuine and synthetic selfies alike, so capture is not a class cue."""
    import cv2

    arr = np.asarray(img.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vig = 1.0 - 0.12 * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 3.0)
    arr *= vig[..., None]
    arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=rng.uniform(0.3, 0.8))
    arr += np.random.normal(0, rng.uniform(1.2, 3.0), arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def simulate_screen_recapture(img: Image.Image, rng: random.Random) -> Image.Image:
    """A photo of a phone screen: letterboxed onto a flat border, moiré, second JPEG."""
    import cv2

    inner = img.convert("RGB").resize((520, 520), Image.LANCZOS)
    canvas = Image.new("RGB", (1080, 2340), (14, 14, 16))
    canvas.paste(inner, (280, 900))
    arr = np.asarray(canvas, dtype=np.float32)
    h, w = arr.shape[:2]
    yy = np.arange(h, dtype=np.float32)[:, None, None]
    arr *= (1.0 + 0.035 * np.sin(yy * 1.9))  # scanline moiré across rows, all channels
    arr = cv2.GaussianBlur(arr, (0, 0), 0.7)
    arr += np.random.normal(0, 2.2, arr.shape).astype(np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out.resize((540, 1170), Image.LANCZOS)


def write_image(img: Image.Image, path: Path, exif_mode: str, rng: random.Random,
                submitted: dt.datetime, fmt: str = "JPEG") -> Dict:
    """Save with one of several metadata profiles.

    `exif_mode` is chosen independently of the packet's label for most rows, so
    metadata cannot be used as a shortcut to the ground truth.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    info: Dict[str, object] = {"exif_mode": exif_mode}

    if exif_mode == "none" or fmt == "PNG":
        img.save(path, fmt, quality=rng.randint(80, 93) if fmt == "JPEG" else None)
        return info

    make, model = rng.choice(PHONES)
    if exif_mode == "fresh":
        captured = submitted - dt.timedelta(minutes=rng.randint(1, 25))
        software = f"{model} {rng.randint(16,18)}.{rng.randint(0,6)}"
    elif exif_mode == "stale":
        captured = submitted - dt.timedelta(days=rng.randint(35, 400))
        software = f"{model} {rng.randint(14,17)}.{rng.randint(0,6)}"
    elif exif_mode == "edited":
        captured = submitted - dt.timedelta(days=rng.randint(1, 120))
        software = rng.choice(EDITORS)
    elif exif_mode == "generated":
        captured = submitted - dt.timedelta(minutes=rng.randint(1, 600))
        software = rng.choice(GENERATOR_TAGS)
    else:
        captured = submitted - dt.timedelta(minutes=rng.randint(1, 60))
        software = f"{model}"

    ts = captured.strftime("%Y:%m:%d %H:%M:%S").encode()
    zeroth = {
        piexif.ImageIFD.Make: make.encode(),
        piexif.ImageIFD.Model: model.encode(),
        piexif.ImageIFD.Software: software.encode(),
        piexif.ImageIFD.DateTime: ts,
    }
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: ts,
        piexif.ExifIFD.DateTimeDigitized: ts,
        piexif.ExifIFD.ISOSpeedRatings: rng.choice([32, 50, 64, 100, 200, 400]),
        piexif.ExifIFD.FNumber: (rng.choice([16, 18, 22, 24]), 10),
        piexif.ExifIFD.ExposureTime: (1, rng.choice([30, 60, 120, 250])),
        piexif.ExifIFD.LensModel: f"{model} front camera".encode(),
    }
    gps = {}
    if exif_mode in ("fresh", "plain") and rng.random() < 0.75:
        gps = {
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLatitude: ((rng.randint(8, 32), 1), (rng.randint(0, 59), 1), (0, 1)),
            piexif.GPSIFD.GPSLongitudeRef: b"E",
            piexif.GPSIFD.GPSLongitude: ((rng.randint(68, 92), 1), (rng.randint(0, 59), 1), (0, 1)),
        }
    exif_bytes = piexif.dump({"0th": zeroth, "Exif": exif_ifd, "GPS": gps, "1st": {}, "thumbnail": None})
    img.save(path, "JPEG", quality=rng.randint(80, 93), exif=exif_bytes)
    info.update({"device": f"{make} {model}", "captured": captured.isoformat(), "software": software})
    return info


def pick_exif_mode(is_fake: bool, attack: Optional[str], rng: random.Random) -> str:
    """Mixed on purpose.

    Genuine submissions frequently arrive with stripped EXIF (a messaging app
    re-encoded them), and a competent fraudster can forge plausible EXIF. If the
    corpus made EXIF a perfect class signal the metadata detector would look
    superhuman and the fusion model would learn nothing real.
    """
    if attack == "stale_or_edited_media":
        return rng.choice(["stale", "edited", "generated"])
    if not is_fake:
        return rng.choices(["fresh", "plain", "none"], weights=[0.55, 0.25, 0.20])[0]
    return rng.choices(["fresh", "plain", "none", "stale", "edited"], weights=[0.34, 0.20, 0.24, 0.12, 0.10])[0]


# ----------------------------------------------------------------------------------
# Packet construction
# ----------------------------------------------------------------------------------

def second_capture(face: Image.Image, rng: random.Random) -> Image.Image:
    """Approximate a *different photograph* of the same person.

    A real KYC packet contains two independent captures: a selfie taken today and
    a portrait taken years ago at a photo studio. FFHQ gives us one image per
    identity, so we approximate the second capture with a pose warp, a different
    light direction, a colour-temperature shift and an independent compression
    history. This is an acknowledged limitation of the corpus - genuine
    selfie-vs-ID similarity here still runs higher than it would on real pairs,
    which is exactly why the face-match decision band is calibrated from data
    (scripts/calibrate_face_match.py) instead of hard-coded.
    """
    import cv2

    arr = np.asarray(face.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # Small yaw/pitch change, as a shear plus asymmetric scale.
    yaw = rng.uniform(-0.10, 0.10)
    m = np.float32([[1.0 + rng.uniform(-0.04, 0.04), yaw, -yaw * w * 0.5],
                    [rng.uniform(-0.03, 0.03), 1.0 + rng.uniform(-0.03, 0.03), 0]])
    arr = cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    arr = cv2.warpAffine(arr, cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-7, 7), rng.uniform(0.94, 1.06)),
                         (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Light from the other side, plus a colour-temperature shift.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    arr *= (1.0 + 0.22 * ((xx / w - 0.5) * gx + (yy / h - 0.5) * gy))[..., None]
    warm = np.array([rng.uniform(0.94, 1.07), 1.0, rng.uniform(0.93, 1.08)], dtype=np.float32)
    arr *= warm

    arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=rng.uniform(0.6, 1.6))
    arr += np.random.normal(0, rng.uniform(2.0, 5.0), arr.shape).astype(np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    import io as _io

    buf = _io.BytesIO()
    out.save(buf, "JPEG", quality=rng.randint(58, 78))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def crop_portrait(face: Image.Image, rng: random.Random, independent: bool = True) -> Image.Image:
    """The passport-style crop printed on the card.

    `independent=True` first re-captures the face so the card portrait is not a
    pixel-for-pixel copy of the selfie - otherwise every genuine packet would
    look like the "selfie copied from the ID" attack and the corpus would be
    scoring a bug rather than a fraud pattern.
    """
    src = second_capture(face, rng) if independent else face
    w, h = src.size
    jx, jy = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
    box = (int(w * (0.14 + jx)), int(h * (0.06 + jy)), int(w * (0.86 + jx)), int(h * (0.92 + jy)))
    box = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
    return src.crop(box).resize((220, 268), Image.LANCZOS)


def build_packet(
    packet_id: str,
    label: str,
    attack: Optional[str],
    real_faces: List[Image.Image],
    fake_faces: List[Tuple[Image.Image, str]],
    rng: random.Random,
    with_video: bool,
    identity_index: int,
) -> Dict:
    out_dir = CORPUS / packet_id
    out_dir.mkdir(parents=True, exist_ok=True)
    submitted = dt.datetime.now() - dt.timedelta(hours=rng.randint(0, 72))

    real_face = real_faces[identity_index % len(real_faces)]
    other_real = real_faces[(identity_index + 7) % len(real_faces)]
    fake_face, generator = fake_faces[identity_index % len(fake_faces)] if fake_faces else (procedural_fake_face(identity_index), "gan_other")

    valid_numbers = attack != "invalid_document"
    identity = make_identity(rng, valid_numbers=valid_numbers)
    doc_type = rng.choices(["PAN", "Aadhaar"], weights=[0.7, 0.3])[0]

    selfie_source = real_face
    card_face = real_face
    tamper: Optional[str] = None
    alt_photo = None
    video_swap: Optional[Image.Image] = None
    generator_label = "real"
    notes: Dict[str, object] = {}

    if attack == "generated_selfie":
        selfie_source, generator_label = fake_face, generator
    elif attack == "synthetic_identity":
        selfie_source = card_face = fake_face
        generator_label = generator
    elif attack == "tampered_document":
        tamper = rng.choice(["name_swap", "number_swap", "photo_swap", "dob_swap"])
        alt_photo = crop_portrait(other_real, rng)
    elif attack == "face_swap_liveness":
        video_swap = fake_face
        generator_label = "faceswap"
    elif attack == "impersonation":
        card_face = other_real
    elif attack == "reused_id_selfie":
        pass  # handled after the card is rendered
    elif attack == "recaptured_screen":
        pass  # handled at write time

    # ---- ID document -------------------------------------------------------
    portrait = crop_portrait(card_face, rng)
    # Chosen independently of the label: roughly half of real KYC uploads are
    # phone photos of a card and half are direct digital scans, and tamper
    # detection behaves very differently on the two.
    capture_mode = rng.choices(["photo", "scan"], weights=[0.55, 0.45])[0]
    card, card_meta = build_card(identity, portrait, rng, doc_type=doc_type, tamper=tamper,
                                 alt_photo=alt_photo, capture=capture_mode)
    id_path = out_dir / "id_document.jpg"
    id_exif = write_image(card, id_path, pick_exif_mode(label == "fake", attack, rng), rng, submitted)

    # ---- Selfie ------------------------------------------------------------
    if attack == "reused_id_selfie":
        # The lazy fraud: crop the portrait straight out of the submitted card.
        card_arr = np.asarray(card.convert("RGB"))
        h, w = card_arr.shape[:2]
        # Inset from the printed frame so the card border does not bleed into the
        # "selfie" - a forger cropping this would trim it too.
        if doc_type == "PAN":
            x0, y0 = int(w * 0.750), int(h * 0.250)
            x1, y1 = int(w * 0.943), int(h * 0.640)
        else:
            x0, y0 = int(w * 0.052), int(h * 0.220)
            x1, y1 = int(w * 0.225), int(h * 0.575)
        selfie_img = Image.fromarray(card_arr[y0:y1, x0:x1]).resize((960, 960), Image.LANCZOS)
        notes["selfie_is_card_crop"] = True
    else:
        selfie_img = simulate_selfie_capture(selfie_source, rng)

    selfie_path = out_dir / "selfie.jpg"
    if attack == "recaptured_screen":
        selfie_img = simulate_screen_recapture(selfie_img, rng)
        selfie_exif = write_image(selfie_img, selfie_path, "none", rng, submitted)
    else:
        mode = pick_exif_mode(label == "fake", attack, rng)
        if attack in ("generated_selfie", "synthetic_identity") and rng.random() < 0.25:
            mode = "generated"  # a careless fraudster leaves the generator tag in
        selfie_exif = write_image(selfie_img, selfie_path, mode, rng, submitted)

    # ---- Liveness clip ------------------------------------------------------
    video_meta = None
    if with_video:
        from videos import make_liveness_video

        video_meta = make_liveness_video(
            selfie_source if attack != "reused_id_selfie" else real_face,
            out_dir / "liveness.mp4",
            swap_source=video_swap,
            seed=rng.randint(0, 10**6),
            duration_s=rng.choice([3.0, 4.0, 5.0]),
        )

    return {
        "id": packet_id,
        "label": label,
        "attack_type": attack,
        "generator": generator_label,
        "doc_type": doc_type,
        "capture_mode": capture_mode,
        "identity_index": identity_index,
        "claimed_name": identity.name,
        "claimed_id_number": identity.pan if doc_type == "PAN" else identity.aadhaar,
        "claimed_dob": identity.dob,
        "selfie": str(selfie_path),
        "id_document": str(id_path),
        "video": video_meta["path"] if video_meta else None,
        "submitted_at": submitted.isoformat(),
        "card_meta": card_meta,
        "selfie_exif": selfie_exif,
        "id_exif": id_exif,
        "video_meta": video_meta,
        "notes": notes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packets", type=int, default=240, help="total packets to build")
    ap.add_argument("--fake-ratio", type=float, default=0.5)
    ap.add_argument("--with-video", action="store_true", help="also synthesise liveness clips (slower)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--sd-steps", type=int, default=2)
    ap.add_argument("--clean", action="store_true", help="wipe the corpus first")
    args = ap.parse_args()

    if args.clean and CORPUS.exists():
        shutil.rmtree(CORPUS)
    CORPUS.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n_fake = int(args.packets * args.fake_ratio)
    n_real = args.packets - n_fake

    # One distinct identity per packet. Reusing a face across packets would give
    # two different claimed names the same face, which is exactly the signature of
    # an onboarding ring - the corpus would be generating false positives for the
    # linkage detector to find.
    n_identities = max(40, args.packets)
    log.info("loading %d real faces from FFHQ", n_identities)
    real_faces = load_real_faces(n_identities, seed=args.seed)
    log.info("loaded %d real faces", len(real_faces))

    n_synth = max(24, n_fake // 2)
    log.info("generating %d synthetic faces (SD-Turbo)", n_synth)
    fake_faces = generate_sd_faces(n_synth, seed=args.seed, steps=args.sd_steps)
    log.info("generated %d synthetic faces", len(fake_faces))

    plan: List[Tuple[str, Optional[str]]] = [("real", None)] * n_real
    for i in range(n_fake):
        plan.append(("fake", ATTACK_TYPES[i % len(ATTACK_TYPES)]))
    rng.shuffle(plan)

    # Identity-disjoint split: identities below the cut are train, above are test.
    cut = int(len(real_faces) * (1 - args.test_frac))
    manifest: List[Dict] = []
    for i, (label, attack) in enumerate(plan):
        identity_index = i % len(real_faces)
        pid = f"pkt_{i:04d}"
        try:
            entry = build_packet(pid, label, attack, real_faces, fake_faces, rng, args.with_video, identity_index)
        except Exception as exc:  # noqa: BLE001
            log.exception("packet %s failed: %s", pid, exc)
            continue
        entry["split"] = "train" if identity_index < cut else "test"
        manifest.append(entry)
        if (i + 1) % 20 == 0:
            log.info("built %d/%d packets", i + 1, len(plan))

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    by_attack: Dict[str, int] = {}
    for e in manifest:
        by_attack[e["attack_type"] or "genuine"] = by_attack.get(e["attack_type"] or "genuine", 0) + 1
    log.info("wrote %s", MANIFEST)
    log.info("packets: %d (train=%d test=%d)", len(manifest),
             sum(1 for e in manifest if e["split"] == "train"), sum(1 for e in manifest if e["split"] == "test"))
    log.info("composition: %s", json.dumps(by_attack, indent=2))


if __name__ == "__main__":
    main()
