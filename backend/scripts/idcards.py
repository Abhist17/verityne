"""Synthetic Indian ID card generator.

Hard ethical line: Verityne never trains on, stores, or evaluates against a real
citizen's identity document. Every card in this project is generated here, from
fictional names and structurally-valid-but-unissued numbers.

Two things matter for the eval set to be honest:

  1. Genuine and tampered cards go through the *same* capture simulation
     (perspective warp, lighting gradient, sensor noise, single JPEG pass). If
     genuine cards were pristine renders and fakes were JPEG-mangled, a detector
     would learn "is it a clean PNG" and report a fake 0.99 AUC.
  2. Tampering is applied the way a real forger does it - paste a region sourced
     from a differently-compressed image onto an already-compressed card, then
     re-save. That is what leaves a genuine ELA signature. Drawing the "tampered"
     text directly onto the render would leave none, and we would be grading
     ourselves on a fiction.
"""
from __future__ import annotations

import io
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/freefont",
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF",
]

CARD_W, CARD_H = 1012, 638  # ~CR80 at 300dpi-ish

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan", "Rohan",
    "Ananya", "Diya", "Aadhya", "Saanvi", "Myra", "Anika", "Navya", "Kiara", "Priya", "Meera",
    "Rahul", "Karthik", "Nikhil", "Farhan", "Imran", "Zoya", "Fatima", "Neha", "Sneha", "Pooja",
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Menon", "Bose", "Gupta", "Malhotra",
    "Chatterjee", "Desai", "Joshi", "Kulkarni", "Rao", "Singh", "Khan", "Pillai", "Banerjee", "Shetty",
]


def _font(names: List[str], size: int) -> ImageFont.FreeTypeFont:
    for d in FONT_DIRS:
        for n in names:
            p = Path(d) / n
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    continue
    return ImageFont.load_default(size=size)


def font_bold(size: int) -> ImageFont.FreeTypeFont:
    return _font(["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf", "Roboto-Bold.ttf"], size)


def font_regular(size: int) -> ImageFont.FreeTypeFont:
    return _font(["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf", "Roboto-Regular.ttf"], size)


def font_mono(size: int) -> ImageFont.FreeTypeFont:
    return _font(["DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "FreeMonoBold.ttf"], size)


# --------------------------------------------------------------------------------------
# Identity generation
# --------------------------------------------------------------------------------------

PAN_HOLDER_TYPES = "ABCFGHJLPT"


def make_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def make_pan(name: str, rng: random.Random, valid: bool = True) -> str:
    """Structurally valid PAN: AAAAA9999A, 4th char a holder-type code, 5th the surname initial.

    `valid=False` reproduces the mistakes open-source PAN generators actually make.
    """
    surname_initial = name.split()[-1][0].upper()
    if valid:
        head = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        return f"{head}P{surname_initial}{rng.randint(1000, 9999)}{rng.choice(string.ascii_uppercase)}"
    mode = rng.choice(["bad_holder", "bad_surname", "bad_shape"])
    head = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    if mode == "bad_holder":
        bad = rng.choice([c for c in string.ascii_uppercase if c not in PAN_HOLDER_TYPES])
        return f"{head}{bad}{surname_initial}{rng.randint(1000, 9999)}{rng.choice(string.ascii_uppercase)}"
    if mode == "bad_surname":
        wrong = rng.choice([c for c in string.ascii_uppercase if c != surname_initial])
        return f"{head}P{wrong}{rng.randint(1000, 9999)}{rng.choice(string.ascii_uppercase)}"
    return f"{head}P{surname_initial}{rng.randint(100, 999)}{rng.choice(string.ascii_uppercase)}"


def make_aadhaar(rng: random.Random, valid: bool = True) -> str:
    from verityne.utils.verhoeff import append_verhoeff

    body = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    if valid:
        return append_verhoeff(body)
    # Wrong check digit: exactly the failure a generated Aadhaar exhibits.
    correct = append_verhoeff(body)[-1]
    wrong = rng.choice([d for d in "0123456789" if d != correct])
    return body + wrong


def make_dob(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1965, 2003)}"


@dataclass
class Identity:
    name: str
    father_name: str
    dob: str
    pan: str
    aadhaar: str
    gender: str
    address: str


def make_identity(rng: random.Random, valid_numbers: bool = True) -> Identity:
    name = make_name(rng)
    return Identity(
        name=name,
        father_name=make_name(rng),
        dob=make_dob(rng),
        pan=make_pan(name, rng, valid=valid_numbers),
        aadhaar=make_aadhaar(rng, valid=valid_numbers),
        gender=rng.choice(["MALE", "FEMALE"]),
        address=f"{rng.randint(1, 400)}, {rng.choice(['MG Road','Nehru Nagar','Sector 12','Gandhi Path','Lake View'])}, "
                f"{rng.choice(['Bengaluru','Mumbai','Pune','Hyderabad','Jaipur','Kochi'])} - {rng.randint(400001, 700099)}",
    )


# --------------------------------------------------------------------------------------
# Card rendering
# --------------------------------------------------------------------------------------

def _guilloche(draw: ImageDraw.ImageDraw, w: int, h: int, colour, rng: random.Random) -> None:
    """The fine wavy security pattern printed on real cards."""
    for k in range(26):
        pts = []
        phase, amp = rng.uniform(0, 6.28), rng.uniform(6, 18)
        freq = rng.uniform(0.01, 0.03)
        y0 = h * k / 26.0
        for x in range(0, w, 6):
            pts.append((x, y0 + amp * np.sin(freq * x + phase)))
        draw.line(pts, fill=colour, width=1)


def render_pan_card(identity: Identity, photo: Optional[Image.Image], rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (CARD_W, CARD_H), (238, 240, 233))
    d = ImageDraw.Draw(img)
    _guilloche(d, CARD_W, CARD_H, (222, 227, 214), rng)

    d.rectangle([0, 0, CARD_W, 96], fill=(24, 61, 122))
    d.text((28, 20), "INCOME TAX DEPARTMENT", font=font_bold(30), fill=(255, 255, 255))
    d.text((28, 58), "GOVT. OF INDIA", font=font_regular(24), fill=(214, 224, 244))
    d.text((CARD_W - 250, 30), "भारत सरकार", font=font_regular(28), fill=(255, 255, 255))

    d.text((28, 126), "Permanent Account Number Card", font=font_regular(22), fill=(70, 78, 92))

    label, value = font_regular(21), font_bold(30)
    rows = [
        ("Name", identity.name),
        ("Father's Name", identity.father_name),
        ("Date of Birth", identity.dob),
    ]
    y = 190
    for lab, val in rows:
        d.text((28, y), lab, font=label, fill=(96, 104, 118))
        d.text((28, y + 26), val, font=value, fill=(18, 22, 30))
        y += 88

    d.text((28, y + 6), "Permanent Account Number", font=label, fill=(96, 104, 118))
    d.text((28, y + 32), identity.pan, font=font_mono(38), fill=(12, 16, 24))

    # Portrait well
    px, py, pw, ph = CARD_W - 268, 150, 220, 268
    d.rectangle([px - 3, py - 3, px + pw + 3, py + ph + 3], fill=(255, 255, 255), outline=(150, 158, 170), width=2)
    if photo is not None:
        img.paste(photo.resize((pw, ph), Image.LANCZOS), (px, py))
    else:
        d.rectangle([px, py, px + pw, py + ph], fill=(206, 212, 220))

    # Signature strip
    sy = py + ph + 26
    d.line([(px, sy + 34), (px + pw, sy + 34)], fill=(90, 98, 112), width=2)
    d.text((px, sy + 38), "Signature", font=font_regular(18), fill=(110, 118, 132))
    sig = [(px + 12 + i * 9, sy + 24 - int(14 * np.sin(i * 0.9 + rng.random()))) for i in range(22)]
    d.line(sig, fill=(20, 30, 90), width=3)
    return img


def render_aadhaar_card(identity: Identity, photo: Optional[Image.Image], rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (CARD_W, CARD_H), (253, 250, 244))
    d = ImageDraw.Draw(img)
    _guilloche(d, CARD_W, CARD_H, (243, 231, 219), rng)

    d.rectangle([0, 0, CARD_W, 88], fill=(255, 255, 255))
    d.text((110, 18), "भारत सरकार", font=font_regular(26), fill=(20, 24, 32))
    d.text((110, 50), "GOVERNMENT OF INDIA", font=font_bold(26), fill=(20, 24, 32))
    d.ellipse([26, 14, 92, 80], outline=(200, 90, 40), width=4)
    d.rectangle([0, 88, CARD_W, 94], fill=(226, 106, 44))

    px, py, pw, ph = 40, 130, 200, 246
    d.rectangle([px - 3, py - 3, px + pw + 3, py + ph + 3], fill=(255, 255, 255), outline=(170, 176, 186), width=2)
    if photo is not None:
        img.paste(photo.resize((pw, ph), Image.LANCZOS), (px, py))
    else:
        d.rectangle([px, py, px + pw, py + ph], fill=(210, 214, 220))

    tx = px + pw + 42
    d.text((tx, 140), identity.name, font=font_bold(34), fill=(16, 20, 28))
    d.text((tx, 190), f"DOB: {identity.dob}", font=font_regular(26), fill=(40, 46, 58))
    d.text((tx, 226), identity.gender, font=font_regular(26), fill=(40, 46, 58))
    d.text((tx, 272), identity.address[:44], font=font_regular(20), fill=(70, 76, 90))
    d.text((tx, 298), identity.address[44:88], font=font_regular(20), fill=(70, 76, 90))

    grouped = f"{identity.aadhaar[:4]} {identity.aadhaar[4:8]} {identity.aadhaar[8:]}"
    d.rectangle([0, CARD_H - 118, CARD_W, CARD_H], fill=(226, 106, 44))
    d.text((CARD_W // 2 - 200, CARD_H - 100), grouped, font=font_mono(52), fill=(255, 255, 255))
    d.text((CARD_W // 2 - 210, CARD_H - 40), "आधार - आम आदमी का अधिकार", font=font_regular(22), fill=(255, 238, 226))
    return img


# --------------------------------------------------------------------------------------
# Tampering + capture simulation
# --------------------------------------------------------------------------------------

def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def tamper_card(
    card: Image.Image, identity: Identity, rng: random.Random, kind: str, alt_photo: Optional[Image.Image] = None
) -> Tuple[Image.Image, Dict]:
    """Edit an already-compressed card the way a forger would, leaving a real ELA trail."""
    base = _jpeg_roundtrip(card, quality=94)  # the "original" the forger obtained
    info: Dict[str, object] = {"tamper_kind": kind}

    if kind == "name_swap":
        new_name = make_name(rng)
        box = (24, 210, 620, 258)
        patch = base.crop(box)
        # Re-compress the patch on its own so its history diverges from the page.
        patch = _jpeg_roundtrip(patch, quality=rng.choice([62, 70, 78]))
        pd = ImageDraw.Draw(patch)
        pd.rectangle([0, 0, patch.width, patch.height], fill=(240, 242, 236))
        pd.text((4, 2), new_name, font=font_bold(30), fill=(20, 24, 32))
        base.paste(patch, box[:2])
        info.update({"field": "name", "original": identity.name, "replacement": new_name})

    elif kind == "number_swap":
        box = (24, 452, 560, 512)
        patch = _jpeg_roundtrip(base.crop(box), quality=rng.choice([58, 68]))
        pd = ImageDraw.Draw(patch)
        pd.rectangle([0, 0, patch.width, patch.height], fill=(238, 240, 234))
        pd.text((4, 4), make_pan(identity.name, rng, valid=False), font=font_mono(38), fill=(12, 16, 24))
        base.paste(patch, box[:2])
        info.update({"field": "pan_number"})

    elif kind == "photo_swap" and alt_photo is not None:
        px, py, pw, ph = CARD_W - 268, 150, 220, 268
        swapped = _jpeg_roundtrip(alt_photo.resize((pw, ph), Image.LANCZOS), quality=rng.choice([55, 65, 75]))
        base.paste(swapped, (px, py))
        info.update({"field": "portrait"})

    elif kind == "dob_swap":
        box = (24, 386, 420, 434)
        patch = _jpeg_roundtrip(base.crop(box), quality=rng.choice([60, 72]))
        pd = ImageDraw.Draw(patch)
        pd.rectangle([0, 0, patch.width, patch.height], fill=(239, 241, 235))
        pd.text((4, 2), make_dob(rng), font=font_bold(30), fill=(18, 22, 30))
        base.paste(patch, box[:2])
        info.update({"field": "dob"})

    return base, info


def simulate_capture(card: Image.Image, rng: random.Random, strength: float = 1.0) -> Image.Image:
    """Make a rendered card look like a phone photo of a card.

    Applied identically to genuine and tampered cards - this is what stops the
    eval set from being trivially separable on capture artefacts alone.
    """
    import cv2

    arr = np.asarray(card.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # Mild perspective, as if held at an angle.
    j = 0.018 * strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(0, j) * w, rng.uniform(0, j) * h],
                      [w - rng.uniform(0, j) * w, rng.uniform(0, j) * h],
                      [w - rng.uniform(0, j) * w, h - rng.uniform(0, j) * h],
                      [rng.uniform(0, j) * w, h - rng.uniform(0, j) * h]])
    arr = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Uneven lighting from an off-axis light source.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    grad = 1.0 + 0.13 * strength * ((xx / w - 0.5) * gx + (yy / h - 0.5) * gy)
    vig = 1.0 - 0.16 * strength * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 3.2)
    arr *= (grad * vig)[..., None]

    arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=rng.uniform(0.4, 1.1) * strength)
    arr += np.random.normal(0, rng.uniform(1.6, 4.2) * strength, arr.shape).astype(np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    out = ImageEnhance.Color(out).enhance(rng.uniform(0.92, 1.08))
    return _jpeg_roundtrip(out, quality=rng.randint(78, 92))


def simulate_scan(card: Image.Image, rng: random.Random) -> Image.Image:
    """A flatbed scan or a direct digital upload: no perspective, light noise, one JPEG pass.

    This distinction matters for the eval. Error Level Analysis reads a splice
    from an image's compression history, and photographing a card through a phone
    camera re-encodes the whole frame and largely erases that history. Digital
    uploads - which are extremely common in real KYC flows - keep it. Reporting a
    single ELA number across both would hide where the technique does and does
    not work, so the corpus carries both populations and the report breaks them out.
    """
    import cv2

    arr = np.asarray(card.convert("RGB"), dtype=np.float32)
    arr += np.random.normal(0, rng.uniform(0.6, 1.8), arr.shape).astype(np.float32)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return _jpeg_roundtrip(out, quality=rng.randint(84, 95))


def build_card(
    identity: Identity,
    photo: Optional[Image.Image],
    rng: random.Random,
    doc_type: str = "PAN",
    tamper: Optional[str] = None,
    alt_photo: Optional[Image.Image] = None,
    capture: str = "photo",
) -> Tuple[Image.Image, Dict]:
    card = render_pan_card(identity, photo, rng) if doc_type == "PAN" else render_aadhaar_card(identity, photo, rng)
    meta: Dict[str, object] = {"doc_type": doc_type, "tampered": bool(tamper), "capture_mode": capture}
    if tamper:
        card, info = tamper_card(card, identity, rng, tamper, alt_photo=alt_photo)
        meta.update(info)
    rendered = simulate_scan(card, rng) if capture == "scan" else simulate_capture(card, rng)
    return rendered, meta
