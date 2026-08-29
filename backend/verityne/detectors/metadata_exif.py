"""Detector 5 - Metadata / EXIF Auditor.

Every check here is deterministic and costs microseconds. Individually each is
weak - plenty of honest submissions have stripped EXIF because a chat app
re-encoded the file. Together they are a strong prior, and they catch the lazy
end of the fraud market before a single tensor is allocated.

Each rule contributes a weighted, capped amount so no single soft signal (like
"no GPS") can push a submission into REJECT on its own.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import piexif
from PIL import Image

from ..schemas import DetectorOutput
from ..utils.images import is_probably_screenshot, load_rgb, upscale_evidence
from .base import Detector, SubmissionPayload, clamp

EDITOR_TOKENS = (
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator", "paint.net",
    "krita", "canva", "figma", "inkscape",
)
#: Fitted on the training split - see eval/calibration.json. Below the hard
#: threshold the file is almost certainly an enlargement of something smaller.
UPSCALE_HARD = 0.57
UPSCALE_SOFT = 0.62

GENERATOR_TOKENS = (
    "stable diffusion", "stablediffusion", "midjourney", "dall", "dalle", "comfyui",
    "automatic1111", "invokeai", "firefly", "novelai", "flux", "sdxl", "gan",
)


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore").strip("\x00 ").strip()
    return str(v).strip()


def read_exif(path: str | Path) -> Dict[str, object]:
    """Flatten the EXIF blocks we care about into a plain dict."""
    out: Dict[str, object] = {"has_exif": False, "raw": {}}
    try:
        img = Image.open(str(path))
        out["format"] = img.format
        out["width"], out["height"] = img.size
        out["mode"] = img.mode
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out

    # PNG/WebP text chunks are where diffusion UIs stash their generation prompts.
    png_text = {k.lower(): str(v)[:400] for k, v in (getattr(img, "text", {}) or {}).items()}
    if png_text:
        out["png_text"] = png_text

    try:
        exif = piexif.load(str(path))
    except Exception:
        return out

    tagmap = {"0th": piexif.ImageIFD, "Exif": piexif.ExifIFD, "GPS": piexif.GPSIFD}
    flat: Dict[str, object] = {}
    for block, ifd in tagmap.items():
        data = exif.get(block) or {}
        names = {v: k for k, v in vars(ifd).items() if isinstance(v, int)}
        for tag, val in data.items():
            name = names.get(tag, f"{block}_{tag}")
            if isinstance(val, (bytes, str)) and len(str(val)) < 500:
                flat[name] = _decode(val)
            elif isinstance(val, (int, float, tuple)):
                flat[name] = val
    out["has_exif"] = bool(flat)
    out["raw"] = flat
    out["gps_present"] = bool(exif.get("GPS"))
    return out


def _parse_exif_dt(value: object) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


class MetadataExifDetector(Detector):
    name = "metadata_exif"
    heavy = False

    def applicable(self, payload: SubmissionPayload) -> bool:
        return payload.selfie_path is not None or payload.id_doc_path is not None

    def _run(self, payload: SubmissionPayload) -> DetectorOutput:
        reasons: List[str] = []
        signals: Dict[str, object] = {}
        hits: List[Tuple[str, float]] = []

        targets = [("selfie", payload.selfie_path), ("id_document", payload.id_doc_path)]
        submitted_at = dt.datetime.fromtimestamp(payload.submitted_at) if payload.submitted_at else dt.datetime.now()

        for tag, path in targets:
            if path is None:
                continue
            meta = read_exif(path)
            signals[tag] = {k: v for k, v in meta.items() if k != "raw"}
            raw = meta.get("raw", {}) or {}
            signals[f"{tag}_exif"] = {k: str(v)[:120] for k, v in list(raw.items())[:24]}

            software = _decode(raw.get("Software", "")).lower()
            png_text = " ".join((meta.get("png_text") or {}).values()).lower()
            haystack = f"{software} {png_text}"

            for token in GENERATOR_TOKENS:
                if token in haystack:
                    hits.append((f"{tag}:generator_tag", 0.95))
                    reasons.append(f"The {tag} file carries generation metadata naming '{token}' - it was produced by an image generator")
                    break
            else:
                for token in EDITOR_TOKENS:
                    if token in haystack:
                        hits.append((f"{tag}:editor_tag", 0.55))
                        reasons.append(f"The {tag} was last written by image-editing software ('{_decode(raw.get('Software',''))}')")
                        break

            if not meta.get("has_exif") and not meta.get("png_text"):
                hits.append((f"{tag}:no_exif", 0.22))
                reasons.append(f"The {tag} has no camera metadata at all - consistent with a generated or re-saved file (also common after messaging apps)")

            fmt = str(meta.get("format", "")).upper()
            if fmt == "PNG" and tag == "selfie":
                hits.append((f"{tag}:png_selfie", 0.30))
                reasons.append("The selfie is a PNG; phone cameras save JPEG, so this file did not come straight off a camera")

            captured = _parse_exif_dt(raw.get("DateTimeOriginal") or raw.get("DateTime"))
            if captured:
                age_days = (submitted_at - captured).total_seconds() / 86400.0
                signals[f"{tag}_age_days"] = round(age_days, 2)
                if age_days > 30:
                    hits.append((f"{tag}:stale", min(0.6, 0.25 + age_days / 400.0)))
                    reasons.append(f"The {tag} was captured {age_days:.0f} days before submission - a live KYC capture should be minutes old")
                elif age_days < -1:
                    hits.append((f"{tag}:future", 0.5))
                    reasons.append(f"The {tag}'s capture timestamp is in the future relative to submission - metadata has been altered")

            make, model = _decode(raw.get("Make", "")), _decode(raw.get("Model", ""))
            if make or model:
                signals[f"{tag}_device"] = f"{make} {model}".strip()
                w, h = meta.get("width", 0), meta.get("height", 0)
                if w and h and max(w, h) < 900 and "iphone" in f"{make} {model}".lower():
                    hits.append((f"{tag}:device_mismatch", 0.45))
                    reasons.append(f"EXIF claims an {make} {model} but the image is only {w}x{h} - too small for that camera")

            if meta.get("has_exif") and not meta.get("gps_present") and tag == "selfie":
                hits.append((f"{tag}:no_gps", 0.12))

            try:
                rgb = payload.cache.get(f"{tag}_rgb")
                if rgb is None:
                    rgb = load_rgb(path)
                shot, why = is_probably_screenshot(rgb)
                if shot:
                    hits.append((f"{tag}:screenshot", 0.4))
                    reasons.append(f"The {tag} looks like a screenshot rather than a direct capture ({why})")

                # Provenance: does the file carry the detail its resolution claims?
                ev = upscale_evidence(rgb)
                signals[f"{tag}_provenance"] = ev
                scale = ev.get("effective_scale")
                if scale is not None and ev.get("nominal_px", 0) >= 400:
                    if scale < UPSCALE_HARD:
                        hits.append((f"{tag}:upscaled", 0.62))
                        reasons.append(
                            f"The {tag} is {ev['nominal_px']}px wide but carries detail only to about "
                            f"{ev['effective_px']}px - it was enlarged from a much smaller source, which is what "
                            "happens when an ID-card portrait or a screen capture is passed off as a fresh photo"
                        )
                    elif scale < UPSCALE_SOFT:
                        hits.append((f"{tag}:soft_detail", 0.24))
                        reasons.append(
                            f"The {tag} is softer than its {ev['nominal_px']}px resolution implies "
                            f"(real detail to ~{ev['effective_px']}px)"
                        )
            except Exception:
                pass

            try:
                signals[f"{tag}_bytes"] = os.path.getsize(path)
            except OSError:
                pass

        signals["rule_hits"] = [{"rule": r, "weight": round(w, 3)} for r, w in hits]

        # Noisy-OR: each independent hit erodes the "clean" probability, so several
        # weak signals compound without any single one dominating.
        clean = 1.0
        for _, w in hits:
            clean *= (1.0 - w)
        score = clamp(1.0 - clean)

        # Metadata is corroborating evidence, not proof. Cap its influence unless a
        # hard tell (an explicit generator tag) fired.
        hard = any(w >= 0.9 for _, w in hits)
        if not hard:
            score = min(score, 0.82)
        confidence = 0.9 if hard else (0.6 if hits else 0.5)

        if not reasons:
            reasons.append("Image metadata is internally consistent with a recent, unedited camera capture")

        return DetectorOutput(
            name=self.name, label=self.label, score=score, confidence=confidence,
            reasons=reasons, signals=signals,
        )
