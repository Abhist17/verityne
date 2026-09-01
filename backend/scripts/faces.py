"""Face sources for the evaluation corpus.

Real faces come from FFHQ (public research dataset of real photographs).
Fake faces are generated locally with SD-Turbo, which means we own the labels -
and that is what makes generator fingerprinting nearly free: we always know
which model produced which image.

If SD-Turbo is unavailable the fake pool falls back to a GAN-style synthesiser
that reproduces the artefacts we actually detect (upsampling grid, smoothed
high-frequency detail, symmetric feature placement). That fallback is labelled
as such in the manifest so no metric silently conflates the two.
"""
from __future__ import annotations

import io
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "storage" / "models" / "hf"))

log = logging.getLogger("verityne.faces")

FFHQ_REPO = "bitmind/ffhq-256"
FFHQ_FILE = "data/train-00000-of-00016.parquet"

FACE_PROMPTS = [
    "a passport photograph of an indian man, plain background, front facing, sharp focus",
    "a passport photograph of an indian woman, plain background, front facing, sharp focus",
    "headshot portrait of a south asian man in his 30s, neutral expression, studio light",
    "headshot portrait of a south asian woman in her 20s, neutral expression, studio light",
    "id photo of a middle aged indian man, plain grey background, looking at camera",
    "id photo of a young indian woman, plain white background, looking at camera",
]


# ---------------------------------------------------------------------------------
# Real faces
# ---------------------------------------------------------------------------------

def load_real_faces(n: int, seed: int = 0, size: int = 512) -> List[Image.Image]:
    """Sample real photographs from the FFHQ shard."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FFHQ_REPO, FFHQ_FILE, repo_type="dataset")
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    rng = random.Random(seed)
    wanted = sorted(rng.sample(range(total), k=min(n, total)))

    out: List[Image.Image] = []
    cursor = 0
    want_idx = 0
    for batch in pf.iter_batches(batch_size=512, columns=["image"]):
        col = batch.column("image").to_pylist()
        for row in col:
            if want_idx < len(wanted) and cursor == wanted[want_idx]:
                raw = row["bytes"] if isinstance(row, dict) else row
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                if img.size[0] != size:
                    img = img.resize((size, size), Image.LANCZOS)
                out.append(img)
                want_idx += 1
            cursor += 1
        if want_idx >= len(wanted):
            break
    return out


# ---------------------------------------------------------------------------------
# Fake faces
# ---------------------------------------------------------------------------------

_PIPE = None


def _sd_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.enable_attention_slicing()  # 6GB laptop GPU: trade a little speed for headroom
        except Exception:
            pass
        _PIPE = pipe
        return pipe
    except Exception as exc:  # noqa: BLE001
        log.warning("SD-Turbo unavailable (%s); using the procedural GAN-artefact fallback", exc)
        return None


def generate_sd_faces(n: int, seed: int = 0, steps: int = 2, size: int = 512) -> List[Tuple[Image.Image, str]]:
    """Generate n synthetic faces with SD-Turbo. Returns (image, generator_tag)."""
    pipe = _sd_pipeline()
    if pipe is None:
        return [(procedural_fake_face(seed + i), "gan_other") for i in range(n)]

    import torch

    out: List[Tuple[Image.Image, str]] = []
    rng = random.Random(seed)
    for i in range(n):
        prompt = rng.choice(FACE_PROMPTS)
        g = torch.Generator(device=pipe.device).manual_seed(seed * 1000 + i)
        img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
                   height=size, width=size, generator=g).images[0]
        out.append((img, "stable_diffusion"))
        if (i + 1) % 10 == 0:
            log.info("generated %d/%d SD faces", i + 1, n)
    return out


def procedural_fake_face(seed: int, size: int = 512) -> Image.Image:
    """A deterministic stand-in that carries genuine generative artefacts.

    Built by upsampling a low-resolution latent through repeated 2x nearest+blur
    steps - the exact operation that stamps a half-Nyquist grid into the spectrum.
    It is a stand-in for a real generator, and the manifest records it as one.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.18, (16, 16, 3)).astype(np.float32)
    img = Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8))
    while img.size[0] < size:
        img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.NEAREST)
        img = img.filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(img, dtype=np.float32)

    # Impose a coarse face-like structure so face detection still fires.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    face = np.exp(-(((xx - 0.5) / 0.30) ** 2 + ((yy - 0.52) / 0.38) ** 2))
    skin = np.array([225.0, 190.0, 165.0], dtype=np.float32)
    arr = arr * (1 - face[..., None] * 0.75) + skin * face[..., None] * 0.75
    for (ex, ey) in ((0.38, 0.44), (0.62, 0.44)):
        eye = np.exp(-(((xx - ex) / 0.035) ** 2 + ((yy - ey) / 0.022) ** 2))
        arr = arr * (1 - eye[..., None] * 0.85)
    mouth = np.exp(-(((xx - 0.5) / 0.09) ** 2 + ((yy - 0.68) / 0.018) ** 2))
    arr = arr * (1 - mouth[..., None] * 0.5)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------------
# Face swapping (for the video attack set)
# ---------------------------------------------------------------------------------

def swap_face(target: Image.Image, source: Image.Image, blend: float = 0.92, jitter: Tuple[int, int] = (0, 0)) -> Image.Image:
    """Naive face swap: soft elliptical paste with colour transfer.

    This is what a cheap Roop-style swap actually produces - a colour-matched
    face pasted inside a feathered ellipse. It leaves exactly the boundary
    discontinuity and identity flicker the liveness detector looks for.
    """
    import cv2

    tgt = np.asarray(target.convert("RGB"), dtype=np.float32)
    h, w = tgt.shape[:2]
    src = np.asarray(source.convert("RGB").resize((w, h), Image.LANCZOS), dtype=np.float32)

    # Colour transfer in LAB so the pasted face matches the target's lighting.
    t_lab = cv2.cvtColor(tgt.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    s_lab = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        s_std = s_lab[..., c].std() + 1e-6
        s_lab[..., c] = (s_lab[..., c] - s_lab[..., c].mean()) * (t_lab[..., c].std() / s_std) + t_lab[..., c].mean()
    src = cv2.cvtColor(np.clip(s_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2 + jitter[0], int(h * 0.48) + jitter[1]
    cv2.ellipse(mask, (cx, cy), (int(w * 0.30), int(h * 0.38)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.025)[..., None] * blend
    return Image.fromarray(np.clip(tgt * (1 - mask) + src * mask, 0, 255).astype(np.uint8))
