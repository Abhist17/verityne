"""Lazily-loaded, process-local model singletons.

Everything here is loaded on first use and cached. If a model cannot be loaded
(no weights, no network, no GPU) the accessor returns None and the calling
detector degrades to its classical-signal path instead of failing.
"""
from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Optional

import numpy as np

from ..config import MODEL_ROOT, resolve_device

log = logging.getLogger("verityne.models")

os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

#: Guards *construction* of the singletons below, so two threads racing to warm
#: the same model do not both build it.
_LOCK = threading.Lock()

# Candidate deepfake classifiers, tried in order. `scripts/benchmark_models.py`
# measures these on our own eval set and writes the winner to models/active_detector.txt.
DEEPFAKE_CANDIDATES = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "dima806/deepfake_vs_real_image_detection",
]

ACTIVE_MARKER = MODEL_ROOT / "active_detector.txt"


def _preferred_repo() -> list[str]:
    if ACTIVE_MARKER.exists():
        pinned = ACTIVE_MARKER.read_text().strip()
        if pinned:
            return [pinned] + [r for r in DEEPFAKE_CANDIDATES if r != pinned]
    return list(DEEPFAKE_CANDIDATES)


class DeepfakeClassifier:
    """Wraps a HF image-classification checkpoint and exposes P(fake) + Grad-CAM."""

    def __init__(self, repo_id: str):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        # One lock per model object, held across every forward pass.
        #
        # These singletons are shared by a ThreadPoolExecutor - `pipeline.py`
        # runs stage one concurrently and `scripts/score_corpus.py` scores
        # several packets at once - and a HuggingFace processor plus a torch
        # module are not safe to call concurrently on one instance. Left
        # unguarded this corrupts the heap: scoring the corpus aborted with
        # "double free or corruption" and "corrupted size vs. prev_size" partway
        # through, at both two and four workers, non-deterministically.
        #
        # The lock is per model rather than global on purpose. Serialising one
        # model's forward passes is what fixes the crash; OCR, face embedding
        # and the deepfake head can still overlap with each other, which is
        # where the concurrency actually pays.
        self._lock = threading.Lock()
        self.repo_id = repo_id
        self.device = resolve_device()
        self.processor = AutoImageProcessor.from_pretrained(repo_id)
        self.model = AutoModelForImageClassification.from_pretrained(repo_id).to(self.device).eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.fake_index = self._resolve_fake_index()

    def _resolve_fake_index(self) -> int:
        for idx, name in self.id2label.items():
            low = name.lower()
            if any(tok in low for tok in ("fake", "deepfake", "ai", "synth", "spoof")):
                return idx
        # Two-class model with unhelpful labels: assume index 1 is the positive class.
        return 1 if len(self.id2label) > 1 else 0

    def predict(self, rgb: np.ndarray) -> float:
        import torch

        with self._lock, torch.no_grad():
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1)[0]
        return float(probs[self.fake_index])

    def predict_batch(self, images: list[np.ndarray]) -> list[float]:
        import torch

        if not images:
            return []
        with self._lock, torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**inputs).logits.float(), dim=-1)
        return [float(p[self.fake_index]) for p in probs]

    def saliency(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Grad-CAM over the 'fake' logit. None if hooking the graph fails."""
        from ..utils.gradcam import GradCAM, find_target_layer

        layer = find_target_layer(self.model)
        if layer is None:
            return None
        try:
            with self._lock:
                inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
                pixel_values = inputs["pixel_values"].requires_grad_(True)
                with GradCAM(self.model, layer) as cam:
                    return cam(pixel_values, class_idx=self.fake_index)
        except Exception as exc:  # noqa: BLE001
            log.warning("grad-cam failed: %s", exc)
            return None


@functools.lru_cache(maxsize=1)
def deepfake_classifier() -> Optional[DeepfakeClassifier]:
    with _LOCK:
        for repo in _preferred_repo():
            try:
                clf = DeepfakeClassifier(repo)
                log.info("loaded deepfake classifier %s on %s", repo, clf.device)
                return clf
            except Exception as exc:  # noqa: BLE001
                log.warning("could not load %s: %s", repo, exc)
        return None


class FaceEmbedder:
    """512-d FaceNet (VGGFace2) embeddings via facenet-pytorch."""

    def __init__(self):
        import torch
        from facenet_pytorch import InceptionResnetV1

        self._lock = threading.Lock()  # see DeepfakeClassifier.__init__
        self.device = resolve_device()
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        face = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(face).permute(2, 0, 1).float()
        t = (t - 127.5) / 128.0  # facenet's expected normalisation
        with self._lock, torch.no_grad():
            v = self.model(t.unsqueeze(0).to(self.device))[0]
        v = v.detach().float().cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-9)


@functools.lru_cache(maxsize=1)
def face_embedder() -> Optional[FaceEmbedder]:
    with _LOCK:
        try:
            return FaceEmbedder()
        except Exception as exc:  # noqa: BLE001
            log.warning("face embedder unavailable: %s", exc)
            return None


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


@functools.lru_cache(maxsize=1)
def spectral_scorer():
    """Optional logistic-regression head over spectral features (trained by scripts/train_fusion.py)."""
    path = MODEL_ROOT / "spectral_lr.joblib"
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def fingerprint_model():
    """Optional multiclass head that names the generator behind a fake."""
    path = MODEL_ROOT / "generator_fingerprint.joblib"
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


def warmup() -> dict:
    """Touch every model once so the first real request isn't paying load time."""
    status = {}
    clf = deepfake_classifier()
    status["deepfake_classifier"] = clf.repo_id if clf else None
    emb = face_embedder()
    status["face_embedder"] = "facenet-vggface2" if emb else None
    status["spectral_scorer"] = bool(spectral_scorer())
    status["fingerprint_model"] = bool(fingerprint_model())
    status["device"] = resolve_device()
    return status
