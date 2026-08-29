"""Grad-CAM saliency for the deepfake classifier.

Works with both CNNs (hook the last conv block) and ViTs (hook the last encoder
block and reshape patch tokens back to a grid). The ViT path matters because the
best available pretrained detectors are ViT-based.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def _reshape_vit_tokens(t):
    """(B, tokens, C) -> (B, C, H, W), dropping the CLS token."""
    import torch

    if t.dim() != 3:
        return t
    b, n, c = t.shape
    n_patch = n - 1 if int((n - 1) ** 0.5) ** 2 == (n - 1) else n
    side = int(round(n_patch**0.5))
    if side * side != n_patch:
        return t
    tokens = t[:, n - n_patch:, :]
    return tokens.transpose(1, 2).reshape(b, c, side, side)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._acts = None
        self._grads = None
        self._handles = [
            target_layer.register_forward_hook(self._save_act),
            target_layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, _m, _i, out):
        self._acts = out[0] if isinstance(out, (tuple, list)) else out

    def _save_grad(self, _m, _gi, gout):
        self._grads = gout[0]

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, inputs, class_idx: Optional[int] = None) -> np.ndarray:
        import torch

        self.model.zero_grad(set_to_none=True)
        out = self.model(inputs)
        logits = out.logits if hasattr(out, "logits") else out
        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1)[0])
        logits[:, class_idx].sum().backward()

        acts, grads = self._acts, self._grads
        if acts is None or grads is None:
            return np.zeros((7, 7), dtype=np.float32)
        acts, grads = _reshape_vit_tokens(acts), _reshape_vit_tokens(grads)
        if acts.dim() != 4:
            return np.zeros((7, 7), dtype=np.float32)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))[0]
        cam = cam - cam.min()
        peak = float(cam.max())
        if peak > 1e-8:
            cam = cam / peak
        return cam.detach().float().cpu().numpy()


def find_target_layer(model) -> Optional[object]:
    """Best-effort: last ViT encoder block, else the last Conv2d in the graph."""
    import torch.nn as nn

    for attr in ("vit", "deit", "beit", "swin"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "encoder"):
            layers = getattr(sub.encoder, "layer", None) or getattr(sub.encoder, "layers", None)
            if layers is not None and len(layers):
                return layers[-1]
    convs: List = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1] if convs else None
