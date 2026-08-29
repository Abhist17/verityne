#!/usr/bin/env python3
"""Benchmark candidate deepfake checkpoints on *our* data before trusting one.

    python backend/scripts/benchmark_models.py

Pretrained deepfake detectors are trained on one generator family and often
collapse on another - a DFDC-trained model sees face-swap video frames, not
diffusion output. Picking a checkpoint by reputation instead of measurement is
how a project ends up reporting a confident number for a model that is guessing.

This measures every candidate on the training split and pins the winner in
storage/models/active_detector.txt, which detectors/models.py reads at load time.
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

from verityne.config import DATASET_ROOT, EVAL_ROOT, MODEL_ROOT  # noqa: E402
from verityne.detectors.models import DEEPFAKE_CANDIDATES, DeepfakeClassifier  # noqa: E402
from verityne.detectors.selfie_deepfake import heuristic_spectral_score  # noqa: E402
from verityne.utils.images import largest_face, load_rgb  # noqa: E402
from verityne.utils.spectral import spectral_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark")


def load_faces(entries: List[Dict]) -> tuple[list, np.ndarray, list]:
    faces, y, attacks = [], [], []
    for e in entries:
        try:
            f = largest_face(load_rgb(e["selfie"]), size=256)
        except Exception:
            f = None
        if f is None:
            continue
        faces.append(f)
        y.append(1 if e.get("attack_type") in ("generated_selfie", "synthetic_identity") else 0)
        attacks.append(e.get("attack_type"))
    return faces, np.asarray(y, dtype=int), attacks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text())
    entries = [e for e in manifest if e.get("split") == args.split]
    if args.limit:
        entries = entries[: args.limit]

    faces, y, _ = load_faces(entries)
    log.info("benchmarking on %d selfie crops (%d synthetic)", len(faces), int(y.sum()))
    if len(set(y.tolist())) < 2:
        raise SystemExit("need both classes to benchmark")

    results: List[Dict] = []

    t0 = time.perf_counter()
    spec = np.asarray([heuristic_spectral_score(spectral_features(f)) for f in faces])
    results.append(
        {
            "model": "spectral_heuristic",
            "auc": round(float(roc_auc_score(y, spec)), 4),
            "ms_per_image": round((time.perf_counter() - t0) * 1000 / max(1, len(faces)), 2),
            "kind": "classical",
        }
    )

    for repo in DEEPFAKE_CANDIDATES:
        try:
            clf = DeepfakeClassifier(repo)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", repo, exc)
            results.append({"model": repo, "auc": None, "error": str(exc), "kind": "neural"})
            continue
        t0 = time.perf_counter()
        scores = []
        for i in range(0, len(faces), 16):
            scores.extend(clf.predict_batch(faces[i: i + 16]))
        elapsed = time.perf_counter() - t0
        arr = np.asarray(scores)
        auc = float(roc_auc_score(y, arr))
        results.append(
            {
                "model": repo,
                "auc": round(auc, 4),
                "ms_per_image": round(elapsed * 1000 / max(1, len(faces)), 2),
                "mean_real": round(float(arr[y == 0].mean()), 4),
                "mean_fake": round(float(arr[y == 1].mean()), 4),
                "labels": clf.id2label,
                "fake_index": clf.fake_index,
                "kind": "neural",
            }
        )
        log.info("%s: AUC=%.4f (real %.3f vs fake %.3f)", repo, auc, arr[y == 0].mean(), arr[y == 1].mean())

    scored = [r for r in results if r.get("auc") is not None and r["kind"] == "neural"]
    report = {"n_images": len(faces), "n_synthetic": int(y.sum()), "results": results}

    if scored:
        best = max(scored, key=lambda r: r["auc"])
        report["selected"] = best["model"]
        spectral_auc = results[0]["auc"]
        if best["auc"] < 0.6:
            report["warning"] = (
                f"Best neural checkpoint only reaches AUC {best['auc']:.3f} on this corpus - it is close to "
                f"guessing on these generators. The spectral head (AUC {spectral_auc:.3f}) is carrying this "
                "detector, and the ensemble weighting in selfie_deepfake.py reflects that."
            )
            log.warning(report["warning"])
        (MODEL_ROOT / "active_detector.txt").write_text(best["model"])
        log.info("pinned active detector: %s (AUC %.4f)", best["model"], best["auc"])

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "model_benchmark.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %s", EVAL_ROOT / "model_benchmark.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
