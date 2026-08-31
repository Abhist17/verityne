#!/usr/bin/env python3
"""Real photographs of Indian people, for a check the corpus cannot make.

`build_dataset.py` draws its *synthetic* faces from SD-Turbo with prompts that
are explicitly Indian - "a passport photograph of an indian man", "headshot
portrait of a south asian woman" - and its *real* faces from FFHQ, which is
Flickr photographs and is predominantly not South Asian.

That asymmetry is a demographic shortcut waiting to happen. A detector that
learned any part of "South Asian features" as evidence of synthesis would score
well on this corpus and reject Indian merchants in production, and no number in
`eval/metrics.json` could tell the two apart, because every genuine face in that
corpus is FFHQ.

So this module fetches real photographs of Indian people from a third-party
dataset - portrait framing, phone capture, plain backgrounds, which is close to
what a KYC selfie actually looks like - and `evaluate_indian_faces.py` scores
them beside FFHQ faces through an identical capture path.

Nothing is vendored: the shards are fetched to `datasets/indian_faces/`, which
is git-ignored, exactly as LFW and MIDV-2020 are.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Iterator, List

from PIL import Image

log = logging.getLogger("indian_faces")

#: Captioned portraits of Indian people. Chosen because the images are ordinary
#: phone photographs of ordinary people rather than web-scraped public figures,
#: which is the population a KYC queue actually sees.
REPO = "lokesh6309/indian_face-caption"
SHARD = "data/train-{:05d}-of-00014.parquet"
N_SHARDS = 14


def ensure_dataset(root: Path, shards: int = 2) -> List[Path]:
    """Download `shards` parquet shards, skipping any already on disk."""
    from huggingface_hub import hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for i in range(min(shards, N_SHARDS)):
        name = SHARD.format(i)
        local = root / Path(name).name
        if local.exists():
            log.info("have %s", local.name)
        else:
            log.info("fetching %s (~430 MB)", name)
            src = hf_hub_download(REPO, name, repo_type="dataset")
            local.write_bytes(Path(src).read_bytes())
        out.append(local)
    return out


def iter_faces(paths: List[Path], limit: int | None = None) -> Iterator[Image.Image]:
    """Deduplicated images, largest-first is not required - order is as stored.

    The shards contain exact duplicates (the same photograph captioned twice), so
    every image is hashed and repeats are dropped. Counting one face twice would
    quietly halve the effective sample size.
    """
    import pyarrow.parquet as pq

    seen: set[str] = set()
    n = 0
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64, columns=["image"]):
            for rec in batch.to_pylist():
                blob = (rec.get("image") or {}).get("bytes")
                if not blob:
                    continue
                digest = hashlib.sha256(blob).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                try:
                    yield Image.open(io.BytesIO(blob)).convert("RGB")
                except Exception:  # noqa: BLE001
                    continue
                n += 1
                if limit and n >= limit:
                    return
