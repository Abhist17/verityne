#!/usr/bin/env bash
# Assemble the single-container Render deploy: dashboard + read-only API.
#
# The demo images are the bulk of the payload, so they are downscaled rather
# than dropped. At 640px and JPEG q72 the 989 files go from 150 MB to about
# 26 MB, which is still every thumbnail and every heatmap overlay - and those
# are what make the evidence pages worth looking at. Full resolution only
# matters if you are inspecting an artefact pixel by pixel, which is not what a
# reviewer is doing on a demo link.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/deploy/.build/render"
cd "$REPO"

[ -e storage/verityne.db ] || { echo "missing storage/verityne.db - run 'make gauntlet'" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/seed"

# Source the image builds from. The Dockerfile reads backend/, frontend/, eval/
# and deploy/ out of this tree, so the context is the repo itself; only the
# payload has to be assembled.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/storage/models"

# Only the fitted artefacts. `cp -r storage/models` would drag in models/hf,
# which is a 4 GB cache of upstream checkpoints - it filled the disk before this
# was a find rather than a recursive copy.
find storage/models -maxdepth 1 -type f -exec cp {} "$STAGE/storage/models/" \;
cp storage/verityne.db "$STAGE/storage/verityne.db"

echo "downscaling evidence images ..."
"$REPO/.venv/bin/python" - "$STAGE" <<'PY'
import io, pathlib, sys
from PIL import Image

stage = pathlib.Path(sys.argv[1])
kept = skipped = 0
for d in ("storage/uploads", "storage/heatmaps"):
    for src in pathlib.Path(d).rglob("*"):
        if not src.is_file():
            continue
        dst = stage / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            dst.write_bytes(src.read_bytes()); skipped += 1; continue
        try:
            im = Image.open(src)
            im.thumbnail((640, 640))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=72, optimize=True)
            # Keep the original name: the database and the verdict rows store
            # these paths, and a .png that is really a JPEG still decodes in
            # every browser. Renaming would mean rewriting stored evidence.
            dst.write_bytes(buf.getvalue()); kept += 1
        except Exception:
            dst.write_bytes(src.read_bytes()); skipped += 1
print(f"  {kept} downscaled, {skipped} copied as-is")
PY

# The Dockerfile builds with the repo as its context, so the payload has to
# live inside the repo. It is gitignored: it is a build output.
mkdir -p "$REPO/seed"
tar -czf "$REPO/seed/demo-data-render.tar.gz" -C "$STAGE" storage
rmdir "$OUT/seed" "$OUT" 2>/dev/null || true

subs=$(python3 -c "
import sqlite3
try: print(sqlite3.connect('$REPO/storage/verityne.db').execute('select count(*) from submissions').fetchone()[0])
except Exception: print('?')
")
cat <<MSG

Payload: seed/demo-data-render.tar.gz  ($(du -h "$REPO/seed/demo-data-render.tar.gz" | cut -f1), $subs submissions)

Build and run it, from the repo root:

  docker build -f deploy/render/Dockerfile -t verityne:render .
  docker run -e PORT=10000 -p 10000:10000 verityne:render

Deploy: push to GitHub, then on Render choose New > Blueprint and point it at
this repo. deploy/render/render.yaml describes the service; nothing to fill in.
MSG
