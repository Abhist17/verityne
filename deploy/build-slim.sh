#!/usr/bin/env bash
# Assemble the read-only API bundle - the one that fits a free tier.
#
# Same idea as build-space.sh, minus the scoring stack and minus the images.
# See deploy/slim/README.md for what that costs.
set -euo pipefail

# --with-images ships the 166 MB of uploads and heatmaps too.
#
# Worth knowing before you decide: what a free tier caps is RAM, and these are
# static files on disk. The container serves them at the same 71 MiB it serves
# everything else - the only cost is a bigger image to push, once. Without them
# the evidence cards keep their scores and reasons and show "overlay not
# deployed" where the picture would be.
WITH_IMAGES=0
[ "${1:-}" = "--with-images" ] && WITH_IMAGES=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/deploy/.build/slim"
cd "$REPO"

[ -e storage/verityne.db ] || { echo "missing storage/verityne.db - run 'make gauntlet'" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/seed"
cp deploy/slim/Dockerfile deploy/slim/requirements-slim.txt "$OUT/"

tar -c --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -C "$REPO" backend eval | tar -x -C "$OUT"

# The database and the fitted-model metadata always; the images only when
# asked. No read endpoint needs the images to answer - they feed thumbnails and
# the heatmap overlays, and the UI shows a labelled frame when they are absent.
PAYLOAD=(storage/models storage/verityne.db)
[ "$WITH_IMAGES" = 1 ] && PAYLOAD+=(storage/uploads storage/heatmaps)

tar -czf "$OUT/seed/demo-data-slim.tar.gz" \
    --exclude='storage/models/hf' \
    --exclude='storage/models/torch' \
    -C "$REPO" "${PAYLOAD[@]}"

# Every local path the Dockerfile COPYs must exist in the bundle.
while read -r src; do
  [ -e "$OUT/$src" ] || { echo "bundle is missing $src, which the Dockerfile COPYs" >&2; exit 1; }
done < <(grep -E '^COPY ' "$OUT/Dockerfile" | sed -E 's/^COPY +//; s/ +[^ ]+$//' | tr ' ' '\n' | grep -v '^$')

subs=$(python3 -c "
import sqlite3
try: print(sqlite3.connect('$REPO/storage/verityne.db').execute('select count(*) from submissions').fetchone()[0])
except Exception: print('?')
")
cat <<MSG

Slim bundle: deploy/.build/slim  ($(du -sh "$OUT" | cut -f1), $subs submissions)

  cd deploy/.build/slim
  docker build -t verityne-api:slim .
  docker run -p 8080:8080 verityne-api:slim

~71 MiB RAM either way, boots in about a second. All sixteen read endpoints
answer; POST /verify and the other scoring routes do not.
$([ "$WITH_IMAGES" = 1 ] \
  && echo "Images included: ~1.3 GB image, evidence cards fully illustrated." \
  || echo "No images: ~791 MB image. Re-run with --with-images to include them.")
MSG
