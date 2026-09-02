#!/usr/bin/env bash
# Assemble the read-only API bundle - the one that fits a free tier.
#
# Same idea as build-space.sh, minus the scoring stack and minus the images.
# See deploy/slim/README.md for what that costs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/deploy/.build/slim"
cd "$REPO"

[ -e storage/verityne.db ] || { echo "missing storage/verityne.db - run 'make gauntlet'" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/seed"
cp deploy/slim/Dockerfile deploy/slim/requirements-slim.txt "$OUT/"

tar -c --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -C "$REPO" backend eval | tar -x -C "$OUT"

# The database and the fitted-model metadata. No uploads or heatmaps: they are
# 166 MB and only feed thumbnails, and no read endpoint needs them to answer.
tar -czf "$OUT/seed/demo-data-slim.tar.gz" \
    --exclude='storage/models/hf' \
    --exclude='storage/models/torch' \
    -C "$REPO" storage/models storage/verityne.db

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

~791 MB image, ~71 MiB RAM, boots in about a second. All sixteen read
endpoints answer; POST /verify and the other scoring routes do not.
MSG
