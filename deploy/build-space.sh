#!/usr/bin/env bash
# Assemble a ready-to-push Hugging Face Space from this working tree.
#
# The Space needs three things this repository deliberately does not track:
# the fitted models, the seeded database, and the uploads and heatmaps behind
# every evidence card. They are gitignored because they are build outputs of a
# GPU pipeline, not source - but a deploy without them is a dashboard with no
# fusion layer and three empty pages. So this script reads them out of
# ./storage and packs them into one archive the image extracts at build time.
#
# One archive rather than 1,015 loose files: Git LFS bills per object, and a
# thousand small pointers make a Space repo slow to clone and miserable to
# inspect.
#
#   ./deploy/build-space.sh          -> deploy/.build/space
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/deploy/.build/space"
SEED="$OUT/seed/demo-data.tar.gz"

cd "$REPO"

# ---------------------------------------------------------------- preflight
missing=0
for f in storage/verityne.db storage/models/fusion.joblib storage/models/active_detector.txt; do
  [ -e "$f" ] || { echo "missing: $f" >&2; missing=1; }
done
if [ "$missing" = 1 ]; then
  cat >&2 <<'MSG'

The demo payload is not on this machine. It is produced by the pipeline:

  make pipeline    # corpus, calibration, scoring, fusion  (needs a GPU, ~hours)
  make gauntlet    # the 20 demo fixtures
  make redteam     # the held-out attack pool

Deploying without it gives you a dashboard with no fusion layer and empty
Attacks, Threat and Review pages.
MSG
  exit 1
fi

# ---------------------------------------------------------------- assemble
rm -rf "$OUT"
mkdir -p "$OUT/seed"

# Every file the Dockerfile COPYs has to be here, prefetch_models.py included -
# a missing one fails the build ~10 minutes in, on Hugging Face's builder rather
# than on this laptop.
cp deploy/hf-space/Dockerfile \
   deploy/hf-space/README.md \
   deploy/hf-space/.gitattributes \
   deploy/hf-space/prefetch_models.py "$OUT/"

# Source the image needs. `backend` and `eval` are tracked, so they come from
# the working tree as-is.
mkdir -p "$OUT/backend"
tar -c --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -C "$REPO" backend eval | tar -x -C "$OUT"

# The untracked payload. `models/hf` and `models/torch` are excluded on
# purpose: those are upstream checkpoints, and the Dockerfile pulls them at
# build time so they never ride through Git LFS.
echo "packing demo payload ..."
tar -czf "$SEED" \
    --exclude='storage/models/hf' \
    --exclude='storage/models/torch' \
    --exclude='storage/private' \
    --exclude='storage/tmp_collector' \
    --exclude='storage/*.db-journal' \
    --exclude='storage/verityne-prepollution.db' \
    -C "$REPO" \
    storage/models storage/verityne.db storage/uploads storage/heatmaps

# ---------------------------------------------------------------- verify
# Every local path the Dockerfile COPYs must exist in the bundle.
while read -r src; do
  [ -e "$OUT/$src" ] || { echo "assembled bundle is missing $src, which the Dockerfile COPYs" >&2; exit 1; }
done < <(grep -E '^COPY ' "$OUT/Dockerfile" | sed -E 's/^COPY +//; s/ +[^ ]+$//' | tr ' ' '\n' | grep -v '^$')

# ---------------------------------------------------------------- report
size=$(du -h "$SEED" | cut -f1)
subs=$(python3 -c "
import sqlite3,sys
try: print(sqlite3.connect('$REPO/storage/verityne.db').execute('select count(*) from submissions').fetchone()[0])
except Exception: print('?')
")
cat <<MSG

Space assembled: deploy/.build/space
  seed archive : $size  ($(find "$REPO/storage/uploads" "$REPO/storage/heatmaps" -type f | wc -l) images, $subs submissions)

Push it (needs git-lfs and a Hugging Face account):

  cd deploy/.build/space
  git init -b main
  git lfs install && git lfs track "seed/demo-data.tar.gz"
  git add -A && git commit -m "Verityne API"
  git remote add origin https://huggingface.co/spaces/<your-username>/verityne-api
  git push -u origin main

Create the Space first at https://huggingface.co/new-space with SDK "Docker".
The build takes ~10-15 minutes; watch it in the Space's Logs tab.
MSG
