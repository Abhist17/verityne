# Verityne API, read-only. The build that actually fits a free tier.
#
# The full image is 4.36 GB: 2.3 GB of Python packages, 435 MB of pretrained
# checkpoints, 166 MB of uploads and heatmaps. Nothing free will run that -
# the no-card tiers cap at 512 MB of RAM, and everything with enough RAM wants
# a card. Dropping the images alone saves 166 MB of 4.36 GB, which is not the
# problem; PyTorch is.
#
# So this drops the scoring stack instead. `verityne.main` imports its
# detectors lazily, which is what makes that possible: the app starts, and all
# sixteen read endpoints answer from eval/*.json and the SQLite database.
# POST /verify, /face-match/live and /gauntlet/run are the cost - they need the
# models, and no amount of packaging avoids that. Verdicts already computed are
# in the database and read back in full, with their reasons and detector
# breakdowns.
FROM python:3.12-slim

# libgl and libglib are opencv's runtime deps; the import chain reaches
# opencv through utils/hashing whether or not anything scores.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    VERITYNE_STORAGE=/app/storage \
    VERITYNE_EVAL=/app/eval \
    VERITYNE_DATASETS=/app/datasets \
    VERITYNE_DEVICE=cpu \
    PORT=8080 \
    # No models to warm. Left on, the app would import the detector package at
    # boot and die on the first missing import instead of serving.
    VERITYNE_WARMUP=0

COPY requirements-slim.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-slim.txt

COPY backend /app/backend
COPY eval /app/eval

# The seeded database, and the fitted-model metadata the API reports even when
# it cannot load the models themselves. No uploads or heatmaps: they are 166 MB
# of the payload and only feed thumbnails.
COPY seed/demo-data-slim.tar.gz /tmp/seed.tar.gz
RUN mkdir -p /app/storage \
    && tar -xzf /tmp/seed.tar.gz -C /app \
    && rm /tmp/seed.tar.gz

RUN mkdir -p /app/storage/uploads /app/storage/heatmaps /app/datasets \
    && chown -R user:user /app

USER user
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD exec uvicorn verityne.main:app --host 0.0.0.0 --port "${PORT}" --app-dir /app/backend
