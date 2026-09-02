# Verityne API, as a Hugging Face Docker Space.
#
# This is backend/Dockerfile with three deployment facts folded in:
#
#   * Spaces route to one port and run the container as UID 1000, so the app
#     listens on 7860 and everything it writes is owned by `user`. The API
#     writes uploads, heatmaps and a SQLite file at runtime, so a root-owned
#     /app/storage would fail on the first upload rather than at boot.
#   * A Space's disk is ephemeral on the free tier: it resets to the image on
#     every restart. So the demo payload - the trained models, the seeded
#     database, and the 164 MB of uploads and heatmaps behind the evidence
#     cards - is baked into the image rather than mounted beside it.
#   * A Space sleeps after inactivity and cold-starts on the next visit. The
#     two checkpoints that load at boot are pulled at build time, so waking up
#     is a container start rather than a 600 MB download over someone's click.
FROM python:3.12-slim

# tesseract is the preferred OCR backend; without it the ID detector falls back
# to easyocr (pip-installed, slower) and then to a no-OCR degraded mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr libgl1 libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    HF_HOME=/app/storage/models/hf \
    TORCH_HOME=/app/storage/models/torch \
    VERITYNE_STORAGE=/app/storage \
    VERITYNE_EVAL=/app/eval \
    VERITYNE_DATASETS=/app/datasets \
    VERITYNE_DEVICE=cpu \
    VERITYNE_WARMUP=1

# CPU wheels: the image stays ~2 GB instead of ~7 GB, and a Space has no GPU on
# the free tier anyway. requirements.txt pins torch with a +cu124 local tag that
# exists only on the CUDA index, so those two lines are dropped and the same
# versions come from the CPU index - the same trick CI and backend/Dockerfile
# use, for the same reason.
COPY backend/requirements.txt backend/requirements-nodeps.txt /app/backend/
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^(torch|torchvision)==' /app/backend/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && pip install --no-cache-dir --no-deps -r /app/backend/requirements-nodeps.txt

COPY backend /app/backend
COPY eval /app/eval

# The demo payload. Extracted before the model pre-pull because it carries
# models/active_detector.txt, which names the checkpoint to pull.
COPY seed/demo-data.tar.gz /tmp/demo-data.tar.gz
RUN mkdir -p /app/storage \
    && tar -xzf /tmp/demo-data.tar.gz -C /app \
    && rm /tmp/demo-data.tar.gz

# Warm the two checkpoints into the image. A copied script rather than a RUN
# heredoc: heredocs need a BuildKit frontend, and a Space is not built by the
# Docker on this laptop.
COPY prefetch_models.py /tmp/prefetch_models.py
RUN python /tmp/prefetch_models.py && rm /tmp/prefetch_models.py

RUN mkdir -p /app/storage/uploads /app/storage/heatmaps /app/datasets \
    && chown -R user:user /app

USER user

# The port is taken from the environment, defaulting to the 7860 a Hugging Face
# Space routes to. Cloud Run injects its own PORT and will not route to a
# container that ignores it, so hardcoding 7860 in the CMD made this image
# deployable to exactly one host. `exec` so uvicorn is PID 1 and receives the
# platform's shutdown signal rather than leaving the shell to swallow it.
ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD exec uvicorn verityne.main:app --host 0.0.0.0 --port "${PORT}" --app-dir /app/backend
