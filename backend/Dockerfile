FROM python:3.12-slim

# tesseract is the preferred OCR backend; without it the ID detector falls back
# to easyocr (pip-installed, slower) and then to a no-OCR degraded mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr libgl1 libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend HF_HOME=/app/storage/models/hf

COPY backend/requirements.txt backend/requirements-nodeps.txt /app/backend/
# CPU wheels in the container: the image stays ~2GB instead of ~7GB, and the
# detectors are small enough to run acceptably on CPU. Mount a GPU and swap the
# index URL if you want CUDA.
#
# requirements.txt pins torch with a +cu124 local tag that exists only on the
# CUDA index, so those two lines are dropped here and the same versions come
# from the CPU index — the same trick CI uses, for the same reason.
# facenet-pytorch goes last and without deps; see requirements-nodeps.txt.
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^(torch|torchvision)==' /app/backend/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && pip install --no-cache-dir --no-deps -r /app/backend/requirements-nodeps.txt

COPY backend /app/backend
RUN mkdir -p /app/storage/uploads /app/storage/heatmaps /app/storage/models /app/datasets /app/eval

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "verityne.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
