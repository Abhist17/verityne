# Verityne, whole product, one container. Built for Render's free tier.
#
# Two hosts was the wrong shape for this. The dashboard already proxies /api and
# /static to the API server-side, so putting both in one container means that
# proxy is a loopback call: no second service, no public API origin, no CORS,
# and one URL to hand someone. Render gives the container one $PORT; Next takes
# it and uvicorn sits behind it on 8000, which nothing outside ever reaches.
#
# The API here is the read-only build - no torch, no checkpoints. See
# deploy/slim/README.md for what that costs (live scoring) and what it does not
# (everything already scored, which is all 27 review cases and 20 fixtures).

# ---------------------------------------------------------------- dashboard
FROM node:20-slim AS web
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
# `next build` bakes rewrites() into routes-manifest.json, so the API origin is
# fixed here, not at runtime. localhost:8000 is right because the API is in this
# same container - the browser never resolves it, the Next server does.
ENV NEXT_PUBLIC_API_URL=http://localhost:8000 \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      nodejs libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    VERITYNE_STORAGE=/app/storage \
    VERITYNE_EVAL=/app/eval \
    VERITYNE_DATASETS=/app/datasets \
    VERITYNE_DEVICE=cpu \
    VERITYNE_WARMUP=0 \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=10000

COPY deploy/slim/requirements-slim.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-slim.txt

COPY backend /app/backend
COPY eval /app/eval

# The dashboard, already built.
COPY --from=web /app/.next /app/web/.next
COPY --from=web /app/node_modules /app/web/node_modules
COPY --from=web /app/package.json /app/web/package.json
COPY --from=web /app/next.config.mjs /app/web/next.config.mjs
COPY --from=web /app/public /app/web/public

# Database, fitted-model metadata, and the downscaled evidence images.
COPY seed/demo-data-render.tar.gz /tmp/seed.tar.gz
RUN mkdir -p /app/storage \
    && tar -xzf /tmp/seed.tar.gz -C /app \
    && rm /tmp/seed.tar.gz \
    && mkdir -p /app/storage/uploads /app/storage/heatmaps /app/datasets

COPY deploy/render/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 10000
CMD ["/app/start.sh"]
