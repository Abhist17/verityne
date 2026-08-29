"""Verityne API.

    uvicorn verityne.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import routes_gauntlet, routes_live, routes_metrics, routes_ops, routes_verify
from .config import HEATMAP_DIR, UPLOAD_DIR, get_policy, resolve_device
from .db import init_db

logging.basicConfig(level=os.getenv("VERITYNE_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verityne")

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("database ready")
    if os.getenv("VERITYNE_WARMUP", "1") == "1":
        # Pay model load time at boot, not on the first judge's click.
        import asyncio

        from .detectors.models import warmup

        status = await asyncio.get_running_loop().run_in_executor(None, warmup)
        app.state.model_status = status
        log.info("models warm: %s", status)
    else:
        app.state.model_status = {"warmup": "skipped"}
    yield


app = FastAPI(
    title="Verityne",
    version="1.0.0",
    description=(
        "Deepfake-aware KYC verification. Five independent detectors, a calibrated "
        "fusion layer, and an explanation for every verdict."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("VERITYNE_CORS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


app.include_router(routes_verify.router, tags=["verification"])
app.include_router(routes_gauntlet.router, tags=["gauntlet"])
app.include_router(routes_metrics.router, tags=["metrics"])
app.include_router(routes_live.router, tags=["live"])
app.include_router(routes_ops.router, tags=["ops"])

app.mount("/static/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - START_TIME, 1),
        "device": resolve_device(),
        "models": getattr(app.state, "model_status", {}),
        "default_policy": get_policy().model_dump(),
    }


@app.get("/", tags=["ops"])
def root():
    return {
        "name": "Verityne",
        "docs": "/docs",
        "endpoints": ["/verify", "/face-match/live", "/batch-verify", "/gauntlet/stream", "/metrics", "/attacks", "/review-queue"],
    }
