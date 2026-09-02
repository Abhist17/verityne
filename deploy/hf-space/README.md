---
title: Verityne API
emoji: 🛡️
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Deepfake-aware KYC verification - six detectors and a fusion layer
---

# Verityne API

The scoring backend for [Verityne](https://github.com/Abhist17/verityne): six
independent detectors, a calibrated fusion layer, and a human-readable
explanation behind every verdict.

This Space serves the API only. The dashboard that reads it is deployed
separately and proxies to this origin, so there is no CORS story here.

- `GET /health` - liveness, loaded models, and the active policy
- `GET /docs` - the full OpenAPI surface
- `GET /metrics` - the held-out evaluation report
- `POST /verify` - score a packet

It runs on CPU, so a verdict takes a few seconds rather than the ~2.5s the
project's own metrics page reports from a CUDA machine. The Space also sleeps
after inactivity; the first request after that pays a container start.
