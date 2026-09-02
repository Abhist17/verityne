[← Verityne](../README.md)

> Every endpoint, every environment variable, how to run the tests, and what to do when it breaks.

# API

| Endpoint | Purpose |
| --- | --- |
| `POST /verify` | Score one KYC packet (multipart: `selfie`, `liveness_video`, `id_document`, optional `behavioral_token`). |
| `POST /behavioral` | Accept a form-fill telemetry buffer against a token the page minted on load, reduce it to features, and store only those. Posted before the files, because uploads fail and the buffer should not die with them. |
| `GET /behavioral/{token}` | Read back one stored telemetry session with its score and the rules that fired. |
| `GET /threat/graph` | The fraud-ring graph the Threat Intelligence page reads: `{nodes, edges, rings, threshold}`, where each edge carries `kind: "face" \| "asset_exact" \| "asset_near"`. A ring built only from face edges is an inference and is labelled one; `has_exact_asset_reuse` marks the rings that rest on a byte-identical file. Edges are built at the *search* threshold (0.8169), not the verification point (0.5198) - at the latter every genuine applicant wires to a stranger ([§2b](real-data.md#2b-the-linkage-threshold-was-answering-the-wrong-question)). Computed per request from the audit log and capped at 400 nodes; at production volume this becomes a job and a table. |
| `POST /batch-verify` | Retroactive sweep - re-score history with the current model to find fakes that were let through. |
| `GET /submissions` · `GET /submissions/{id}` | Browse the audit log; full record for one submission. |
| `POST /submissions/{id}/rescore` | Re-run one stored packet against the current model. |
| `GET /gauntlet` · `POST /gauntlet/run` · `GET /gauntlet/stream` | Fixtures, batch scoreboard, and the SSE live feed. |
| `GET /metrics` | Held-out report plus live operational stats. |
| `GET /metrics/corrections` | The audit trail - every belief this project measured and lost, generated from the evidence files each entry cites. |
| `GET /metrics/behavioral` | Detector 6's evaluation: corpus sizes, the feature-set ablation, per-strategy generalisation, and the attack that beats it. |
| `GET /metrics/real` | The same detectors measured on real third-party data; says which reports exist rather than treating a missing dataset as a zero. |
| `GET /metrics/cost-curve` · `GET /metrics/thresholds` | Friction/fraud tradeoff, and where each merchant policy sits on it. |
| `GET /attacks` | Flagged submissions grouped by attack pattern. |
| `GET /review-queue` · `POST /review/{id}/decision` · `GET /review/agreement` | Human-in-the-loop queue, analyst decisions, model/analyst agreement rate. |
| `POST /redteam/generate` | Score a never-before-seen synthetic attack. |
| `GET /policy` · `POST /admin/reload-policy` | Effective merchant policy; hot-reload `policy.yaml`. |
| `GET /audit` | Raw audit event stream. |
| `GET /health` | Uptime, resolved device, warm model status, active policy. |

Auth is a single `X-API-Key` header - this is a demo, not a tenancy model.

<details>
<summary>Example request and response</summary>

```bash
curl -X POST http://localhost:8000/verify \
  -H "X-API-Key: verityne-demo-key" \
  -F "selfie=@selfie.jpg" \
  -F "id_document=@pan.jpg" \
  -F "liveness_video=@liveness.mp4" \
  -F "merchant_id=default" \
  -F "claimed_name=Ravi Kumar"
```

```jsonc
{
  "submission_id": "ed0f5f3eb2cf40e8bbf853ff8b0396fb",
  "verdict": "REJECT",
  "final_score": 0.92,
  "abstained": false,
  "attack_pattern": "reused_kyc_kit",
  "generator_guess": null,
  "top_reasons": [
    "This face matches 1 earlier submission(s) under the same name (possible duplicate application)",
    "The exact same image file was used in 4 earlier submission(s) - consistent with a purchased, pre-made KYC kit",
    "The id_document is 1012px wide but carries detail only to about 569px - it was enlarged from a much smaller source"
  ],
  "explanation": "This submission was rejected with an overall risk score of 0.92. …",
  "detector_breakdown": { "selfie_deepfake": {…}, "id_forensics": {…}, … },
  "heatmaps": { "selfie_deepfake": "/static/heatmaps/…png", … },
  "latency_ms": 2136.51,
  "fusion_model": "logreg",
  "policy": {…},
  "created_at": "2026-08-29T08:31:11.538746Z"
}
```
</details>

---

## Configuration

Copy `.env.example` to `.env`. Every value has a working default.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VERITYNE_API_KEY` | `verityne-demo-key` | Single-tenant demo auth (`X-API-Key`). |
| `VERITYNE_DB` | `sqlite:///storage/verityne.db` | Any SQLAlchemy URL; `postgresql+psycopg://…` works, schema is identical. |
| `VERITYNE_STORAGE` | `./storage` | Uploads, heatmaps, model cache, SQLite file. |
| `VERITYNE_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu`. |
| `VERITYNE_MAX_UPLOAD_MB` | `40` | Per-file upload ceiling. |
| `VERITYNE_MAX_VIDEO_FRAMES` | `24` | Frames sampled from a liveness clip. |
| `VERITYNE_FRAME_STRIDE` | `5` | Sampling stride within the clip. |
| `VERITYNE_WARMUP` | `1` | Load models at boot instead of on the first request. |
| `VERITYNE_CORS` | `http://localhost:3000` | Comma-separated allowed origins. |
| `VERITYNE_SLACK_WEBHOOK` | *(empty)* | High-confidence rejections POST here if set. |
| `VERITYNE_LOG` | `INFO` | Log level. |
| `HF_HOME` | `./storage/models/hf` | Keeps model weights inside the repo, not `~/.cache`. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Where the dashboard looks for the API. |
| `NEXT_PUBLIC_API_KEY` | `verityne-demo-key` | Key the dashboard sends. |

---

## Testing

```bash
make test                                  # 238 tests, ~2 s
.venv/bin/python -m pytest backend/tests -q -k verhoeff   # one group
cd frontend && npx tsc --noEmit && npm run build          # dashboard gates
```

The suite covers the deterministic parts - Verhoeff check digits, PAN structural
validation, OCR confusion repair, ELA tamper scoring, spectral features, fusion
arithmetic, policy decisions, the detector base contract, and the real-data
ingest: LFW's fold protocol, MIDV-2020 quad ordering, threshold arithmetic,
tamper placement and localisation scoring, the Gauntlet's fixture selection and
scoreboard arithmetic, the corpus leak guards, the linkage cap and the search-vs-pair
threshold contract, and the README's own headline numbers against the evidence
files they cite. It deliberately
does *not* assert on model outputs: those belong in `eval/metrics.json`, where a
regression shows up as a number rather than a red test. CI
(`.github/workflows/ci.yml`) runs the same suite on CPU torch plus the dashboard
typecheck and production build.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `torch==2.6.0+cu124` won't install | You have no CUDA. Drop the local tag: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu`, then run the two requirements steps below. |
| `ResolutionImpossible` mentioning `facenet-pytorch` and `numpy` | You installed `requirements-nodeps.txt` with dependency resolution on. facenet-pytorch declares `torch<2.3`, `numpy<2` and `Pillow<10.3`; those bounds are stale, not real. Install it second and with `--no-deps` - `make setup` already does. |
| First request takes 60 s+ | Model weights downloading. Set `VERITYNE_WARMUP=1` so the cost is paid at boot, and check `GET /health` for `models`. |
| `cv2.CascadeClassifier` missing | OpenCV 5 removed it and the face-detection fallback needs it. Stay on `opencv-python-headless==4.11.x`. |
| `/metrics` returns empty | `eval/metrics.json` is committed, but a `make clean-data` removes it. Re-run `make evaluate` (or the whole `make pipeline`). |
| `/metrics/real` lists things under `missing` | That dataset was not on disk when the evaluation last ran. `make data-real` fetches LFW and prints the manual steps for MIDV-2020; the video datasets are gated and cannot be scripted. |
| `No supported deepfake video dataset found` | `make eval-real-video` needs FF++, Celeb-DF v2 or the DFDC preview extracted into the directory you point `DATA=` at. Nothing downloads them. |
| Dashboard shows "failed to fetch" | `NEXT_PUBLIC_API_URL` points somewhere the browser can't reach, or the origin isn't in `VERITYNE_CORS`. |
| Gauntlet page is empty | Fixtures aren't loaded: `make gauntlet`. |
| Everything gets rejected as a "reused KYC kit" | The linkage index has accumulated repeat submissions of the same files, which is what re-scoring the corpus during testing looks like. `make clean` drops the database, then `make gauntlet` re-seeds. Note that an *exact* pixel match is a real signal - a near-match no longer rejects on its own. |
| CUDA out of memory | `VERITYNE_DEVICE=cpu`, or lower `VERITYNE_MAX_VIDEO_FRAMES`. |
| `double free or corruption` / `corrupted size vs. prev_size` during `make score` | Fixed. The shared model singletons were being called from several threads without a lock; see [the thread-safety section](results.md#a-thread-safety-bug-the-real-data-work-uncovered). If you see it again, `--workers 1` isolates it, but the locks in `detectors/models.py`, `utils/ocr.py` and `utils/images.py` should have settled it. |

---

---

[← Verityne](../README.md) - [Architecture](architecture.md) · [Corrections](corrections.md) · [Results](results.md) · [Measured on real data](real-data.md) · [Detector 6](behavioral.md) · [Fusion & policy](fusion.md) · [What it proves](evaluation.md) · **API & config**
