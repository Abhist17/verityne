# Deploying Verityne

Two hosts, because the two halves want different things. The API is a 4.4 GB
PyTorch image that needs 2-4 GB of RAM and does not need to be fast to start.
The dashboard is a Next.js app that wants a CDN and near-instant loads.

    browser ──► Vercel (dashboard) ──► Cloud Run (API)
                    │
                    └── /api/* and /static/* are rewritten server-side, so the
                        browser only ever talks to the Vercel origin. There is
                        no CORS configuration to get wrong, and the API origin
                        is never visible to the page.

**Hugging Face Spaces is no longer an option for this image.** Docker and
Gradio Spaces require a PRO subscription; only Static Spaces are free, and a
Static Space serves HTML and never runs a Dockerfile. The Space recipe below
still works if you have PRO - `deploy/hf-space/` and `build-space.sh` are
unchanged - but the free path is Cloud Run.

Cloud Run scales to zero, so an idle demo costs nothing and the free tier
(2M requests, 360k GiB-seconds/month) covers a review link comfortably. It
does require a billing account on the project. The trade is a cold start of
roughly a minute after a quiet period, because the image is large.

---

## 1. The API, on Cloud Run

```bash
./deploy/build-space.sh                  # assembles deploy/.build/space
cd deploy/.build/space
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create verityne --repository-format=docker --location=asia-south1
gcloud auth configure-docker asia-south1-docker.pkg.dev

IMG=asia-south1-docker.pkg.dev/<PROJECT_ID>/verityne/api:latest
docker build -t "$IMG" . && docker push "$IMG"

gcloud run deploy verityne-api --image "$IMG" --region asia-south1 \
  --allow-unauthenticated --memory 4Gi --cpu 2 --timeout 300 \
  --min-instances 0 --max-instances 2
```

The container reads `$PORT`, which Cloud Run injects, so no port flag is
needed. `--min-instances 0` is what keeps it free.

## Appendix: the same image on a Hugging Face Space (needs PRO)

**Assemble the Space.** The image needs three things this repo does not track -
the fitted models, the seeded database, and the 164 MB of uploads and heatmaps
behind the evidence cards. They are build outputs of a GPU pipeline, not
source. This packs them out of your working tree:

```bash
./deploy/build-space.sh          # writes deploy/.build/space
```

**Create the Space** at <https://huggingface.co/new-space>:

| Field | Value |
|---|---|
| Owner | your username |
| Space name | `verityne-api` |
| License | `mit` |
| SDK | **Docker** → Blank |
| Hardware | CPU basic (**requires PRO**) |
| Visibility | Public |

**Push it.** Needs `git-lfs` (`sudo apt install git-lfs`), because the demo
payload is a 161 MB object:

```bash
cd deploy/.build/space
git init -b main
git lfs install && git lfs track "seed/demo-data.tar.gz"
git add -A && git commit -m "Verityne API"
git remote add origin https://huggingface.co/spaces/<username>/verityne-api
git push -u origin main
```

Git will ask for a username and a password: the password is a **write access
token** from <https://huggingface.co/settings/tokens>, not your account
password.

The build takes 10-15 minutes - most of it installing torch and pre-pulling the
two checkpoints so that later cold starts are a container start rather than a
600 MB download. Watch the Space's **Logs** tab. When it is done:

```bash
curl https://<username>-verityne-api.hf.space/health
```

That should report `"status":"ok"` and list the loaded models.

## 2. The dashboard, on Vercel

Import the GitHub repo at <https://vercel.com/new>, then set:

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework | Next.js (auto-detected) |
| Environment variable | `NEXT_PUBLIC_API_URL` = your Cloud Run URL, e.g. `https://verityne-api-xxxx.a.run.app` |

**That variable is read at build time, not at runtime.** `next build` resolves
`rewrites()` and writes the API origin into `routes-manifest.json`; `next start`
then serves the manifest and never re-reads the config. So it has to be set
before the first build, and **changing it later requires a redeploy**, not a
restart. Setting it only in the runtime environment silently leaves the
dashboard proxying to `http://localhost:8000`.

Deploy. The dashboard comes up at `https://<project>.vercel.app`, and that is
the link to send.

---

## Checking it end to end

```bash
curl -s https://<project>.vercel.app/api/health | head -c 200      # via the proxy
```

If that returns the same JSON as calling the API directly, the rewrite is
working and every page will load. Then open the site and check the three pages
that depend on the seeded database - **Attacks**, **Threat**, **Review**. If
they are populated with thumbnails, the demo payload made it into the image.

## What reviewers will notice

- **The first load after a quiet period is slow.** Cloud Run scaled to zero
  and has to pull a 4.4 GB image back. Once warm it stays warm for a while.
- **Verdicts take a few seconds, not ~2.5s.** The metrics page reports latency
  from a CUDA machine; Cloud Run is CPU-only. The verdicts are identical,
  only slower.
- **The API is open.** `NEXT_PUBLIC_API_KEY` defaults to `verityne-demo-key`,
  and anything prefixed `NEXT_PUBLIC_` is visible in the browser bundle by
  definition. That is fine for a review link and wrong for anything else: the
  service is public and accepts uploads. Delete it when you are finished
  collecting feedback, or set a real key on both sides.

## Redeploying

- **Dashboard**: push to `main`; Vercel rebuilds automatically.
- **API**: re-run `./deploy/build-space.sh`, then rebuild and `docker push` the
  image and `gcloud run deploy` again. Only re-run it when the backend or the
  demo data actually changed - it is a 4.4 GB push each time.
