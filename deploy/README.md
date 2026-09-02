# Deploying Verityne

Two hosts, because the two halves want different things. The API is a ~2 GB
PyTorch image that needs 2-4 GB of RAM and does not need to be fast to start.
The dashboard is a Next.js app that wants a CDN and near-instant loads.

    browser ──► Vercel (dashboard) ──► Hugging Face Space (API)
                    │
                    └── /api/* and /static/* are rewritten server-side, so the
                        browser only ever talks to the Vercel origin. There is
                        no CORS configuration to get wrong, and the API origin
                        is never visible to the page.

Both tiers are free. The Space sleeps after inactivity and cold-starts on the
next request; for a link you are sending someone to review at their leisure,
that is the right trade.

---

## 1. The API, on a Hugging Face Space

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
| Hardware | CPU basic (free, 16 GB) |
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
| Environment variable | `NEXT_PUBLIC_API_URL` = `https://<username>-verityne-api.hf.space` |

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

If that returns the same JSON as calling the Space directly, the rewrite is
working and every page will load. Then open the site and check the three pages
that depend on the seeded database - **Attacks**, **Threat**, **Review**. If
they are populated with thumbnails, the demo payload made it into the image.

## What reviewers will notice

- **The first load after a quiet period is slow.** The Space has to wake up.
  Once warm it stays warm for a while.
- **Verdicts take a few seconds, not ~2.5s.** The metrics page reports latency
  from a CUDA machine; a free Space is CPU-only. The verdicts are identical,
  only slower.
- **The API is open.** `NEXT_PUBLIC_API_KEY` defaults to `verityne-demo-key`,
  and anything prefixed `NEXT_PUBLIC_` is visible in the browser bundle by
  definition. That is fine for a review link and wrong for anything else: the
  Space is public and accepts uploads. Take it down when you are finished
  collecting feedback, or set a real key on both sides.

## Redeploying

- **Dashboard**: push to `main`; Vercel rebuilds automatically.
- **API**: re-run `./deploy/build-space.sh`, then commit and push inside
  `deploy/.build/space` again. Only re-run it when the backend or the demo data
  actually changed - it is a 161 MB upload each time.
