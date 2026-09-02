# The deployable build

The full API image is 4.36 GB: 2.3 GB of Python packages, 435 MB of pretrained
checkpoints, 166 MB of uploads and heatmaps. Nothing free will run it. The
no-card free tiers cap at 512 MB of RAM, which PyTorch will not start in, and
every host with enough RAM wants a card on file.

Dropping the photos is not the fix. They are 166 MB of 4.36 GB. PyTorch is the
problem.

So this build drops the scoring stack instead, which is possible only because
`verityne.main` imports its detectors lazily.

|                     | full        | slim   |
|---------------------|-------------|--------|
| image               | 4.36 GB     | 791 MB |
| RAM at rest         | ~1.4 GB     | 71 MiB |
| boot to serving     | ~12 s       | ~1 s   |
| payload shipped     | 163 MB      | 3.6 MB |
| read endpoints (16) | all         | all    |
| live scoring        | yes         | no     |

## What works

All sixteen read endpoints, verified in the running container: `/metrics` and
its six sub-reports, `/gauntlet`, `/review-queue`, `/review/agreement`,
`/attacks`, `/threat/graph`, `/submissions`, `/policy`, `/audit`. Every verdict
already computed is in the database and reads back in full - score, reasons,
detector breakdown, linkage, attack pattern.

So Metrics, Corrections, Attacks, Threat, Review and Gauntlet are complete.

## What does not

`POST /verify`, `/face-match/live` and `/gauntlet/run` need the models. No
amount of packaging avoids that. Thumbnails and heatmap overlays are also gone,
since the images are what the payload dropped - the evidence cards render with
their text and scores but no picture.

## Build it

```bash
./deploy/build-slim.sh          # writes deploy/.build/slim
cd deploy/.build/slim
docker build -t verityne-api:slim .
docker run -p 8080:8080 verityne-api:slim
```

The container reads `$PORT`, so it deploys unchanged to anything that injects
one. `VERITYNE_WARMUP=0` is set in the image and must stay: left on, the app
imports the detector package at boot and dies on the first missing import
instead of serving.
