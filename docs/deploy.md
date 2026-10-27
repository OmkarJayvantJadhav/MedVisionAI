# Deployment

The app ships as one Docker image: the React frontend, the FastAPI backend and
the model, which is downloaded from the GitHub release and checksum-verified
while the image is built. The image runs as a non-root user, listens on
`$PORT` (7860 by default) and has a health check.

## Run it locally

```bash
docker compose up --build        # or: make docker-build docker-run
# open http://localhost:7860
make smoke                       # checks /ready, the home page, /predict and /explain
```

## Hosting on Render

`render.yaml` describes the service for a Render web service. Render
builds the Dockerfile and **deploys `main` automatically once GitHub CI has
passed** (`autoDeployTrigger: checksPass`).

### One-time setup

1. Sign up at <https://render.com> with your GitHub account.
2. **New → Blueprint**, pick the `MedVisionAI` repository and confirm. Render
   reads `render.yaml` and creates the `medvision-ai` web service on the free
   plan. The first build takes about 5–10 minutes.
3. Copy the service URL from the Render dashboard (for example
   `https://medvision-ai.onrender.com`).
4. In GitHub, open **Settings → Secrets and variables → Actions → Variables →
   New repository variable**: name `LIVE_URL`, value the service URL. The
   **Live site** workflow then smoke-tests the live app daily and on demand.

### Hosting limits

- **Sleeps when idle.** The service stops after 15 minutes without visitors;
  the next visit wakes it in about a minute.
- **A fraction of one CPU and 512 MB of memory.** The app uses about 170 MB.
  With `MEDVISION_INFERENCE_THREADS=1` an analysis with heatmap takes about
  1–3 s (measured in the image limited to 0.1 CPU).
- **Shared rate limit.** Visitors arrive through Render's proxy, so the app sees
  one address for all of them; `MEDVISION_RATE_LIMIT_PER_MINUTE=60` is a
  shared ceiling that protects the CPU.

Hugging Face Spaces was the first choice (D5) but no longer hosts Docker Spaces
on its free CPU tier (D44).
