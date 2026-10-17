# Deployment

The app ships as one Docker image: the React frontend, the FastAPI backend and
the model, which is downloaded from the GitHub release and checksum-verified
while the image is built. The image runs as a non-root user on port 7860 and
has a health check.

## Run it locally

```bash
docker compose up --build        # or: make docker-build docker-run
# open http://localhost:7860
make smoke                       # checks /ready, the home page, /predict and /explain
```

## Live demo on Hugging Face Spaces

`.github/workflows/deploy.yml` deploys `main` to a free Hugging Face Space
(Docker SDK, CPU) after CI passes:

1. `.github/scripts/deploy_space.py` copies the Dockerfile, `backend/` and
   `frontend/` (tracked files only) into a staging folder with a Space README,
   creates the Space if needed and uploads it as one commit.
2. It waits until Hugging Face has rebuilt the Space and reports `RUNNING`;
   a build or runtime error fails the job.
3. `.github/scripts/smoke_test.py` then tests the live URL.

### One-time setup

1. Create a free account at <https://huggingface.co/join>.
2. Create a token at <https://huggingface.co/settings/tokens>: **Create new
   token → Write**, and copy it.
3. In GitHub, open the repository's **Settings → Secrets and variables →
   Actions**:
   - **Secrets** tab → **New repository secret**: name `HF_TOKEN`, value the token.
   - **Variables** tab → **New repository variable**: name `HF_SPACE`, value
     `<your-hf-username>/medvision-ai`.
4. Open **Actions → Deploy → Run workflow**. The first build takes about 5–10
   minutes. The site then lives at
   `https://<your-hf-username>-medvision-ai.hf.space`.

Without the secret and variable the Deploy workflow only prints a notice.

### Rate limit behind the Space proxy

Every visitor reaches the Space through Hugging Face's proxy, so the app sees
one client address for all of them. The deploy sets
`MEDVISION_RATE_LIMIT_PER_MINUTE=60` as a Space variable: a shared ceiling that
protects the free CPU rather than a per-person limit (D43).
