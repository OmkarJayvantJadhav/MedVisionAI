PACKAGES := ml backend

.PHONY: help setup lint format typecheck test check clean download-model predict run frontend-setup frontend-check frontend-dev api-schema docker-build docker-run smoke evaluate

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install dependencies for every package and the git hooks
	@for p in $(PACKAGES); do uv sync --project $$p --locked || exit 1; done
	uv tool run pre-commit install

lint: ## Run ruff lint and format checks
	@for p in $(PACKAGES); do uv run --project $$p ruff check $$p && uv run --project $$p ruff format --check $$p || exit 1; done

format: ## Auto-fix lint issues and format code
	@for p in $(PACKAGES); do uv run --project $$p ruff check --fix $$p && uv run --project $$p ruff format $$p || exit 1; done

typecheck: ## Run mypy in strict mode
	@for p in $(PACKAGES); do (cd $$p && uv run mypy) || exit 1; done

test: ## Run the test suites (excludes data and model tests)
	@for p in $(PACKAGES); do (cd $$p && uv run pytest) || exit 1; done

check: lint typecheck test ## Run every check CI runs

download-model: ## Fetch and verify model artifacts: make download-model BASE_URL=<release folder URL>
	cd backend && uv run python -m app.inference.download --base-url "$(BASE_URL)" --dest ../models

run: ## Start the API on http://localhost:8000 (docs at /docs) using ../models
	cd backend && MEDVISION_MODEL_DIR=$(CURDIR)/models uv run uvicorn app.main:create_app --factory --reload --port 8000

predict: ## Classify images with the served model: make predict IMAGES="a.jpg b.png"
	cd backend && uv run python -m app.inference.cli $(abspath $(IMAGES)) --model-dir ../models

frontend-setup: ## Install frontend dependencies
	cd frontend && npm ci

frontend-check: ## Lint, test and build the frontend (same as CI)
	cd frontend && npm run lint && npm test -- --run && npm run build

frontend-dev: ## Start the frontend dev server on http://localhost:5173 (run `make run` too)
	cd frontend && npm run dev

api-schema: ## Regenerate frontend/src/api/openapi.json from the backend
	cd backend && uv run python -m app.openapi

# make evaluate: every responsible-AI report (plan section 8) for one trained model.
# Reports go to ml/outputs/<report>/<run>, where <run> is the model's folder name.
EVAL_MODEL = $(abspath $(MODEL))
EVAL_RUN = $(notdir $(patsubst %/,%,$(dir $(EVAL_MODEL))))
EVAL_OUT = $(CURDIR)/ml/outputs
NON_SKIN ?= $(CURDIR)/ml/data/non_skin
ML = cd ml && uv run medvision-ml

evaluate: ## All model reports: make evaluate MODEL=ml/artifacts/<run>/model.keras METADATA=<ISIC metadata CSV>
	@test -n "$(MODEL)" -a -n "$(METADATA)" || { echo "usage: make evaluate MODEL=... METADATA=..."; exit 2; }
	$(ML) evaluate --model $(EVAL_MODEL)
	$(ML) calibrate --model $(EVAL_MODEL)
	test -f $(NON_SKIN)/ATTRIBUTION.csv || ($(ML) fetch-non-skin --dest $(NON_SKIN) --count 500)
	$(ML) fit-ood --model $(EVAL_MODEL) --non-skin $(NON_SKIN) \
		--calibration $(EVAL_OUT)/calibration/$(EVAL_RUN)/calibration.json
	$(ML) robustness --model $(EVAL_MODEL) \
		--calibration $(EVAL_OUT)/calibration/$(EVAL_RUN)/calibration.json \
		--ood $(EVAL_OUT)/ood/$(EVAL_RUN)/ood.json
	$(ML) subgroups --model $(EVAL_MODEL) --metadata $(abspath $(METADATA)) \
		--calibration $(EVAL_OUT)/calibration/$(EVAL_RUN)/calibration.json

docker-build: ## Build the app image (frontend + API + verified model)
	docker build -t medvisionai:local .

docker-run: ## Run the app image on http://localhost:7860
	docker run --rm -p 7860:7860 medvisionai:local

smoke: ## Smoke-test a running app: make smoke URL=http://localhost:7860
	python3 .github/scripts/smoke_test.py $(or $(URL),http://localhost:7860) frontend/public/samples/melanoma.jpg

clean: ## Remove caches and build outputs
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
