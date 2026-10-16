PACKAGES := ml backend

.PHONY: help setup lint format typecheck test check clean download-model predict run

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

clean: ## Remove caches and build outputs
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
