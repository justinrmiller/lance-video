# video-lance — common developer tasks.
#
# Everything runs through `uv run` so targets use the locked environment
# without needing an activated venv. Run `make` (or `make help`) for the list.

# Override on the command line, e.g. `make ingest DB=./demo.db DIR=./clips`.
DB  ?= ./video-lance.db
DIR ?= ./videos

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: sync
sync: ## Install/refresh the locked environment (uv sync)
	uv sync

.PHONY: lock
lock: ## Re-resolve dependencies to the newest compatible versions
	uv lock --upgrade

.PHONY: hooks
hooks: ## Install the pre-commit git hook (one-time)
	uv run pre-commit install

# ---------------------------------------------------------------------------
# Quality gates  (mirror .github/workflows/ci.yml)
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Lint with ruff
	uv run ruff check src tests

.PHONY: fix
fix: ## Lint and auto-fix with ruff
	uv run ruff check --fix src tests

.PHONY: fmt
fmt: ## Auto-format with ruff
	uv run ruff format src tests scripts

.PHONY: fmt-check
fmt-check: ## Verify formatting without modifying files
	uv run ruff format --check src tests scripts

.PHONY: typecheck
typecheck: ## Static type check with ty
	uv run ty check

.PHONY: test
test: ## Run the unit test suite
	uv run pytest -v

.PHONY: cov
cov: ## Run tests with a terminal coverage report
	uv run pytest --cov

.PHONY: cov-html
cov-html: ## Run tests and write an HTML coverage report to htmlcov/
	uv run pytest --cov --cov-report=html

.PHONY: integration
integration: ## Run the opt-in real-model integration suite (~6 GB of weights)
	VL_INTEGRATION=1 uv run pytest tests/test_integration_real_models.py -v

.PHONY: pre-commit
pre-commit: ## Run every pre-commit hook against the whole tree
	uv run pre-commit run --all-files

.PHONY: check
check: lint fmt-check typecheck test ## Run the full CI gate locally (lint + format + typecheck + test)

# ---------------------------------------------------------------------------
# Application — see `uv run video-lance --help` for every flag.
# ---------------------------------------------------------------------------

.PHONY: ingest
ingest: ## Ingest DIR into DB (override DIR=… DB=…)
	uv run video-lance ingest $(DIR) --db-path $(DB)

.PHONY: info
info: ## Report row counts, embedding models, and indexes for DB
	uv run video-lance info --db-path $(DB)

.PHONY: reindex
reindex: ## Drop and rebuild the FTS + vector indexes on DB
	uv run video-lance reindex --db-path $(DB)

.PHONY: ui
ui: ## Launch the Gradio web UI against DB
	uv run video-lance ui --db-path $(DB)

.PHONY: demo
demo: ## One-shot ingest + search demo (scripts/demo.sh, builds ./demo.db)
	./scripts/demo.sh $(DIR)

.PHONY: verify-blob
verify-blob: ## Check the blob-column encoding on DB (scripts/verify_blob.py)
	uv run python scripts/verify_blob.py --db-path $(DB)

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove caches and coverage artefacts (leaves .venv and any DB)
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage .coverage.*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
