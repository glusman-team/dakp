# DAKP pipeline — minimal run controls.
#
# All run configuration lives in `.envrc` (direnv) — edit that file, not these targets.
# Quality gates run via pre-commit / `uv run` directly (no make wrappers). `make help` lists targets.

.DEFAULT_GOAL := help
.PHONY: help install up-mock up-sample up-prod down clean

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/{printf "  %-12s %s\n",$$1,$$2}' $(MAKEFILE_LIST)

install: ## Install everything (uv sync — runtime + dev; there are no extras)
	uv sync

up-mock: ## Run the pipeline end-to-end on the mock profile (fixtures; no network)
	PROFILE=mock bash scripts/dakp_up.sh

up-sample: ## Run on the sample profile (real sources, bounded scope)
	PROFILE=sample bash scripts/dakp_up.sh

up-prod: ## Run on the prod profile (real build; scope set in .envrc)
	PROFILE=prod bash scripts/dakp_up.sh

down: ## Stop the local Airflow started by the up-* targets
	bash scripts/dakp_down.sh

clean: ## Remove caches, coverage data, the Go binary, and tmp/
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov tmp/
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	rm -f go/dakp-worker
