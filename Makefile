# DAKP pipeline rebuild — routine command shortcuts (mirrors the Tablassert Makefile).
# All Python runs through uv; Go runs through the go/ module. `make help` lists targets.

.DEFAULT_GOAL := help
.PHONY: help setup install install-ner install-airflow install-kg install-kg-qc install-all test cov coverage \
	lint lint-fix fmt fmt-check typecheck check pre-commit \
	build-go test-go vet-go fmt-go-check check-go check-all run-mock clean

help: ## Show this help
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/{printf "  %-18s %s\n",$$1,$$2}' $(MAKEFILE_LIST) | sort

# ---- Environment / install -------------------------------------------------

setup install: ## Install base + dev dependencies (uv sync)
	uv sync

install-ner: ## Install the heavy biomedical NER backend (GLiNER zero-shot; pulls torch)
	uv sync --extra ner

install-airflow: ## Install the optional Airflow orchestration extra
	uv sync --extra airflow

install-kg: ## Install the Tablassert KG-build extra (PyPI tablassert; laptop-safe)
	uv sync --extra kg

install-kg-qc: ## Install the heavy Tablassert QC-audit extra (adds torch/sentence-transformers)
	uv sync --extra kg-qc

install-all: ## Install every optional extra (ner + airflow + kg + kg-qc)
	uv sync --all-extras

# ---- Python quality gate ---------------------------------------------------

test: ## Run the Python test suite
	uv run pytest

cov coverage: ## Run tests with branch coverage (fail_under=100 configured)
	uv run pytest --cov --cov-report=term-missing

lint: ## Ruff check (Tablassert-style broad select)
	uv run ruff check src tests

lint-fix: ## Ruff check --fix
	uv run ruff check --fix src tests

fmt: ## Ruff format
	uv run ruff format src tests

fmt-check: ## Ruff format --check
	uv run ruff format --check src tests

typecheck: ## Pyright (reportUnusedImport/Variable on)
	uv run pyright

check: lint fmt-check typecheck test ## Full Python gate (lint + format + typecheck + tests)

pre-commit: ## Run all pre-commit hooks
	uv run pre-commit run --all-files

# ---- Go quality gate -------------------------------------------------------

build-go: ## Build the dakp-worker Go binary
	cd go && go build -o dakp-worker ./cmd/dakp-worker

test-go: ## Run Go tests (includes Python-parity golden tests)
	cd go && go test ./...

vet-go: ## Go vet
	cd go && go vet ./...

fmt-go-check: ## Assert gofmt-clean
	cd go && test -z "$$(gofmt -l .)" || (gofmt -l . && exit 1)

check-go: build-go vet-go test-go fmt-go-check ## Full Go gate (build + vet + test + gofmt)

check-all: check check-go ## Full gate: Python + Go

# ---- Run -------------------------------------------------------------------

run-mock: ## Run the mocked end-to-end pipeline (no network, no real Tablassert)
	uv run dakp run --profile mock --fixture-root tests/fixtures/pipeline --workdir tmp/mock-run

# ---- Hygiene ---------------------------------------------------------------

clean: ## Remove caches, coverage data, the Go binary, and tmp/
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov tmp/
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	rm -f go/dakp-worker
