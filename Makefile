.PHONY: setup test test-go vet lint fmt fmt-check typecheck check precommit clean

# Install Python dev dependencies.
setup:
	uv sync --group dev

# Run the Python test suite. Two phases, because they have opposite parallelism needs:
#   unit        - pure in-process tests; pytest-xdist fans them out (`-n auto` from addopts).
#   integration - spawn NER workers, real `tablassert build-kg` subprocesses, and coverage's
#                 .pth-based child instrumentation. Under xdist this mix deadlocked the CI
#                 runner non-deterministically (worker execnet + spawn children + subprocess
#                 flocks), so it runs sequentially: `-n 0` on the CLI overrides the addopts.
# Coverage data ACCUMULATES across the two runs (--cov-append), and the fail_under gate in
# pyproject.toml is evaluated by coverage at the END of each run over the combined data.
test:
	uv run pytest tests/unit
	uv run pytest tests/integration --cov-append

# Run the Go test suite.
test-go:
	cd go && go test ./...

# Run go vet over the Go module.
vet:
	cd go && go vet ./...

# Run Python lint checks.
lint:
	uv run ruff check .

# Format Python and Go code.
fmt:
	uv run ruff format .
	gofmt -w go

# Check Python and Go formatting without changing files.
fmt-check:
	uv run ruff format --check .
	test -z "$$(gofmt -l go)"

# Run Python type checks.
typecheck:
	uv run pyright

# Run the full local quality gate (mirrors CI).
check: lint fmt-check typecheck test test-go vet

# Run the pre-commit hooks over all files (the `pre-commit` CI job runs this same target).
precommit:
	uv run pre-commit run --all-files

# Remove local build, test, and cache artifacts.
clean:
	rm -rf build dist htmlcov .coverage .coverage.* .pytest_cache .ruff_cache
