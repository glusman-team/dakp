.PHONY: setup test test-go lint fmt fmt-check typecheck check clean

# Install Python dev dependencies.
setup:
	uv sync --group dev

# Run the Python test suite (parallel, with the 100% coverage gate from pyproject.toml).
test:
	uv run pytest

# Run the Go test suite.
test-go:
	cd go && go test ./...

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
check: lint fmt-check typecheck test test-go
	cd go && go vet ./...

# Remove local build, test, and cache artifacts.
clean:
	rm -rf build dist htmlcov .coverage .coverage.* .pytest_cache .ruff_cache
