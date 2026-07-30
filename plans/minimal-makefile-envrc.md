# Minimal Makefile + documented `.envrc`

## Context

The dev surface is overwhelming: the `Makefile` has ~25 targets (many redundant — `setup`/`install`,
`cov`/`coverage`, `check`/`check-all`/`check-go`), and the run orchestrator (`scripts/dakp_up.sh`)
scatters config across env vars **and hardcodes** the run scope (`quarter_limit=1`, `release_limit=1`,
`log_level=INFO`) into the `dakp_config` Airflow Variable — so changing scope means editing the script
(the wenceslaus runbook calls this out as a pain point for full prod builds).

Goal (user-approved, "full recommendations"):
- A **super minimal** Makefile: `up-mock` / `up-sample` / `up-prod` + `down` (+ `install`, `help`, `clean`).
- A single committed, documented **`.envrc`** (direnv) centralizing every knob with comments + options.
- Make the orchestrator **fully env-driven** (no hardcoded limits) so nothing in `scripts/` needs hand-editing.
- Update the user-facing docs to the new target names.

Confirmed safe: no CI / justfile / tox / nox exists, and no YAML references any make target; pre-commit
calls `uv run` directly. Dropping the quality-gate make targets breaks nothing automated.

## Profiles (`src/dakp_pipeline/config.py`)

| Profile  | Sources         | quarter/release default | Tablassert          |
| -------- | --------------- | ----------------------- | ------------------- |
| `mock`   | fixtures only   | q=1 / rel=fixtures      | deferred (manifest) |
| `sample` | real, bounded   | q=1 / rel=all           | deferred            |
| `prod`   | real full build | **all / all (unbounded)** | installed CLI     |

`runtime.build_context_from_config` applies a limit override **only when the Variable value is not
`null`**; `null` ⇒ use the profile default. So "empty = profile default" (for `prod` = full build).
An empty *string* would crash `int("")` — the script must emit JSON `null`, not `""`.

## Deliverable 1 — `Makefile` (rewrite, minimal)

```makefile
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
```

Single `down` (one `AIRFLOW_HOME` ⇒ one Airflow instance; "down per profile" isn't meaningful).

## Deliverable 2 — `.envrc` (new, committed, direnv-loaded)

Safe defaults so committing is harmless; active knobs exported, rare ones commented with defaults.

```bash
# .envrc — DAKP run configuration, auto-loaded by direnv (https://direnv.net).
# Run `direnv allow` once after editing. Every variable is optional: unset/empty means
# "use the documented default". The make targets (up-mock/up-sample/up-prod/down) read these.
# No direnv? `source .envrc` manually, or pass vars inline: WORKDIR=/x make up-mock.

# --- Profile -----------------------------------------------------------------
# Pipeline profile: sources + concurrency + Tablassert handoff mode.
#   mock   — tiny fixtures, no network, no Tablassert (laptop/CI-safe default)
#   sample — real sources, bounded scope
#   prod   — real full build + installed Tablassert CLI (workstation-class host)
# `make up-mock|up-sample|up-prod` OVERRIDE this; it's the default for direct script runs.
export PROFILE=mock

# --- Run scope (empty = profile default) -------------------------------------
# QUARTER_LIMIT  — max FAERS quarters to process.   empty -> mock/sample: 1, prod: ALL
# RELEASE_LIMIT  — max DailyMed releases to process. empty -> prod: ALL
# Set a number to bound a run (laptop-safe prod smoke: set both to 1).
export QUARTER_LIMIT=
export RELEASE_LIMIT=

# --- Logging -----------------------------------------------------------------
# Pipeline log level: DEBUG | INFO | WARNING | ERROR. Default: INFO.
export LOG_LEVEL=INFO

# --- Workdir / fixtures (uncomment to override) ------------------------------
# WORKDIR      — root for ALL run artifacts (raw store, interim parquet, TSVs, KGX, reports, logs).
#                Default: <repo>/tmp/airflow-run/data. Use a big disk for real builds.
# export WORKDIR=/local_raid1/dakp/work
# FIXTURE_ROOT — mock-profile fixtures. Default: <repo>/tests/fixtures/pipeline.
# export FIXTURE_ROOT=/path/to/tests/fixtures/pipeline

# --- Airflow home + port (uncomment to override) -----------------------------
# AIRFLOW_HOME — Airflow state (db, logs, bundles, pid). Default: <repo>/tmp/airflow-home.
# export AIRFLOW_HOME=/local_raid1/dakp/airflow-home
# AIRFLOW_PORT — API/webserver port. Default: 8090 (8080 is often taken by the aoe daemon).
# export AIRFLOW_PORT=8090

# --- Airflow task pools (advanced; rarely changed) ---------------------------
# Concurrency-bounding pools the DAG schedules onto (match the DAG constants).
# export DOWNLOAD_POOL=dakp_download
# export EXTRACT_POOL=dakp_extract
```

## Deliverable 3 — `scripts/dakp_up.sh` (env-drive the hardcoded bits)

Replace the hardcoded `quarter_limit: 1, release_limit: 1, log_level: "INFO"` in the `dakp_config`
Variable with env-driven values; empty ⇒ JSON `null` (⇒ profile default).

- Add near the other var defaults:
  ```bash
  LOG_LEVEL="${LOG_LEVEL:-INFO}"
  # empty -> JSON null -> profile default (prod = unbounded full build)
  ql_json() { [[ -n "${1:-}" ]] && printf '%s' "$1" || printf 'null'; }
  QUARTER_JSON="$(ql_json "${QUARTER_LIMIT:-}")"
  RELEASE_JSON="$(ql_json "${RELEASE_LIMIT:-}")"
  ```
- Change the `airflow variables set dakp_config` JSON to embed `$QUARTER_JSON` / `$RELEASE_JSON`
  unquoted (so `null` or `123` embeds raw) and `\"log_level\": \"$LOG_LEVEL\"`.
- Update the header usage comment (line 4) to mention `.envrc` + the new vars.
- (Optional, small) numeric guard on QUARTER_LIMIT/RELEASE_LIMIT to fail fast with a clear message.

`scripts/dakp_down.sh`: no change needed (already reads only `AIRFLOW_HOME`).

## Deliverable 4 — `.gitignore`

Add `.direnv/` (direnv cache). Do **not** ignore `.envrc` (it's committed with safe defaults).

## Deliverable 5 — doc updates (new target names; fix stale install-* refs)

Scope = make-target / `.envrc` / install-extras references only. **Not** touching `plans/` or
`PLAN.md` (historical records) or the runbook's pre-existing retired-CLI (`cli.py`/`--flags`) rot.

- **`README.md`** — Quickstart (`make install` / `direnv allow` / `make up-mock` / `make down`);
  rewrite the "Makefile targets" table to the minimal set; replace `make run`→`make up-*`,
  `make install-all`→`make install`; prod-smoke section → `make up-prod` + `.envrc` scope (drop the
  "edit scripts/dakp_up.sh to unbound" note); verification `make check-all`→`uv run pre-commit run --all-files` (+ `cd go && go test ./...`).
- **`docs/runbook.md`** — Prerequisites (drop `install-ner/kg/kg-qc/all`; `uv sync` is complete);
  `make run`→`make up-mock`, `make down` kept; prod-smoke `PROFILE=prod … make run`→`make up-prod`
  + `.envrc` (drop the "pins quarter/release to 1, adjust it" note); `make install-kg`/`install-ner`
  → `uv sync`; Tests `make check-all`→ pre-commit/`uv run` + `go test`.
- **`docs/wenceslaus-runbook.md`** — Prerequisites (`uv sync`; `cd go && go build ./cmd/dakp-worker`
  instead of `make build-go`; drop `--all-extras`/`--extra` since no extras); `PROFILE=prod WORKDIR=…
  make run`→`make up-prod` + `.envrc` WORKDIR; `make check-all`→ pre-commit/`uv run` + `go test`;
  laptop-loop + troubleshooting `make run`→`make up-*`.
- **`go/README.md`** — `make build-go`/`make check-go`→`cd go && go build ./...` / `go test ./...`;
  `make run`→`make up-mock`.
- **`src/dakp_pipeline/ner/README.md`** — drop `# or: make install-ner` (`uv sync` is complete).
- **`docs/tablassert-handoff.md`** — `make install-kg`→`uv sync`.
- **Docstring consistency pass** (`make run`→`make up-mock`, comments only, no behavior change):
  `src/dakp_pipeline/pipeline.py`, `src/dakp_pipeline/dags/__init__.py`,
  `src/dakp_pipeline/dags/dakp_build.py`, `tests/integration/test_mock_pipeline.py`.

## Files to modify

`Makefile`, `.envrc` (new), `scripts/dakp_up.sh`, `.gitignore`, `README.md`, `docs/runbook.md`,
`docs/wenceslaus-runbook.md`, `go/README.md`, `src/dakp_pipeline/ner/README.md`,
`docs/tablassert-handoff.md`, + 4 docstring-only files listed above.

## Reuse

- Existing `scripts/dakp_up.sh` / `scripts/dakp_down.sh` stay the implementation; only the config
  sourcing changes. No new scripts.
- `runtime.build_context_from_config` already treats `null` limits as "profile default" — no Python
  changes needed for the env-driven scope.

## Verification

1. `make help` lists exactly: help, install, up-mock, up-sample, up-prod, down, clean.
2. `direnv allow` loads `.envrc` (or `source .envrc`); `echo $PROFILE` → `mock`.
3. `make up-mock` runs the mocked DAG end-to-end (no network), prints `SUCCESS` + `build_summary.json`;
   works even with `.envrc` unloaded (script defaults).
4. `make down` stops Airflow (no lingering `airflow` processes: `pgrep -af airflow` empty).
5. Scope is env-driven: set `QUARTER_LIMIT=1 RELEASE_LIMIT=1` → `dakp_config` Variable carries
   `"quarter_limit": 1, "release_limit": 1`; leave empty → carries `null` (check via
   `uv run airflow variables get dakp_config`).
6. `make clean` removes `tmp/` + caches.
7. Quality gates still pass via their direct commands (unchanged): `uv run pre-commit run --all-files`
   and `cd go && go test ./...` — confirms the docstring edits + script change broke nothing.
8. `grep -rn "make run\|make install-\|make check-all\|make build-go" README.md docs/ Makefile` → no
   stale references remain (outside `plans/`, `PLAN.md`).
