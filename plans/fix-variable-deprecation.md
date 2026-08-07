# Fix: `airflow.models.Variable` DeprecationWarning in task logs

## Context

Task logs from the `dakp_build` DAG emit:

```
WARNING - Using Variable.get from `airflow.models` is deprecated. Please use `get` on
Variable from sdk (`airflow.sdk.Variable`) instead category=DeprecationWarning
```

It appears in **several tasks** because every Python task calls `_ctx()` → `_cfg()` →
`Variable.get(...)` (`src/dakp_pipeline/dags/dakp_build.py:47-49`), and `_cfg` imports
`Variable` from the deprecated `airflow.models` path (line 21). Under the Airflow 3 task
runtime, `airflow.models.Variable` is a shim that warns; the task-side API is
`airflow.sdk.Variable`.

The project pins `apache-airflow>=3,<4` (pyproject.toml) and runs Airflow 3.3.0, where
`airflow.sdk.Variable` has existed since 3.0 — no compat concern.

## Approach

Swap the import to the SDK path. Verified drop-in: `airflow.sdk.Variable.get` has the
identical signature `(key, default=NOTSET, deserialize_json=False)`
(`.venv/.../airflow/sdk/definitions/variable.py`), so the call site
`Variable.get(CONFIG_VARIABLE, deserialize_json=True)` is unchanged.

## Files to modify

- `src/dakp_pipeline/dags/dakp_build.py` — the only place in the codebase importing
  `airflow.models` (verified by grep).

Not affected (checked):
- `src/dakp_pipeline/cli.py` — sets the Variable via the `airflow variables set`
  subprocess, no Python import.
- `go/cmd/dakp-bundle` — reads the Variable through the Go SDK client.
- `src/dakp_pipeline/dags/__init__.py`, tests — no `airflow.models` references.

## Steps

- [x] In `src/dakp_pipeline/dags/dakp_build.py`:
  - Delete `from airflow.models import Variable` (line 21).
  - Fold `Variable` into the existing SDK import:
    `from airflow.sdk import Variable, dag, task`
    (ruff/isort case-sensitive ordering puts `Variable` first).
- [x] No change to `_cfg()` — the call already matches the SDK signature.

## Verification

1. [x] `uv run ruff check src/dakp_pipeline/dags/dakp_build.py` and
   `uv run ruff format --check src/dakp_pipeline/dags/dakp_build.py` — clean.
2. [x] `uv run pytest tests/unit` — 640 passed (two consecutive green runs).
3. [x] Deprecation probe: module imports clean under `-W error::DeprecationWarning`,
   and `dakp_build.Variable is airflow.sdk.Variable` asserted. Also inspected the
   installed Airflow 3.3.0 source: the warning is emitted by
   `airflow/models/variable.py:Variable.get` only when a task execution context
   exists (`SUPERVISOR_COMMS`), after which it delegates to `airflow.sdk.Variable.get`
   anyway — so the new import is the exact same code path minus the warning.
4. [x→user] End-to-end: no standalone Airflow running on this box; starting `dakp up`
   (downloads + Go bundle) is disproportionate for a one-line import swap. Confirm on
   the next run: grep task logs for `DeprecationWarning` — expect none.

### Pre-existing issues observed (NOT caused by this change, verified on stashed baseline)

- Coverage gate reports only `tests/` files (never `src/`, despite
  `[tool.coverage.run] source`) and fails at 99.8% — identical on the fully-stashed
  baseline, so it predates this change.
- `test_tablassert_configs.py::test_real_runner_records_failure` failed once flakily
  (passed on both subsequent full runs); it exercises `tablassert.py`, which another
  session is actively modifying.
