# Logging and observability

How DAKP logs, what it records, and how to read a failed run. Airflow is the primary log
surface; everything else feeds into it (PLAN.md "Logging and observability"). All logging
code lives in [`src/dakp_pipeline/logging_setup.py`](../src/dakp_pipeline/logging_setup.py).

> **Status.** The loguru + stdlib bridge, the file/stderr sinks, and the Go-worker JSON log
relay are **implemented** — a run produces structured `logs/dakp.log`, `build_summary.json`,
`tablassert_handoff.json`, per-artifact manifests, and (when Go workers run) relayed `log/slog`
records. The per-task `task_report.json` and failure **bundles** remain design targets (below).

## Design: three sinks, one record

```text
  Python (loguru)  ──┐                                 ┌── stderr (human, structured)
                    │                                 │
  stdlib logging ───┼── InterceptHandler ──► loguru ──┼── <workdir>/logs/dakp.log (rotating file)
                    │                                 │
                    └── (Airflow only) loguru ──► airflow.task stdlib logger ──► Airflow task files

  Go workers ── log/slog JSON lines ──► relayed into the loguru logger line-by-line
```

### loguru is primary; stdlib is bridged in

`loguru` is the single structured logger. An [`InterceptHandler`](../src/dakp_pipeline/logging_setup.py)
is installed on the stdlib root logger (`logging.basicConfig(handlers=[InterceptHandler()], force=True)`)
so third-party libraries that emit through `logging` flow into loguru and share its sinks
and formatting. This is the canonical loguru integration recipe.

### The Airflow bridge

When the runner or DAG runs under Airflow, `configure_logging(..., for_airflow=True)`
adds a forwarder ([`_stdlib_record_sink`](../src/dakp_pipeline/logging_setup.py)) that
re-emits each loguru record into `logging.getLogger("airflow.task")`, so loguru output
appears in Airflow's per-task log files under
`logs/dag_id=dakp_build/run_id=.../task_id=.../`. `airflow.task.propagate = False` prevents
records looping back through the root `InterceptHandler`. Tests never set `for_airflow`.

### Sinks

`configure_logging(workdir, level, *, for_airflow)` is **idempotent** — repeated calls
replace sinks rather than stacking them (`logger.remove()` first). It installs:

- **stderr** — structured, human-facing. `colorize=None` (auto-colorize only on a real TTY,
  avoiding a terminfo lookup that prints a spurious warning under pytest capture).
- **rotating file** — `<workdir>/logs/dakp.log`, `rotation="20 MB"`, `retention=5`,
  `compression="gz"`. Created only when a workdir is passed.
- **airflow.task forwarder** — only when `for_airflow=True`.

`logging.root` and noisy loggers (`urllib3`, `botocore`, `airflow.task`) are pinned to the
configured `level`.

## Structured fields

Structured context is attached with [`bind(**fields)`](../src/dakp_pipeline/logging_setup.py),
which returns a `logger.bind(...)` logger. Attach task context at task entry and reuse it:

```python
from dakp_pipeline.logging_setup import bind

log = bind(task_id="extract_faers", shard_id="24Q3", artifact_id=ref.blake3)
log.info("parsed quarter", rows=12, cache_hit=False)
```

The shared field schema (Python and the future Go workers use the same names):

| Field | Meaning |
| --- | --- |
| `task_id` | DAG/task name (e.g. `extract_faers`) |
| `shard_id` | release/quarter/bin the record concerns (e.g. `24Q3`) |
| `artifact_id` | `b3:<hex>` of the artifact being produced |
| `input_hash` / `output_hash` | upstream/downstream `b3:<hex>` |
| `rows` / `partitions` | table row/partition counts |
| `elapsed_ms` | stage timing |
| `cache_hit` | whether the artifact was reused from the store |
| `warning_count` | lossy-decision / parse-warning count |

> `run_pipeline` binds `task_id`, `profile`, and `workdir` at the top level
> ([`pipeline.py`](../src/dakp_pipeline/pipeline.py)); each stage and shaper binds its own
> `task_id` plus `artifact_id` / `rows` context as it produces artifacts.

## Go-worker JSON logging (implemented)

The heavy parsing/extraction runs as **native Go workers** in an Airflow Go SDK bundle
([`../go/cmd/dakp-bundle`](../go/cmd/dakp-bundle)). Each Go task is handed a `*slog.Logger` by the
SDK whose output is routed **directly into the Airflow task log** (no Python relay shim): the Go
worker emits `log/slog` JSON using the same field schema above, and the supervisor surfaces it in
the per-task log file alongside the Python tasks' logs. The workers use `golang.org/x/sync/errgroup`
with `SetLimit` for bounded, cancellation-on-first-error shard processing (honouring the task's
`sdk.TIRunContext` cancellation).

> The Go extractors are parity-locked to the pure-Python reference extractors (golden-file parity in
> `go test ./...`); the pure-Python extractors are retained as the reference/test oracle and run in
> the Airflow-free test harness ([`../src/dakp_pipeline/pipeline.py`](../src/dakp_pipeline/pipeline.py)).

## Reports

### Build summary (implemented)

Every run writes `<workdir>/data/reports/build_summary.json` (schema `dakp.build_summary.v1`),
produced by [`pipeline._write_build_summary`](../src/dakp_pipeline/pipeline.py). It records
the profile, generated-at timestamp, workdir, per-table summary (name/path/rows/`artifact_id`),
the Tablassert handoff refs, and the **Translator-readiness contract** result
(`translator_contract.ok`, `problems[]`, per-table `rows` + `missing_columns`):

```jsonc
{
  "schema_version": "dakp.build_summary.v1",
  "profile": "mock",
  "tables": [ { "name": "approved_treats_assertions", "path": "...", "rows": 3, "artifact_id": "b3:…" } ],
  "tablassert": { "handoff_refs": ["…/tablassert_handoff.json"] },
  "translator_contract": { "ok": true, "problems": [], "tables": { "approved_treats_assertions": { "rows": 3, "missing_columns": [] } } }
}
```

### Tablassert handoff manifest (implemented, mock)

`<workdir>/data/reports/tablassert_handoff.json` — written by
[`tablassert/run.py`](../src/dakp_pipeline/tablassert/run.py) in the mock profile. Records
`mode`, `status: "deferred"`, the assertion inputs (table/`artifact_id`/rows), and the
generated config paths. In a full run this is replaced by real KGX outputs.

### Per-task `task_report.json` (design, not yet emitted)

PLAN.md specifies that every task writes a small `task_report.json` alongside its artifact
manifests with timings, row counts, cache hit/miss status, warning summaries, and output
paths, plus **failure bundles**: when a shard fails, write the exact input manifest,
command args, stderr/stdout path, and first N parse warnings, and log the bundle path.
The `data/reports/` directory ([`paths.py`](../src/dakp_pipeline/paths.py)) is reserved for
these; they are not yet emitted (today the per-artifact manifests + `build_summary.json` carry
the row counts, schema fingerprints, and contract results). Track via
[PLAN.md](../PLAN.md) "Reports and failure handling".

## How to read a failed run

Work outward from the human-facing summary to the raw provenance:

1. **`<workdir>/data/reports/build_summary.json`** — check
   `translator_contract.ok`. If `false`, `problems[]` names the missing table or columns;
   per-table `missing_columns` pin the exact contract drift.
2. **`<workdir>/logs/dakp.log`** — structured, time-ordered records. Grep for the failing
   `task_id` / `shard_id`; the bound context (`artifact_id`, `rows`) tells you which
   content-addressed artifact the stage reached. (Under Airflow, read the per-task file at
   `logs/dag_id=dakp_build/run_id=.../task_id=.../`.)
3. **`<workdir>/data/reports/tablassert_handoff.json`** — if the failure is at/after the
   handoff, the `assertion_inputs` and `config_inputs` show exactly what was fed to
   Tablassert.
4. **`<workdir>/data/manifests/<hex>.json`** — the failing artifact's manifest. Its
   `inputs[]` are the upstream artifact ids; walk them to find where the content-addressed
   chain breaks. The `operation.name` tells you which stage produced it.
5. **Rerun from the boundary** — because every stage is content-addressed and
   monkeypatchable, you can rerun a single stage against the cached upstream artifacts
   (see [`runbook.md`](./runbook.md) for cache invalidation and shard reruns).

Progress is intended to be emitted at **shard boundaries**, not per row, to keep logs
useful at scale (PLAN.md). In the single-threaded mock profile the two INFO lines
(`pipeline start` → `pipeline complete`) bracket the whole run.

## Related

- [`architecture.md`](./architecture.md) — the layered stages these logs describe.
- [`runbook.md`](./runbook.md) — reruns, BLAKE3 cache invalidation, shard-level debugging.
