# Runbook

Common failures, reruns, BLAKE3 cache invalidation, and debugging — for the Milestone-1
scaffold and the patterns that carry forward to the full build.

## Prerequisites

```bash
uv sync                       # base install (polars, loguru, blake3, pydantic); no Airflow
uv run pytest -q              # unit + integration tests, no network
```

## Running the mock pipeline

```bash
uv run dakp run --profile mock \
  --fixture-root tests/fixtures/pipeline \
  --workdir /tmp/dakp-mock
```

Outputs land under `/tmp/dakp-mock/` (see [`README.md`](../README.md#where-things-land)).
A successful run prints `Pipeline complete` and the `build_summary.json` path.

## Running a bounded `prod` smoke run

The `prod` profile runs the **real** fetchers/extractors. Bound the scope so a smoke run
stays tiny (one FAERS quarter, one DailyMed release):

```bash
uv run dakp run --profile prod \
  --quarter-limit 1 --release-limit 1 \
  --workdir /tmp/dakp-prod-smoke
```

This hits the real FDA/DailyMed network endpoints. For an **offline** exercise of the same
real code path (HTTP layer mocked, fixtures served "as if downloaded"), run
`uv run pytest -q tests/integration/test_prod_smoke.py` — it passes in CI with no network and
no real Tablassert (the Tablassert *handoff* runs its real runner; only the `../Tablassert`
subprocess is faked).

## Common failures

### Network / download errors on a non-mock profile

`sample` and `prod` run the real stdlib-HTTP downloaders (DailyMed full releases, FAERS
quarterly zips, Drugs@FDA data files), so they need network access to the FDA/DailyMed
endpoints. A fetch failure raises loudly (no silent fixture fallback).
**Fix:** check connectivity/proxy, or validate the real path offline with the bounded smoke
test above. Bound scope with `--quarter-limit` / `--release-limit` to keep a real run small.

### `--fixture-root is required for the mock profile`

The CLI enforces this ([`cli.py`](../src/dakp_pipeline/cli.py)). **Fix:** pass
`--fixture-root tests/fixtures/pipeline`.

### `fixture not found: <path>` / `cannot ingest missing file`

A fetcher's fixture tuple names a file that is not under `--fixture-root`. **Fix:** check
the fixture path in the relevant [`sources/<x>.py`](../src/dakp_pipeline/sources/) module
against `tests/fixtures/pipeline/`.

### `RuntimeError: run_airflow=True requires the airflow extra`

You passed `--run-airflow` without Airflow installed. **Fix:** `uv sync --extra airflow`.

### `NotImplementedError: real ../Tablassert integration lands in Milestone 7`

A full run reached `run_tablassert` with `run_tablassert=True` on a non-mock profile.
There is deliberately **no local KGX fallback**. **Fix:** stay on the mock profile (deferred
handoff) until Milestone 7, or monkeypatch `dakp_pipeline.tablassert.run` as the
[integration test](../tests/integration/test_mock_pipeline.py) does.

### `subject_curie` / `subject_name` are empty in the output TSVs

Expected in the scaffold. The lexical dictionary baseline resolves only disease **objects**;
subject CURIEs are resolved by fullmap/Tablassert during modeling (Milestone 4+). See
[`tabular-contracts.md`](./tabular-contracts.md).

### `translator_contract.ok == false` in `build_summary.json`

A contract check failed. **Fix:** read `problems[]` (names the missing table or the missing
columns) and the per-table `missing_columns` list, then check the corresponding shaper in
[`assertions/`](../src/dakp_pipeline/assertions/) and its column contract in
[`schemas.py`](../src/dakp_pipeline/io/schemas.py).

### The DAG did not register / Airflow does not see `dakp_build`

The DAG module is **import-safe without Airflow** (guarded imports + no-op fallbacks), so
it loads under `uv sync`, but it only registers a real DAG when Airflow is present (the
module-level `dag_obj = dakp_build()` runs only when `_AIRFLOW_AVAILABLE`). **Fix:**
`uv sync --extra airflow`, then ensure `src/dakp_pipeline/dags` is on Airflow's DAG folder
path or imported by your Airflow config.

## Reruns

DAKP is content-addressed and idempotent. Because reuse is keyed by BLAKE3 content hash
(not filename or mtime), **re-running the identical mock pipeline re-ingests the identical
fixtures as cache hits** — the second run is cheap and reproduces the same artifact ids.

```bash
uv run dakp run --profile mock --fixture-root tests/fixtures/pipeline --workdir /tmp/dakp-mock
# second invocation: fixture ingests are cache hits; outputs regenerate deterministically
```

> **`force` is declared but not yet wired.** The `Profile.force` field exists and is
> threaded into `ctx.params["force"]`, but the Milestone-1 `ArtifactStore` does **not**
> consult it — cache logic is purely `dest.exists()`. Force-rerun today means deleting the
> relevant store entries (next section). Wiring `force` into the store is a milestone target.

## BLAKE3 cache invalidation

Artifacts live at `data/raw/by-hash/<hex>/<name>` (copied inputs) or in place
(interim/TSV/configs); manifests live at `data/manifests/<hex>.json`. To invalidate:

| Goal | Action |
| --- | --- |
| Re-fetch one raw input | delete `data/raw/by-hash/<hex>/` (and its alias); the next run re-ingests |
| Re-extract one interim table | delete the interim parquet **and** its manifest; the extractor reruns against the cached upstream |
| Re-shape one assertion table | delete `data/tabular/<table>.tsv` (and its manifest); the shaper reruns |
| Start completely fresh | delete the workdir (or pass a new `--workdir`) |

To find the `<hex>` for an artifact, read `build_summary.json` (`tables[].artifact_id`) or
the manifest's `artifact_id` / `inputs[]` fields. The `inputs[]` chain lets you trace which
**downstream** artifacts depend on a changed upstream: change a fixture → its `b3:<hex>`
changes → every manifest listing it in `inputs[]` is now stale and will be regenerated.

Because the schema is also fingerprinted (`table.schema_fingerprint`), a column-contract
change is itself a signal that downstream artifacts must regenerate.

## Inspecting outputs

```bash
# A table by suffix (parquet vs TSV) — schemas.read_table handles both
uv run python -c "import polars as pl; print(pl.read_parquet('/tmp/dakp-mock/data/interim/faers/cases.parquet'))"

# Assertion TSVs are plain text
column -t -s$'\t' /tmp/dakp-mock/data/tabular/approved_treats_assertions.tsv | less -S

# Read any manifest
cat /tmp/dakp-mock/data/manifests/<hex>.json | jq .

# The build summary (contract result + per-table rows/artifact ids)
cat /tmp/dakp-mock/data/reports/build_summary.json | jq .

# The Tablassert handoff (what would be fed to ../Tablassert)
cat /tmp/dakp-mock/data/reports/tablassert_handoff.json | jq .
```

## Shard-level debugging

> The `mock` profile is **single-threaded** (`threads=1`); Airflow dynamic task mapping and
> Go workers are not yet wired (see [`architecture.md`](./architecture.md#sharding-and-concurrency)).
> Today "shard-level" debugging is effectively **stage-level** debugging.

To debug a single stage against cached upstream artifacts without re-running the whole
pipeline:

1. Run the full mock pipeline once to populate the store.
2. In a Python REPL or test, build a `TaskContext` (see
   [`pipeline._build_context`](../src/dakp_pipeline/pipeline.py)) pointing at the same
   workdir, then call the individual stage function directly (e.g.
   `approved_treats.transform([...refs], ctx)`), passing the cached upstream `ArtifactRef`s.
   Because stages are pure functions over paths/config, this reproduces the stage in
   isolation.
3. Inspect the produced parquet/TSV/manifest as above.

Every stage logs via [`bind(task_id=..., ...)`](../src/dakp_pipeline/logging_setup.py); grep
`logs/dakp.log` for the `task_id` to see the bound `artifact_id` / `rows` context (see
[`logging.md`](./logging.md#how-to-read-a-failed-run) for the full read-a-failed-run recipe).

## Tests

```bash
uv run pytest -q                    # all tests
uv run pytest tests/unit -q         # store, hashing, config, CLI, tablassert configs
uv run pytest tests/integration -q  # mocked end-to-end (monkeypatches every boundary)
```

The integration test is the canonical example of substituting every external boundary
(fetchers → `ctx.fixture(...)`, Tablassert → a fake). Mirror it when prototyping a new
stage against fixtures.

## Related

- [`logging.md`](./logging.md) — reading a failed run from logs, summary, and manifests.
- [`architecture.md`](./architecture.md) — the content-addressed store and provenance DAG.
- [`README.md`](../README.md) — profiles, quickstart, and the milestone roadmap.
