# Runbook

Common failures, reruns, BLAKE3 cache invalidation, and debugging for the DAKP pipeline. For the
**full production build** on `wenceslaus` (fullmap + prod KG), see
[`wenceslaus-runbook.md`](./wenceslaus-runbook.md).

## Prerequisites

```bash
uv sync                       # base install (polars, loguru, blake3, pydantic); no Airflow/NER/Tablassert
uv run pytest -q              # unit + integration tests, no network
```

Optional extras (never required for the base install or the test suite):

```bash
make install-ner       # [ner] GLiNER production NER (pulls torch)
make install-kg        # [kg] PyPI tablassert (KG build; laptop-safe)
make install-kg-qc     # [kg-qc] tablassert[qc] audit (pulls torch; beefy hosts)
make install-airflow   # [airflow] orchestration
```

## Running the mock pipeline

```bash
uv run dakp run --profile mock \
  --fixture-root tests/fixtures/pipeline \
  --workdir /tmp/dakp-mock
# or: make run-mock
```

Outputs land under `/tmp/dakp-mock/` (see [`../README.md`](../README.md#where-things-land)). A
successful run prints `Pipeline complete` and the `build_summary.json` path. It needs no network and
no real Tablassert (the mock profile defers the handoff).

## Running a bounded `prod` smoke run

The `prod` profile runs the **real** fetchers/extractors/NER/aggregation. Bound the scope so a smoke
run stays tiny (one FAERS quarter, one DailyMed release):

```bash
uv run dakp run --profile prod \
  --quarter-limit 1 --release-limit 1 \
  --workdir /tmp/dakp-prod-smoke
```

This hits the real FDA/DailyMed network endpoints. For an **offline** exercise of the same real code
path (HTTP layer mocked, fixtures served "as if downloaded"), run
`uv run pytest -q tests/integration/test_prod_smoke.py` — it passes in CI with no network. The
Tablassert *handoff* runs its real runner; only the `tablassert` subprocess is faked unless the
`[kg]` extra is installed.

## Common failures

### Network / download errors on a non-mock profile

`sample` and `prod` run the real stdlib-HTTP downloaders (DailyMed full releases, FAERS quarterly
zips, Drugs@FDA data files), so they need network access to the FDA/DailyMed endpoints. A fetch
failure raises loudly (no silent fixture fallback). **Fix:** check connectivity/proxy, or validate
the real path offline with the bounded smoke test above. Bound scope with `--quarter-limit` /
`--release-limit` to keep a real run small.

### `--fixture-root is required for the mock profile`

The CLI enforces this ([`cli.py`](../src/dakp_pipeline/cli.py)). **Fix:** pass
`--fixture-root tests/fixtures/pipeline`.

### `fixture not found: <path>` / `cannot ingest missing file`

A fetcher's fixture tuple names a file that is not under `--fixture-root`. **Fix:** check the fixture
path in the relevant [`sources/<x>.py`](../src/dakp_pipeline/sources/) module against
`tests/fixtures/pipeline/`.

### `RuntimeError: run_airflow=True requires the airflow extra`

You passed `--run-airflow` without Airflow installed. **Fix:** `uv sync --extra airflow`.

### `RuntimeError: tablassert is not available: install the [kg] extra …`

A full (non-mock) run reached `run_tablassert` with `run_tablassert=True` but `tablassert` is not
importable and no editable-checkout override is set. There is deliberately **no local KGX
fallback**. **Fix:** `uv sync --extra kg` (or `make install-kg`), or point at a local checkout via
`DAKP_TABLASERT_DIR` / the `tablassert_dir` param. The mock profile defers the handoff and never
needs Tablassert.

### `NERDependencyError: … install the [ner] extra`

A production-mode `DiseaseNER(offline=False)` ran without the `[ner]` extra (gliner missing).
**Fix:** `uv sync --extra ner` (or `make install-ner`). Offline mode (the default, used by tests and
the mock pipeline) needs no NER deps.

### `subject_curie` / `object_curie` empty in the output TSVs

Expected by design. FAERS `applied_to_treat` subjects carry no source drug id, and
`contraindicated_in` objects are mined **mention text** — both are resolved to CURIEs by
Tablassert/fullmap at `build-kg`. `treats` subjects carry the DailyMed UNII; resolved disease
objects carry MONDO/HP from the lexical baseline. See
[`semantic-equivalence.md`](./semantic-equivalence.md) and [`tabular-contracts.md`](./tabular-contracts.md).

### `translator_contract.ok == false` in `build_summary.json`

A contract check failed. **Fix:** read `problems[]` (names the missing table or columns) and the
per-table `missing_columns` list, then check the corresponding shaper in
[`assertions/`](../src/dakp_pipeline/assertions/) and its column contract in
[`schemas.py`](../src/dakp_pipeline/io/schemas.py).

### The DAG did not register / Airflow does not see `dakp_build`

The DAG module is **import-safe without Airflow** (guarded imports + no-op fallbacks), so it loads
under `uv sync`, but it only registers a real DAG when Airflow is present. **Fix:**
`uv sync --extra airflow`, then ensure `src/dakp_pipeline/dags` is on Airflow's DAG folder path.

## Reruns

DAKP is content-addressed and idempotent. Because reuse is keyed by BLAKE3 content hash (not
filename or mtime), **re-running the identical pipeline re-ingests the identical inputs as cache
hits** — the second run is cheap and reproduces the same artifact ids.

`--force` forces the **acquisition** layer to re-download sources unconditionally (and re-fetch NER
models). Because the store is content-addressed, re-ingesting byte-identical content is still a
cache hit; `--force` matters when the upstream content may have changed and you want to re-pull it.

## BLAKE3 cache invalidation

Artifacts live at `data/raw/by-hash/<hex>/<name>` (copied inputs) or in place
(interim/TSV/configs); manifests live at `data/manifests/<hex>.json`. To invalidate:

| Goal | Action |
| --- | --- |
| Re-fetch one raw input | delete `data/raw/by-hash/<hex>/` (and its alias); the next run re-ingests |
| Re-extract one interim table | delete the interim parquet **and** its manifest; the extractor reruns against the cached upstream |
| Re-shape one assertion table | delete `data/tabular/<table>.tsv` (and its manifest); the shaper reruns |
| Regenerate Tablassert configs | delete `<workdir>/tables/*.yaml` (and their manifests) |
| Start completely fresh | delete the workdir (or pass a new `--workdir`) |

To find the `<hex>` for an artifact, read `build_summary.json` (`tables[].artifact_id`) or the
manifest's `artifact_id` / `inputs[]` fields. The `inputs[]` chain lets you trace which
**downstream** artifacts depend on a changed upstream. Because the schema is fingerprinted
(`table.schema_fingerprint`), a column-contract change is itself a signal to regenerate downstream.

## Inspecting outputs

```bash
# An interim table by suffix (parquet vs TSV) — schemas.read_table handles both
uv run python -c "import polars as pl; print(pl.read_parquet('/tmp/dakp-mock/data/interim/faers/cases.parquet'))"

# Assertion TSVs are plain text
column -t -s$'\t' /tmp/dakp-mock/data/tabular/approved_treats_assertions.tsv | less -S

# Generated Tablassert configs (graph + per-table)
cat /tmp/dakp-mock/tables/graph.yaml

# Read any manifest
cat /tmp/dakp-mock/data/manifests/<hex>.json | jq .

# The build summary (contract result + per-table rows/artifact ids)
cat /tmp/dakp-mock/data/reports/build_summary.json | jq .

# The Tablassert handoff (what would be fed to `tablassert build-kg`)
cat /tmp/dakp-mock/data/reports/tablassert_handoff.json | jq .
```

## Shard-level debugging

The `mock` profile is single-threaded (`threads=1`); prod shards via Airflow dynamic task mapping
and the Go workers. To debug a single stage against cached upstream artifacts without re-running the
whole pipeline:

1. Run the pipeline once to populate the store.
2. In a Python REPL or test, build a `TaskContext` (see
   [`pipeline._build_context`](../src/dakp_pipeline/pipeline.py)) pointing at the same workdir, then
   call the individual stage function directly (e.g. `approved_treats.transform([...refs], ctx)`),
   passing the cached upstream `ArtifactRef`s.
3. Inspect the produced parquet/TSV/manifest as above.

Every stage logs via [`bind(task_id=..., ...)`](../src/dakp_pipeline/logging_setup.py); grep
`logs/dakp.log` for the `task_id` to see the bound `artifact_id` / `rows` context (see
[`logging.md`](./logging.md#how-to-read-a-failed-run)). Go workers emit structured `log/slog` JSON
on stderr that Python relays into the same log.

## Tests

```bash
uv run pytest -q                                          # all tests
uv run pytest tests/unit -q                               # store, hashing, config, CLI, NER, assertions, tablassert
uv run pytest tests/integration -q                        # mocked end-to-end + semantic-equivalence + prod smoke
uv run pytest tests/integration/test_semantic_equivalence.py -q   # preserved-semantics guardrail
make check-all                                            # Python gate + Go parity gate
```

The integration tests are the canonical examples of substituting every external boundary (fetchers →
`ctx.fixture(...)`, Tablassert → a fake). Mirror them when prototyping a new stage against fixtures.

## Related

- [`wenceslaus-runbook.md`](./wenceslaus-runbook.md) — the full production build (fullmap + prod KG).
- [`logging.md`](./logging.md) — reading a failed run from logs, summary, and manifests.
- [`architecture.md`](./architecture.md) — the content-addressed store and provenance DAG.
- [`../README.md`](../README.md) — profiles, quickstart, and Makefile targets.
