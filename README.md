# DAKP — Drug Approvals Knowledge Provider

Reproducible `uv` Python pipeline that builds **treatment**, **observed-use**, and
**contraindication** assertion tables from DailyMed, Drugs\@FDA, FAERS, and MEDI, ready
to hand off to [Tablassert](https://github.com/SkyeAv/Tablassert) for Translator KGX
modeling. See [`PLAN.md`](./PLAN.md) for the full approved specification.

> **Status — Milestone 1 (scaffold).** The full DAG shape and BLAKE3 content-addressed
> store are in place and the pipeline runs end-to-end on tiny mocked fixtures with **no
> network and no real Tablassert/Airflow installed**. Real source downloaders/extractors,
> NER, and the Tablassert integration land in later milestones.

## Edge families produced

| Predicate | Subject → Object | Primary sources |
| --- | --- | --- |
| `biolink:treats` | drug → disease/phenotype | DailyMed + Drugs\@FDA approvals, FAERS support |
| `biolink:applied_to_treat` | drug → disease/phenotype | FAERS observed use, DailyMed support |
| `biolink:contraindicated_in` | drug → disease/phenotype | MEDI, DailyMed support |

## Quickstart (mocked, laptop-safe)

```bash
uv sync                       # base install; no Airflow required
uv run pytest -q              # unit + mocked integration
uv run dakp run --profile mock \
  --fixture-root tests/fixtures/pipeline \
  --workdir /tmp/dakp-mock
```

The mock run writes uncompressed TSV assertion tables under
`<workdir>/data/tabular/` plus a build summary, with every external call mocked.

## Profiles

| Profile | Concurrency | Sources | Tablassert |
| --- | --- | --- | --- |
| `mock` | 1 thread, 1 GiB | fixtures only | deferred |
| `sample` | 4 threads, 8 GiB | real, bounded sample | deferred |
| `wenceslaus_full` | 64 threads, 128 GiB | real full build | delegates to `../Tablassert` |

## Running under Airflow (optional)

```bash
uv sync --extra airflow
uv run airflow standalone   # then trigger the dakp_build DAG
```

The DAG (`src/dakp_pipeline/dags/dakp_build.py`) is a thin TaskFlow wrapper around the
pure-Python `run_pipeline` runner; the runner is the source of truth and is what tests
exercise.

## Where things land

```
<workdir>/
  data/raw/by-hash/<hash>/   # content-addressed immutable store (BLAKE3)
  data/raw/aliases/          # human-readable names -> store hashes
  data/interim/              # partitioned parquet interim tables
  data/tabular/              # uncompressed TSV assertion tables (Tablassert-facing)
  data/kgx/                  # KGX NDJSON (written by Tablassert in full builds)
  data/manifests/            # per-artifact JSON manifests
  data/reports/              # task_report.json + build_summary.json
  logs/dakp.log              # structured logs (loguru -> stdlib bridge)
```

## Dependency philosophy

Stdlib first. Approved runtime deps only: **polars, loguru, blake3, pydantic**. Airflow
is an optional extra. No web frameworks, ORMs, or generic ETL engines. See
`PLAN.md` → "Dependency philosophy".

## Tablassert handoff

DAKP does everything *up to* the shape Tablassert consumes: acquire → extract → shape
into assertion tables, then generate Tablassert Graph + table configs. Canonical entity
resolution, KGX compilation, dedup, deterministic IDs, and RIG generation are delegated
to `../Tablassert` (Milestone 7). DAKP ships **no** local fallback KGX compiler.
