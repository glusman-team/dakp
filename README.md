# DAKP — Drug Approvals Knowledge Provider

Reproducible `uv` Python pipeline that builds **treatment**, **observed-use**, and
**contraindication** assertion tables from DailyMed, Drugs@FDA, FAERS, and MEDI, then
hands them to [Tablassert](https://github.com/SkyeAv/Tablassert) for Translator KGX
modeling. The full approved specification lives in [`PLAN.md`](./PLAN.md); this README is
the operational entry point and links into [`docs/`](./docs).

> **Status — Milestone 1 (scaffold).** The full DAG shape and the BLAKE3 content-addressed
> store are in place, and the pipeline runs end-to-end on tiny mocked fixtures with **no
> network and no real Tablassert/Airflow installed**. Real source downloaders/extractors,
> NER/canonical resolution, and the live Tablassert integration land in later milestones
> (see [Roadmap](#roadmap)). The `sample` and `wenceslaus_full` profiles exist as
> declarations but their fetchers and Tablassert call are stubs that fail loudly today.

## Why a rebuild

The legacy DAKP build is a collection of Perl/Python scripts (`DailyMed/`, `FAERS/`,
`DrugsFDA/`, `matrix/`, `bin/`) that mix downloading, parsing, lexical matching, edge
modeling, and KGX serialization in monolithic scripts. It is brittle: hardcoded local
paths (`/ssd2/sqlite/BABEL.db`, …), missing Perl libraries, shell side effects, and no
checkpoints. The rebuild replaces it with:

- A reproducible `uv` project with pinned dependencies and `uv.lock`.
- A typed, monkeypatchable pipeline where every stage communicates through
  **BLAKE3 content-addressed `ArtifactRef` handles** — restartable, cacheable by hash,
  and trivially fakeable in tests.
- An **Apache Airflow DAG** (`dakp_build`) as the orchestration surface, with a
  pure-Python runner as the source of truth that tests exercise directly.
- A clear **delegation boundary**: DAKP acquires, extracts, and shapes tables; canonical
  entity resolution, KGX compilation, dedup, deterministic IDs, and RIG generation are
  delegated to Tablassert/fullmap. DAKP ships **no local fallback KGX compiler**.

Legacy scripts are retained in-tree only for audit; the new DAG never imports them.

## Edge families produced

Three Translator edge families are first-scope (see [Provenance semantics](#provenance-semantics)):

| Predicate | Subject → Object | Primary source | DAKP role |
| --- | --- | --- | --- |
| `biolink:treats` | drug → disease/phenotype | DailyMed + Drugs@FDA approvals; FAERS support | **primary** knowledge source |
| `biolink:applied_to_treat` | drug → disease/phenotype | FAERS observed use; DailyMed support | **aggregator** over FAERS |
| `biolink:contraindicated_in` | drug → disease/phenotype | MEDI; DailyMed support | **aggregator** over MEDI |

## Quickstart (mocked, laptop-safe)

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # base install; Airflow NOT required
uv run pytest -q              # unit + mocked integration (no network)
uv run dakp run --profile mock \
  --fixture-root tests/fixtures/pipeline \
  --workdir /tmp/dakp-mock
```

The mock run needs no network and no real Tablassert. It writes three uncompressed TSV
assertion tables, generated Tablassert configs, a build summary, and a deferred-handoff
manifest (see [Where things land](#where-things-land)). All fetchers are monkeypatchable;
`tests/integration/test_mock_pipeline.py` shows the boundary.

## Profiles

Profiles are defined in Python at [`src/dakp_pipeline/config.py`](./src/dakp_pipeline/config.py)
(`load_profile`) — the base install needs no `pyyaml`. An unknown profile name raises at
startup rather than silently defaulting.

| Profile | Threads | Memory | Sources | Tablassert |
| --- | --- | --- | --- | --- |
| `mock` | 1 | 1 GiB | fixtures only | deferred (writes handoff manifest) |
| `sample` | 4 | 8 GiB | real, bounded sample | deferred |
| `wenceslaus_full` | 64 | 128 GiB | real full build | delegates to `../Tablassert` |

> `sample` and `wenceslaus_full` fetchers are **Milestone-2 stubs**: calling them today
> raises `NotImplementedError` ("only the mock profile is implemented"). Real acquisition
> and the live Tablassert call land in Milestones 2 and 7 respectively.

## Running the full build on `wenceslaus`

Target host: `wenceslaus` — Ubuntu 24.04, dual Xeon Gold 6230 (80 logical CPUs), 187 GiB
RAM, `/local_raid1` ~1.75 TiB, optional NVIDIA P100 GPUs. The full profile targets this
workstation-class host; design also keeps laptop-safe modes.

```bash
uv sync --extra airflow          # Airflow is an optional dependency group
uv run dakp run --profile wenceslaus_full \
  --workdir /local_raid1/dakp
# or via the Airflow scheduler: trigger the dakp_build DAG
```

Until Milestones 2 + 7 land, this raises at the first real acquisition step. The
concurrency budgets, sharding model, and `/local_raid1` I/O guidance are documented in
[`docs/architecture.md`](./docs/architecture.md).

## The DAG

`dakp_build` is the orchestration DAG
([`src/dakp_pipeline/dags/dakp_build.py`](./src/dakp_pipeline/dags/dakp_build.py)),
implemented with the Airflow TaskFlow API. It is a thin wrapper around the same stage
functions the pure-Python runner (`src/dakp_pipeline/pipeline.py`) calls, and is
**import-safe without Airflow installed** (imports are guarded; no-op decorator
fallbacks let `uv sync` + tests load the module). The 11-task graph:

```text
acquire_sources ─┬─▶ extract_dailymed  ─┐
                 ├─▶ extract_faers      ─┤
                 ├─▶ extract_drugsfda   ─┼─▶ {shape_treatment_tables,
                 └─▶ extract_medi       ─┘    shape_faers_use_tables,
                                                 shape_contraindication_tables}
                                                ─▶ generate_tablassert_configs
                                                ─▶ run_tablassert
                                                ─▶ write_build_summary
```

What each task produces (all stages pass `ArtifactRef` handles / paths, never large
in-memory dataframes):

| Task | Produces | Output location |
| --- | --- | --- |
| `acquire_sources` | content-addressed raw fixtures/downloads | `data/raw/by-hash/<hex>/`, `data/raw/aliases/` |
| `extract_dailymed` | `spl_documents` interim parquet | `data/interim/dailymed/` |
| `extract_faers` | joined `cases` interim parquet | `data/interim/faers/` |
| `extract_drugsfda` | normalized `products` interim parquet | `data/interim/drugsfda/` |
| `extract_medi` | `contraindications` interim parquet | `data/interim/medi/` |
| `shape_*_tables` | uncompressed TSV assertion tables | `data/tabular/` |
| `generate_tablassert_configs` | `graph.yaml` + per-table configs | `data/store/tablassert/tables/` |
| `run_tablassert` | KGX NDJSON (full) / deferred handoff manifest (mock) | `data/kgx/` or `data/reports/` |
| `write_build_summary` | `build_summary.json` | `data/reports/` |

See [`docs/architecture.md`](./docs/architecture.md) for the layered model and sharding,
and [`docs/sources.md`](./docs/sources.md) for per-source extraction.

## Where things land

All paths derive from a single workdir root (CLI `--workdir`, DAG param `workdir`, or
`tmp_path` in tests). No absolute paths are embedded in code
([`src/dakp_pipeline/paths.py`](./src/dakp_pipeline/paths.py)).

```
<workdir>/
  data/raw/by-hash/<hex>/     # immutable content-addressed store (BLAKE3)
  data/raw/aliases/           # human-readable names -> store artifact ids
  data/interim/               # partitioned parquet interim tables
  data/tabular/               # uncompressed TSV assertion tables (Tablassert-facing)
  data/kgx/                   # KGX NDJSON (written by Tablassert in full builds)
  data/manifests/<hex>.json   # per-artifact JSON manifests (dakp.artifact.v1)
  data/store/tablassert/      # generated Graph + table config YAMLs
  data/reports/               # build_summary.json + tablassert_handoff.json
  logs/dakp.log               # structured logs (loguru -> stdlib bridge)
```

A verified mock run writes exactly: 4 interim parquet tables, 3 tabular TSVs, 4 generated
config YAMLs (`graph.yaml` + 3 tables), `build_summary.json`,
`tablassert_handoff.json`, one manifest JSON per artifact, and `logs/dakp.log`. See
[`docs/runbook.md`](./docs/runbook.md) for how to read a run.

## Provenance semantics

Translator provenance is modeled to match the `../DINGO` translator-ingests reference
(local sibling repo: `../DINGO/src/translator_ingest/ingests/dakp/`).
Each assertion table carries `primary_knowledge_source`, `upstream_resource_ids`
(`|`-joined infores ids), `knowledge_level`, and `agent_type` columns
([`src/dakp_pipeline/io/schemas.py`](./src/dakp_pipeline/io/schemas.py)); the generated
Tablassert config emits a matching `provenance.override` block
([`src/dakp_pipeline/tablassert/configs.py`](./src/dakp_pipeline/tablassert/configs.py)).

| Family | `primary_knowledge_source` | `upstream_resource_ids` | `knowledge_level` | DAKP role in DINGO ingest |
| --- | --- | --- | --- | --- |
| `treats` | `infores:multiomics-drugapprovals` | `infores:dailymed\|infores:faers` | `knowledge_assertion` | DAKP = **primary**; DailyMed + FAERS supporting |
| `applied_to_treat` | `infores:multiomics-drugapprovals` | `infores:faers\|infores:dailymed` | `observation` | DAKP = **aggregator**; FAERS primary, DailyMed supporting |
| `contraindicated_in` | `infores:multiomics-drugapprovals` | `infores:medi\|infores:dailymed` | `knowledge_assertion` | DAKP = **aggregator**; MEDI primary, DailyMed supporting |

All three use `agent_type = manual_validation_of_automated_agent`. `clinical_approval_status`
is `approved_for_condition` for `treats` and `observed_use` for `applied_to_treat` (FAERS
label/status behavior kept stable for the first rebuild; the exact legacy value is
confirmed during the Milestone-5 audit). See
[`docs/tablassert-handoff.md`](./docs/tablassert-handoff.md) for the config/provenance
mapping.

## Tablassert / fullmap handoff

DAKP does everything *up to* the shape Tablassert consumes: acquire → extract → shape
into assertion tables, then generate a Tablassert **Graph config** plus one **table
config** per assertion table. Canonical entity resolution (CURIE/name/category),
category assignment, node normalization, KGX NDJSON writing, deduplication, deterministic
UUIDs, and RIG generation are delegated to `../Tablassert`. DAKP deliberately ships **no**
parallel KGX compiler — if a Biolink slot is missing, it is upstreamed into Tablassert
rather than reimplemented here.

In the mock profile, `run_tablassert` writes a deferred-handoff manifest recording the
assertion inputs + generated configs and returns; it does not compile a graph. See
[`docs/tablassert-handoff.md`](./docs/tablassert-handoff.md).

## How to add a new source

The fetcher/extractor/shaper pattern is uniform and monkeypatchable. To add source `X`:

1. **Add a fixture** under `tests/fixtures/pipeline/<x>/` mirroring the real source's
   shape (tiny, deterministic).
2. **Add a fetcher** at `src/dakp_pipeline/sources/<x>.py` — a `<X>Fetcher` class with a
   `fetch(ctx)` method plus a module-level `fetch = <X>Fetcher().fetch` binding (so tests
   can `monkeypatch.setattr(x, "fetch", ...)`). Use `require_mock(ctx, "x")` +
   `ingest_fixtures(ctx, _FIXTURES, namespace="x")` from
   [`sources/__init__.py`](./src/dakp_pipeline/sources/__init__.py). Real acquisition is
   Milestone 2.
3. **Add an extractor** at `src/dakp_pipeline/extract/<x>.py` if parsing is needed
   (return parquet interim refs; register them with `ArtifactStore.register`).
4. **Wire it into the runner** in [`pipeline.py`](./src/dakp_pipeline/pipeline.py)
   (`x_raw = x.fetch(ctx)` → `x_ext = …extract(x_raw, ctx)` → feed into the relevant
   shaper) and into the DAG if it should be an Airflow task.
5. **If it defines a new edge family**, add the column contract + entry to
   `ASSERTION_TABLES` in [`schemas.py`](./src/dakp_pipeline/io/schemas.py), add a shaper
   under [`assertions/`](./src/dakp_pipeline/assertions/), add its provenance tuple to
   `_TABLE_PROVENANCE` in [`tablassert/configs.py`](./src/dakp_pipeline/tablassert/configs.py),
   and (if it must pass the readiness gate) it is auto-checked by the contract in
   [`translator/contract.py`](./src/dakp_pipeline/translator/contract.py).

## Dependency philosophy

Stdlib first (per `PLAN.md` → "Dependency philosophy"). Approved runtime deps only:
**polars, loguru, blake3, pydantic**. Airflow is an optional extra. No web frameworks,
ORMs, or generic ETL engines. Go workers (`workers/go_runner.py` is a stub today) will be
added only where profiling justifies native speed — the `go/` tree lands in a later
milestone.

## Roadmap (milestones)

| Milestone | Scope | Status |
| --- | --- | --- |
| 1 — Scaffold | project skeleton, BLAKE3 store, full mocked DAG, fixtures | ✅ this branch |
| 2 — Acquisition | real DailyMed/FAERS/Drugs@FDA/MEDI downloaders w/ manifests | planned |
| 3 — Extraction | streaming DailyMed/FAERS/Drugs@FDA/MEDI extraction + rejects/warnings | planned |
| 4 — NER/mapping | dictionary + candidates + fullmap/Tablassert canonical resolution | planned |
| 5 — Assertions | evidence-rich assertion aggregation rules + tests | planned |
| 6 — Airflow DAG | TaskFlow wiring, XCom serialization, task reruns | scaffolded |
| 7 — Tablassert | Graph/table config generation + live `../Tablassert` run + RIG | scaffolded (mock handoff) |
| 8 — Validation/perf/release | KGX validation, benchmarks, publish layout | planned |

## Further reading

- [`docs/architecture.md`](./docs/architecture.md) — layered pipeline, sharding/concurrency, BLAKE3 store, Tablassert boundary.
- [`docs/logging.md`](./docs/logging.md) — Airflow + loguru + Go logging, structured fields, reading a failed run.
- [`docs/sources.md`](./docs/sources.md) — per-source acquisition, extraction, schema notes.
- [`docs/tabular-contracts.md`](./docs/tabular-contracts.md) — every tabular contract table with columns + example rows.
- [`docs/tablassert-handoff.md`](./docs/tablassert-handoff.md) — assertion tables, Graph/table config generation, provenance overrides.
- [`docs/runbook.md`](./docs/runbook.md) — common failures, reruns, cache invalidation, shard debugging.
- [`PLAN.md`](./PLAN.md) — the full approved specification.
