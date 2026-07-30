# DAKP — Drug Approvals Knowledge Provider

Reproducible `uv` Python pipeline that builds **treatment**, **observed-use**, and
**contraindication** assertion tables from DailyMed, Drugs@FDA, and FAERS, then hands them
to [Tablassert](https://pypi.org/project/tablassert/) (PyPI, `8.0.0`) for Translator KGX
modeling. The full approved specification lives in [`PLAN.md`](./PLAN.md); this README is the
operational entry point and links into [`docs/`](./docs).

> **Status — final architecture.** The pipeline runs end-to-end: real stdlib-HTTP source
> downloaders, Python extractors with **byte-for-byte Go parity** workers, a single
> benchmarked NER backend, evidence-rich assertion aggregation, Airflow orchestration, and a
> delegated Tablassert KGX handoff. The mock profile runs the whole DAG on tiny fixtures with
> **no network and no Tablassert/Airflow installed**; the full production build runs on the
> `wenceslaus` host (see [`docs/wenceslaus-runbook.md`](./docs/wenceslaus-runbook.md)). What
> changed relative to the legacy build — and why it is equivalent-or-better — is documented in
> [`docs/semantic-equivalence.md`](./docs/semantic-equivalence.md).

## The pipeline

```text
acquire ─▶ extract ─▶ NER ─▶ aggregate ─▶ Tablassert KGX
  │          │         │        │              │
  │          │         │        │              └─ graph.yaml + per-table configs ─▶ installed
  │          │         │        │                 `tablassert build-kg --fullmap …` (KGX NDJSON)
  │          │         │        └─ 3 uncompressed TSV assertion tables (Tablassert-facing)
  │          │         └─ single composite DiseaseNER (gazetteer + GLiNER): disease/phenotype
  │          │            MENTIONS (text + type only) from DailyMed contraindication sections
  │          └─ interim parquet (spl_documents/sets/approvals/ingredients/sections, faers cases,
  │             drugsfda products) — Python extractors, or byte-identical Go workers in prod
  └─ content-addressed raw downloads (BLAKE3 store); mock profile ingests fixtures instead
```

- **acquire** — real stdlib-HTTP downloaders ([`sources/`](./src/dakp_pipeline/sources/),
  [`acquire.py`](./src/dakp_pipeline/acquire.py)) for DailyMed full releases, Drugs@FDA, and
  FAERS quarterly extracts; content-addressed, idempotent, manifest-recorded.
- **extract** — [`extract/`](./src/dakp_pipeline/extract/) parses raw artifacts into interim
  parquet. The heavy parsers have **Go ports** ([`go/`](./go/)) that are byte-for-byte identical
  to the Python output (golden-file parity tests); prod opts into them with `use_go_workers`.
- **NER** — one composite backend ([`ner/`](./src/dakp_pipeline/ner/), `DiseaseNER`): a curated
  gazetteer anchoring high-precision spans plus GLiNER zero-shot filling the gaps. Emits
  **mentions only** (text span + entity type) — never ontology CURIEs.
- **aggregate** — [`assertions/`](./src/dakp_pipeline/assertions/) joins extracted tables + NER
  mentions into three uncompressed TSV assertion tables.
- **Tablassert KGX** — [`tablassert/`](./src/dakp_pipeline/tablassert/) generates a Graph config
  + one table config per assertion table, then delegates canonical resolution + KGX compilation
  to the **installed `tablassert` CLI** (`[kg]` extra). DAKP ships no local KGX compiler.

## Why a rebuild

The legacy DAKP build is a collection of Perl/Python scripts ([`ref/legacy/`](./ref/legacy))
that mix downloading, parsing, lexical matching, edge modeling, and KGX serialization in
monolithic scripts — brittle, with hardcoded local paths (`/ssd2/sqlite/BABEL.db`), missing
Perl libraries, shell side effects, and no tests. The rebuild replaces it with:

- A reproducible `uv` project with pinned dependencies and `uv.lock`.
- A typed, monkeypatchable pipeline where every stage communicates through **BLAKE3
  content-addressed `ArtifactRef` handles** — restartable, cacheable by hash, trivially fakeable.
- An **Apache Airflow DAG** (`dakp_build`) as the orchestration surface, with a pure-Python
  runner as the source of truth that tests exercise directly.
- A clear **delegation boundary**: DAKP acquires, extracts, NER-mines, and shapes tables;
  canonical entity resolution, KGX compilation, dedup, deterministic IDs, and RIG generation are
  delegated to Tablassert/fullmap. DAKP ships **no local fallback KGX compiler**.
- **100% branch coverage** (`uv run pytest --cov`, `fail_under = 100`), ruff lint + format, and
  pyright all clean — zero dead code.

Legacy scripts are retained in-tree only for audit; the new DAG never imports them.

## Edge families produced

Three Translator edge families (see [`docs/semantic-equivalence.md`](./docs/semantic-equivalence.md)
for the preserved-vs-improved accounting and [`docs/tablassert-handoff.md`](./docs/tablassert-handoff.md)
for the provenance mapping):

| Predicate | Subject → Object | Upstream sources | DAKP role |
| --- | --- | --- | --- |
| `biolink:treats` | drug → disease/phenotype | DailyMed + Drugs@FDA approvals; FAERS support | **primary** knowledge source |
| `biolink:applied_to_treat` | drug → disease/phenotype | FAERS observed use; DailyMed support | **aggregator** over FAERS |
| `biolink:contraindicated_in` | drug → disease/phenotype | DailyMed SPL contraindications (NER-mined) | **aggregator**; text-mined from DailyMed |

All three aggregate under `infores:multiomics-drugapprovals`. Subjects are chemical/drug
categories (ChemicalEntity/SmallMolecule/MolecularMixture/ComplexMolecularMixture/Drug); objects
are Disease/PhenotypicFeature — matching the DINGO reference ingest
(`../DINGO/src/translator_ingest/ingests/dakp/dakp_rig.yaml`).

## Quickstart (mocked, laptop-safe)

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # base install; Airflow / NER / Tablassert NOT required
uv run pytest -q              # unit + mocked integration (no network)
uv run dakp run --profile mock \
  --fixture-root tests/fixtures/pipeline \
  --workdir /tmp/dakp-mock
```

The mock run needs no network and no real Tablassert. It writes three uncompressed TSV assertion
tables, generated Tablassert configs, a build summary, and a deferred-handoff manifest (see
[Where things land](#where-things-land)). All fetchers are monkeypatchable;
[`tests/integration/test_mock_pipeline.py`](./tests/integration/test_mock_pipeline.py) shows the
boundary. The base install and the **entire test suite run without the `[ner]`, `[kg]`, or
`[airflow]` extras** — NER defaults to a deterministic offline gazetteer, and the mock profile
defers the Tablassert handoff.

## Profiles

Profiles are defined in Python at [`src/dakp_pipeline/config.py`](./src/dakp_pipeline/config.py)
(`load_profile`) — the base install needs no `pyyaml`. An unknown profile name raises at startup
rather than silently defaulting.

| Profile | Threads | Memory | Sources | Tablassert | Go workers |
| --- | --- | --- | --- | --- | --- |
| `mock` | 1 | 1 GiB | fixtures only | deferred (writes handoff manifest) | off |
| `sample` | 4 | 8 GiB | real, bounded sample | deferred | off |
| `prod` | 64 | 128 GiB | real full build | installed `tablassert` CLI (`[kg]`) | opt-in (`use_go_workers`) |

The real fetchers use stdlib HTTP (no `requests`) and are content-addressed and idempotent.
`prod` defaults to the full scope (`quarter_limit` / `release_limit` unset = all quarters/releases)
and runs the real Tablassert handoff; bound it with `--quarter-limit` / `--release-limit` for a
tiny real smoke run (below).

## Makefile targets

All Python runs through `uv`; Go runs through the [`go/`](./go/) module. `make help` lists everything.

| Target | What it does |
| --- | --- |
| `make setup` / `make install` | `uv sync` — base + dev deps |
| `make install-ner` | `uv sync --extra ner` — GLiNER production NER (pulls torch) |
| `make install-airflow` | `uv sync --extra airflow` — orchestration extra |
| `make install-kg` | `uv sync --extra kg` — PyPI `tablassert` (KG build; laptop-safe) |
| `make install-kg-qc` | `uv sync --extra kg-qc` — `tablassert[qc]` audit (pulls torch; beefy hosts) |
| `make install-all` | `uv sync --all-extras` |
| `make test` / `make cov` | pytest / pytest with branch coverage (`fail_under = 100`) |
| `make lint` / `make fmt` / `make fmt-check` / `make typecheck` | ruff check / ruff format / format check / pyright |
| `make check` | lint + fmt-check + typecheck + test |
| `make build-go` / `make test-go` / `make check-go` | Go build / test (incl. Python-parity goldens) / full Go gate |
| `make check-all` | `make check` + `make check-go` (the full Python + Go gate) |
| `make run-mock` | the mocked end-to-end pipeline (no network, no real Tablassert) |
| `make clean` | remove caches, coverage data, the Go binary, and `tmp/` |

## Running the full build on `wenceslaus`

The full profile targets `wenceslaus` (Ubuntu 24.04, dual Xeon Gold 6230 / 80 logical CPUs,
187 GiB RAM, `/local_raid1` ~1.75 TiB). **This laptop cannot build the fullmap** (~120 GiB RAM);
the fullmap build and the full prod KG are wenceslaus-only, while mock/sample runs and NER are
laptop-safe. The exact commands are in [`docs/wenceslaus-runbook.md`](./docs/wenceslaus-runbook.md).

### Bounded `prod` smoke run (laptop-safe with network)

To validate the **real** fetcher → extractor → NER → aggregation → Tablassert-handoff path without
the multi-TB full build, bound the scope so only one FAERS quarter and one DailyMed release are
processed:

```bash
uv run dakp run --profile prod \
  --quarter-limit 1 --release-limit 1 \
  --workdir /tmp/dakp-prod-smoke
```

`--quarter-limit` caps FAERS quarters (most-recent first) and `--release-limit` caps DailyMed full
releases; both default to the profile value (`prod` = all). `--force` reruns every stage ignoring
the BLAKE3 cache. The offline integration test
[`tests/integration/test_prod_smoke.py`](./tests/integration/test_prod_smoke.py) exercises the exact
same real code path with the HTTP layer mocked, so it passes in CI with no network.

## The DAG

`dakp_build` ([`src/dakp_pipeline/dags/dakp_build.py`](./src/dakp_pipeline/dags/dakp_build.py)) is
the orchestration DAG, implemented with the Airflow TaskFlow API. It is a thin wrapper around the
same stage functions the pure-Python runner ([`pipeline.py`](./src/dakp_pipeline/pipeline.py)) calls,
and is **import-safe without Airflow installed** (guarded imports + no-op decorator fallbacks).

```text
acquire_sources ─┬─▶ extract_dailymed  ─┐
                 ├─▶ extract_faers      ─┼─▶ {shape_treatment_tables,
                 └─▶ extract_drugsfda   ─┘    shape_faers_use_tables,
                                                 shape_contraindication_tables}
                                                ─▶ generate_tablassert_configs
                                                ─▶ run_tablassert
                                                ─▶ write_build_summary
```

| Task | Produces | Output location |
| --- | --- | --- |
| `acquire_sources` | content-addressed raw fixtures/downloads | `data/raw/by-hash/<hex>/`, `data/raw/aliases/` |
| `extract_dailymed` | `spl_documents/sets/approvals/ingredients/sections` parquet | `data/interim/dailymed/` |
| `extract_faers` | joined `cases` parquet | `data/interim/faers/` |
| `extract_drugsfda` | normalized `products` parquet | `data/interim/drugsfda/` |
| `shape_*_tables` | uncompressed TSV assertion tables (contraindications NER-mined) | `data/tabular/` |
| `generate_tablassert_configs` | `graph.yaml` + per-table configs | `tables/` (workdir-relative) |
| `run_tablassert` | KGX NDJSON (full) / deferred handoff manifest (mock) | `data/kgx/` or `data/reports/` |
| `write_build_summary` | `build_summary.json` | `data/reports/` |

Contraindications are **mined from the DailyMed SPL contraindication sections** during
`shape_contraindication_tables` (no separate MEDI source). See
[`docs/architecture.md`](./docs/architecture.md) and [`docs/sources.md`](./docs/sources.md).

## Where things land

All paths derive from a single workdir root (CLI `--workdir`, DAG param `workdir`, or `tmp_path`
in tests). No absolute paths are embedded in code ([`paths.py`](./src/dakp_pipeline/paths.py)).

```
<workdir>/
  data/raw/by-hash/<hex>/     # immutable content-addressed store (BLAKE3)
  data/raw/aliases/           # human-readable names -> store artifact ids
  data/interim/               # partitioned parquet interim tables
  data/tabular/               # uncompressed TSV assertion tables (Tablassert-facing)
  data/kgx/                   # KGX NDJSON (written by Tablassert in full builds)
  data/manifests/<hex>.json   # per-artifact JSON manifests (dakp.artifact.v1)
  tables/                     # generated graph.yaml + per-table config YAMLs
  data/reports/               # build_summary.json + tablassert_handoff.json
  logs/dakp.log               # structured logs (loguru -> stdlib bridge)
```

See [`docs/runbook.md`](./docs/runbook.md) for how to read a run.

## The single NER backend

There is **one** NER backend ([`ner/ner.py`](./src/dakp_pipeline/ner/ner.py), `DiseaseNER`) with
**one** entry point — no pluggable backend selector. It was settled by a labeled benchmark
([`ner/BENCHMARK.md`](./src/dakp_pipeline/ner/BENCHMARK.md)): the **gazetteer + GLiNER composite**
won (precision 0.972 / recall 1.000 / F1 0.986 on 27 cases / 35 gold spans); SciSpacy was dropped
(no phenotype label, coarse spans).

- **Offline mode (default):** curated gazetteer + deterministic lexical matcher. Precision 1.000 /
  F1 0.955, zero heavy deps, fully deterministic. Used by tests + the mock pipeline.
- **Production mode (`offline=False`):** the same gazetteer anchors spans and GLiNER zero-shot
  (`urchade/gliner_small-v2.1`) fills out-of-gazetteer gaps. Needs the `[ner]` extra (lazy-imported;
  the base install never loads torch).

DAKP extracts **mentions only** (text span + type); ontology CURIE resolution is exclusively
Tablassert/fullmap's job at `tablassert build-kg`. See [`ner/README.md`](./src/dakp_pipeline/ner/README.md).

## Tablassert / fullmap handoff

DAKP does everything *up to* the shape Tablassert consumes: acquire → extract → NER → aggregate
into assertion tables, then generate a Tablassert **Graph config** plus one **table config** per
assertion table. Canonical entity resolution (CURIE/name/category), category assignment, node
normalization, KGX NDJSON writing, deduplication, deterministic UUIDs, and RIG generation are
delegated to the **installed `tablassert` package** (PyPI `8.0.0`, the `[kg]` extra). DAKP
deliberately ships **no** parallel KGX compiler — if a Biolink slot is missing, it is upstreamed
into Tablassert rather than reimplemented here.

The runner ([`tablassert/run.py`](./src/dakp_pipeline/tablassert/run.py)) shells out to
`tablassert build-kg tables/graph.yaml --fullmap <path> [--qc] [--release]`. In the mock profile it
writes a deferred-handoff manifest instead. See [`docs/tablassert-handoff.md`](./docs/tablassert-handoff.md).

## How to add a new source

The fetcher/extractor/shaper pattern is uniform and monkeypatchable. To add source `X`:

1. **Add a fixture** under `tests/fixtures/pipeline/<x>/` mirroring the real source's shape.
2. **Add a fetcher** at `src/dakp_pipeline/sources/<x>.py` — a `<X>Fetcher` class with a
   `fetch(ctx)` method plus a module-level `fetch` binding (so tests can monkeypatch it). Use
   `require_mock(ctx, "x")` + `ingest_fixtures(ctx, _FIXTURES, namespace="x")` from
   [`sources/__init__.py`](./src/dakp_pipeline/sources/__init__.py).
3. **Add an extractor** at `src/dakp_pipeline/extract/<x>.py` if parsing is needed (return parquet
   interim refs; register them with `ArtifactStore.register`). Add a byte-parity Go port under
   [`go/internal/<x>/`](./go/) if the parser is hot.
4. **Wire it into the runner** in [`pipeline.py`](./src/dakp_pipeline/pipeline.py) and into the DAG
   if it should be an Airflow task.
5. **If it defines a new edge family**, add the column contract + entry to `ASSERTION_TABLES` in
   [`schemas.py`](./src/dakp_pipeline/io/schemas.py), add a shaper under
   [`assertions/`](./src/dakp_pipeline/assertions/), add its provenance tuple to `_TABLE_SPECS` in
   [`tablassert/configs.py`](./src/dakp_pipeline/tablassert/configs.py), and (if it must pass the
   readiness gate) it is auto-checked by [`translator/contract.py`](./src/dakp_pipeline/translator/contract.py).

## Dependency philosophy

Stdlib first (per `PLAN.md` → "Dependency philosophy"). Approved runtime deps only: **polars,
loguru, blake3, pydantic**. Everything heavy is an optional extra: `[airflow]` (orchestration),
`[ner]` (GLiNER, pulls torch), `[kg]` (PyPI `tablassert`), `[kg-qc]` (`tablassert[qc]` audit, pulls
torch). Go workers ([`go/`](./go/)) cover the hot extraction paths with byte-for-byte parity.

## Verification

```bash
make check-all        # Python (lint + format + pyright + tests @ 100% coverage) + Go (build + vet + parity tests + gofmt)
```

The semantic-preservation suite
[`tests/integration/test_semantic_equivalence.py`](./tests/integration/test_semantic_equivalence.py)
asserts the rebuild preserves the legacy DAKP semantics (edge families, categories, provenance,
`clinical_approval_status`, evidence fields, deterministic output) and cross-checks the Translator
contract against the DINGO reference ingest.

## Further reading

- [`docs/semantic-equivalence.md`](./docs/semantic-equivalence.md) — preserved-vs-improved semantics vs the old DAKP.
- [`docs/wenceslaus-runbook.md`](./docs/wenceslaus-runbook.md) — the full production build (fullmap + prod KG).
- [`docs/architecture.md`](./docs/architecture.md) — layered pipeline, sharding/concurrency, BLAKE3 store, Tablassert boundary.
- [`docs/sources.md`](./docs/sources.md) — per-source acquisition, extraction, DailyMed-NER contraindications.
- [`docs/tablassert-handoff.md`](./docs/tablassert-handoff.md) — assertion tables, config generation, provenance overrides.
- [`docs/tabular-contracts.md`](./docs/tabular-contracts.md) — every tabular contract table with columns.
- [`docs/runbook.md`](./docs/runbook.md) — common failures, reruns, cache invalidation.
- [`docs/logging.md`](./docs/logging.md) — Airflow + loguru + Go logging, reading a failed run.
- [`PLAN.md`](./PLAN.md) — the full approved specification.
