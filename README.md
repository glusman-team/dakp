# DAKP — Drug Approvals Knowledge Provider

Reproducible `uv` Python pipeline that builds **treatment**, **observed-use**, and
**contraindication** assertion tables from DailyMed, Drugs@FDA, and FAERS, then hands them
to [Tablassert](https://pypi.org/project/tablassert/) (PyPI, `8.0.0`) for Translator KGX
modeling. The full approved specification lives in [`PLAN.md`](./PLAN.md); this README is the
operational entry point and links into [`docs/`](./docs).

> **Status — final architecture.** The pipeline is **Airflow-native**: the `dakp_build` DAG is the
> only orchestrator, and the heavy DailyMed/FAERS/Drugs@FDA extraction runs as **native Airflow Go
> SDK bundle workers** (no subprocess/OS commands). One command runs the whole pipeline end to end:
> `make install` then `make run` (see the [Quickstart](#quickstart-mocked-laptop-safe)). Real
> stdlib-HTTP source downloaders, a single benchmarked NER backend, evidence-rich assertion
> aggregation, and a delegated Tablassert KGX handoff. The mock profile runs the whole DAG on tiny
> fixtures with **no network and no Tablassert** (Airflow runs locally via `make run`); the full
> production build runs on the `wenceslaus` host (see
> [`docs/wenceslaus-runbook.md`](./docs/wenceslaus-runbook.md)). What changed relative to the legacy
> build — and why it is equivalent-or-better — is documented in
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
  │             drugsfda products) — native Go workers (Airflow Go SDK bundle), parity-locked to the
  │             pure-Python reference extractors
  └─ content-addressed raw downloads (BLAKE3 store); mock profile ingests fixtures instead
```

- **acquire** — real stdlib-HTTP downloaders ([`sources/`](./src/dakp_pipeline/sources/),
  [`acquire.py`](./src/dakp_pipeline/acquire.py)) for DailyMed full releases, Drugs@FDA, and
  FAERS quarterly extracts; content-addressed, idempotent, manifest-recorded.
- **extract** — the heavy parsers run as **native Go workers** in an Airflow Go SDK bundle
  ([`go/cmd/dakp-bundle`](./go/cmd/dakp-bundle), parsing libraries in [`go/internal/`](./go/)); the
  DAG's `extract_*` tasks are `@task.stub(queue="golang")` declarations the ExecutableCoordinator
  forks per task instance. They are parity-locked to the pure-Python reference extractors
  ([`extract/`](./src/dakp_pipeline/extract/), kept as the reference/test oracle) via golden-file
  parity tests.
- **NER** — one composite backend ([`ner/`](./src/dakp_pipeline/ner/), `DiseaseNER`): a curated
  gazetteer anchoring high-precision spans plus GLiNER zero-shot filling the gaps. Emits
  **mentions only** (text span + entity type) — never ontology CURIEs.
- **aggregate** — [`assertions/`](./src/dakp_pipeline/assertions/) joins extracted tables + NER
  mentions into three uncompressed TSV assertion tables.
- **Tablassert KGX** — [`tablassert/`](./src/dakp_pipeline/tablassert/) generates a Graph config
  + one table config per assertion table, then delegates canonical resolution + KGX compilation
  to the **installed `tablassert` CLI** (a core dependency, installed by `uv sync`). DAKP ships no local KGX compiler.

## Why a rebuild

The legacy DAKP build is a collection of Perl/Python scripts ([`ref/legacy/`](./ref/legacy))
that mix downloading, parsing, lexical matching, edge modeling, and KGX serialization in
monolithic scripts — brittle, with hardcoded local paths (`/ssd2/sqlite/BABEL.db`), missing
Perl libraries, shell side effects, and no tests. The rebuild replaces it with:

- A reproducible `uv` project with pinned dependencies and `uv.lock`.
- A typed, monkeypatchable pipeline where every stage communicates through **BLAKE3
  content-addressed `ArtifactRef` handles** — restartable, cacheable by hash, trivially fakeable.
- An **Apache Airflow DAG** (`dakp_build`) as the sole orchestration surface (Airflow 3 is a hard
  dependency); the heavy extraction runs as native Go SDK bundle workers, the other stages as
  Python TaskFlow tasks. A pure-Python `run_pipeline` harness ([`pipeline.py`](./src/dakp_pipeline/pipeline.py))
  exercises the stage functions + reference extractors in tests without Airflow.
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

Requires Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), and a Go toolchain ≥ 1.24 (to build the
native worker bundle).

```bash
make install        # uv sync — ONE command installs ALL runtime + dev deps (Airflow 3, GLiNER, tablassert[qc])
uv run pytest -q    # unit + mocked integration (no network; the tests need no running Airflow)
make run            # ONE command: build+pack the Go bundle, start Airflow, run dakp_build, wait
make down           # stop the local Airflow
```

`make run` builds + packs the native Go bundle, starts a local Airflow (standalone, port 8090) with
the Go coordinator configured, provisions the task pools, sets the per-run `dakp_config` Variable,
triggers the `dakp_build` DAG, waits for it to finish, and prints the build summary. Override with
`PROFILE=`, `WORKDIR=`, `FIXTURE_ROOT=`, `AIRFLOW_PORT=` env vars (e.g. `PROFILE=sample make run`).

The mock run needs no network and no real Tablassert. It writes three uncompressed TSV assertion
tables, generated Tablassert configs, a build summary, and a deferred-handoff manifest (see
[Where things land](#where-things-land)). The **test suite runs offline** — NER defaults to a
deterministic gazetteer and the mock profile defers the Tablassert handoff — but a single
`make install` (`uv sync`) installs the entire production runtime in one command (there are no
optional extras).

## Profiles

Profiles are defined in Python at [`src/dakp_pipeline/config.py`](./src/dakp_pipeline/config.py)
(`load_profile`) — the base install needs no `pyyaml`. An unknown profile name raises at startup
rather than silently defaulting.

| Profile | Threads | Memory | Sources | Tablassert |
| --- | --- | --- | --- | --- |
| `mock` | 1 | 1 GiB | fixtures only | deferred (writes handoff manifest) |
| `sample` | 4 | 8 GiB | real, bounded sample | deferred |
| `prod` | 64 | 128 GiB | real full build | installed `tablassert[qc]` CLI |

The heavy extraction always runs as **native Go workers** (the Airflow Go SDK bundle); the profile
only sizes concurrency/memory and selects sources + the Tablassert handoff mode.

The real fetchers use stdlib HTTP (no `requests`) and are content-addressed and idempotent.
`prod` defaults to the full scope (`quarter_limit` / `release_limit` unset = all quarters/releases)
and runs the real Tablassert handoff; bound it with `--quarter-limit` / `--release-limit` for a
tiny real smoke run (below).

## Makefile targets

All Python runs through `uv`; Go runs through the [`go/`](./go/) module. `make help` lists everything.

| Target | What it does |
| --- | --- |
| `make setup` / `make install` | `uv sync` — ONE command installs every runtime + dev dep (Airflow 3, GLiNER NER, `tablassert[qc]`) |
| `make test` / `make cov` | pytest / pytest with branch coverage (`fail_under = 100`) |
| `make lint` / `make fmt` / `make fmt-check` / `make typecheck` | ruff check / ruff format / format check / pyright |
| `make check` | lint + fmt-check + typecheck + test |
| `make build-go` / `make test-go` / `make check-go` | Go build / test (incl. Python-parity goldens) / full Go gate |
| `make check-all` | `make check` + `make check-go` (the full Python + Go gate) |
| `make bundle` | build + pack the native Go bundle into the coordinator's `executables_root` |
| `make run` | ONE-COMMAND end-to-end run via Airflow (bundle + Airflow + trigger + wait); `PROFILE`/`WORKDIR`/`FIXTURE_ROOT`/`AIRFLOW_PORT` env override |
| `make down` | stop the local Airflow started by `make run` |
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
PROFILE=prod WORKDIR=/tmp/dakp-prod-smoke make run
```

The `dakp_config` Variable that `make run` sets carries the profile + scope. The orchestrator
(`scripts/dakp_up.sh`) currently pins `quarter_limit` / `release_limit` to 1 (a bounded smoke run);
for a full-scope prod build, unset them there (or pass full scope via the Variable). The offline
integration test [`tests/integration/test_prod_smoke.py`](./tests/integration/test_prod_smoke.py)
exercises the exact same real stage code path (via the `run_pipeline` harness) with the HTTP layer
mocked, so it passes in CI with no network.

## The DAG

`dakp_build` ([`src/dakp_pipeline/dags/dakp_build.py`](./src/dakp_pipeline/dags/dakp_build.py)) is
the orchestration DAG (Airflow 3 TaskFlow API) and the **sole** way to run the pipeline. The three
`extract_*` tasks are **native Go SDK workers** — `@task.stub(queue="golang")` declarations whose Go
implementations ([`go/cmd/dakp-bundle`](./go/cmd/dakp-bundle)) the ExecutableCoordinator forks per
task instance; acquisition, shaping, Tablassert handoff, and the build summary are Python TaskFlow
tasks. Tasks pass `ArtifactRef` manifests over XCom (JSON dicts; see
[`io/xcom.py`](./src/dakp_pipeline/io/xcom.py)); run config comes from the `dakp_config` Variable.

```text
acquire_dailymed ─▶ extract_dailymed ─┐  (extract_* are native Go SDK
acquire_faers    ─▶ extract_faers    ─┼─▶  bundle workers, queue=golang)
acquire_drugsfda ─▶ extract_drugsfda ─┘
        ─▶ {shape_treatment_tables, shape_faers_use_tables, shape_contraindication_tables}
acquire_ner_models ─▶ shape_contraindication_tables
        ─▶ generate_tablassert_configs
acquire_ontologies ─▶ run_tablassert ─▶ write_build_summary
```

| Task | Produces | Output location |
| --- | --- | --- |
| `acquire_*` (dailymed/faers/drugsfda/ner_models/ontologies) | content-addressed raw fixtures/downloads | `data/raw/by-hash/<hex>/`, `data/raw/aliases/` |
| `extract_dailymed` *(native Go)* | `spl_documents/sets/approvals/ingredients/sections` parquet | `data/interim/dailymed/` |
| `extract_faers` *(native Go)* | joined `cases` parquet + audits | `data/interim/faers/` |
| `extract_drugsfda` *(native Go)* | normalized `products/applications/submissions/lookups` parquet | `data/interim/drugsfda/` |
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
  (`urchade/gliner_small-v2.1`) fills out-of-gazetteer gaps. GLiNER is a core dependency but
  lazy-imported (module load never touches torch).

DAKP extracts **mentions only** (text span + type); ontology CURIE resolution is exclusively
Tablassert/fullmap's job at `tablassert build-kg`. See [`ner/README.md`](./src/dakp_pipeline/ner/README.md).

## Tablassert / fullmap handoff

DAKP does everything *up to* the shape Tablassert consumes: acquire → extract → NER → aggregate
into assertion tables, then generate a Tablassert **Graph config** plus one **table config** per
assertion table. Canonical entity resolution (CURIE/name/category), category assignment, node
normalization, KGX NDJSON writing, deduplication, deterministic UUIDs, and RIG generation are
delegated to the **installed `tablassert` package** (PyPI `8.0.0`, the `tablassert[qc]` core
dependency). DAKP
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
4. **Wire it into the DAG** ([`dags/dakp_build.py`](./src/dakp_pipeline/dags/dakp_build.py)) as a
   TaskFlow task (and into the [`pipeline.py`](./src/dakp_pipeline/pipeline.py) test harness if it
   should run in the Airflow-free tests). A hot parser becomes a native Go worker in
   [`go/cmd/dakp-bundle`](./go/cmd/dakp-bundle) exposed as a `@task.stub(queue="golang")` task.
5. **If it defines a new edge family**, add the column contract + entry to `ASSERTION_TABLES` in
   [`schemas.py`](./src/dakp_pipeline/io/schemas.py), add a shaper under
   [`assertions/`](./src/dakp_pipeline/assertions/), add its provenance tuple to `_TABLE_SPECS` in
   [`tablassert/configs.py`](./src/dakp_pipeline/tablassert/configs.py), and (if it must pass the
   readiness gate) it is auto-checked by [`translator/contract.py`](./src/dakp_pipeline/translator/contract.py).

## Dependency philosophy

Lean runtime, stdlib-first where practical. There are **no optional extras** — one `uv sync`
(`make install`) installs the entire production runtime: **apache-airflow (3.x — the pipeline is
Airflow-native), polars, loguru, blake3, pydantic, pendulum**, the biomedical NER backend
(**gliner** zero-shot, pulls torch), and **`tablassert[qc]`** (the KG build plus its embedding-based
`--qc` audit). GLiNER is still lazy-imported (module load never touches torch) and the mock profile
+ tests stay offline. The hot extraction paths run as **native Go workers** ([`go/`](./go/)) in an Airflow Go SDK
bundle, parity-locked to the pure-Python reference extractors.

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
