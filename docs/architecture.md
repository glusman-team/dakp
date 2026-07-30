# Architecture

How DAKP is layered, how it caches and parallelizes, and exactly where the Tablassert
boundary falls. See [`../PLAN.md`](../PLAN.md) "Recommended architecture" for the design
rationale; this document describes the **implemented final architecture**.

## At a glance

DAKP is a staged, content-addressed pipeline: **acquire → extract → NER → aggregate →
Tablassert KGX**. Every stage reads and writes
[`ArtifactRef`](../src/dakp_pipeline/io/contracts.py) handles (a path + a BLAKE3 id + optional
manifest/schema metadata) — never large in-memory dataframes. The **Airflow DAG is the sole
orchestrator** (Airflow 3 is a hard dependency); a pure-Python harness exercises the same stage
functions in tests:

- **Airflow DAG** — [`src/dakp_pipeline/dags/dakp_build.py`](../src/dakp_pipeline/dags/dakp_build.py)
  `dakp_build`. The three `extract_*` tasks are **native Go SDK workers**
  (`@task.stub(queue="golang")`, forked by the ExecutableCoordinator); acquisition, shaping,
  Tablassert handoff, and the summary are Python TaskFlow tasks. Run config comes from the
  `dakp_config` Variable; tasks pass `ArtifactRef` manifests over XCom.
- **Pure-Python harness** — [`src/dakp_pipeline/pipeline.py`](../src/dakp_pipeline/pipeline.py)
  `run_pipeline(...)`, an Airflow-free test/dev harness that runs the stage functions with the
  pure-Python reference extractors (what the unit/integration tests exercise).

```text
              acquire → extract → NER → aggregate → generate configs → Tablassert handoff → summary
                │          │       │        │              │                    │                 │
 fixtures/net   │   parquet interim │   assertion TSV  graph.yaml +      resolution/KGX     build_summary.json
                │   (by-hash store; │   (3 families)   per-table .yaml   (installed         tablassert_handoff.json
                │    Go parity)     │                                   tablassert CLI)
```

## Layered pipeline

The rebuild is organized into six layers (PLAN.md "Expanded layered pipeline"), all implemented.

### 1. Acquisition / immutable raw layer

Source-specific fetchers ([`sources/`](../src/dakp_pipeline/sources/)) acquire raw artifacts;
[`acquire.py`](../src/dakp_pipeline/acquire.py) coordinates concurrent downloads. Each fetcher is a
class with a `fetch(ctx)` method plus a module-level `fetch` binding so tests can
`monkeypatch.setattr(<module>, "fetch", ...)`. Real acquisition uses **stdlib HTTP** (no
`requests`) — DailyMed full releases, Drugs@FDA data files, FAERS quarterly extracts — and is
content-addressed and idempotent. In the `mock` profile, fetchers call
`ingest_fixtures(ctx, names, namespace=...)` which content-addresses each fixture into the store.
Acquisition never deletes or mutates a previously fetched release — it only adds to the immutable
store.

### 2. Extraction layer

Extractors ([`extract/`](../src/dakp_pipeline/extract/)) parse raw artifacts into normalized
**interim parquet** tables, registered in place with `ArtifactStore.register`:

- [`spl_xml.extract`](../src/dakp_pipeline/extract/spl_xml.py) — streams gzipped SPL (HL7 v3 and
  the namespace-free mock shape) into `spl_documents` / `spl_sets` / `spl_approvals` /
  `spl_ingredients` / `spl_sections`. Recognizes LOINC section codes (`34067-9` indications,
  `34070-3` contraindications, `34066-1` boxed warning, `42229-5` warnings/precautions).
- [`faers_ascii.extract`](../src/dakp_pipeline/extract/faers_ascii.py) — parses `$`-delimited FAERS
  files, partitions by family (`DEMO`/`DRUG`/`INDI`/`REAC`/`RPSR`/`DELETE`), joins within the
  quarter on `primaryid`, dedups across quarters, and emits a case-level table. NDA numbers are
  digit-normalized to join consistently with Drugs@FDA `ApplNo`.
- [`drugsfda_products.extract`](../src/dakp_pipeline/extract/drugsfda_products.py) — normalizes the
  Drugs@FDA products/applications/submissions tables into parquet.

**Native Go workers.** The hot extractors run as **native Airflow Go SDK bundle workers**: the
parsing libraries live under [`go/internal/`](../go/) (`dailymed`, `faers`, `drugsfda`) and the
bundle entrypoint [`go/cmd/dakp-bundle`](../go/cmd/dakp-bundle) registers them as the DAG's
`extract_*` tasks. Each is **parity-locked** to the pure-Python reference extractor — golden-file
parity tests in `go test ./...` assert byte-for-byte TSV equality (see
[`../go/README.md`](../go/README.md)). The Airflow worker's ExecutableCoordinator forks the packed
bundle per task instance (no subprocess/OS-command shim); the bundle reads its upstream
`ArtifactRef` manifests from XCom, writes the interim parquet + TSV handoff into the BLAKE3 store,
and pushes the output manifests back as XCom. The pure-Python extractors are retained as the
reference/test oracle.

### 3. NER layer — one composite backend, mentions only

There is **one** NER backend ([`ner/ner.py`](../src/dakp_pipeline/ner/ner.py), `DiseaseNER`) with
**one** entry point (`extract_disease_mentions` / `extract_contraindication_diseases`) — no
pluggable backend selector. It extracts disease/phenotype **mentions** (text span + entity type)
from DailyMed SPL "Contraindications" sections (LOINC `34070-3`) and FAERS indication strings.

The settled backend is a **gazetteer + GLiNER composite**, chosen by a labeled benchmark
([`ner/BENCHMARK.md`](../src/dakp_pipeline/ner/BENCHMARK.md): composite precision 0.972 / recall
1.000 / F1 0.986; SciSpacy dropped). Offline mode (default) is the deterministic, dep-free
gazetteer; production mode (`offline=False`, `[ner]` extra) adds GLiNER zero-shot recall. DAKP
**never resolves terms to ontology CURIEs** — mentions are text + type only; canonical resolution
is exclusively Tablassert/fullmap's job at `build-kg`. See [`ner/README.md`](../src/dakp_pipeline/ner/README.md).

### 4. Assertion table layer

Shapers ([`assertions/`](../src/dakp_pipeline/assertions/)) join extracted tables (and NER mentions)
into **uncompressed TSV** assertion tables under `data/tabular/`. Each shaper is a thin, auditable
join built on shared helpers ([`evidence.py`](../src/dakp_pipeline/assertions/evidence.py):
NDA normalization, SPL-support joining, provenance assembly). The three families:

| Shaper | Table | Joins |
| --- | --- | --- |
| [`approved_treats`](../src/dakp_pipeline/assertions/approved_treats.py) | `approved_treats_assertions` | NDA → Drugs@FDA ingredient → DailyMed SPL approval + indication section |
| [`observed_uses`](../src/dakp_pipeline/assertions/observed_uses.py) | `faers_applied_to_treat_assertions` | FAERS case-level (drugname × indication) distinct-case counts |
| [`contraindications`](../src/dakp_pipeline/assertions/contraindications.py) | `contraindication_assertions` | DailyMed SPL contraindication sections, **NER-mined** |

Columns are declared in [`schemas.py`](../src/dakp_pipeline/io/schemas.py) (`ASSERTION_TABLES`) and
enforced by the [`translator/contract.py`](../src/dakp_pipeline/translator/contract.py) readiness
gate. See [`tabular-contracts.md`](./tabular-contracts.md).

### 5. Tablassert modeling layer

DAKP generates a Tablassert **Graph config** plus one **table config** per assertion table
([`tablassert/configs.py`](../src/dakp_pipeline/tablassert/configs.py)), then hands off
([`tablassert/run.py`](../src/dakp_pipeline/tablassert/run.py)) to the **installed `tablassert`
CLI** (PyPI `8.0.0`, the `[kg]` extra): `tablassert build-kg tables/graph.yaml --fullmap <path>
[--qc] [--release]`. DAKP does **not** implement fullmap resolution, subgraph compilation, node
normalization, the KGX writer, dedup, UUID machinery, or the RIG compiler — those belong to
Tablassert. See [`tablassert-handoff.md`](./tablassert-handoff.md).

### 6. Translator-readiness layer

A dependency-free contract check ([`translator/contract.py`](../src/dakp_pipeline/translator/contract.py))
verifies each assertion table exists with its declared column contract (`validate`) and validates
KGX node/edge records against the DAKP Translator contract (`validate_kgx`: node coverage,
biolink-prefixed categories, the three edge families with chemical/drug subjects +
disease/phenotype objects, and the per-family infores provenance chain). The legacy-informed
regression guardrail ([`translator/regression.py`](../src/dakp_pipeline/translator/regression.py))
re-checks the family/provenance/`clinical_approval_status` invariants on every build.

## Sharding and concurrency

The DAG targets `prod` (80 logical CPUs, 187 GiB RAM, `/local_raid1`) with dynamic task mapping and
partitioned artifacts so no step becomes a serial bottleneck. Concurrency is configurable per task
class; the profile carries the budgets (`threads`, `memory_budget_gb`, `quarter_limit`,
`release_limit`, `download.concurrency`).

| Source | Shard key | Notes |
| --- | --- | --- |
| DailyMed | full-release ZIP → inner ZIP/XML bin or SPL prefix | streaming XML workers; parquet by release/bin |
| FAERS | quarter × file family (`DEMO`/`DRUG`/`INDI`/`REAC`/`RPSR`/`DELETE`) | join within quarter first; aggregate across quarters last |
| Drugs@FDA | source table | join on normalized application/product IDs |
| Assertion aggregation | per-shard → global reduce | deterministic sorted keys |

**Concurrency controls:** Airflow pools per task class (network downloads, XML parsing, FAERS
parsing, Tablassert handoff); per-task `threads`/`processes`/`batch_size`/`memory_budget_gb`;
bounded download parallelism (`download.concurrency`) to avoid `/local_raid1` I/O saturation; Polars
lazy/streaming scans for Python-side joins; per-shard BLAKE3 checkpoints so failed shards rerun
independently; Go workers (`-jobs`/`-limit`) for the hot extractors in prod.

## BLAKE3 content-addressed store

Borrowed from Nix: immutable store paths, hash-addressed artifacts, manifests, and reuse keyed by
content hash — but using **BLAKE3** as the primary hash for speed on large files and extracted
trees. All hashing lives in [`io/content_hash.py`](../src/dakp_pipeline/io/content_hash.py):

| Function | Purpose | Returns |
| --- | --- | --- |
| `hash_file(path)` | streaming BLAKE3 of file bytes (1 MiB read window) | `b3:<hex>` |
| `hash_tree(root)` | deterministic tree hash: stable over sorted relative paths, file sizes, contents; mtimes/order/empty dirs do not affect it | `b3:<hex>` |
| `hash_bytes(data)` | BLAKE3 of a bytes blob | `b3:<hex>` |
| `sha256_sri(path)` | optional secondary SHA-256 in W3C SRI form for interop | `sha256-…` |

Canonical artifact ids are always `b3:<hex>`. The Go workers use `zeebo/blake3` (32-byte output)
and a Nix-NAR-like tree hash that is **byte-for-byte identical** to Python's `hash_tree`, so Python
and Go agree on every artifact id (golden fixtures in `go/internal/blake3store/testdata/`).

### Store layout and reuse

[`ArtifactStore`](../src/dakp_pipeline/io/artifact_store.py) is bound to a
[`Workdir`](../src/dakp_pipeline/paths.py) and provides two ingest modes:

- **`ingest(src, ...)`** — copies an external file (a fixture or download) into
  `data/raw/by-hash/<hex>/<name>`. An identical artifact is a **cache hit** (no copy, manifest
  reused). Also writes a human-readable alias under `data/raw/aliases/`.
- **`register(path, ...)`** — hashes an artifact already living in the workdir (an interim parquet
  or generated TSV/config) **in place**; no copy. Use `is_tree=True` to hash a directory.

Reuse is keyed by content hash, **never** by filename or mtime. Every artifact gets a JSON manifest
([`io/manifests.py`](../src/dakp_pipeline/io/manifests.py), schema `dakp.artifact.v1`) at
`data/manifests/<hex>.json` recording its `inputs[]` (upstream artifact ids) — a content-addressed
provenance DAG.

## The delegation boundary

This is the most important architectural rule: **DAKP shapes tables; Tablassert does the graph.**

| DAKP owns | Tablassert owns |
| --- | --- |
| source acquisition + manifests | canonical entity resolution (fullmap) |
| extraction into interim parquet (Python + Go parity) | category assignment / node normalization |
| NER mention extraction (text + type only) | ontology CURIE/name/category resolution |
| shaping assertion-ready TSV | KGX NDJSON compilation + writing |
| generating Graph/table config YAML | deduplication + deterministic UUIDs |
| content-addressed caching of raw + tabular artifacts | build-cache / QC artifacts |
| the Translator-readiness column/KGX contract | full Biolink/Translator validation/QC |

If a needed Biolink slot is not exposed cleanly by Tablassert, the fix is upstreamed into
Tablassert — DAKP does not grow a parallel KGX postprocessor.

## Deployment topology & shared filesystem

The pipeline is Airflow-native: the `dakp_build` DAG is the sole orchestrator and the three
`extract_*` tasks run as **native Go SDK bundle workers** (the ExecutableCoordinator forks the packed
bundle, [`go/cmd/dakp-bundle`](../go/cmd/dakp-bundle), once per task instance).

**Data plane = filesystem, not XCom.** Tasks pass only small `ArtifactRef` manifests (path + BLAKE3
id + metadata) over XCom; the heavy bytes move through the BLAKE3 content-addressed store on disk.
Consequently, **every worker that runs any task must see the same workdir / content-addressed
store**, and **every worker that runs an `extract_*` task must have the packed bundle in its
`executables_root`**:

- **LocalExecutor (`make run`, single host):** automatic — one machine, one filesystem. The
  orchestrator builds the bundle into `$AIRFLOW_HOME/executable-bundles` and points the coordinator
  at it; the workdir is local.
- **CeleryExecutor / distributed workers:** the workdir + store must live on a **shared/networked
  filesystem** (e.g. NFS) mounted at the same path on all workers, and the packed bundle must be
  deployed to `executables_root` on every worker that serves the `golang` queue. Otherwise a Go
  extract task on worker B cannot read the raw artifacts worker A acquired.

Required Airflow config (set by `scripts/dakp_up.sh` via `AIRFLOW__*` env vars; equivalently
`airflow.cfg`):

- `[sdk] coordinators` — register `airflow.sdk.coordinators.executable.ExecutableCoordinator` with
  `executables_root` pointing at the packed-bundle directory.
- `[sdk] queue_to_coordinator = {"golang": "go"}` — route the `extract_*` stub tasks to the Go
  coordinator.
- `[core] execution_api_server_url` — **must** be set to the API server's `/execution/` URL; without
  it the task supervisor defaults to `localhost:8080`.
- The `dakp_download` / `dakp_extract` **pools** must exist (the orchestrator provisions them);
  tasks on a non-existent pool are never scheduled.

## Related

- [`semantic-equivalence.md`](./semantic-equivalence.md) — preserved-vs-improved semantics vs the old DAKP.
- [`wenceslaus-runbook.md`](./wenceslaus-runbook.md) — the full production build (fullmap + prod KG).
- [`logging.md`](./logging.md) — how the layered stages log and how to read failures.
- [`tablassert-handoff.md`](./tablassert-handoff.md) — the generated configs and provenance overrides.
- [`runbook.md`](./runbook.md) — cache invalidation, reruns, shard debugging.
