# Architecture

How DAKP is layered, how it caches and parallelizes, and exactly where the Tablassert
boundary falls. See [`../PLAN.md`](../PLAN.md) "Recommended architecture" for the design
rationale; this document describes the implemented scaffold and its target shape.

## At a glance

DAKP is a staged, content-addressed pipeline. Every stage reads and writes
[`ArtifactRef`](../src/dakp_pipeline/io/contracts.py) handles (a path + a BLAKE3 id +
optional manifest/schema metadata) — never large in-memory dataframes. Two runners drive
the same stage functions:

- **Pure-Python runner** — [`src/dakp_pipeline/pipeline.py`](../src/dakp_pipeline/pipeline.py)
  `run_pipeline(...)`. This is the source of truth and what the CLI and tests exercise.
- **Airflow DAG** — [`src/dakp_pipeline/dags/dakp_build.py`](../src/dakp_pipeline/dags/dakp_build.py)
  `dakp_build`, a thin TaskFlow wrapper around those same functions. Import-safe without
  Airflow (guarded imports + no-op decorator fallbacks).

```text
                acquire → extract → shape assertions → generate configs → Tablassert handoff → summary
                  │          │           │                   │                    │                 │
   fixtures/net   │   parquet interim    │             graph.yaml +        resolution/KGX     build_summary.json
                  │   (by-hash store)    │             per-table .yaml      (delegated)        tablassert_handoff.json
```

## Layered pipeline

The rebuild is organized into six layers (PLAN.md "Expanded layered pipeline"). The
scaffold implements the first four plus the Tablassert handoff wiring; layers 4–6 are
partially scaffolded and grow in later milestones.

### 1. Acquisition / immutable raw layer

Source-specific fetchers ([`sources/`](../src/dakp_pipeline/sources/)) acquire raw
artifacts. Each fetcher is a class with a `fetch(ctx)` method plus a module-level `fetch`
binding so tests can `monkeypatch.setattr(<module>, "fetch", ...)`. In the `mock` profile,
fetchers call `ingest_fixtures(ctx, names, namespace=...)` which content-addresses each
fixture into the store; any non-mock profile hits `require_mock(ctx, ...)` and raises
`NotImplementedError` (real acquisition is Milestone 2). Acquisition never deletes or
mutates a previously fetched release — it only adds to the immutable store.

### 2. Extraction layer

Extractors ([`extract/`](../src/dakp_pipeline/extract/)) parse raw artifacts into
normalized **interim parquet** tables, registered in place with `ArtifactStore.register`.
The Milestone-1 extractors are faithful-but-tiny:

- [`spl_xml.extract`](../src/dakp_pipeline/extract/spl_xml.py) — streams the gzipped mock
  SPL batch (`<splBatch>` of `<document>` elements) into the `spl_documents` table with
  the PLAN.md column contract. Recognizes LOINC section codes
  (`34067-9` indications, `34070-3` contraindications, `34066-1` boxed warning,
  `42229-5` warnings/precautions).
- [`faers_ascii.extract`](../src/dakp_pipeline/extract/faers_ascii.py) — parses the
  `$`-delimited mock FAERS files, partitions by family (`DEMO`/`DRUG`/`INDI`/`REAC`/
  `RPSR`/`DELETE`), joins within the quarter on `primaryid`, and emits a case-level
  table. NDA numbers are digit-normalized to join consistently with Drugs@FDA `ApplNo`.
- [`drugsfda_products.extract`](../src/dakp_pipeline/extract/drugsfda_products.py) —
  normalizes the products TSV (lowercase columns, digit-only `appl_no`) into parquet.

> **Target vs now.** PLAN.md Phase 3 specifies three extraction output levels
> (raw-normalized → entity/evidence → assertion-ready) and partitioned parquet by
> release/quarter. The scaffold writes single parquet files per table; full partitioning,
> rejects/warnings tables, and `source_record_id` derivation land in Milestone 3.

### 3. NER / canonical mapping layer (scaffolded)

Disease mapping today is a fast **exact-match dictionary baseline**
([`assertions/__init__.py`](../src/dakp_pipeline/assertions/__init__.py) `match_diseases`)
loaded from `tests/fixtures/pipeline/ontology/disease_map.tsv` (text → curie/name/category).
Canonical resolution via Tablassert/fullmap replaces this in Milestone 4; DAKP-specific NER
will be limited to mention/span generation from unstructured text, with fullmap performing
the canonical CURIE/name/category assignment. BABEL is treated as legacy/reference only.

### 4. Assertion table layer

Shapers ([`assertions/`](../src/dakp_pipeline/assertions/)) join extracted tables into
**uncompressed TSV** assertion tables under `data/tabular/`. Each shaper is a thin,
auditable join built on shared helpers (`match_diseases`, `row_for`, `join_pipe`,
provenance constants). The three first-scope families:

| Shaper | Table | Joins |
| --- | --- | --- |
| [`approved_treats`](../src/dakp_pipeline/assertions/approved_treats.py) | `approved_treats_assertions` | DailyMed SPL indications + Drugs@FDA approvals |
| [`observed_uses`](../src/dakp_pipeline/assertions/observed_uses.py) | `faers_applied_to_treat_assertions` | FAERS case-level (drugname × indication) counts |
| [`contraindications`](../src/dakp_pipeline/assertions/contraindications.py) | `contraindication_assertions` | DailyMed SPL contraindication sections (NER-mined) |

Columns are declared in [`schemas.py`](../src/dakp_pipeline/io/schemas.py)
(`ASSERTION_TABLES`) and enforced by the [`translator/contract.py`](../src/dakp_pipeline/translator/contract.py)
readiness gate. See [`tabular-contracts.md`](./tabular-contracts.md).

### 5. Tablassert modeling layer

DAKP generates a Tablassert **Graph config** plus one **table config** per assertion table
([`tablassert/configs.py`](../src/dakp_pipeline/tablassert/configs.py)), then hands off
([`tablassert/run.py`](../src/dakp_pipeline/tablassert/run.py)). DAKP does **not**
implement TCode, fullmap resolution, subgraph compilation, node normalization, the KGX
writer, dedup, UUID machinery, or the RIG compiler — those belong to `../Tablassert`.
See [`tablassert-handoff.md`](./tablassert-handoff.md).

### 6. Translator-readiness layer

A dependency-free contract check ([`translator/contract.py`](../src/dakp_pipeline/translator/contract.py))
verifies each assertion table exists with its declared column contract and records row
counts + any missing columns into `build_summary.json`. Full Biolink/Translator validation
(predicate/category compatibility, dangling-node detection, KGX filename checks) is largely
delegated to Tablassert's QC in Milestone 6+.

## Sharding and concurrency

The DAG targets `prod` (80 logical CPUs, 187 GiB RAM, `/local_raid1`) with
dynamic task mapping and partitioned artifacts so no step becomes a serial bottleneck.
Concurrency is configurable per task class; the profile carries the budgets
(`threads`, `memory_budget_gb`, `quarter_limit`).

**Sharding plan (target, PLAN.md):**

| Source | Shard key | Notes |
| --- | --- | --- |
| DailyMed | full-release ZIP → inner ZIP/XML bin or SPL prefix | streaming XML workers; parquet by release/bin |
| FAERS | quarter × file family (`DEMO`/`DRUG`/`INDI`/`REAC`/`RPSR`/`DELETE`) | join within quarter first; aggregate across quarters last |
| Drugs@FDA | source table | join on normalized application/product IDs |
| Assertion aggregation | per-shard → global reduce | deterministic keys |

**Concurrency controls:** Airflow pools per task class (network downloads, XML parsing,
FAERS parsing, Tablassert handoff); per-task `threads`/`processes`/`batch_size`/
`partition_size`/`memory_budget_gb`; bounded parallelism to avoid `/local_raid1` I/O
saturation; Polars lazy/streaming scans for Python-side joins; per-shard BLAKE3
checkpoints so failed shards rerun independently.

> **Milestone-1 reality.** The `mock` profile runs single-threaded (`threads=1`,
> `memory_budget_gb=1`). The interim outputs are single parquet files, not partitioned
> trees. Airflow dynamic task mapping and Go workers are not yet wired —
> [`workers/go_runner.py`](../src/dakp_pipeline/workers/go_runner.py) raises
> `NotImplementedError` and the `go/` tree lands later.

## BLAKE3 content-addressed store

Borrowed from Nix: immutable store paths, hash-addressed artifacts, manifests, and reuse
keyed by content hash — but using **BLAKE3** as the primary hash for speed on large files
and extracted trees (PLAN.md "Nix-store-inspired artifact and cryptography model").

All hashing lives in [`io/content_hash.py`](../src/dakp_pipeline/io/content_hash.py):

| Function | Purpose | Returns |
| --- | --- | --- |
| `hash_file(path)` | streaming BLAKE3 of file bytes (1 MiB read window) | `b3:<hex>` |
| `hash_tree(root)` | deterministic tree hash: stable over sorted relative paths, file sizes, contents; mtimes/order/empty dirs do not affect it | `b3:<hex>` |
| `hash_bytes(data)` | BLAKE3 of a bytes blob | `b3:<hex>` |
| `sha256_sri(path)` | optional secondary SHA-256 in W3C SRI form (`sha256-<base64>`) for interop | `sha256-…` |
| `artifact_id(hex)` / `digest_dirname(id)` | canonical id form and bare-hex store dir name | `b3:<hex>` / `<hex>` |

Canonical artifact ids are always `b3:<hex>`. SHA-256/SRI/Nix hashes are stored only as
interop metadata, never as the primary key. The pure code path uses the `blake3`
Rust-extension wheel, so tests/CI need no external CLI (`b3sum`, `nix-hash`).

### Store layout and reuse

[`ArtifactStore`](../src/dakp_pipeline/io/artifact_store.py) is bound to a
[`Workdir`](../src/dakp_pipeline/paths.py) and provides two ingest modes:

- **`ingest(src, ...)`** — copies an external file (a fixture or download) into
  `data/raw/by-hash/<hex>/<name>`. If an identical artifact is already present it is a
  **cache hit** (no copy, manifest reused). Also writes a human-readable alias under
  `data/raw/aliases/` and returns `(ArtifactRef, cache_hit)`.
- **`register(path, ...)`** — hashes an artifact already living in the workdir (an
  interim parquet or generated TSV/config) **in place**; no copy. Use `is_tree=True` to
  hash a directory with `hash_tree`.

Reuse is keyed by content hash, **never** by filename or mtime. Re-running the mock
pipeline re-ingests the identical fixtures as cache hits.

### Manifest schema

Every artifact gets a JSON manifest ([`io/manifests.py`](../src/dakp_pipeline/io/manifests.py),
schema version `dakp.artifact.v1`) at `data/manifests/<hex>.json`:

```jsonc
{
  "schema_version": "dakp.artifact.v1",
  "artifact_id": "b3:da80e425…",
  "path": "data/tabular/faers_applied_to_treat_assertions.tsv",
  "media_type": "text/tab-separated-values",
  "hash": { "algorithm": "BLAKE3", "file": "b3:…", "tree": null, "sha256_sri": "sha256-…" },
  "inputs": ["b3:<input1>", "b3:<input2>"],            // upstream artifact ids
  "operation": { "name": "shape_faers_applied_to_treat_assertions", "version": "v1" },
  "source":     { "url": null, "etag": null, "last_modified": null, "retrieved_at": null },
  "environment":{ "git_commit": null, "uv_lock_hash": null, "tablassert_commit": null, "fullmap_hash": null },
  "table":      { "rows": 2, "partitions": null, "schema_fingerprint": "b3:…", "warnings": null }
}
```

`inputs` records the upstream artifact ids, giving a content-addressed provenance DAG.
> **Milestone-1 note:** the `source` and `environment` blocks are declared but mostly
> `null` in the scaffold — real network metadata (ETag/Last-Modified/retrieved_at) and
> git/`uv.lock`/Tablassert/fullmap capture are wired in Milestones 2 and 7. The `table`
> block (rows + `schema_fingerprint`) is populated today.

## The delegation boundary

This is the most important architectural rule: **DAKP shapes tables; Tablassert does the
graph.**

| DAKP owns | Tablassert owns |
| --- | --- |
| source acquisition + manifests | canonical entity resolution (fullmap) |
| extraction into interim parquet | category assignment / node normalization |
| shaping assertion-ready TSV | KGX NDJSON compilation + writing |
| generating Graph/table config YAML | deduplication + deterministic UUIDs |
| content-addressed caching of raw + tabular artifacts | RIG generation |
| the Translator-readiness column contract | full Biolink/Translator validation/QC |

If a needed Biolink slot is not exposed cleanly by Tablassert, the fix is upstreamed into
`../Tablassert` — DAKP does not grow a parallel KGX postprocessor. This keeps the ralph
loop deterministic: inspect a failed extractor/shaper, patch one source-shaping operation,
rerun from the content-addressed tabular boundary, then hand the result to Tablassert.

## Related

- [`logging.md`](./logging.md) — how the layered stages log and how to read failures.
- [`tablassert-handoff.md`](./tablassert-handoff.md) — the generated configs and provenance overrides.
- [`runbook.md`](./runbook.md) — cache invalidation, reruns, shard debugging.
