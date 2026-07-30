# DAKP Pipeline Rebuild Plan

## Context

The current DAKP repository is a collection of legacy Perl/Python scripts for building the Drug Approvals Knowledge Provider KG from:

- DailyMed SPL XML full releases (`DailyMed/bin/getFullRelease.pl`, `DailyMed/bin/parseXML-xtree.py`)
- FAERS quarterly extracts (`FAERS/bin/getLatest.pl`, `FAERS/bin/listCases.pl`, `FAERS/bin/drug2indi.pl`, `FAERS/bin/findIndicationTerms.py`)
- Drugs@FDA product/NDA data (`DrugsFDA/bin/download.pl`)
- HPO/MONDO OBO term lists (`HPO/bin/obo2terms.pl`, `MONDO/bin/obo2terms.pl`)
- MEDI / matrix contraindication list (`matrix/bin/studyContraindications.py`, `matrix/bin/contraindications2kg.py`)

The current build is brittle because it relies on hardcoded local paths (`/ssd2/sqlite/BABEL.db`, `/15TB_1/users/...`), missing local Perl libraries (`../lib/libSystem.pl`, `../lib/libText.pl`), shell side effects, and monolithic scripts that mix downloading, parsing, NER/entity mapping, edge modeling, and KGX serialization.

Target outcome: rebuild as a reproducible `uv` Python project with an Apache Airflow DAG for the full pipeline. The first executable version should run end-to-end on mocked inputs with monkeypatch-friendly task boundaries for future tests, then scale to real source downloads and full local builds. The DAG should first produce clean tabular outputs for most stages, then model those tables into KGX/Translator-compatible outputs using active-development Tablassert/fullmap from `../Tablassert` once the Tablassert 8.x API is ready. Do not add the Tablassert dependency until that API stabilizes.

## User decisions captured

- Build the **full pipeline skeleton first**, but with mocked inputs and monkeypatchable download/extraction/modeling boundaries for future testing.
- Target deployment/build machine is `wenceslaus`: Ubuntu 24.04, dual Intel Xeon Gold 6230 CPUs (80 logical CPUs), 187 GiB RAM, `/local_raid1` ~1.75 TiB, `/home` ~1.5 TiB, and four NVIDIA Tesla P100 16 GiB GPUs. Design should still have laptop-safe modes, but the main full build can assume this workstation-class host.
- Tablassert/fullmap should be the **canonical mapping backend**. BABEL should be treated as legacy/reference/comparison only unless temporarily needed during migration.
- FAERS labels and current `applied_to_treat` clinical status behavior should stay constant for now.
- `../DINGO` is the local translator-ingests reference to model against.
- Contraindications remain **first-scope**, not deferred.

## Dependency philosophy

This project follows the same minimal-dependency philosophy as Tablassert and the rest of the team's projects: **prefer the standard library of each language and add a dependency only when it is essential and already justified.**

- In Python, lean on the standard library (`pathlib`, `hashlib`, `json`, `subprocess`, `logging`, `argparse`/`cyclopts`-style CLIs, `dataclasses`, `concurrent.futures`, `gzip`, `zipfile`, `xml.etree`/streaming) before reaching for third-party packages.
- In Go, lean on the standard library (`os`, `io`, `bufio`, `encoding/json`, `encoding/xml`, `archive/zip`, `log/slog`, `context`) plus `golang.org/x/sync/errgroup` for bounded concurrency.
- Only add a dependency when it materially reduces risk or effort and there is no good stdlib path. The expected, pre-approved dependencies are:
  - Python: Airflow (orchestration), Polars (large-table transforms), loguru (logging), blake3 (hashing), pydantic (config/schema validation), pytest + ruff + pyright + pre-commit (dev tooling).
  - Go: `golang.org/x/sync` (errgroup); BLAKE3 via a maintained Go module.
- Anything beyond that list must justify itself in the PR that introduces it and should be revisited if the stdlib catches up.
- Do not add heavyweight frameworks (web servers, ORMs, generic ETL engines) just to structure code; the DAG + typed task interfaces + content-addressed store are the structure.

## Current pipeline inventory

### Source acquisition

- `DailyMed/bin/getFullRelease.pl`
  - Fetches DailyMed full release index, downloads all release ZIPs, extracts XMLs into `DailyMed/xmls/<bin>/...xml.gz`, then runs downstream extractors.
  - Needs refactor: destructive/stash behavior, hardcoded external download directory, shelling out to `wget`, `unzip`, `gzip`, and missing `countCodes.pl`.
- `DrugsFDA/bin/download.pl`
  - Downloads FDA Drugs@FDA data ZIP into `new.zip` and extracts to `DrugsFDA/data/tmp`.
- `FAERS/bin/getLatest.pl`
  - Downloads quarterly FAERS ASCII ZIPs, normalizes filenames, runs case listing, indication extraction, and FAERS term mapping.
  - Needs refactor: missing Perl libs, shell side effects, no manifest/checkpointing beyond ad hoc file existence.

### Extraction / aggregation

- `DailyMed/bin/parseXML-xtree.py`
  - Parses SPL XML with `lxml`, extracts set IDs, approvals, active/inactive ingredients, and LOINC-coded text sections.
- `DailyMed/bin/selectActiveIngredientSingletons.pl`
  - Produces singleton active ingredient table used downstream.
- `DailyMed/bin/findTermsInIndications.pl`
  - Uses MONDO/HPO term lists, word frequency, and custom lexical matching to find disease/phenotype terms in DailyMed indication text.
- `FAERS/bin/listCases.pl`
  - Aggregates DEMO/DRUG/RPSR/REAC/INDI into case-level table plus indication and drug frequency tables.
- `FAERS/bin/drug2indi.pl`
  - Joins FAERS drug/indication counts to Drugs@FDA NDA product data to produce drug/indication tables by NDA and by name.
- `FAERS/bin/caseList2uses.pl`
  - Produces use counts by ingredient/indication from case-level data.
- `matrix/bin/studyContraindications.py`
  - Links MEDI contraindication rows to DailyMed SPL evidence using lexical overlap over DailyMed contraindication text.

### Current KG generation

- `bin/drug2indi2kg.py` and `bin/uselist2kg.py`
  - Read FAERS and DailyMed intermediates, use BABEL sqlite normalization, emit gzipped node/edge TSVs.
  - Predicates include `biolink:treats` and `biolink:applied_to_treat`.
- `matrix/bin/contraindications2kg.py`
  - Emits contraindication nodes/edges from MEDI/matrix rows.
- `bin/dakp-postprocess2jsonlBL.py`
  - Re-normalizes TSV outputs through BABEL, merges treatments and contraindications, assigns deterministic UUIDs, emits KGX JSONL gz files.
- `bin/jsonlines2tsv.py`
  - Converts generated JSONL back to simple TSV, useful as a comparison artifact but not a primary target.

### Translator ingest reference found in `../DINGO`

`../DINGO` appears to contain the NCATS Translator ingests repo or a local clone with DAKP ingest materials:

- `../DINGO/src/translator_ingest/ingests/dakp/dakp.yaml`
  - Reads `drug_approvals_kg_edges.jsonl.gz`.
- `../DINGO/src/translator_ingest/ingests/dakp/dakp.py`
  - Loads `drug_approvals_kg_nodes.jsonl.gz`, creates Biolink Pydantic nodes and associations, converts DAKP `sources` into `RetrievalSource` objects, maps `N_cases` to `number_of_cases`, approvals to `FDA_regulatory_approvals`, and evidence to publications.
- `../DINGO/tests/unit/ingests/dakp/test_dakp.py`
  - Documents expected association families and source provenance combinations for `treats`, `applied_to_treat`, and `contraindicated_in`.
- `../DINGO/src/translator_ingest/ingests/dakp/dakp_rig.yaml`
  - Reference Ingest Guide content to preserve/update.

### Tablassert 8.x notes from `../Tablassert`

- Current source kinds in `../Tablassert/src/tablassert/models.py` are `Text` and `Excel`; I did not find a `ManualSource` class by name.
- I found `ManualProvenance`, exposed as `provenance.override`, for non-PMC/PMID sources.
- Plan should use Tablassert from a local editable path later, but not add it as a dependency yet.
- Likely target: produce clean tabular source files plus Tablassert YAML table configs using `source.kind: text` and `provenance.override` unless Tablassert introduces a new `ManualSource`/manual table source API before implementation.

## Recommended architecture

The rebuild should combine Airflow orchestration with Tablassert's KG-building architecture and Nix-style content-addressed artifact discipline.

### Architecture inspiration from Tablassert

Tablassert's current `build_pipeline` in `../Tablassert/src/tablassert/cli.py` is a useful pattern: it runs a staged pipeline of **load tables → extract sections → build TCode → collect instructions → build subgraphs → compile graph**. Internally, `../Tablassert/src/tablassert/lib.py` keeps the work declarative: source operations, node-prep/fullmap resolution, provenance operations, per-section parquet subgraphs, graph compilation, KGX NDJSON writing, Rust-backed deduplication/UUID generation, and RIG generation.

DAKP should **not** reimplement Tablassert's TCode, fullmap resolution, subgraph compilation, node normalization, KGX writer, deduper, UUID machinery, or RIG compiler. The DAKP rebuild should do everything *up to* the shape Tablassert already knows how to consume:

1. **Load/acquire source manifests** rather than blindly downloading files.
2. **Extract source sections** from DailyMed, FAERS, Drugs@FDA, ontologies, and MEDI into stable source-section tables.
3. **Transform source sections into assertion-ready tables** with the right columns, evidence fields, provenance values, and Tablassert config YAMLs.
4. **Delegate entity resolution, NER/resolution where supported, graph compilation, KGX writing, deduplication, deterministic IDs, and RIG generation to Tablassert/fullmap.**
5. **Only build DAKP-specific caching around raw and tabular artifacts**, not a parallel graph-build system.

This gives the ralph loop something deterministic to drive without duplicating Tablassert: inspect a failed extractor/transform, patch one source-shaping operation, rerun from the content-addressed tabular boundary, then hand the result to Tablassert.

### Nix-store-inspired artifact and cryptography model with BLAKE3

Do not invent a bespoke artifact-cache model if Nix-like mechanisms can cover the same job. Borrow Nix's ideas: immutable store paths, hash-addressed artifacts, manifests, verification, optional signing, and no mtime-based reuse. But use **BLAKE3 as the primary DAKP content hash** for speed on large source files and extracted trees.

Plan:

- Use **BLAKE3 content-addressed artifact keys** for every raw download, extracted directory, normalized table, and assertion-ready table.
- Record both **file BLAKE3 hashes** and **directory/tree BLAKE3 hashes**. Directory hashes should be deterministic over relative paths, file types, sizes, and content, similar in spirit to Nix's NAR/path hashing but using BLAKE3.
- Use canonical artifact IDs like `b3:<hex>` or `b3:<base32/base64url>` and store any optional SHA-256/SRI/Nix hashes only as interoperability metadata, not as the primary key.
- Include in each artifact manifest:
  - source URL(s), ETag, Last-Modified, and retrieval timestamp
  - content hash and, for extracted trees, deterministic tree hash
  - input artifact hashes
  - operation name and operation config hash
  - git commit and `uv.lock` hash
  - fullmap redb path/version/hash
  - Tablassert checkout commit/hash when used
  - row counts, partition counts, schema fingerprint, and validation status
- Make tasks skip/reuse outputs by manifest hash, not by filename or mtime.
- Add optional Nix-backed verification/prefetch metadata later if useful, but keep BLAKE3 pure-Python/Rust-extension hashing as the normal code path so tests and CI do not require Nix.

### Logging and observability

The pipeline should be easy to debug from Airflow, the command line, and ralph-loop logs. Airflow is the primary log surface; everything else feeds into it.

Airflow logging design:

- Use Airflow's built-in task-instance logs as the canonical record. Configure `airflow.cfg` / env so logs land under `logs/dag_id=.../run_id=.../task_id=.../` and are easy to grep.
- Set a DAG-level default `logging_config` and per-task log levels via DAG params (`log_level`, `verbose`).
- In Python orchestration code, use `loguru` for structured, readable logs and bridge it into the standard `logging` root logger so Airflow's file handler captures it. A single `InterceptHandler` forwards loguru records to `logging.getLogger("airflow.task")`.
- Attach Airflow context to every log line: `dag_id`, `run_id`, `task_id`, `try_number`, `profile`. Use a loguru `bind()`/contextualize helper set at task entry.
- Go workers are subprocesses; capture their stdout/stderr through Airflow's task logging by streaming them line-by-line into the task logger (not just dumping at the end), so interleaved shard logs appear in order.

Structured logging fields (Python and Go share a schema):

- `task_id`, `shard_id`, `artifact_id`, `input_hash`, `output_hash`, `rows`, `partitions`, `elapsed_ms`, `cache_hit`, `warning_count`.
- Go workers emit JSON lines (`slog` or `zerolog`); Python parses/relays them so Airflow shows them uniformly.

Reports and failure handling:

- Every task writes a small `task_report.json` alongside artifact manifests with timings, row counts, cache hit/miss status, warning summaries, and output paths.
- Emit progress at shard boundaries, not per row, to keep logs useful.
- Add failure bundles: when a shard fails, write the exact input manifest, command args, stderr/stdout path, and first N parse warnings, and log the bundle path.
- Add optional metrics later (Prometheus/OpenTelemetry) only if Airflow/loguru/task reports are not enough.

### Expanded layered pipeline with sharding and concurrency

The Airflow DAG should use dynamic task mapping and partitioned artifacts so `wenceslaus_full` can use available CPU, memory, and disk bandwidth without turning any one step into a serial bottleneck. Concurrency should be configurable per task class, with conservative defaults and high-throughput settings for `wenceslaus`.

Sharding plan:

- **DailyMed**: shard by full-release ZIP, then by inner ZIP/XML bin or SPL document prefix; parse XML with streaming workers; write partitioned parquet by release and bin.
- **FAERS**: shard by quarter and file family (`DEMO`, `DRUG`, `INDI`, `REAC`, `RPSR`, `DELETE`); join within quarter first; aggregate across quarters only after per-quarter artifacts are complete.
- **Drugs@FDA**: shard by source table for parsing, then join on normalized application/product IDs.
- **Ontology/fullmap/term indexes**: build once per fullmap/source hash; query in batches; cache resolved unique strings before joining back to large tables.
- **DailyMed/FAERS NER**: shard text by document/section bins and indication-string hash buckets; resolve unique mention strings rather than every occurrence.
- **Assertion aggregation**: aggregate locally per shard, then reduce globally by deterministic keys.
- **Tablassert handoff**: produce fewer, well-shaped assertion tables and configs; let Tablassert handle its own internal parallelism and graph build phases.

Concurrency controls:

- Airflow pools for network downloads, XML parsing, FAERS parsing, fullmap/Tablassert handoff, and optional model NER.
- Per-task `threads`, `processes`, `batch_size`, `partition_size`, and `memory_budget_gb` config.
- Bounded parallelism on `/local_raid1` to maximize throughput without I/O thrash.
- Polars lazy/streaming scans for Python-side joins and reductions where Python remains appropriate.
- Checkpoint every shard by BLAKE3 manifest so failed shards rerun independently.

Go worker plan:

- Heavy parsing/extraction workers should be native Go CLIs where that is likely faster/simpler than Python: FAERS ASCII parsing, DailyMed XML shard extraction, BLAKE3 tree hashing, and high-volume text normalization.
- Airflow tasks remain thin Python orchestrators that call `go run ./go/cmd/dakp-worker ...` in development or a built `dakp-worker` binary in full runs.
- Go workers should use `golang.org/x/sync/errgroup` for bounded concurrent shard processing and cancellation-on-first-error. Use `SetLimit` to respect Airflow task concurrency and memory settings.
- Go workers should emit structured JSON logs and final artifact manifests to stdout/files so Airflow can collect them.
- Python implementations can exist for tiny fixtures where useful, but production-heavy code paths should prefer Go workers rather than making every Airflow worker a Python script.

1. **Acquisition and immutable raw-source layer**
   - Source-specific fetchers for DailyMed, FAERS, Drugs@FDA, MONDO, HPO, fullmap/BABEL-derived resources, and MEDI.
   - Every fetcher has a real implementation plus a fixture/mock implementation selected by config or monkeypatch.
   - Downloads go into an immutable cache under `data/raw/by-hash/<hash>/...`; human-readable aliases point to the hash.
   - Acquisition never deletes or mutates previously fetched releases.

2. **Dramatically improved extraction layer**
   - Extraction is the most important rebuild target. It should become a typed, schema-validated, partitioned table factory rather than a collection of side-effect scripts.
   - DailyMed extraction should stream SPL XML and emit normalized document, product, approval, ingredient, section, set, LOINC, and evidence tables. It should retain XML provenance, section code/title/text, set IDs, approval IDs, application types, active/inactive ingredient roles, UNII codes, and parse warnings.
   - FAERS extraction should parse each quarterly ASCII file into normalized parquet tables for DEMO, DRUG, INDI, REAC, RPSR, DELETE, and derived case-level joins. It should preserve quarter, source file, primary/case IDs, drug sequence, reporter metadata, suspect/concomitant role codes, drug names, ingredient strings, NDA numbers, indications, reactions, and delete/dedup decisions.
   - Drugs@FDA extraction should normalize product/application/action/submission tables, preserve NDA/BLA/ANDA variants with and without leading zeroes, and create lookup tables for proprietary names, nonproprietary names, ingredients, application numbers, marketing status, and product NDCs.
   - Ontology/fullmap extraction should produce term/alias/category/source-version tables or consume the fullmap redb directly, with clear hash/version capture.
   - MEDI/matrix extraction should preserve original contraindication strings, normalized provided IDs/labels, version, sheet/row provenance, and DailyMed support matching evidence.
   - All extractors should emit: output path, schema version, row counts, input hashes, parse warnings, and sample QA summaries.
   - Extraction tasks should be streaming or partitioned by release/quarter/bin so they can use `wenceslaus` parallelism without requiring one huge in-memory dataframe.

3. **NER, mention, and canonical mapping layer**
   - Fold in Tablassert's NER/resolution machinery wherever possible instead of duplicating it. Tablassert already has `level_one`, `level_two`, `resolve_batch`, `resolve_many`, fullmap audits, category prioritization/avoid lists, taxon filtering, and Rust-backed fullmap lookup.
   - DAKP-specific NER should focus on source-aware mention generation: extracting spans from DailyMed section text and FAERS indication strings, preserving offsets, section context, and candidate evidence.
   - Tablassert/fullmap should resolve text/candidate columns to canonical CURIE/name/category/source metadata during the modeling stage or a shared pre-modeling mapping task.
   - Optional model NER/reranking can propose mention spans or candidate ranking, but Tablassert/fullmap remains the canonical resolver.

4. **Assertion table layer**
   - Aggregates mapped/candidate evidence into explicit assertion tables for approved treatments, observed FAERS applied-to-treat uses, and contraindications.
   - Keeps evidence fields denormalized enough for Tablassert annotations/provenance and normalized enough for QA: approval IDs, SPL set IDs, case counts, source rows, mapping decision IDs, and confidence/extraction notes.

5. **Tablassert modeling and graph compilation layer**
   - Use Tablassert table configs to encode subject/object/predicate/provenance/annotations from DAKP assertion tables.
   - Let Tablassert/fullmap perform canonical node resolution, category assignment, node normalization, edge-category derivation, deduplication, deterministic UUIDs, KGX NDJSON writing, and RIG generation where the API supports it.
   - If DAKP needs custom Biolink slots that Tablassert does not yet expose cleanly, upstream that support into `../Tablassert` rather than building a parallel KGX compiler here.

6. **Translator-readiness and quality layer**
   - Do **not** require byte-for-byte or edge-for-edge equivalence with the current DAKP ingest, because this rebuild is expected to improve extraction, NER, and modeling.
   - Instead, validate against a Translator-readiness contract: Biolink-valid nodes/edges, required provenance/source fields, expected edge families, stable clinical labels for FAERS in the first rebuild, required FDA approval/evidence/case-count fields where applicable, no missing node references, and ingestable KGX filenames.
   - Use the old output and `../DINGO` tests as regression guardrails and examples, not as a ceiling on output quality.

## Proposed repository layout

```text
pyproject.toml
uv.lock
go.mod
go.sum
README.md
src/dakp_pipeline/
  __init__.py
  cli.py
  config.py
  paths.py
  io/
    artifact_store.py
    content_hash.py
    downloads.py
    manifests.py
    schemas.py
  sources/
    dailymed.py
    drugsfda.py
    faers.py
    hpo.py
    mondo.py
    medi.py
  extract/
    spl_xml.py
    faers_ascii.py
    drugsfda_products.py
  ner/
    lexical.py
    dictionary.py
    model.py
    candidates.py
  assertions/
    approved_treats.py
    observed_uses.py
    contraindications.py
    evidence.py
  tablassert/
    configs.py
    run.py
  translator/
    contract.py
    rig.py
  workers/
    go_runner.py          # thin Airflow/Python wrapper around compiled Go workers
  dags/
    dakp_build.py
go/
  cmd/
    dakp-worker/
      main.go
  internal/
    blake3store/
    dailymed/
    faers/
    drugsfda/
    medi/
    pipeline/
configs/
  local.example.yaml
  pipeline.yaml
data/
  raw/
    by-hash/
    aliases/
  interim/
  tabular/
  kgx/
  manifests/
  store/
tables/
  graph.yaml                 # Tablassert Graph config referencing the table configs below
  approved_treats.yaml
  faers_applied_to_treat.yaml
  contraindications.yaml
docs/
  README.md
  architecture.md
  logging.md
  sources.md
  tabular-contracts.md
  tablassert-handoff.md
  runbook.md
tests/
  unit/
  integration/
  fixtures/
ref/
  legacy/                # old scripts copied/moved here for audit only; never invoked by DAG
```

### Documentation

The repo must ship comprehensive docs, not just code. `README.md` at the root should be the entry point and link into `docs/`.

`README.md` should cover:

- What DAKP is and why the rebuild exists.
- Quickstart for the mocked pipeline on a development laptop (`uv sync`, `uv run pytest`, `uv run dakp run --profile mock`).
- How to run the full build on `wenceslaus` (`--profile wenceslaus_full`).
- The DAG shape and what each task produces.
- Where artifacts, manifests, and logs land.
- How Tablassert/fullmap is used and where the handoff happens.
- Provenance/source semantics for the three edge families.
- How to extend with a new source.

`docs/` should expand each area:

- `architecture.md`: the layered pipeline, sharding/concurrency, BLAKE3 store, and Tablassert delegation boundary.
- `logging.md`: the Airflow/loguru/Go logging design and how to read a failed run.
- `sources.md`: per-source acquisition, extraction, and schema notes.
- `tabular-contracts.md`: every tabular contract with columns and examples.
- `tablassert-handoff.md`: how assertion tables and the graph/table configs are generated and consumed.
- `runbook.md`: common failures, reruns, cache invalidation, and shard-level debugging.

## Approach

### Phase 0: Ground truth and acceptance criteria

- Implement the whole DAG shape immediately, with mocked fixtures/monkeypatched task implementations available from day one so the complete pipeline can be tested before real source acquisition is stable.
- Treat `../DINGO` as the active translator-ingests reference contract.
- Treat Tablassert/fullmap as canonical mapping; retain BABEL behavior only for legacy regression comparison and temporary migration shims.
- Preserve the three current Translator edge families:
  - `chemical/drug --biolink:treats--> disease/phenotype`
  - `chemical/drug --biolink:applied_to_treat--> disease/phenotype`
  - `chemical/drug --biolink:contraindicated_in--> disease/phenotype`
- Preserve key attributes/provenance from the current output and translator ingest:
  - deterministic edge IDs
  - subject/object names and Biolink categories
  - `clinical_approval_status`, keeping current FAERS/applied-to-treat label behavior constant for the first rebuild
  - FDA approval/NDA/BLA/ANDA identifiers
  - FAERS case counts as `number_of_cases` or equivalent annotation
  - DailyMed SPL set/evidence identifiers
  - source chain involving `infores:multiomics-drugapprovals`, `infores:dailymed`, `infores:faers`, and `infores:medi`
- Make every intermediate table inspectable and restartable.

### Phase 1: Project bootstrap

- First implementation action: create a new git branch before any code changes.
- Create a `uv` Python project with pinned Python version, `ruff`, `pytest`, `pyright`, `pre-commit`, and package entrypoints.
- Add project-specific pre-commit hooks using the Tablassert repo as a model: ruff, ruff-format, pyright via `uv run`, pytest via `uv run`, plus Go formatting/testing hooks once Go workers are added.
- Add Airflow dependencies only after deciding whether to run Airflow directly in-process or via optional dependency group.
- Add basic config/paths system so no absolute paths are embedded in code.
- Add a fixture/mocking framework immediately: every downloader, parser, source-shaping worker, and Tablassert runner should have dependency-injected interfaces that tests can monkeypatch.
- Do **not** run the real Tablassert build or real full-data build in the first mocked milestone on the current development laptop; the Tablassert task should be monkeypatched there. Production/full profile assumes Tablassert is available and should not include fallback KGX logic.
- Keep legacy scripts only under `ref/` for comparison during migration; never invoke them from the new DAG.

### Phase 2: Airflow DAG design

DAG name: `dakp_build`.

Initial full DAG shape, with every task runnable against tiny mocked/fixture inputs before real full-data execution:

```text
start
  -> acquire_dailymed
  -> extract_dailymed_spl
  -> acquire_drugsfda
  -> extract_drugsfda_products
  -> acquire_faers
  -> extract_faers_cases
  -> acquire_medi_matrix
  -> extract_contraindications
  -> shape_treatment_tables
  -> shape_faers_use_tables
  -> shape_contraindication_tables
  -> generate_tablassert_configs
  -> run_tablassert
  -> write_build_summary
end
```

Implementation details:

- Use Airflow `@task` Python TaskFlow API for readability.
- Each task accepts/returns manifests or paths, not big in-memory dataframes.
- Use partitioned Parquet for large interim tables and **uncompressed TSV/CSV** for final tabular outputs meant for Tablassert, because Tablassert table sources should read plain text files.
- Store source versions/checksums in `data/manifests/*.json`.
- Add `force`, `since`, `quarter_limit`, `mock_sources`, and `fixture_root` config options.
- Do not acquire ontologies or build custom NER indexes in DAKP first-scope; Tablassert/fullmap/BABEL-derived fullmap owns canonical resolution resources.
- `run_tablassert` is assumed to exist for real/full runs; tests monkeypatch it, but no fallback local KGX compiler should be built.
- `write_build_summary` only writes local file manifests, task reports, and output paths. It does not publish artifacts or validate KGX; Tablassert handles its own validation/QC.
- Provide two execution profiles:
  - `mock`: tiny fixture inputs, all external calls monkeypatchable, suitable for CI and ralph loops.
  - `wenceslaus_full`: real full build on `/local_raid1` targeting the 80-thread / 187 GiB RAM workstation.

### Phase 3: Extraction and tabular output contracts

Extraction should be rebuilt as a first-class data engineering layer, not a thin port of the legacy scripts. Each extractor should have three output levels:

1. **Raw-normalized tables**: faithful row-level representations of source files with source file, row number, release/quarter, parse warnings, and no semantic collapsing.
2. **Entity/evidence tables**: section text, ingredients, approvals, indications, reactions, product/application links, and SPL support evidence with stable source record IDs.
3. **Assertion-ready tables**: joined, deduplicated, and evidence-rich records that are ready for fullmap/Tablassert mapping/modeling.

Extraction requirements:

- Every output table has an explicit schema version and a schema fingerprint in its manifest.
- Every row has a stable `source_record_id` derived from source hash + source-local row/document identifiers, not from mutable row order alone.
- Every lossy decision emits a QA counter and, where feasible, a rejects/warnings table.
- Extraction is partitioned by source release or quarter (`dailymed_release`, `faers_quarter`, `medi_version`) and can be rerun independently.
- Text extraction preserves both raw text and cleaned text so NER changes do not require re-downloading or reparsing XML/ASCII files.
- Parsing code should be pure functions over paths/config where possible, making it easy to monkeypatch file inputs in tests.

Create tabular outputs before final KG modeling. Early tables should be text-first and mapping-auditable; after the fullmap/Tablassert mapping step, assertion tables may include canonical CURIE/name/category columns used for KGX modeling. Proposed table families:

#### Raw/normalized extraction tables

- `data/interim/dailymed/spl_documents.parquet`
- `data/interim/dailymed/spl_sections.parquet`
- `data/interim/dailymed/spl_approvals.parquet`
- `data/interim/dailymed/spl_ingredients.parquet`
- `data/interim/dailymed/spl_sets.parquet`
- `data/interim/faers/demo.parquet`
- `data/interim/faers/drug.parquet`
- `data/interim/faers/indi.parquet`
- `data/interim/faers/reac.parquet`
- `data/interim/faers/rpsr.parquet`
- `data/interim/faers/delete.parquet`
- `data/interim/drugsfda/products.parquet`
- `data/interim/drugsfda/applications.parquet`
- `data/interim/drugsfda/submissions.parquet`
- `data/interim/medi/contraindications.parquet`

#### Public tabular contracts

#### `data/tabular/dailymed_spl_documents.tsv`

- `spl_document_id`
- `spl_set_id`
- `xml_path`
- `release_file`
- `approval_code`
- `approval_type`
- `loinc_code`
- `section_name`
- `section_text`
- `active_ingredient_name`
- `active_ingredient_unii`

#### `data/tabular/faers_cases.tsv`

- `quarter`
- `primaryid`
- `caseid`
- `source`
- `occp_cod`
- `reporter_country`
- `drugname`
- `ingredient`
- `nda`
- `indication`
- `effects`

#### `data/tabular/faers_drug_indication_counts.tsv`

- `case_count`
- `nda`
- `drug_name`
- `ingredient_name`
- `indication_text`
- `source_quarters`

#### `data/tabular/mention_candidates.tsv`

- `source_table`
- `source_record_id`
- `text_field`
- `mention_text`
- `mention_start`
- `mention_end`
- `semantic_group` (`drug`, `disease`, `phenotype`)
- `candidate_curie`
- `candidate_name`
- `candidate_category`
- `candidate_source` (`BABEL`, `MONDO`, `HPO`, `SapBERT`, `scispaCy`, etc.)
- `score`
- `rank`
- `normalization_notes`

#### `data/tabular/approved_treats_assertions.tsv`

- `subject_text`
- `subject_curie` (nullable until mapping stage is finalized)
- `subject_name`
- `subject_category`
- `predicate` = `biolink:treats`
- `object_text`
- `object_curie`
- `object_name`
- `object_category`
- `approval_ids`
- `supporting_spl_sets`
- `supporting_spl_documents`
- `clinical_approval_status` = `approved_for_condition`
- `knowledge_level` = `knowledge_assertion`
- `agent_type` = `manual_validation_of_automated_agent`
- `primary_knowledge_source` = `infores:multiomics-drugapprovals`
- `upstream_resource_ids` = `infores:dailymed|infores:faers`

#### `data/tabular/faers_applied_to_treat_assertions.tsv`

- Same subject/object/predicate fields, with:
  - `predicate` = `biolink:applied_to_treat`
  - `case_count`
  - `clinical_approval_status` preserving the current FAERS/applied-to-treat behavior for the first rebuild
  - FAERS as primary upstream source and DailyMed as supporting upstream source.

#### `data/tabular/contraindication_assertions.tsv`

- Same subject/object/predicate fields, with:
  - `predicate` = `biolink:contraindicated_in`
  - `supporting_spl_sets`
  - `medi_version`
  - `source_score`
  - upstream sources `infores:medi` and `infores:dailymed`.

### Phase 4: NER / entity resolution strategy

Goal: better NER while keeping a fixture/mock mode, laptop-safe mode, and a high-throughput `wenceslaus_full` mode.

Recommended staged approach:

1. **Fast dictionary baseline**
   - Port existing MONDO/HPO lexical matcher into typed Python.
   - Treat BABEL lookup as legacy/reference behavior, not the canonical mapping backend.
   - Precompile dictionary indexes to SQLite/Parquet/DAWG/trie artifacts.
   - Use normalized strings and aliases from MONDO/HPO plus drug aliases from Drugs@FDA/DailyMed/fullmap inputs.
2. **Candidate generation**
   - Use exact/normalized phrase matching first.
   - Add abbreviation/synonym handling, section-aware matching, and blacklist/ignore terms from legacy scripts.
   - Emit all candidates with ranks, not just the chosen ID.
3. **Canonical fullmap/Tablassert resolution**
   - Feed candidate/entity text columns into Tablassert/fullmap for canonical IDs, categories, names, and taxon-aware resolution.
   - Store final mapping decisions separately from raw mention candidates so mapping upgrades can be rerun without reparsing raw sources.
4. **Optional model reranker**
   - Evaluate SciSpacy small models, GLiNER small biomedical-compatible model, or SapBERT embedding rerank on candidate shortlist.
   - Keep model use optional behind config flags.
   - Process in batches; cap candidate list; cache embeddings/candidates.
   - On `wenceslaus_full`, allow GPU-backed model experiments on the P100s, but do not require GPU for correctness.
5. **Mapping decoupling**
   - Do not let NER directly emit final KG edges. It should emit candidate tables and decisions so failures are auditable.

Potential optimization path:

- First optimize Python with Polars, SQLite indexes, multiprocessing, and streaming XML parsing.
- For `wenceslaus_full`, default to bounded parallelism rather than unbounded 80-thread fanout: make worker counts configurable per task to avoid I/O saturation on `/local_raid1` and memory spikes.
- Only after profiling, consider Rust/Go worker(s) for:
  - XML section extraction
  - dictionary phrase matching over millions of labels
  - FAERS ASCII parsing
- Keep worker boundary file-based or IPC-based with stable schemas so Python/Airflow remains orchestration layer.

### Phase 5: Tablassert modeling and NER/resolution

- Generate a Tablassert Graph config (`tables/graph.yaml`) plus one table config per assertion table.
- Fold Tablassert NER/resolution into this layer rather than duplicating it in DAKP:
  - use Tablassert's `level_one`/`level_two` normalization behavior for configured node columns;
  - use `resolve_batch`/`resolve_many` against fullmap for canonical CURIE/name/category/source metadata;
  - use Tablassert QC/fullmap audit outputs to identify unmapped or suspicious rows;
  - keep DAKP-specific NER limited to mention/span generation from unstructured DailyMed/FAERS text and hand Tablassert the text/candidate columns for final resolution.
- Based on current `../Tablassert`, use `source.kind: text` and `provenance.override` (`ManualProvenance`) for non-PMC sources unless the intended 8.0.0 `ManualSource` API appears.
- Avoid adding Tablassert to `pyproject.toml` initially; create a local wrapper that can run via:

```bash
uv run --with-editable ../Tablassert dakp-tablassert ...
```

or a config setting pointing to the local checkout.

Translator provenance conventions to preserve from `../DINGO`:

- `biolink:treats`: DAKP/Multiomics Drug Approvals is the primary/owning KP source, with DailyMed and FAERS upstream/supporting evidence.
- `biolink:applied_to_treat`: DAKP is an aggregator over FAERS primary observations plus DailyMed support; keep current FAERS labels/status behavior.
- `biolink:contraindicated_in`: DAKP aggregates MEDI primary assertions with DailyMed support.
- Assertion tables should include explicit source columns (`primary_knowledge_source`, `aggregator_knowledge_source`, `supporting_data_sources`, `upstream_resource_ids`, `source_record_urls`) so Tablassert can emit Translator-conventional provenance. If Tablassert needs a small upstream enhancement to express the exact DINGO source chain, make that change in `../Tablassert` rather than encoding a parallel KGX postprocessor in DAKP.

Example intended provenance override shape for a DAKP-owned treatment assertion:

```yaml
provenance:
  override:
    infores: infores:multiomics-drugapprovals
    upstream_resource_ids:
      - infores:dailymed
      - infores:faers
    knowledge_level: knowledge_assertion
    agent_type: manual_validation_of_automated_agent
```

### Phase 6: Translator-readiness compatibility

- Use `../DINGO/src/translator_ingest/ingests/dakp/dakp.py`, DAKP RIG, and unit tests as a contract/reference, not as a strict output comparator.
- Produce KGX node/edge JSONL gzip files with names compatible with the ingest:
  - `drug_approvals_kg_nodes.jsonl.gz`
  - `drug_approvals_kg_edges.jsonl.gz`
- Preserve required Translator semantics while allowing improved coverage and improved mappings:
  - expected predicates and association categories
  - source/provenance chain using DAKP, DailyMed, FAERS, and MEDI infores IDs
  - FAERS label/status behavior kept stable for first rebuild
  - FDA approval, SPL evidence, and case-count fields present where applicable
- Preserve or update DAKP RIG content from `../DINGO/src/translator_ingest/ingests/dakp/dakp_rig.yaml`.
- Add regression tests based on the association fixtures in `../DINGO/tests/unit/ingests/dakp/test_dakp.py`, plus new positive tests for improvements.

## Files to modify/create

Primary new files:

- `pyproject.toml`
- `README.md`
- `configs/local.example.yaml`
- `configs/pipeline.yaml`
- `src/dakp_pipeline/**`
- `src/dakp_pipeline/dags/dakp_build.py`
- `tables/*.yaml`
- `tests/**`

Legacy files to move/copy under `ref/legacy/` for audit only; the new DAG must never invoke them:

- `DailyMed/bin/getFullRelease.pl`
- `DailyMed/bin/parseXML-xtree.py`
- `FAERS/bin/getLatest.pl`
- `FAERS/bin/listCases.pl`
- `FAERS/bin/drug2indi.pl`
- `FAERS/bin/findIndicationTerms.py`
- `matrix/bin/studyContraindications.py`
- `matrix/bin/contraindications2kg.py`
- `bin/dakp-postprocess2jsonlBL.py`

## Reuse

- Reuse XML extraction logic from `DailyMed/bin/parseXML-xtree.py` but convert to typed, streaming/idempotent Python.
- Reuse FAERS table joining semantics from `FAERS/bin/listCases.pl` and `FAERS/bin/drug2indi.pl`.
- Reuse ignore term lists from `FAERS/bin/listCases.pl` and `FAERS/bin/drug2indi.pl`.
- Reuse DailyMed term finding concepts from `DailyMed/bin/findTermsInIndications.pl`, but replace ad hoc matching with a tested dictionary/candidate pipeline.
- Reuse deterministic UUID namespace behavior from `bin/dakp-postprocess2jsonlBL.py`.
- Reuse Translator source/provenance expectations from `../DINGO/src/translator_ingest/ingests/dakp/dakp.py` and `../DINGO/tests/unit/ingests/dakp/test_dakp.py`.
- Reuse Tablassert config concepts from `../Tablassert/docs/configuration/table.md` and `../Tablassert/src/tablassert/models.py` (`Text` source and `ManualProvenance`).
- Reuse Tablassert build architecture from `../Tablassert/src/tablassert/cli.py` (`build_pipeline`: load tables → extract sections → build TCode → collect instructions → build subgraphs → compile graph).
- Reuse Tablassert execution concepts from `../Tablassert/src/tablassert/lib.py`: content-hashed stores, node prep, `resolve_batch`, `fullmap_audit`, per-section parquet subgraphs, lazy graph compilation, Rust `dedup_ndjson`, and RIG generation.
- Reuse Nix store/hash concepts rather than inventing artifact semantics: deterministic path/tree hashing, immutable store paths, optional signing/verification, and content-addressed reuse — but use BLAKE3 as DAKP's primary content hash for speed.

## Audit-oriented implementation sketches

These code blocks are intentionally implementation-shaped so future development loops can audit the architecture before writing production code.

### Core source-shaping interfaces

```python
# src/dakp_pipeline/io/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Mapping, Any, Iterable

@dataclass(frozen=True)
class ArtifactRef:
    uri: Path
    blake3: str
    media_type: str
    rows: int | None = None
    schema_fingerprint: str | None = None
    manifest: Path | None = None

@dataclass(frozen=True)
class TaskContext:
    profile: str                 # mock | sample | wenceslaus_full
    workdir: Path
    fixture_root: Path | None
    threads: int
    memory_budget_gb: int
    params: Mapping[str, Any]

class Fetcher(Protocol):
    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]: ...

class Extractor(Protocol):
    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]: ...

class Transformer(Protocol):
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]: ...
```

### BLAKE3 artifact manifest shape

```json
{
  "schema_version": "dakp.artifact.v1",
  "artifact_id": "b3:4f8c...",
  "path": "data/interim/faers/quarter=24Q3/drug.parquet",
  "media_type": "application/vnd.apache.parquet",
  "hash": {
    "algorithm": "BLAKE3",
    "file": "b3:...",
    "tree": null
  },
  "inputs": ["b3:raw-faers-zip", "b3:extractor-config"],
  "operation": {
    "name": "extract_faers_drug",
    "version": "v1",
    "config_hash": "b3:..."
  },
  "source": {
    "url": "https://fis.fda.gov/content/Exports/faers_ascii_2024q3.zip",
    "etag": "...",
    "last_modified": "...",
    "retrieved_at": "2026-..."
  },
  "environment": {
    "git_commit": "...",
    "uv_lock_hash": "b3:...",
    "tablassert_commit": null,
    "fullmap_hash": "b3:..."
  },
  "table": {
    "rows": 123456,
    "partitions": 8,
    "schema_fingerprint": "b3:...",
    "warnings": 12
  }
}
```

### Airflow DAG skeleton

```python
# src/dakp_pipeline/dags/dakp_build.py
from __future__ import annotations

from airflow.decorators import dag, task
from pendulum import datetime

@dag(
    dag_id="dakp_build",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "profile": "mock",
        "fixture_root": "tests/fixtures/pipeline",
        "workdir": "data",
        "quarter_limit": 1,
        "threads": 8,
    },
)
def dakp_build():
    @task
    def acquire_sources(params=None):
        # returns manifest refs only; mocked tests monkeypatch fetchers
        ...

    @task
    def extract_dailymed(source_refs): ...

    @task
    def extract_faers(source_refs): ...

    @task
    def extract_drugsfda(source_refs): ...

    @task
    def extract_medi(source_refs): ...

    @task
    def shape_treatment_tables(extracted_refs): ...

    @task
    def shape_faers_use_tables(extracted_refs): ...

    @task
    def shape_contraindication_tables(extracted_refs): ...

    @task
    def generate_tablassert_configs(assertion_refs): ...

    @task
    def run_tablassert(assertion_refs, config_refs):
        # delegates mapping/NER-resolution/KGX/RIG/QC to ../Tablassert; no local KGX compiler
        ...

    @task
    def write_build_summary(kgx_refs):
        # writes local manifests/reports only; no publishing
        ...

    sources = acquire_sources()
    dm = extract_dailymed(sources)
    faers = extract_faers(sources)
    drugsfda = extract_drugsfda(sources)
    medi = extract_medi(sources)
    approved = shape_treatment_tables([dm, faers, drugsfda, medi])
    uses = shape_faers_use_tables([dm, faers, drugsfda, medi])
    contra = shape_contraindication_tables([dm, drugsfda, medi])
    configs = generate_tablassert_configs([approved, uses, contra])
    kgx = run_tablassert([approved, uses, contra], configs)
    write_build_summary(kgx)

dakp_build()
```

### Sharded DailyMed extraction sketch

```python
# src/dakp_pipeline/extract/spl_xml.py
from pathlib import Path
from collections.abc import Iterator

DAILYMED_SECTION_CODES = {
    "34067-9": "indications_and_usage",
    "34070-3": "contraindications",
    "34066-1": "boxed_warning",
    "42229-5": "warnings_and_precautions",
}

def iter_spl_documents(xml_shard: Path) -> Iterator[dict]:
    """Stream one shard of gzipped SPL XML files and emit document-level records."""
    ...

def extract_spl_sections(xml_shard: Path, out_dir: Path) -> list[Path]:
    """Write partitioned parquet tables: documents, sets, approvals, ingredients, sections."""
    # no global state; safe for Airflow dynamic task mapping
    # partition keys: release_id, xml_bin, section_code
    ...
```

### Sharded FAERS extraction sketch

```python
# src/dakp_pipeline/extract/faers_ascii.py
from pathlib import Path

FAERS_FILES = ("DEMO", "DRUG", "INDI", "REAC", "RPSR", "DELETE")

def parse_faers_file(path: Path, quarter: str, family: str, out_dir: Path) -> Path:
    """Parse one FAERS ASCII family for one quarter into parquet with source row IDs."""
    ...

def build_quarter_case_table(quarter_parts: dict[str, Path], out_dir: Path) -> Path:
    """Join parsed quarter tables locally before global aggregation."""
    ...

def reduce_faers_counts(case_tables: list[Path], out_dir: Path) -> Path:
    """Global reduce over quarter-level case tables to drug/indication count tables."""
    ...
```

### Tablassert handoff config sketch

Tablassert builds from a **Graph config** that references one or more table configs. DAKP generates both: a single `tables/graph.yaml` plus one table config per assertion table.

```yaml
# tables/graph.yaml
name: dakp
version: "0.6.0"
description: >-
  Drug Approvals Knowledge Provider: FDA-approved treatment relationships,
  FAERS-observed applied-to-treat uses, and contraindications, modeled from
  DailyMed, Drugs@FDA, FAERS, and MEDI.
infores: infores:multiomics-drugapprovals
fullmap: .fullmap
tables:
  - tables/approved_treats.yaml
  - tables/faers_applied_to_treat.yaml
  - tables/contraindications.yaml
```

```yaml
# tables/faers_applied_to_treat.yaml
source:
  kind: text
  local: data/tabular/faers_applied_to_treat_assertions.tsv
  url: https://example.invalid/dakp/generated/faers_applied_to_treat_assertions.tsv
  delimiter: "\t"
statement:
  subject:
    method: column
    encoding: A   # subject_text or mapped subject column, finalized during implementation
    prioritize: [Drug, SmallMolecule, ChemicalEntity]
  predicate: applied_to_treat
  object:
    method: column
    encoding: F
    prioritize: [Disease, PhenotypicFeature]
provenance:
  override:
    infores: infores:multiomics-drugapprovals
    upstream_resource_ids:
      - infores:faers
      - infores:dailymed
    knowledge_level: observation
    agent_type: manual_validation_of_automated_agent
annotations:
  - annotation: number_of_cases
    method: column
    encoding: K
  - annotation: clinical_approval_status
    method: column
    encoding: L
  - annotation: FDA_regulatory_approvals
    method: column
    encoding: M
```

### Monkeypatch-first full-pipeline test sketch

```python
# tests/integration/test_mock_pipeline.py
from pathlib import Path


def test_full_pipeline_uses_mocked_sources(monkeypatch, tmp_path):
    from dakp_pipeline.cli import run_pipeline
    from dakp_pipeline.sources import dailymed, faers, drugsfda, medi

    monkeypatch.setattr(dailymed, "fetch", lambda ctx: [ctx.fixture("dailymed_release.zip")])
    monkeypatch.setattr(faers, "fetch", lambda ctx: [ctx.fixture("faers_24q3.zip")])
    monkeypatch.setattr(drugsfda, "fetch", lambda ctx: [ctx.fixture("drugsfda.zip")])
    monkeypatch.setattr(medi, "fetch", lambda ctx: [ctx.fixture("medi.xlsx")])
    monkeypatch.setattr("dakp_pipeline.tablassert.run", fake_tablassert_run)  # no real Tablassert on dev laptop

    result = run_pipeline(
        profile="mock",
        fixture_root=Path("tests/fixtures/pipeline"),
        workdir=tmp_path,
        run_airflow=False,
    )

    assert result.table("approved_treats_assertions").rows > 0
    assert result.table("faers_applied_to_treat_assertions").rows > 0
    assert result.table("contraindication_assertions").rows > 0
    assert result.build_summary.exists()
```

## Implementation steps

### Milestone 1: Project skeleton, full mocked DAG, and fixtures

- [ ] Create a new git branch before implementation begins.
- [ ] Initialize `uv` project structure.
- [ ] Add `.pre-commit-config.yaml` modeled on `../Tablassert/.pre-commit-config.yaml`: ruff, ruff-format, pyright via `uv run`, pytest via `uv run`, and later Go fmt/test hooks.
- [ ] Add minimal package, CLI, config, and path helpers.
- [ ] Add Tablassert-compatible source-shaping primitives: load manifests, extract source sections, transform to assertion-ready tables, and generate Tablassert configs without reimplementing Tablassert graph-building internals.
- [ ] Add content-addressed artifact store helpers using BLAKE3 file/tree hashes, with optional secondary Nix/SRI metadata only for interoperability.
- [ ] Add the complete Airflow DAG skeleton early, even before real implementations are complete.
- [ ] Add tiny fixture files for DailyMed XML, FAERS ASCII tables, Drugs@FDA product rows, ontology rows, fullmap/Tablassert-like mapping outputs, and MEDI contraindication rows.
- [ ] Add monkeypatchable interfaces for downloaders, parsers, NER backends, fullmap/Tablassert mapping, KGX writing, and validation.
- [ ] Add smoke tests for config loading, CLI help, content-addressed cache hits, and a full mocked DAG/CLI pipeline run that works on the current development computer without real Tablassert/full build dependencies.
- [ ] Commit and push after each coherent step/milestone to preserve strict team provenance.

### Milestone 2: Source acquisition

- [ ] Implement idempotent DailyMed downloader with manifest/checksums and no destructive stashing.
- [ ] Implement Drugs@FDA downloader/extractor.
- [ ] Implement FAERS quarterly downloader/extractor with quarter discovery and `quarter_limit` dev mode.
- [ ] Implement ontology/source-term/fullmap acquisition or consume prebuilt local fullmap files with hash/version capture.
- [ ] Add optional Nix integration points for verification/prefetch metadata if useful, but keep BLAKE3 hashing as the primary artifact identity.
- [ ] Add unit tests using local fixture ZIPs and mocked HTTP.

### Milestone 3: Raw extraction to tables

- [ ] Port DailyMed SPL XML extraction to streaming Python producing document, set, approval, ingredient, section, LOINC, product, and evidence tables.
- [ ] Port FAERS ASCII extraction to partitioned Python/Polars producing raw DEMO/DRUG/INDI/REAC/RPSR/DELETE tables plus case-level joins and dedup/delete audit tables.
- [ ] Port Drugs@FDA product/application/submission extraction and normalize NDA/BLA/ANDA identifiers with and without leading zeroes.
- [ ] Port MEDI/matrix extraction while preserving sheet/row/version provenance and DailyMed support-score evidence.
- [ ] Add rejects/warnings tables and row-count/schema/content-hash validations for each extractor.

### Milestone 4: NER/candidate and fullmap mapping pipeline

- [ ] Build normalized dictionary indexes from MONDO/HPO/drug aliases.
- [ ] Port and improve lexical matching with deterministic scoring.
- [ ] Emit `mention_candidates.tsv` with all candidate evidence.
- [ ] Add canonical fullmap/Tablassert mapping step and a mocked mapping backend for tests.
- [ ] Add optional model reranking spike behind config.
- [ ] Benchmark on mock fixtures, laptop-safe samples, and `wenceslaus_full`; profile hotspots before choosing Rust/Go.

### Milestone 5: Assertion aggregation

- [ ] Build approved-treatment assertion table by joining FAERS NDA drug/indication counts with DailyMed approvals/SPL support.
- [ ] Build FAERS applied-to-treat assertion table with case counts while preserving current FAERS label/status behavior.
- [ ] Build contraindication assertion table with MEDI and DailyMed evidence.
- [ ] Make aggregation rules explicit and tested.

### Milestone 6: Airflow DAG

- [ ] Wrap each acquisition/extraction/NER/assertion task as an Airflow TaskFlow task.
- [ ] Ensure tasks pass manifest/path metadata only.
- [ ] Add local run instructions and task-level rerun behavior.
- [ ] Add DAG-level params for dev/full builds.

### Milestone 7: Tablassert integration

- [ ] Generate the Tablassert Graph config plus per-assertion table configs.
- [ ] Confirm actual Tablassert 8.0.0 class/API name for manual source/provenance.
- [ ] Add optional local editable Tablassert execution wrapper without adding dependency yet.
- [ ] Verify generated KGX matches DAKP translator ingest expectations.

### Milestone 8: Validation, performance, and release artifacts

- [ ] Add KGX validation: node coverage, required Biolink fields, source provenance, category/predicate compatibility.
- [ ] Add legacy-informed regression guardrails where available, without requiring edge-for-edge equality.
- [ ] Add performance benchmark reports for mock, sample/laptop-safe, and `wenceslaus_full` profiles.
- [ ] Add artifact publishing layout with manifest/version metadata.
- [ ] Update RIG content and examples.

## Verification

- Unit tests for each parser and aggregation rule with tiny fixtures.
- Full mocked end-to-end pipeline test using monkeypatched source acquisition, NER, mapping, and Tablassert/KGX modeling.
- Integration test for a mini end-to-end DAG run using local fixtures.
- Schema tests for all tabular outputs.
- KGX validation against Biolink/Translator expectations.
- Contract checks against current DAKP edge families and source semantics, allowing improved coverage/mappings:
  - `treats` = FDA-approved condition assertion.
  - `applied_to_treat` = FAERS-observed use without approval, preserving current label/status behavior for the first rebuild.
  - `contraindicated_in` = MEDI/DailyMed contraindication assertion.
- Performance tests in three profiles:
  - `mock`: seconds/minutes, CI/ralph-loop friendly.
  - `sample`: laptop-safe bounded sample.
  - `wenceslaus_full`: full real build on `/local_raid1` with configurable parallelism and memory checks.

## Resolved planning decisions

- First milestone: full pipeline skeleton with mocked inputs and monkeypatchable boundaries.
- Target full-build host: `wenceslaus` with 80 logical CPUs, 187 GiB RAM, `/local_raid1`, and optional P100 GPUs.
- Canonical mapping: Tablassert/fullmap, not BABEL.
- FAERS `applied_to_treat` labels/status: keep current behavior for first rebuild.
- Translator reference: `../DINGO`.
- Contraindications: first-scope.

## Remaining design choices for implementation phase

- Check the latest `../Tablassert` API at implementation time and use it directly; do not block this plan on a future renamed ManualSource/ManualProvenance detail.
- For this part of the DAG, only write local files, manifests, and reports. Do not design publication/deployment yet.

## Milestone 9: Edge-case testing round (100% coverage)

Added per user direction as the final phase, after Go integration + Airflow download tasks integrate.

Goal: drive the test suite to **100% coverage** (or document genuinely-unreachable lines with `# pragma: no cover`) via a dedicated edge-case testing round.

Approach:
1. Measure baseline: `uv run pytest --cov=dakp_pipeline --cov-report=term-missing`.
2. Add a coverage gate to `pyproject.toml` (`[tool.coverage.run]` source=dakp_pipeline, `[tool.coverage.report]` fail_under target).
3. Add comprehensive edge-case tests across every module (primarily NEW `tests/unit/test_*_edge.py` files; touch source only for genuine bugs):
   - **io**: artifact_store (cache hit/miss, collisions, tree-hash determinism), content_hash (empty file/dir, unicode, symlink skip), manifests (round-trip, missing/extra fields), contracts.
   - **sources**: fetcher error paths, HTTP retry/backoff, malformed/missing inputs, mock-vs-real selection.
   - **extract**: spl_xml malformed/partial XML + namespace variants; faers_ascii delimiter/`$`-trailing/CRLF/legacy `isr`/missing-column/encoding edge cases; drugsfda NDA/BLA/ANDA normalization corner cases (no prefix, all-zero, mixed).
   - **ner**: backends on empty/short/whitespace text, unknown backend selection error, lazy-import error when `[ner]` extra absent, model_cache idempotency + corrupt cache.
   - **assertions**: empty inputs, missing/extra columns, multi-NDA dedup, no-support edge cases, determinism.
   - **tablassert**: config generation edge cases (missing columns, empty tables), runner subprocess failure path.
   - **translator**: contract validation negative cases (missing node ref, bad category/predicate, missing provenance), RIG generation edge cases, regression guardrail failures.
   - **benchmarks/release**: empty build output, missing tables, report-shape determinism.
   - **pipeline/cli/dags**: unknown profile error, CLI arg validation, DAG task wiring without airflow.
   - **workers/go_runner**: Go-unavailable error, subprocess non-zero exit, stdout/stderr parsing.
4. Fan out as 2–3 parallel workers by module area (disjoint test files) to avoid collisions; integrate + run the full coverage gate.

Acceptance: `uv run pytest --cov` reports the target coverage with no missing branches (or only explicitly-pragma'd unreachable lines); full gate (ruff/format/pyright + Go) stays green.

## Developer DX: Makefile

A root `Makefile` (mirroring Tablassert's) provides routine command shortcuts so developers don't have to remember the exact `uv`/`go` invocations:

- **Install:** `make setup` (base+dev), `make install-ner` (heavy GLiNER + SciSpacy/spacy NER backends), `make install-airflow`, `make install-all`.
- **Python gate:** `make test`, `make cov`/`make coverage` (branch coverage, fail_under=100), `make lint`, `make lint-fix`, `make fmt`, `make fmt-check`, `make typecheck`, `make check` (full Python gate), `make pre-commit`.
- **Go gate:** `make build-go`, `make test-go`, `make vet-go`, `make fmt-go-check`, `make check-go` (full Go gate).
- **Combined:** `make check-all` (Python + Go).
- **Run:** `make run-mock` (mocked end-to-end pipeline).
- **Hygiene:** `make clean`.

`make help` lists all targets. The `install-ner` target installs the optional `[ner]` extra (gliner/scispacy/spacy/huggingface_hub) needed for the real SOTA contraindication NER backends; the base install and full test suite run without it.
