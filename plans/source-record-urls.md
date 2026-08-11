# Plan: Real source record URLs for Tablassert output

## Context

The Tablassert table configs currently hardcode a placeholder source URL:

- `src/dakp_pipeline/tablassert.py` sets `SOURCE_URL_BASE = "https://example.invalid/dakp/generated"`.
- `table_config()` writes `template.source.url` as `https://example.invalid/dakp/generated/<assertion-table>.tsv`.
- The committed configs in `tables/*.yaml` mirror that generator output.

Tablassert uses the section source URL as provenance for every edge produced by that section. With one section per assertion table, every edge gets the fake generated-table URL instead of the real upstream data source(s).

User decisions now locked in:

1. Use **raw downloadable source files** as source-record URLs.
2. When an assertion is backed by multiple sources, carry **all** URLs.
3. Multi-section table configs are acceptable.

Latest Tablassert check (PyPI/GitHub, `v8.2.0`):

- New modeling moved source record URLs under Biolink `sources` retrieval entries (`sources[*].source_record_urls`) instead of a flat edge field.
- `source.url` is still a single `HttpUrl` section constant; Tablassert does not yet support a per-row URL column or multiple URLs per section.
- `build-kg --fullmap` was removed in `8.1.0`; the fullmap path is read from graph YAML.
- `8.2.0` requires rebuilding old local fullmap/redb databases (`tablassert.fullmap.v5` / redb 4+), so do **not** blindly bump DAKP to latest until the local database is rebuilt and the runner command is version-gated.

Important consequence: duplicating one assertion row across one section per URL would not satisfy “all URLs”; Tablassert’s deduper only suppresses byte-identical records, so different source URLs would become duplicate edges, not one edge with a merged URL list. The clean solution is to prepare row-level URL sets in DAKP and use/adopt a small Tablassert modeling extension that lets a section carry multiple source URLs.

## Approach

Recommended approach: add row-level raw URL sets in DAKP, then generate multi-section Tablassert configs grouped by identical URL set. Use the latest Tablassert retrieval-source model once it supports multiple URLs per section; until then, keep a compatibility guard so DAKP never silently drops URLs.

Design details:

1. Add a final `source_record_urls` column to each DAKP assertion TSV. The value is a deterministic pipe-joined, sorted unique list of raw downloadable URLs.
   - Appending the column keeps existing subject/object/predicate column letters stable.
   - Tablassert does not need to read this as an annotation; DAKP’s config generator reads it to build sections.
2. Populate `source_record_urls` in assertion shapers:
   - FAERS observed-use rows: all FAERS quarter ZIP URLs represented by contributing case rows’ `quarter` values.
   - Approved-treats rows: all contributing DailyMed release ZIP URLs, FAERS quarter ZIP URLs, and the Drugs@FDA data-files ZIP URL used for NDA/product confirmation.
   - Contraindication rows: all contributing DailyMed release ZIP URLs for the mined SPL sections.
3. Preserve row-to-URL mapping through extraction/evidence:
   - FAERS already has `quarter`, and quarter → URL can be recovered from raw manifests or derived from `FDA_FAERS_DOWNLOAD_BASE`.
   - DailyMed needs an added interim provenance column (`source_url` or `source_artifact_id`) on SPL normalized tables so `spl_set_id`/`spl_document_id` support can map back to its release ZIP URL in multi-release runs.
   - Drugs@FDA URL can come from extraction input manifests, with `DRUGSFDA_DATA_FILES_URL` as the production default.
4. Update `tablassert.generate()` to read the assertion TSVs and group data rows by exact URL tuple.
   - Tablassert reads text files with `has_header=False`, so data row `0` in a normal TSV is `source.rows` index `1`; row `0` is the header.
   - Emit `template` for shared statement/provenance/annotation config and `sections` for source-specific overrides.
   - Each section selects its rows and carries the full URL set once Tablassert supports section-level URL lists.
5. Tablassert dependency strategy:
   - Do not rely on `8.2.0` immediately for production while the local fullmap DB is incompatible.
   - Plan/coordinate a Tablassert change such as `source.urls: list[HttpUrl]` (or `source.url: HttpUrl | list[HttpUrl]`) where retrieval provenance uses the whole list.
   - Once that is available and the local fullmap is rebuilt, update DAKP’s runner for `>=8.1` by omitting `--fullmap` and relying on `graph.yaml.fullmap`.
   - Before that support exists, fail real Tablassert generation/run if a row has more than one URL and the installed Tablassert can only carry one; do not degrade to a fake or partial URL.

## Files to modify

Critical Python paths:

- `src/dakp_pipeline/io/schemas.py`
- `src/dakp_pipeline/extract/spl_xml.py`
- `src/dakp_pipeline/assertions/evidence.py`
- `src/dakp_pipeline/assertions/observed_uses.py`
- `src/dakp_pipeline/assertions/approved_treats.py`
- `src/dakp_pipeline/assertions/contraindications.py`
- `src/dakp_pipeline/tablassert.py`
- `src/dakp_pipeline/translator.py` if contract/reporting should explicitly validate URL columns
- `pyproject.toml` / `uv.lock` only when the Tablassert/fullmap migration is ready

Critical Go parity paths if DailyMed interim schema changes:

- `go/internal/dailymed/dailymed.go`
- `go/internal/airflow/extract_dailymed.go`
- DailyMed golden/parity tests under `go/internal/dailymed/*` and `go/internal/airflow/*`

Config/test paths:

- `tables/approved_treats.yaml`
- `tables/faers_applied_to_treat.yaml`
- `tables/contraindications.yaml`
- `tests/unit/test_tablassert_configs.py`
- assertion shaper unit tests under `tests/unit/test_assertions_*.py`
- `tests/integration/test_prod_smoke.py`
- `tests/integration/harness.py` if fixture fetchers need real-style source manifests for URL assertions

## Reuse

Existing code and data to reuse:

- `SourceBlock.url` in artifact manifests (`src/dakp_pipeline/io/manifests.py`).
- `ArtifactStore.read_manifest()` / `manifest_path()` (`src/dakp_pipeline/io/artifact_store.py`) to resolve source URLs from artifact IDs.
- FAERS quarter discovery and URL construction in `src/dakp_pipeline/sources/faers.py`.
- FAERS `quarter` in the case table (`src/dakp_pipeline/extract/faers_ascii.py` and Go FAERS output).
- DailyMed `SourceBlock(url=<release zip>)` already attached when SPL XML members are ingested (`src/dakp_pipeline/sources/dailymed.py`).
- DailyMed evidence aggregation in `build_dailymed_evidence()` for set/document support.
- Drugs@FDA default URL constant in `src/dakp_pipeline/sources/drugsfda.py`.
- Tablassert’s `{template, sections}` config shape and `source.rows` filtering.

## Steps

- [ ] Add URL/provenance helpers in `assertions/evidence.py`:
  - [ ] parse pipe URL lists with stable dedupe/sort;
  - [ ] read direct and input-chain `SourceBlock.url` values from manifests;
  - [ ] map FAERS quarter labels to raw quarter ZIP URLs, with official URL derivation fallback;
  - [ ] map DailyMed SPL set/document support to release ZIP URLs;
  - [ ] collect Drugs@FDA raw URL(s), falling back to `DRUGSFDA_DATA_FILES_URL`.
- [ ] Extend DailyMed extraction to preserve per-row source URL (or source artifact ID) in SPL normalized tables; update Python and Go parity/goldens.
- [ ] Append `source_record_urls` to `APPROVED_TREATS_COLUMNS`, `FAERS_APPLIED_TO_TREAT_COLUMNS`, and `CONTRAINDICATION_COLUMNS` in `schemas.py`.
- [ ] Populate `source_record_urls` in all three assertion shapers while preserving deterministic row order.
- [ ] Refactor Tablassert config builders:
  - [ ] split shared template generation from source section generation;
  - [ ] group assertion TSV row numbers by URL tuple;
  - [ ] emit row indices with the header offset (`data row index + 1`);
  - [ ] remove `SOURCE_URL_BASE` and reject/fail if any runtime section would need a fake/empty URL.
- [ ] Coordinate/adopt Tablassert multi-URL section support (`source.urls`) so one section can attach all raw URLs to `sources[*].source_record_urls` without duplicating edges.
- [ ] Add Tablassert version handling in `TablassertRunner` before bumping dependency:
  - [ ] current/local 8.0.x path keeps `--fullmap`;
  - [ ] 8.1+ path omits `--fullmap` and relies on `graph.yaml.fullmap`;
  - [ ] document/rebuild old local fullmap DB before running latest.
- [ ] Update committed `tables/*.yaml` to remove `example.invalid`. Prefer runtime-generated URL sections as the source of truth; static configs can become template-only or use documented real official source pages until concrete row URLs exist.
- [ ] Update tests for schema, shaper URL values, generated section grouping, and absence of `example.invalid`.

## Verification

- Unit tests:
  - `pytest tests/unit/test_tablassert_configs.py`
  - assertion shaper tests for observed uses, approved treats, and contraindications
  - DailyMed extraction tests (Python and Go) for source URL propagation
  - source/acquisition tests around manifest `source.url`
- Integration smoke:
  - `pytest tests/integration/test_prod_smoke.py` should prove bounded real URLs are requested and then carried into assertion TSVs and generated Tablassert sections.
- Manual inspection after a bounded run:
  - inspect `workdir/data/tabular/*assertions.tsv` and confirm `source_record_urls` contains raw URLs;
  - inspect `workdir/tables/*.yaml` and confirm no `example.invalid` remains;
  - confirm FAERS rows refer to `https://fis.fda.gov/content/Exports/faers_ascii_<YYYY>q<N>.zip` (or the discovered equivalent);
  - when running Tablassert latest after fullmap rebuild, inspect KGX edges for structured `sources[*].source_record_urls` containing every URL in the row’s set.

## Decisions

- Source-record URL granularity: raw downloadable files.
- Multi-source assertions: carry all contributing raw URLs.
- Tablassert config shape: multi-section configs are acceptable, but sections must group by URL set, not by individual URL, to avoid duplicate edges.
