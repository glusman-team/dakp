# Plan: Add approval and evidence provenance to every edge

## Context

The current assertion tables expose useful provenance unevenly:

- approved-treatment edges already carry `approval_ids` and DailyMed-backed evidence;
- FAERS observed-use/off-label edges carry case counts and upstream source labels, but no `approval_ids` or edge-level `has_evidence` identifiers;
- contraindication edges carry DailyMed evidence, but no `approval_ids`.

The goal is to make every emitted edge easier to debug by adding explicit approval and evidence provenance without changing assertion semantics or silently inventing evidence.

## Approach

Use one common, append-only `edge_evidence` TSV column as the sole backing column for final Biolink `has_evidence`. It will contain a sorted, deduplicated pipe-separated union of source identifiers and URLs. Keeping one backing column is required because the current Tablassert annotation loader silently overwrites duplicate `has_evidence` annotations.

Keep the existing source-specific debug columns (`supporting_spl_sets`, `supporting_spl_documents`, and `supporting_spl_evidence`) for backward compatibility, and add FAERS-specific debug columns where the edge can have FAERS support (`supporting_faers_records`, `supporting_faers_urls`). These columns make it possible to distinguish a stable report reference from the public quarterly FDA source URL while `edge_evidence` provides the final edge-level union.

The provenance values will be assembled as follows:

- **FAERS records:** use the contributing case rows' `quarter`, `primaryid`, and drug sequence to form a stable, URI-safe `faers:` evidence identifier; retain the exact normalized `source_record_id` in the debug column when available. Add the public FDA quarterly ZIP URL from the input manifest, with the deterministic FDA download URL as fallback.
- **FAERS approvals:** aggregate only the NDA values present on contributing FAERS rows. Normalize for joins, preserve the raw value, and use a DailyMed application display form when the same application is indexed there; never fabricate an application type when FAERS provides only a number.
- **DailyMed:** retain the existing `dailymed:<spl_set_id>` evidence CURIEs and label/document URLs. Reverse-index each supporting SPL set to its attached FDA application IDs so contraindication edges also receive `approval_ids` where the source actually provides them.
- **Approved-treat edges:** union the FAERS evidence/approval values from the contributing candidate rows with the DailyMed SPL evidence/approval values. DailyMed-only fallback candidates get only DailyMed provenance.
- **Observed-use edges:** aggregate FAERS evidence and report-level NDA values over the same `(drugname, indication)` pair used for `case_count`; retain the current approval-status comparison unchanged.
- **Contraindication edges:** aggregate DailyMed evidence and set-linked approval IDs over the existing `(ingredient, condition, context)` key; no FAERS values are added because FAERS does not produce these assertions.

All provenance is optional at the edge level. Blank TSV cells must be filtered before Tablassert annotation so absent provenance becomes an omitted KGX field, not an empty array.

## Files to modify

Core contracts and provenance:

- `src/dakp_pipeline/io/schemas.py` — append `approval_ids`, `edge_evidence`, and source-specific FAERS debug columns to the applicable assertion contracts.
- `src/dakp_pipeline/assertions/evidence.py` — shared deterministic encoding, FAERS source URL/record helpers, DailyMed set→approval reverse index, and source-manifest lookup.
- `src/dakp_pipeline/sources/faers.py` — reuse/export the canonical quarter→ZIP URL derivation if a helper is needed.
- `src/dakp_pipeline/sources/drugsfda.py` — reuse the official Drugs@FDA source URL for supporting provenance where the join contributes it.

Assertion shapers:

- `src/dakp_pipeline/assertions/approved_treats.py`
- `src/dakp_pipeline/assertions/observed_uses.py`
- `src/dakp_pipeline/assertions/contraindications.py`

Extraction/parity required for exact FAERS record provenance:

- `src/dakp_pipeline/extract/faers_ascii.py`
- `go/internal/faers/faers.go`
- `go/internal/airflow/faers_stream.go`
- corresponding Python/Go FAERS tests and golden contracts

Final KG/configuration:

- `src/dakp_pipeline/tablassert.py`
- `tables/approved_treats.yaml`
- `tables/faers_applied_to_treat.yaml`
- `tables/contraindications.yaml`

Tests/documentation:

- assertion unit tests under `tests/unit/test_assertions_*.py`
- `tests/unit/test_tablassert_configs.py`
- `tests/integration/test_semantic_equivalence.py`
- `tests/integration/test_kgx_end_to_end.py`
- `tests/integration/test_prod_smoke.py`
- `plans/all-edge-provenance.md` / relevant README or provenance documentation

## Reuse

- `sorted_pipe()` / `merge_unique()` and `spl_evidence_pipe()` in `src/dakp_pipeline/assertions/evidence.py` for deterministic list encoding.
- `DailyMedEvidence.approval_sets`, `approval_display`, and section indexes for SPL-to-approval joins; extend the index rather than rescanning source tables in each shaper.
- `find_faers_cases()` and the rich `cases.parquet` columns (`quarter`, `primaryid`, `nda`, `drug_seq`, `source_record_id`) for case-level provenance.
- `ArtifactRef.manifest`, `ArtifactStore.read_manifest()`, and `SourceBlock.url` for the actual downloaded FAERS quarter URL.
- `FDA_FAERS_DOWNLOAD_BASE` / `discover_quarters()` and `DRUGSFDA_DATA_FILES_URL` as deterministic URL fallbacks.
- Existing `approval_ids`, DailyMed evidence assembly, and Tablassert `split_by: "|"` mappings.
- Existing semantic-equivalence and KGX tests as regression harnesses for preserving predicates, status logic, counts, and source chains.

## Decisions

- FAERS `approval_ids` means the NDA(s) actually present on the contributing FAERS reports, not every application associated with the normalized drug/ingredient.
- FAERS `has_evidence` carries both per-report stable identifiers and public FDA report/source URLs.
- `has_evidence` is additive across contributing upstream sources; it should retain all directly contributing FAERS and DailyMed evidence rather than selecting only one source.
- When no provenance exists, the final KGX field is omitted rather than emitted as an empty array.
- Contraindication `approval_ids` uses the applications attached to each contributing SPL set for the selected ingredient; if a supporting SPL set has no attached application, the field is omitted for that edge rather than populated from an unrelated application.

## Steps

- [x] Resolve the high-level provenance semantics: FAERS-row NDAs, stable report IDs plus public source URLs, additive evidence, and omitted missing fields.
- [x] Trace the current source-to-edge joins and identify the gaps: observed-use projection drops NDA/quarter/sequence metadata; approved-treat FAERS candidates drop contributing report metadata; DailyMed evidence has no set→approval reverse index; only approved-treats/contraindications currently back `has_evidence`.
- [ ] Define and document URI-safe FAERS evidence-ID formatting and the exact quarterly FDA URL fallback; reject delimiter-unsafe values before pipe encoding.
- [ ] Extend FAERS rich-case output/parity so the Go streaming path preserves the fields needed for exact report provenance instead of blanking `source_record_id` and sequence metadata.
- [ ] Add shared provenance helpers for source manifests, quarter URLs, FAERS record IDs/URLs, DailyMed set URLs, and stable sorted unions.
- [ ] Extend `DailyMedEvidence` with a reverse `spl_set_id -> approval_ids` index and test duplicate/missing approval behavior.
- [ ] Append the common `approval_ids` and `edge_evidence` columns to every assertion contract; append FAERS debug columns to approved-treat and observed-use contracts while preserving all existing column meanings.
- [ ] Update approved-treat candidate aggregation to retain contributing FAERS NDA/report evidence and union it with the existing DailyMed approval/SPL evidence.
- [ ] Update observed-use aggregation to retain per-pair FAERS report IDs, URLs, and actual NDA values while preserving distinct-case counts and approval-status matching.
- [ ] Update contraindication aggregation to emit set-linked approval IDs and unified DailyMed edge evidence without changing NER, context, or singleton-ingredient semantics.
- [ ] Update Tablassert annotations so every table maps `approval_ids` and `edge_evidence` with `split_by: "|"`; verify blank cells are omitted and no duplicate `has_evidence` annotation is declared.
- [ ] Regenerate committed YAML configs and update config/schema expectations.
- [ ] Add focused unit tests, source/parity tests, semantic-equivalence assertions, and final KGX tests for all three families.

## Verification

- Run focused assertion and schema tests for all three shapers, including multi-NDA, multi-quarter, duplicate-primaryid, missing-`source_record_id`, missing-manifest, and no-approval cases.
- Run Python/Go FAERS parity tests and confirm the rich `cases.parquet` output retains the exact fields used to form evidence IDs.
- Run `tests/unit/test_tablassert_configs.py` and inspect generated annotations: exactly one `has_evidence` mapping per table, plus `approval_ids` on all tables, all with pipe splitting.
- Run `tests/integration/test_semantic_equivalence.py`; assert the existing predicates, categories, status logic, case counts, and upstream resource IDs remain unchanged.
- Run `tests/integration/test_kgx_end_to_end.py`; assert approved-treat, observed/off-label, and contraindication edges expose arrays for populated `approval_ids`/`has_evidence`, with stable IDs and URLs present where applicable, and omit absent fields.
- Run the bounded mock/production smoke pipeline and inspect the TSVs and final KGX edges for a known FAERS off-label pair, a DailyMed-approved pair, and a contraindication.
- Run the pipeline twice and compare assertion TSVs/KGX provenance arrays byte-for-byte; verify no approval or evidence value appears unless backed by the source row/SPL/application join.
