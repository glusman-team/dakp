# Tabular contracts

The exact column contracts for every table DAKP emits. All column lists are declared in
[`src/dakp_pipeline/io/schemas.py`](../src/dakp_pipeline/io/schemas.py); a schema
**fingerprint** (`b3:<hex>` of the `"\t"`-joined column list, via
[`schema_fingerprint`](../src/dakp_pipeline/io/content_hash.py)) is recorded in each
artifact manifest so schema drift is detectable by hash.

Two output tiers (see [`architecture.md`](./architecture.md)):

- **Interim tables** — partitioned **parquet** under `data/interim/`, the internal
  extraction layer. Read with `pl.read_parquet`.
- **Assertion tables** — **uncompressed TSV** under `data/tabular/`, the Tablassert-facing
  contracts. Tablassert cannot read compressed inputs, so these are deliberately plain text
  with a header row (written by [`schemas.write_tsv`](../src/dakp_pipeline/io/schemas.py)).

> Example rows below are real output from a verified mock run
> (`uv run dakp run --profile mock --fixture-root tests/fixtures/pipeline --workdir /tmp/dakp-mock`).
> Empty string cells render as `""` (polars quotes empties). `|`-joined cells are
> Translator list-encoding conventions.

## Assertion tables (Tablassert-facing, uncompressed TSV)

### `data/tabular/approved_treats_assertions.tsv`

FDA-approved treatment assertions from DailyMed indications joined to Drugs@FDA approvals.
`predicate = biolink:treats`, `knowledge_level = knowledge_assertion`.

| Column | Example |
| --- | --- |
| `subject_text` | `Examplestatin` |
| `subject_curie` | `""` *(empty until fullmap resolution)* |
| `subject_name` | `""` *(empty until fullmap resolution)* |
| `subject_category` | `ChemicalEntity` |
| `predicate` | `biolink:treats` |
| `object_text` | `hypercholesterolemia` |
| `object_curie` | `MONDO:0005154` |
| `object_name` | `hypercholesterolemia` |
| `object_category` | `Disease` |
| `approval_ids` | `012345` *(DailyMed approval code + `NDA:<n>` from Drugs@FDA, `\|`-joined)* |
| `supporting_spl_sets` | `SETID-EXAMPLESTATIN-001` |
| `supporting_spl_documents` | `SETID-EXAMPLESTATIN-001#34067-9` |
| `clinical_approval_status` | `approved_for_condition` |
| `knowledge_level` | `knowledge_assertion` |
| `agent_type` | `manual_validation_of_automated_agent` |
| `primary_knowledge_source` | `infores:multiomics-drugapprovals` |
| `upstream_resource_ids` | `infores:dailymed\|infores:faers` |

> `subject_curie` / `subject_name` are empty in the scaffold: the lexical baseline maps
> only disease **objects**. Subject resolution is delegated to fullmap (Milestone 4+).

### `data/tabular/faers_applied_to_treat_assertions.tsv`

FAERS-observed drug/indication use, aggregated to case counts. `predicate =
biolink:applied_to_treat`, `knowledge_level = observation` (scaffold value
`statistical_association` pending the Milestone-5 label audit).

| Column | Example |
| --- | --- |
| `subject_text` | `Examplestatin` |
| `subject_curie` | `""` |
| `subject_name` | `""` |
| `subject_category` | `ChemicalEntity` |
| `predicate` | `biolink:applied_to_treat` |
| `object_text` | `hypercholesterolemia` |
| `object_curie` | `MONDO:0005154` |
| `object_name` | `hypercholesterolemia` |
| `object_category` | `Disease` |
| `case_count` | `1` |
| `clinical_approval_status` | `observed_use` *(FAERS label behavior kept stable for first rebuild)* |
| `knowledge_level` | `statistical_association` |
| `agent_type` | `manual_validation_of_automated_agent` |
| `primary_knowledge_source` | `infores:multiomics-drugapprovals` |
| `upstream_resource_ids` | `infores:faers\|infores:dailymed` |

### `data/tabular/contraindication_assertions.tsv`

Contraindication assertions text-mined from DailyMed SPL contraindication sections (LOINC
`34070-3`) via a configurable NER backend. `predicate = biolink:contraindicated_in`,
`knowledge_level = knowledge_assertion`, `agent_type = text_mining_agent`.

| Column | Example |
| --- | --- |
| `subject_text` | `Ibuprofen` |
| `subject_curie` | `UNII:WK2XYI10QM` |
| `subject_name` | `Ibuprofen` |
| `subject_category` | `ChemicalEntity` |
| `predicate` | `biolink:contraindicated_in` |
| `object_text` | `asthma` |
| `object_curie` | `MONDO:0004979` |
| `object_name` | `asthma` |
| `object_category` | `Disease` |
| `supporting_spl_sets` | `SETID-IBUPROFEN-002` |
| `supporting_spl_documents` | `SETID-IBUPROFEN-002#34070-3` |
| `source_score` | `1` *(max NER span score)* |
| `knowledge_level` | `knowledge_assertion` |
| `agent_type` | `text_mining_agent` |
| `primary_knowledge_source` | `infores:multiomics-drugapprovals` |
| `upstream_resource_ids` | `infores:dailymed` |

## Interim tables (parquet)

Internal extraction outputs under `data/interim/`. Not Tablassert-facing.

### `data/interim/dailymed/spl_documents.parquet`

Columns: `spl_document_id`, `spl_set_id`, `xml_path`, `release_file`, `approval_code`,
`approval_type`, `loinc_code`, `section_name`, `section_text`,
`active_ingredient_name`, `active_ingredient_unii`. One row per SPL section. See
[`sources.md`](./sources.md#dailymed-spl).

### `data/interim/faers/cases.parquet`

Columns: `quarter`, `primaryid`, `caseid`, `source`, `occp_cod`, `reporter_country`,
`drugname`, `ingredient`, `nda`, `indication`, `effects`. One row per drug record joined
to its case demographics + indications. See [`sources.md`](./sources.md#faers).

### `data/interim/drugsfda/products.parquet`

Columns: `appl_no`, `appl_type`, `product_no`, `drug_name`, `active_ingredient`,
`marketing_status_name`. `appl_no` digit-normalized for FAERS joins. See
[`sources.md`](./sources.md#drugsfda).

## Planned contracts (not yet emitted)

PLAN.md Phase 3 proposes additional public TSVs for the full build. They are **not**
emitted by the scaffold and have no committed column order yet:

- `data/tabular/faers_drug_indication_counts.tsv` — `case_count`, `nda`, `drug_name`,
  `ingredient_name`, `indication_text`, `source_quarters`. (Today the FAERS shaper folds
  counts directly into `faers_applied_to_treat_assertions` via `case_count`.)
- `data/tabular/mention_candidates.tsv` — source-aware mention spans with candidate CURIEs,
  ranks, and normalization notes. Lands with the NER/candidate pipeline (Milestone 4).

## Validation

The [`translator/contract.validate`](../src/dakp_pipeline/translator/contract.py)
readiness gate checks every table in `ASSERTION_TABLES` is present and readable, and that
no declared column is missing. Results are written into `build_summary.json`
(`translator_contract.ok`, per-table `missing_columns`). Full Biolink/Translator validation
(predicate/category compatibility, dangling-node detection) is largely delegated to
Tablassert's QC in Milestone 6+.

## Related

- [`tablassert-handoff.md`](./tablassert-handoff.md) — how these TSVs are consumed.
- [`sources.md`](./sources.md) — where each table's rows come from.
- [`runbook.md`](./runbook.md) — inspecting interim/TSV outputs when debugging.
