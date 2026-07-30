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
> (the one-command `make up-mock` mock pipeline).
> Empty string cells render as `""` (polars quotes empties). `|`-joined cells are
> Translator list-encoding conventions.

## Assertion tables (Tablassert-facing, uncompressed TSV)

### `data/tabular/approved_treats_assertions.tsv`

FDA-approved treatment assertions from DailyMed indications joined to Drugs@FDA approvals.
`predicate = biolink:treats`, `knowledge_level = knowledge_assertion`.

| Column | Example |
| --- | --- |
| `subject_text` | `Examplestatin` |
| `subject_curie` | `UNII:QFX8B1R4QF` *(DailyMed-provided UNII; fullmap refines the chemical category)* |
| `subject_name` | `Examplestatin` |
| `subject_category` | `ChemicalEntity` |
| `predicate` | `biolink:treats` |
| `object_text` | `hypercholesterolemia` |
| `object_curie` | `MONDO:0005154` |
| `object_name` | `hypercholesterolemia` |
| `object_category` | `Disease` |
| `approval_ids` | `012345` *(DailyMed approval display code, `\|`-joined; digit-normalized for the Drugs@FDA join)* |
| `supporting_spl_sets` | `SETID-EXAMPLESTATIN-001` |
| `supporting_spl_documents` | `SETID-EXAMPLESTATIN-001#34067-9` |
| `clinical_approval_status` | `approved_for_condition` |
| `knowledge_level` | `knowledge_assertion` |
| `agent_type` | `manual_validation_of_automated_agent` |
| `primary_knowledge_source` | `infores:multiomics-drugapprovals` |
| `upstream_resource_ids` | `infores:dailymed\|infores:faers` |

> `treats` subjects carry the DailyMed-provided UNII straight from the SPL source
> (source-provided, not DAKP-mapped); the lexical baseline populates disease **object** CURIEs.
> Canonical category refinement (e.g. SmallMolecule vs ChemicalEntity) is delegated to
> Tablassert/fullmap at `build-kg`.

### `data/tabular/faers_applied_to_treat_assertions.tsv`

FAERS-observed drug/indication use, aggregated to distinct-case counts. `predicate =
biolink:applied_to_treat`, `knowledge_level = statistical_association`,
`clinical_approval_status = observed_use` (the preserved FAERS label/status — see
[`semantic-equivalence.md`](./semantic-equivalence.md#deliberate-refinements). The Tablassert
config's family-level `provenance.override` stamps `knowledge_level: observation`, the DINGO
ingest value). FAERS subjects carry no source drug id, so `subject_curie` is empty for fullmap.

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
`34070-3`) via the single composite NER backend. `predicate = biolink:contraindicated_in`,
`knowledge_level = knowledge_assertion`, `agent_type = text_mining_agent`.

| Column | Example |
| --- | --- |
| `subject_text` | `Ibuprofen` |
| `subject_curie` | `UNII:WK2XYI10QM` |
| `subject_name` | `Ibuprofen` |
| `subject_category` | `ChemicalEntity` |
| `predicate` | `biolink:contraindicated_in` |
| `object_text` | `asthma` *(the mined mention text)* |
| `object_curie` | `""` *(empty by design — fullmap resolves the mention)* |
| `object_name` | `""` *(empty by design)* |
| `object_category` | `""` *(empty by design)* |
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

## Other contracts

- PLAN.md Phase 3 sketches a `data/tabular/faers_drug_indication_counts.tsv`; today the FAERS
  shaper folds those counts directly into `faers_applied_to_treat_assertions` via `case_count`, so
  no separate table is emitted.

## Validation

The [`translator/contract.py`](../src/dakp_pipeline/translator/contract.py) readiness gate has two
layers: `validate` checks every table in `ASSERTION_TABLES` is present and readable with no missing
declared column (results land in `build_summary.json`: `translator_contract.ok`, per-table
`missing_columns`); `validate_kgx` validates KGX node/edge records against the DAKP Translator
contract (node coverage, biolink-prefixed categories, the three edge families with chemical/drug
subjects + disease/phenotype objects, and the per-family infores provenance chain). The legacy
provenance/label invariants are re-checked on every build by
[`translator/regression.py`](../src/dakp_pipeline/translator/regression.py); Tablassert owns the
final full-graph QC.

## Related

- [`tablassert-handoff.md`](./tablassert-handoff.md) — how these TSVs are consumed.
- [`sources.md`](./sources.md) — where each table's rows come from.
- [`runbook.md`](./runbook.md) — inspecting interim/TSV outputs when debugging.
