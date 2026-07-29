# Sources

Per-source notes for the four first-scope sources (DailyMed, Drugs@FDA, FAERS, MEDI) plus
the ontology dictionary baseline. For each: where the fixture lives, what the fetcher and
extractor do today, the schema, and the real-acquisition target. Fetchers live in
[`sources/`](../src/dakp_pipeline/sources/), extractors in [`extract/`](../src/dakp_pipeline/extract/).

> **Milestone-1 status.** Only the `mock` profile is implemented. Each fetcher calls
> `require_mock(ctx, ...)` and raises `NotImplementedError` for any other profile; real
> network acquisition is Milestone 2. Fixtures are tiny and deterministic so the whole
> pipeline runs with no network.

## DailyMed (SPL)

Structured Product Labeling — FDA drug labels, the source of approved indications and
contraindication text.

| | |
| --- | --- |
| Fixture | [`tests/fixtures/pipeline/dailymed/dailymed_spl.xml.gz`](../tests/fixtures/pipeline/dailymed/dailymed_spl.xml.gz) |
| Fetcher | [`sources/dailymed.py`](../src/dakp_pipeline/sources/dailymed.py) `DailyMedFetcher` — ingests `dailymed/dailymed_spl.xml.gz`, namespace `dailymed` |
| Extractor | [`extract/spl_xml.py`](../src/dakp_pipeline/extract/spl_xml.py) `SPLXMLExtractor` |

**Fixture shape.** A simplified, namespace-free analog of HL7 v3 SPL — a gzipped
`<splBatch>` of `<document>` elements, carrying exactly the fields DAKP needs:

```xml
<splBatch>
  <document>
    <setId>SETID-EXAMPLESTATIN-001</setId>
    <activeIngredient name="Examplestatin" unii="QFX8B1R4QF"/>
    <approval code="012345" type="NDA"/>
    <section loinc="34067-9" name="INDICATIONS AND USAGE">
      Examplestatin is indicated for the treatment of hypercholesterolemia ...
    </section>
  </document>
  ...
</splBatch>
```

**Extraction.** `SPLXMLExtractor.extract` streams the gzip (`gzip` + `xml.etree`), emits
one row **per section** into `data/interim/dailymed/spl_documents.parquet`, and registers
it with the store. `spl_document_id` is `<setId>#<loinc>` when a LOINC is present.
Whitespace in section text is collapsed. Recognized LOINC section codes
([`SECTION_CODE_NAMES`](../src/dakp_pipeline/extract/spl_xml.py)):

| LOINC | Name |
| --- | --- |
| `34067-9` | indications_and_usage |
| `34070-3` | contraindications |
| `34066-1` | boxed_warning |
| `42229-5` | warnings_and_precautions |

**Output schema** — `DAILYMED_SPL_DOCUMENTS_COLUMNS` (see
[`tabular-contracts.md`](./tabular-contracts.md)): `spl_document_id`, `spl_set_id`,
`xml_path`, `release_file`, `approval_code`, `approval_type`, `loinc_code`,
`section_name`, `section_text`, `active_ingredient_name`, `active_ingredient_unii`.

**Target real acquisition (Milestone 2–3).** DailyMed full-release ZIPs (legacy
`DailyMed/bin/getFullRelease.pl`), idempotent download with manifest/checksums and no
destructive stashing. Sharded by release ZIP → inner ZIP/XML bin. Streaming extraction into
document/set/approval/ingredient/section/LOINC/evidence tables, partitioned by release/bin,
retaining XML provenance and parse warnings. Source: <https://dailymed.nlm.nih.gov/dailymed/>.

## FAERS

FDA Adverse Event Reporting System quarterly ASCII extracts — the source of observed
(applied-to-treat) drug/indication use counts.

| | |
| --- | --- |
| Fixtures | [`tests/fixtures/pipeline/faers/`](../tests/fixtures/pipeline/faers/) — `DEMO24Q3.txt`, `DRUG24Q3.txt`, `INDI24Q3.txt` |
| Fetcher | [`sources/faers.py`](../src/dakp_pipeline/sources/faers.py) `FAERSFetcher` — ingests the three files, namespace `faers` |
| Extractor | [`extract/faers_ascii.py`](../src/dakp_pipeline/extract/faers_ascii.py) `FAERSASCIIExtractor` |

**Fixture shape.** Modern FAERS ASCII is `$`-delimited; the fixtures mirror that for one
quarter (`24Q3`):

```text
# DEMO24Q3.txt
primaryid$caseid$occp_cod$reporter_country
1001$5001$MD$US
# DRUG24Q3.txt
primaryid$drug_seq$drugname$role_cod$nda$ingredient
1001$1$Examplestatin$PS$012345$Examplestatin
# INDI24Q3.txt
primaryid$indi_drug_seq$indi_pt
1001$1$hypercholesterolemia
```

**Extraction.** `FAERSASCIIExtractor.extract` partitions inputs by family
(`DEMO`/`DRUG`/`INDI`/`REAC`/`RPSR`/`DELETE`), parses the `$`-delimited rows, and joins
**within the quarter** on `primaryid` (indications are indexed by `primaryid`; each drug
row pulls in its case demographics + `"; "`-joined indication text). NDA numbers are
digit-normalized so they join consistently with Drugs@FDA `ApplNo`. Output:
`data/interim/faers/cases.parquet`, schema `FAERS_CASES_COLUMNS`
(`quarter`, `primaryid`, `caseid`, `source`, `occp_cod`, `reporter_country`, `drugname`,
`ingredient`, `nda`, `indication`, `effects`).

**Target real acquisition (Milestone 2–3).** Quarterly FAERS ASCII ZIPs (legacy
`FAERS/bin/getLatest.pl`), quarter discovery with a `quarter_limit` dev mode. Per-quarter
`DEMO`/`DRUG`/`INDI`/`REAC`/`RPSR`/`DELETE` parquet tables plus case-level joins and
dedup/delete audit tables; aggregate across quarters only after per-quarter artifacts are
complete. Source: <https://fis.fda.gov/content/Exports/> (FAERS quarterly data files).

## Drugs@FDA

FDA application/product/action data — the source of NDA/BLA/ANDA approval identifiers and
marketing status.

| | |
| --- | --- |
| Fixture | [`tests/fixtures/pipeline/drugsfda/drugsfda_products.tsv`](../tests/fixtures/pipeline/drugsfda/drugsfda_products.tsv) |
| Fetcher | [`sources/drugsfda.py`](../src/dakp_pipeline/sources/drugsfda.py) `DrugsFDAFetcher` — ingests `drugsfda/drugsfda_products.tsv`, namespace `drugsfda` |
| Extractor | [`extract/drugsfda_products.py`](../src/dakp_pipeline/extract/drugsfda_products.py) `DrugsFDAProductsExtractor` |

**Fixture shape.** A header-cased TSV (the extractor normalizes mixed-case headers):

```text
ApplNo	ApplType	ProductNo	DrugName	ActiveIngredient	MarketingStatusName
012345	NDA	001	Examplestatin	Examplestatin	Prescription
017977	NDA	001	Ibuprofen	Ibuprofen	Prescription
```

**Extraction.** `DrugsFDAProductsExtractor.extract` reads the TSV, renames columns to
canonical lowercase (`ApplNo` → `appl_no`, etc.), selects the product columns, and
digit-normalizes `appl_no` for consistent joins with FAERS `nda`. Output:
`data/interim/drugsfda/products.parquet`, columns `appl_no`, `appl_type`, `product_no`,
`drug_name`, `active_ingredient`, `marketing_status_name`.

**Target real acquisition (Milestone 2–3).** Drugs@FDA download (legacy
`DrugsFDA/bin/download.pl`), normalized product/application/submission tables preserving
NDA/BLA/ANDA variants with and without leading zeroes, plus lookup tables for proprietary
names, ingredients, application numbers, marketing status, and product NDCs. Source:
<https://www.accessdata.fda.gov/scripts/cder/daf/>.

## MEDI

The MEDI / matrix-indication contraindication list — the primary source of
contraindication assertions.

| | |
| --- | --- |
| Fixture | [`tests/fixtures/pipeline/medi/medi_contraindications.tsv`](../tests/fixtures/pipeline/medi/medi_contraindications.tsv) |
| Fetcher | [`sources/medi.py`](../src/dakp_pipeline/sources/medi.py) `MEDIFetcher` — ingests `medi/medi_contraindications.tsv`, namespace `medi` |
| Extractor | [`pipeline._extract_medi`](../src/dakp_pipeline/pipeline.py) (inline passthrough; no `extract/medi` module in Milestone 1) |

**Fixture shape:**

```text
drug_name	contraindicated_condition	source_score
Ibuprofen	asthma	0.9
Ibuprofen	peptic ulcer disease	0.8
```

**Extraction.** The MEDI fixture is already a clean table, so Milestone 1 normalizes it to
`data/interim/medi/contraindications.parquet` with a uniform manifest
(`pipeline._extract_medi`). The [`contraindications`](../src/dakp_pipeline/assertions/contraindications.py)
shaper reads `contraindicated_condition` (falling back to `contraindicated_disease`),
preserves `source_score`, and tags `medi_version = "MEDI-0.x-mock"`.

**Target real acquisition (Milestone 2–3).** The MEDI `contraindicationList.xlsx`, with
sheet/row provenance and DailyMed support-scoring evidence (legacy
`matrix/bin/studyContraindications.py`). Source:
<https://github.com/everycure-org/matrix-indication-list/releases/>.

## Ontology dictionary baseline (not a fetcher source)

| | |
| --- | --- |
| Fixture | [`tests/fixtures/pipeline/ontology/disease_map.tsv`](../tests/fixtures/pipeline/ontology/disease_map.tsv) |
| Loader | [`pipeline._load_disease_map`](../src/dakp_pipeline/pipeline.py) → `ctx.params["disease_map"]` |
| Consumer | [`assertions.match_diseases`](../src/dakp_pipeline/assertions/__init__.py) |

This is the fast **exact-match dictionary baseline** for disease objects, not a fetched
source with its own fetcher. It maps mention text → curie/name/category:

```text
text	curie	name	category
hypercholesterolemia	MONDO:0005154	hypercholesterolemia	Disease
headache	HP:0002315	headache	PhenotypicFeature
asthma	MONDO:0004979	asthma	Disease
```

Per PLAN.md, DAKP does **not** build custom NER indexes in first-scope; Tablassert/fullmap
owns canonical resolution. This fixture exists only to populate object CURIEs during the
scaffold stage; fullmap replaces it in Milestone 4. Subject CURIEs/name are left empty
today (subjects are not dictionary-mapped) and are resolved by fullmap during modeling.

## Media types

[`io/downloads.infer_media_type`](../src/dakp_pipeline/io/downloads.py) maps suffixes to
IANA-ish media types recorded in each artifact manifest (e.g. `.xml.gz` →
`application/gzip`, `.parquet` → `application/vnd.apache.parquet`, `.tsv` →
`text/tab-separated-values`). `http_download(...)` is a Milestone-1 stub that raises
`NotImplementedError`; real acquisition lands in Milestone 2.

## Related

- [`tabular-contracts.md`](./tabular-contracts.md) — the exact column contracts these sources feed.
- [`architecture.md`](./architecture.md) — how acquisition/extraction fit the layered model.
- [`README.md`](../README.md#how-to-add-a-new-source) — the "add a new source" checklist.
