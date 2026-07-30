# Sources

Per-source notes for the three sources (DailyMed, Drugs@FDA, FAERS) plus the lexical disease
baseline. For each: where the fixture lives, what the fetcher and extractor do, the schema, and
the real-acquisition endpoint. Fetchers live in [`sources/`](../src/dakp_pipeline/sources/),
extractors in [`extract/`](../src/dakp_pipeline/extract/).

Real acquisition uses **stdlib HTTP** (no `requests`), is content-addressed and idempotent, and is
exercised offline in CI by [`tests/integration/test_prod_smoke.py`](../tests/integration/test_prod_smoke.py)
(HTTP layer mocked). The `mock` profile ingests fixtures instead of hitting the network.

## DailyMed (SPL)

Structured Product Labeling — FDA drug labels, the source of approved indications **and** the
contraindication text DAKP NER-mines.

| | |
| --- | --- |
| Fixture | [`tests/fixtures/pipeline/dailymed/dailymed_spl.xml.gz`](../tests/fixtures/pipeline/dailymed/dailymed_spl.xml.gz) |
| Fetcher | [`sources/dailymed.py`](../src/dakp_pipeline/sources/dailymed.py) `DailyMedFetcher` — full-release ZIPs (mock: `dailymed/dailymed_spl.xml.gz`) |
| Extractor | [`extract/spl_xml.py`](../src/dakp_pipeline/extract/spl_xml.py) `SPLXMLExtractor` (Go parity: [`go/internal/dailymed`](../go/internal/dailymed)) |

**Extraction.** `SPLXMLExtractor.extract` streams the gzip (`gzip` + `xml.etree`, constant memory
per document), auto-detects HL7 v3 vs the namespace-free mock shape, and emits five normalized
tables: `spl_documents`, `spl_sets`, `spl_approvals`, `spl_ingredients`, `spl_sections`.
Recognized LOINC section codes ([`SECTION_CODE_NAMES`](../src/dakp_pipeline/extract/spl_xml.py)):

| LOINC | Name | Used for |
| --- | --- | --- |
| `34067-9` | indications_and_usage | `treats` SPL support |
| `34070-3` | contraindications | `contraindicated_in` NER mining |
| `34066-1` | boxed_warning | (extracted, not first-scope) |
| `42229-5` | warnings_and_precautions | (extracted, not first-scope) |

**Real acquisition.** DailyMed full-release ZIPs, idempotent download with manifest/checksums,
sharded by release ZIP → inner ZIP/XML bin, `release_limit` bounds scope. Source:
<https://dailymed.nlm.nih.gov/dailymed/>.

## FAERS

FDA Adverse Event Reporting System quarterly ASCII extracts — the source of observed
(applied-to-treat) drug/indication use counts and the NDA-bearing drug-indication pairs that seed
`treats`.

| | |
| --- | --- |
| Fixtures | [`tests/fixtures/pipeline/faers/`](../tests/fixtures/pipeline/faers/) — `DEMO24Q3.txt`, `DRUG24Q3.txt`, `INDI24Q3.txt`, `REAC24Q3.txt` |
| Fetcher | [`sources/faers.py`](../src/dakp_pipeline/sources/faers.py) `FAERSFetcher` — quarterly ASCII ZIPs |
| Extractor | [`extract/faers_ascii.py`](../src/dakp_pipeline/extract/faers_ascii.py) `FAERSASCIIExtractor` (Go parity: [`go/internal/faers`](../go/internal/faers)) |

**Extraction.** Partitions inputs by family (`DEMO`/`DRUG`/`INDI`/`REAC`/`RPSR`/`DELETE`), parses
the `$`-delimited rows, joins **within the quarter** on `primaryid` (INDI×DRUG, left-joining DEMO
metadata + REAC reactions), honors DELETEd primaryids, then reduces across quarters with caseid
dedup (most-recent-wins). NDA numbers are digit-normalized so they join consistently with Drugs@FDA
`ApplNo`. Output: `data/interim/faers/cases.parquet`, schema `FAERS_CASES_COLUMNS` (`quarter`,
`primaryid`, `caseid`, `source`, `occp_cod`, `reporter_country`, `drugname`, `ingredient`, `nda`,
`indication`, `effects`).

**Real acquisition.** Quarterly FAERS ASCII ZIPs, quarter discovery with a `quarter_limit` bound;
per-quarter artifacts complete before cross-quarter aggregation. Source:
<https://fis.fda.gov/content/Exports/> (FAERS quarterly data files).

## Drugs@FDA

FDA application/product/action data — the source of NDA/BLA/ANDA approval identifiers and marketing
status; confirms an NDA is a real FDA application and yields the approved ingredient(s).

| | |
| --- | --- |
| Fixture | [`tests/fixtures/pipeline/drugsfda/drugsfda_products.tsv`](../tests/fixtures/pipeline/drugsfda/drugsfda_products.tsv) |
| Fetcher | [`sources/drugsfda.py`](../src/dakp_pipeline/sources/drugsfda.py) `DrugsFDAFetcher` — data-files ZIP |
| Extractor | [`extract/drugsfda_products.py`](../src/dakp_pipeline/extract/drugsfda_products.py) `DrugsFDAProductsExtractor` (Go parity: [`go/internal/drugsfda`](../go/internal/drugsfda)) |

**Extraction.** Normalizes the products/applications/submissions tables to canonical lowercase
columns, keeps the raw application number **and** both normalized forms (`appl_no` padded,
`appl_no_stripped` digit-only) for consistent joins with FAERS `nda`. Output:
`data/interim/drugsfda/products.parquet` (+ applications/submissions/lookups).

**Real acquisition.** Drugs@FDA data-files download. Source: <https://www.accessdata.fda.gov/scripts/cder/daf/>.

## Contraindications (NER-mined from DailyMed — not a fetched source)

Contraindication assertions are **mined directly** from DailyMed SPL "Contraindications" sections
(LOINC `34070-3`) using DAKP's single composite NER backend — there is no separate contraindication
source to acquire. This **replaces the legacy MEDI/Matrix xlsx** (`infores:medi`); see
[`semantic-equivalence.md`](./semantic-equivalence.md#improvements) for why this is better.

| | |
| --- | --- |
| Input | the DailyMed `spl_sections.parquet` contraindication sections + `spl_ingredients.parquet` active ingredients |
| Miner | [`assertions/contraindications.py`](../src/dakp_pipeline/assertions/contraindications.py) `build_contraindication_rows` |
| NER backend | [`ner/ner.py`](../src/dakp_pipeline/ner/ner.py) `extract_contraindication_diseases` — the single composite `DiseaseNER` (offline gazetteer / production gazetteer+GLiNER) |

**Mining.** For each SPL set with a contraindication section and ≥1 active ingredient, the NER
backend extracts disease/phenotype mentions from the section text; each mention is paired with each
active ingredient of that set to form a `biolink:contraindicated_in` assertion. The **object is the
mined mention TEXT** — `object_curie` / `object_name` / `object_category` are intentionally left
empty for Tablassert/fullmap to resolve at `build-kg` (DAKP does not map mentions to CURIEs). The
subject carries its text + UNII straight from the SPL source.

```text
section: "Contraindicated in patients with asthma or known hypersensitivity to ibuprofen."
  -> Ibuprofen (UNII:WK2XYI10QM) --contraindicated_in--> "asthma" (mention; fullmap resolves the CURIE)
```

**Provenance.** `primary_knowledge_source = infores:multiomics-drugapprovals`,
`upstream_resource_ids = infores:dailymed`, `agent_type = text_mining_agent`,
`knowledge_level = knowledge_assertion`.

## Lexical disease baseline (not a fetched source)

| | |
| --- | --- |
| Fixture | [`tests/fixtures/pipeline/ontology/disease_map.tsv`](../tests/fixtures/pipeline/ontology/disease_map.tsv) |
| Loader | [`pipeline._load_disease_map`](../src/dakp_pipeline/pipeline.py) → `ctx.params["disease_map"]` |
| Consumer | [`assertions.match_diseases`](../src/dakp_pipeline/assertions/__init__.py) |

A fast **exact-match dictionary baseline** (mention text → curie/name/category) used to populate
**object** CURIEs for the `treats` and `applied_to_treat` families where the mention is known.
**Canonical resolution is still Tablassert/fullmap's job**: contraindication objects are left empty
for fullmap, and subject CURIEs come from the source (DailyMed UNII), not from this dictionary. In
production the fullmap resolves everything; this baseline keeps the assertion tables evidence-rich
and the mock pipeline deterministic.

```text
text	curie	name	category
hypercholesterolemia	MONDO:0005154	hypercholesterolemia	Disease
headache	HP:0002315	headache	PhenotypicFeature
asthma	MONDO:0004979	asthma	Disease
```

## Media types

[`io/downloads.infer_media_type`](../src/dakp_pipeline/io/downloads.py) maps suffixes to IANA-ish
media types recorded in each artifact manifest (e.g. `.xml.gz` → `application/gzip`, `.parquet` →
`application/vnd.apache.parquet`, `.tsv` → `text/tab-separated-values`). `http_download(...)`
performs the real stdlib-HTTP fetch with manifest/checksum capture.

## Related

- [`tabular-contracts.md`](./tabular-contracts.md) — the exact column contracts these sources feed.
- [`architecture.md`](./architecture.md) — how acquisition/extraction fit the layered model.
- [`semantic-equivalence.md`](./semantic-equivalence.md) — why contraindications moved off MEDI.
- [`../README.md`](../README.md#how-to-add-a-new-source) — the "add a new source" checklist.
