# EMA centrally-authorised medicines registry as an approved-treats source

## Context

DAKP's approved-treats table has been FDA-only (DailyMed SPL approvals + Drugs@FDA + FAERS
candidates). The first non-US source is the **EMA "Medicines" report** — the bulk export of
every centrally reviewed medicine, a fixed-name xlsx regenerated nightly from the EMA website:

```
https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx
```

Verified against the live export (2026-08-13, ~900 KB, ~2 700 rows x 39 columns):

- The first rows are **banner lines** ("Content type:" / "Medicine", "Output automatically
  generated from content on www.ema.europa.eu on: <date>"); the real header row follows them,
  so the parser locates the row carrying "Name of medicine" + "Medicine status" instead of
  assuming row 0.
- Columns used: `Category` ("Human"/"Veterinary"), `Name of medicine`, `EMA product number`
  (e.g. `EMEA/H/C/006395`), `Medicine status` ("Authorised", "Withdrawn", "Refused",
  "Suspended", ...), `International non-proprietary name (INN) / common name`,
  `Active substance` (semicolon-joined for combos), `Therapeutic area (MeSH)` (semicolon-joined
  MeSH terms; human-only — veterinary rows leave it empty), `Therapeutic indication` (full free
  text), and `Medicine URL` (the EPAR page).

## Locked decisions

- **Filter:** only `Medicine status == "Authorised"` AND `Category == "Human"` rows produce
  edges. Every other status is dropped entirely (no annotated edges); veterinary rows carry no
  usable object (the MeSH column is human-only). Applied at **extract** time.
- **Edges:** one row per `(active substance, MeSH therapeutic-area term)` pair — split
  `Active substance` (falling back to the INN column when empty) and `Therapeutic area (MeSH)`
  on `;`. `clinical_approval_status = approved_for_condition`; `approval_ids` = the EMA product
  number; `supporting_spl_documents` = the EPAR medicine URL; KL/AT and
  `primary_knowledge_source` reuse the existing constants.
- **Provenance:** new constant `INFORES_EMA = "infores:ema"`. EMA-derived rows get
  `upstream_resource_ids = "infores:ema"`; FDA-derived rows keep `infores:dailymed|infores:faers`.
  Per-source rows are NOT merged across sources — the union is one row per source per
  `(subject, object)` pair, sorted deterministically. The treats regression invariant accepts
  either upstream chain (`FamilyInvariant.alternative_upstream`).
- **Parsing:** polars `pl.read_excel` (backed by the new `fastexcel` dependency). Bulk download
  only; no URL scraping.
- **Freshness:** the same seven-day cache-window gate as Drugs@FDA, keyed on
  `ema_max_age_days` (default 7; `force` bypasses; `<= 0` disables) — the export is a
  fixed-name, replace-in-place artifact.

## Files

- `src/dakp_pipeline/sources/ema.py` — fetcher mirroring `sources/drugsfda.py` (BLAKE3 ingest,
  alias `ema/medicines.xlsx`, monkeypatchable `download_ema_table`).
- `src/dakp_pipeline/extract/ema_registry.py` — xlsx -> interim parquet
  `data/interim/ema/ema_registry.parquet`, columns: `medicine_name`, `ema_product_number`,
  `category`, `medicine_status`, `inn`, `active_substance`, `therapeutic_area_mesh`,
  `therapeutic_indication`, `medicine_url` (all UTF-8 strings, sorted by product number).
  `therapeutic_indication` is kept verbatim for Phase 2.
- `src/dakp_pipeline/assertions/approved_treats.py` — `build_ema_treats_rows` (Phase 1,
  `infores:ema`) + `build_epar_treats_rows` (Phase 2, `infores:epar`) unioned in
  `build_approved_treats_rows(..., ema_registry=..., ner=...)` (FDA logic untouched; `ner` is
  optional so direct callers keep Phase 1 behavior).
- `src/dakp_pipeline/dags/dakp_build.py` — `acquire_ema` (download pool) and `extract_ema`
  (a plain Python `@task` in the extract TaskGroup, `EMA_EXTRACT_POOL_SLOTS = 1`; the other
  extracts are Go stubs). `extract_ema` output is an additional input to
  `shape_treatment_tables`.
- `src/dakp_pipeline/tablassert.py` — approved-treats provenance override gains
  `infores:ema`; `source.url` lists both the DailyMed index and the EMA xlsx URL.
  `tables/approved_treats.yaml` + `tables/graph.yaml` regenerated from the generator.
- `src/dakp_pipeline/translator.py` — treats family invariant accepts the EMA upstream chain.
- Tests + fixture: `tests/fixtures/pipeline/ema/medicines-output-medicines-report_en.xlsx`
  (trimmed real export, banner rows included), `tests/unit/test_ema_source.py`,
  `tests/unit/test_ema_extract.py`, `tests/unit/test_assertions_approved_treats_ema.py`.

## Phase 2 (done)

The `therapeutic_indication` free-text column (kept in the Phase 1 interim parquet) is mined
with the composite DiseaseNER, adding **EPAR indication rows** to the same
approved-treats table:

- one row per `(active substance, mined disease/phenotype mention)` — same `;`-split + INN
  fallback subject fan-out as Phase 1; the object is the normalized mention text, so mined rows
  key separately from the MeSH-area rows;
- provenance: `upstream_resource_ids = "infores:epar"` (new `INFORES_EPAR` constant);
  `approval_ids` = EMA product number, `supporting_spl_documents` = EPAR medicine URL;
  `clinical_approval_status = approved_for_condition`; KL/AT/DAKP constants as Phase 1;
- mining runs through `build_epar_treats_rows` inside `build_approved_treats_rows` — the table
  keeps a single writer. The shaper resolves the backend from `ctx.params["ner"]` (injected by
  the DAG / tests) else the offline gazetteer `default_ner`, and only constructs it when the
  EMA registry is among the inputs, so FDA-only runs never touch the NER;
- DAG wiring: `shape_treatment_tables` gained the `acquire_ner_models` ordering dependency (the
  same pattern as `shape_contraindication_tables`) and injects the production
  `DiseaseNER(offline=False)` only when EMA refs are present — deliberately NOT a separate
  `shape_ema_indication_tables` task, which would have created a second writer of the same TSV;
- provenance contracts: the treats family's `alternative_upstream` (both the row-level
  regression `FamilyInvariant` and the KGX-level `EdgeFamily`) accepts `infores:ema` and
  `infores:epar`; the Tablassert override for approved-treats stamps
  `infores:dailymed|infores:faers|infores:ema|infores:epar`.

Known follow-up: `tests/integration/harness.py` still wires only the FDA sources (its
semantic-equivalence assertions hard-code the FDA-only treats chain and SPL evidence on every
treats row). Wiring `ema.fetch` / `ema_registry.extract` into the harness means relaxing those
assertions to per-source chains first — deliberately deferred.

## Verification

- `uv run pytest -q --cov` (100% branch gate), `uv run ruff check`, `uv run ruff format --check`,
  `uv run pyright`.
