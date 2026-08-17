# Standardize DAKP edge output toward the legacy DAKP shape

## Context

DAKP's KGX compilation now runs through Tablassert 12, and the emitted edges have drifted from
the legacy DAKP KG shape (the two reference edges below — one `biolink:treats`, one
`biolink:applied_to_treat`). This plan is a **menu of individually approvable standardization
changes**: approve or deny each numbered proposal; implementation proceeds with the approved set.

Reference (legacy) treats edge:

```json
{"id": "8ccfd259-…", "subject": "CHEBI:4875", "predicate": "biolink:treats", "object": "MONDO:0008383",
 "subject_name": "Etanercept", "object_name": "rheumatoid arthritis",
 "category": ["biolink:EntityToDiseaseAssociation"],
 "knowledge_level": "knowledge_assertion", "agent_type": "manual_validation_of_automated_agent",
 "sources": [
   {"resource_id": "infores:multiomics-drugapprovals", "resource_role": "primary_knowledge_source",
    "upstream_resource_ids": ["infores:dailymed", "infores:faers"],
    "source_record_urls": ["https://db.systemsbiology.net/gestalt/cgi-pub/KGinfo.pl?id=<edge id>"]},
   {"resource_id": "infores:faers", "resource_role": "supporting_data_source"},
   {"resource_id": "infores:dailymed", "resource_role": "supporting_data_source"}],
 "approvals": ["BLA103795"],
 "has_evidence": ["dailymed:2dc2ed4b-…", "dailymed:74bead6e-…"],
 "clinical_approval_status": "approved_for_condition"}
```

Reference (legacy) applied_to_treat edge:

```json
{"id": "408826a1-…", "subject": "CHEBI:4875", "predicate": "biolink:applied_to_treat", "object": "MONDO:0008383",
 "subject_name": "Etanercept", "object_name": "rheumatoid arthritis",
 "category": ["biolink:EntityToDiseaseAssociation"],
 "knowledge_level": "observation", "agent_type": "manual_validation_of_automated_agent",
 "N_cases": 269572,
 "sources": [
   {"resource_id": "infores:multiomics-drugapprovals", "resource_role": "aggregator_knowledge_source",
    "upstream_resource_ids": ["infores:dailymed", "infores:faers"],
    "source_record_urls": ["https://db.systemsbiology.net/gestalt/cgi-pub/KGinfo.pl?id=<edge id>"]},
   {"resource_id": "infores:faers", "resource_role": "primary_knowledge_source"},
   {"resource_id": "infores:dailymed", "resource_role": "supporting_data_source"}],
 "clinical_approval_status": "approved_for_condition"}
```

Current output (verified live: Tablassert 12 e2e run over the fixture pipeline,
`/tmp/pytest-of-skyeav/pytest-2220/…/dakp_0.1.0.edges.ndjson`):

```json
{"approval_ids": "017977",
 "has_evidence": ["https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SETID-IBUPROFEN-002",
                   "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SETID-IBUPROFEN-002#34067-9"],
 "clinical_approval_status": "approved_for_condition",
 "subject": "CHEBI:5855", "original_subject": "Ibuprofen",
 "object": "HP:0002315", "original_object": "headache",
 "predicate": "biolink:treats",
 "category": ["biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation"],
 "knowledge_level": "knowledge_assertion", "agent_type": "manual_validation_of_automated_agent",
 "sources": [
   {"id": "infores:multiomics-drugapprovals", "resource_id": "infores:multiomics-drugapprovals",
    "resource_role": "primary_knowledge_source",
    "upstream_resource_ids": ["infores:dailymed", "infores:faers"],
    "source_record_urls": ["https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"]},
   {"id": "infores:dailymed", "resource_role": "supporting_data_source"},
   {"id": "infores:faers", "resource_role": "supporting_data_source"}],
 "has_supporting_studies": {"approved_treats.yaml": {"id": "…", "name": "…", "has_study_results": [{"id": "…#row3", "name": "row 3"}]}},
 "supporting_text": ["object_nlp_level: 1", "subject_nlp_level: 1"],
 "id": "8084df46-…"}
```

(applied_to_treat edges carry `evidence_count: "1"` instead of `approval_ids`/`has_evidence`;
contraindicated_in edges carry `has_evidence` + `supporting_text` incl. `source_score`.)

## What already matches the legacy shape (no action)

- `id`, `subject`, `predicate`, `object` — present, resolved CURIEs, deterministic ids.
- `knowledge_level` per family — treats/contraindicated `knowledge_assertion`, applied `observation`. ✅
- `agent_type` — `manual_validation_of_automated_agent` (treats/applied), `text_mining_agent` (contraindicated). ✅
- `clinical_approval_status` — first-class on treats/applied edges with valid `ClinicalApprovalStatusEnum` values. ✅
- Structured `sources[]` with the DAKP infores, per-family upstream chains (`dailymed|faers` / `faers|dailymed` / `dailymed`) and `source_record_urls`. ✅
- Family scoping of evidence fields — approvals+has_evidence only on treats; case count only on applied. ✅

## Decision record (review feedback, round 1)

| # | Original proposal | Decision |
|---|---|---|
| 1 | Edge category → legacy `EntityToDisease/PhenotypicFeatureAssociation` | ❌ **Denied** — "don't emit"; keep `ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation` |
| 2 | Approvals field | 🔁 **Rescoped** — keep the `approval_ids` name; emit as a real JSON **list** via `split_by` (verified — see Change A) |
| 3 | Case count → `number_of_cases`/`N_cases` | ⛔ Ignored — keep `evidence_count` |
| 4 | `has_evidence` → `dailymed:<spl_set_id>` CURIEs | ✅ **Approved** — "do this" (Change B) |
| 5 | applied sources roles (DAKP aggregator / FAERS primary) | ⛔ Ignored — keep DAKP primary for all families |
| 6 | `subject_name` / `object_name` | ⛔ Ignored — keep `original_subject` / `original_object` |
| 7 | Legacy gestalt per-edge `source_record_urls` | ⛔ Ignored — keep dataset-level URLs |
| 8 | Legacy uuid3 id parity | ⛔ Ignored — keep Tablassert deterministic ids |
| 9 | Strip Tablassert-only extras | ⛔ Ignored — keep `has_supporting_studies` / `supporting_text` extras |

**Resulting scope: two changes (A and B below), both pure DAKP-side — no Tablassert changes,
no DINGO changes.** Consequence of denying proposal 1: the edge category stays
`ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation`, so `FDA_regulatory_approvals` /
`number_of_cases` could not ride the edge anyway (they'd be rescued into the study description);
the approvals field therefore keeps its curated `approval_ids` name.

---

## Change A — Emit `approval_ids` as a real JSON array (rescoped proposal 2)

Current: `"approval_ids": "012345|017977"` — a pipe-joined **scalar** (Tablassert 12's curated
pass-through emits it verbatim, no split). Target: `"approval_ids": ["012345", "017977"]` — a
real JSON array, like the legacy `approvals` list.

### Verified empirically on stock Tablassert 12.0.0 (no repo changes)

On a **copy** of the e2e build (`/tmp/approval-split-exp/work`) the `approval_ids` annotation
gained `split_by: "|"` and one TSV row got two ids; `tablassert build-kg` was re-run:

- two-value row → `"approval_ids": ["012345", "017977"]`; single-value row → `["017977"]` —
  real JSON arrays in both cases (no wrapped `"a|b"` blob);
- build exits 0; nothing else on the edge changed; DAKP `validate_kgx` passes with 0 problems.

So Tablassert **does** allow it — `split_by` (`lib.split_list`) is a generic per-cell list
encoding; nothing in the pipeline special-cases `approval_ids` against it.

Note: Tablassert's authoring guidance suggests a scalar `approval_ids`, but nothing enforces it
(verified above) — no action needed on that; the list shape is what we want anyway.

### Sub-item A2 — application-type prefix on the values (optional parity polish)

Legacy values carry the FDA application type (`["BLA103795"]`); current values are the bare
numbers (`approval_display` prefers the raw `approval_id`). `spl_approvals.parquet` also carries
`approval_code` (NDA/BLA/ANDA), so `build_dailymed_evidence` could store
`f"{approval_code}{approval_id}"` (falling back to whichever part exists).

- [x] **A2 adopt** (recommended for parity) / [ ] **A2 deny** (keep bare numbers)

### Changes

- [x] `src/dakp_pipeline/tablassert.py` — `_TABLE_ANNOTATIONS["approved_treats_assertions"]`:
      `("approval_ids", "approval_ids", None)` → `("approval_ids", "approval_ids", "|")`.
- [x] (A2) `src/dakp_pipeline/assertions/evidence.py::build_dailymed_evidence` —
      `approval_display` values become `<approval_code><approval_id>` when both are present.
- [x] `tables/approved_treats.yaml` — regenerate (the annotation gains `split_by: "|"`).
- [x] Tests:
  - `tests/unit/test_tablassert_configs.py` — `EXPECTED_ANNOTATIONS["approval_ids"]`
        `("approval_ids", None)` → `("approval_ids", "|")`;
  - `tests/integration/test_kgx_end_to_end.py::test_approval_ids_ride_the_edge_as_a_top_level_field`
        — assert `list[str]`, non-empty, elements free of `"|"` (rename the test/docstring to the
        array contract);
  - (if A2) evidence/approved-treats unit tests that pin display values.

## Change B — Emit `has_evidence` values as `dailymed:<spl_set_id>` CURIEs (approved proposal 4, option 4a)

Current: DailyMed drugInfo **URLs** at two granularities — the SPL-set label URL plus the
section-scoped `…#<loinc>` URL. Target (legacy parity): sorted, deduped
**`dailymed:<spl_set_id>` CURIEs, set granularity only** — e.g.
`["dailymed:2dc2ed4b-…", "dailymed:74bead6e-…"]`. DINGO extends `publications` with
`has_evidence` — CURIEs fit publication semantics better than URLs.

### Changes

- [x] `src/dakp_pipeline/assertions/evidence.py`:
  - add `DAILYMED_SET_CURIE_PREFIX = "dailymed:"` + `dailymed_set_curie(value)` helper (strip;
        accept a bare set id or `<set>#<loinc>` and keep only the set part; idempotent);
  - `spl_evidence_pipe(sets, documents)` → `sorted_pipe` of set CURIEs — sets directly, documents
        reduced to their set id (keeps the function total for rows that pass documents only);
  - keep `dailymed_set_url` / `dailymed_document_url` for the two UN-annotated debug columns
        (`supporting_spl_sets` / `supporting_spl_documents` stay human-readable URLs; Tablassert
        never reads those columns, they never reach the edge).
- [x] Docstrings/comments: `spl_evidence_pipe` itself; the `has_evidence` bullet in
      `tablassert.py::_TABLE_ANNOTATIONS` (drops the "both granularities / set URL + section
      URL" description); `assertions/contraindications.py` module docstring (the
      "unions both granularities" lines); the "Notes / caveats" paragraph in
      `plans/tablassert-output-legacy-parity.md` (state the final CURIE-only form).
- [x] Tests:
  - `tests/unit/test_assertions_evidence.py` — `spl_evidence_pipe` expectations become
        `dailymed:SET-A|dailymed:SET-B` (sorted, deduped, fragment-stripping, idempotence,
        empty → `""`);
  - `tests/unit/test_assertions_approved_treats{,_edge}.py` and
        `tests/unit/test_assertions_contraindications{,_edge}.py` — `supporting_spl_evidence`
        expectations → CURIE form (the `supporting_spl_sets` / `_documents` URL expectations
        stay);
  - `tests/integration/test_kgx_end_to_end.py::test_dailymed_evidence_lands_on_the_edge_not_in_a_study`
        — assert `value.startswith("dailymed:")`; **drop** the "both granularities present"
        sub-assertions (bare-set link + `#`-section link); keep the array / no-`|` checks.

## Denied / ignored items — no work

- **Category stays** `biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation` (proposal 1
  denied; also keeps DINGO's generic-`Association` fallback behavior as today).
- `evidence_count` stays the FAERS count field (proposal 3 ignored — no `N_cases`).
- `sources[]` roles stay: DAKP `primary_knowledge_source` for all families (proposal 5 ignored).
- No `subject_name` / `object_name`; `original_subject` / `original_object` stay (proposal 6).
- `source_record_urls` stay dataset-level (proposal 7); Tablassert deterministic ids stay
  (proposal 8); Tablassert extras stay (proposal 9).

## Steps

- [x] Change B (has_evidence CURIEs): `evidence.py` + docstrings + unit tests.
- [x] Change A (approval_ids `split_by`): `tablassert.py` annotation + config test + e2e test.
- [x] (A2 adopted) `approval_display` type-prefixed values + affected unit tests.
- [x] Regenerate committed `tables/*.yaml`.
- [x] Update `plans/tablassert-output-legacy-parity.md` notes to the final shape.
- [ ] Full suite + lint + type check (100% branch-coverage gate).

## Verification

```bash
uv run pytest tests/unit/test_assertions_evidence.py tests/unit/test_assertions_approved_treats.py \
              tests/unit/test_assertions_approved_treats_edge.py \
              tests/unit/test_assertions_contraindications.py tests/unit/test_assertions_contraindications_edge.py \
              tests/unit/test_tablassert_configs.py -q
uv run pytest tests/integration/test_kgx_end_to_end.py -q
uv run pytest -q --cov          # 100% branch coverage gate
uv run ruff check && uv run ruff format --check && uv run pyright
```

- Inspect a fresh e2e build's `dakp_0.1.0.edges.ndjson`: treats edges must show
  `"approval_ids": [ … ]` (JSON array) and `"has_evidence": ["dailymed:<set>", …]` — the two
  fields now match the legacy reference edge field-by-field; everything else unchanged.
- `validate_kgx` stays green (part of the e2e test) — no contract changes needed; both fields
  remain on the existing allow-lists.
