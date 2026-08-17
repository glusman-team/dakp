# Model Tablassert KGX output like the legacy DAKP jsonlines

## Context

DAKP retired its pure-Python KGX compiler (`ref/legacy/bin/dakp-postprocess2jsonlBL.py`) and now
delegates KGX compilation to **Tablassert**. The emitted `dakp_<version>.edges.ndjson` has drifted
from the legacy modeling. This plan makes the **small DAKP-side changes that work with stock
Tablassert today** (no Tablassert edits) to move the output back toward the legacy shape. Deeper
Tablassert-side gaps (structured `sources[]`, subclass allow-list slots, edge-category choice) are
**explicitly out of scope here** and deferred.

The lever DAKP has today is the **annotation output name**: Tablassert keeps an annotated column as a
first-class edge field only when its name is in `ALLOWED_EDGE_FIELDS`; anything else is folded into
`supporting_text` as `"col: value"` strings (`lib.fold_unknown_to_supporting_text`).

## What survives Tablassert today (verified against the installed allow-list)

DAKP's current evidence annotation names vs the allow-list:

| current annotation name | in `ALLOWED_EDGE_FIELDS`? | effect today |
|---|---|---|
| `supporting_spl_sets` | ❌ | folded into `supporting_text` |
| `supporting_spl_documents` | ❌ | folded into `supporting_text` |
| `clinical_approval_status` | ❌ | folded into `supporting_text` |
| `number_of_cases` (from `case_count`) | ❌ | folded into `supporting_text` |
| `approval_ids` | ❌ | folded into `supporting_text` |
| `source_score` | ❌ | folded into `supporting_text` |
| **`has_evidence`** | ✅ | survives as a first-class edge field |
| **`supporting_documents`** | ✅ | survives as a first-class edge field |

So renaming the two SPL-evidence annotations to `has_evidence` / `supporting_documents` is the one
change that takes effect **today** with no Tablassert modification — and `has_evidence` is exactly the
field name the legacy DAKP used for its SPL-set evidence.

## Legacy reference (why these names)

From `ref/legacy/bin/dakp-postprocess2jsonlBL.py` (`git show c9444e3^:ref/legacy/bin/dakp-postprocess2jsonlBL.py`)
and the downstream consumer `../DINGO/.../ingests/dakp/dakp.py`:

* Legacy edges carried **`has_evidence`** = sorted list of `dailymed:<spl_set_id>` CURIEs. DINGO reads
  `record["has_evidence"]` and extends `publications` with it. DAKP's `supporting_spl_sets` is the same
  SPL-set evidence (pipe-joined `spl_set_id`s) under a non-surviving name.
* Legacy edges denormalized the supporting SPL documents; `supporting_documents` is the sanctioned
  Tablassert/biolink field for document evidence (`TABLASERT_EDGE_EXTRAS`), so
  `supporting_spl_documents` should surface under that name instead of folding away.

The remaining legacy fields (`clinical_approval_status`, `N_cases`/`number_of_cases`,
`approvals`/`FDA_regulatory_approvals`, structured `sources[]`, `EntityTo{Disease,PhenotypicFeature}`
categories, deterministic `uuid3` ids) **cannot** be reproduced with stock Tablassert today and are left
as-is for now (deferred). **2026-08-17 update:** `plans/legacy-edge-parity.md` resolved two of
these with stock Tablassert 12 — `approvals` parity via the `approval_ids` curated pass-through
annotated `split_by: "|"` (edges now carry a real JSON array of `<application type><number>`
values, e.g. `NDA012345`) and `has_evidence` parity via set-granular `dailymed:<spl_set_id>`
CURIEs. The rest were reviewed and deliberately kept as-is: the edge category stays
`ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation`, the FAERS count stays `evidence_count`,
sources roles keep DAKP primary for all families, and gestalt `KGinfo.pl` source-record URLs /
legacy uuid3 id values are not reproduced.

## DAKP-side changes (work with stock Tablassert today)

All in `src/dakp_pipeline/tablassert.py` (`_TABLE_ANNOTATIONS`) + the config test. The assertion-table
**source column** names are unchanged; only the **annotation output name** (second tuple element) changes.

- [ ] `_TABLE_ANNOTATIONS["approved_treats_assertions"]`:
      `("supporting_spl_sets", "supporting_spl_sets")` → `("supporting_spl_sets", "has_evidence")`
- [ ] `_TABLE_ANNOTATIONS["contraindication_assertions"]`:
      `("supporting_spl_sets", "supporting_spl_sets")` → `("supporting_spl_sets", "has_evidence")`
      and `("supporting_spl_documents", "supporting_spl_documents")` → `("supporting_spl_documents", "supporting_documents")`
- [ ] Fix the stale module-docstring claim that `clinical_approval_status` is "written verbatim" (it
      folds today); state that only `has_evidence` / `supporting_documents` are on Tablassert's allow-list
      and the rest fold into `supporting_text`.
- [ ] Update `EXPECTED_ANNOTATIONS` in `tests/unit/test_tablassert_configs.py` to the new output names
      (keys are annotation names; values stay the source columns):
      `has_evidence: supporting_spl_sets`, `supporting_documents: supporting_spl_documents`.

Reuse: `table_config()` already builds annotations from `_TABLE_ANNOTATIONS` via
`column_letter()` (no new code paths); only the name constants change.

### Notes / caveats

* ~~The renamed fields still emit as **pipe-joined scalar strings**~~ — closed by Tablassert 9.1's
  annotation `split_by`, which DAKP now emits on both evidence columns. `has_evidence` reaches the
  edge as a real JSON array (legacy parity), matching what legacy emitted. `supporting_documents` is
  on Tablassert's edge-field allow-list but is not a slot of DAKP's association class, so
  `prune_to_class` routes it onto the inlined supporting study instead of the edge — the values are
  preserved, but as a `supporting_documents=a, b` entry in the study-result description, not an array.
* `has_evidence` values are built in `assertions.evidence.spl_evidence_pipe` as sorted,
  deduplicated `dailymed:<spl_set_id>` CURIEs (set granularity — the legacy DAKP KG form;
  document ids reduce to their set CURIE via `dailymed_set_curie`). The prefixing happens in
  the assertion shapers rather than with Tablassert `Encoding.prefix`, because prefixing the
  joined cell would affect only the first value after `split_by: "|"`. The unannotated
  `supporting_spl_sets` / `supporting_spl_documents` TSV columns keep the human-readable
  DailyMed label URLs as the debuggable split.

## Steps

- [x] Apply the two `_TABLE_ANNOTATIONS` renames + docstring fix in `tablassert.py`.
- [x] Update `EXPECTED_ANNOTATIONS` in `tests/unit/test_tablassert_configs.py`.
- [x] Run unit tests; regenerate configs and confirm `tables/*.yaml` carry `has_evidence` /
      `supporting_documents`.
- [x] (If tablassert is importable) run the KGX end-to-end test and confirm `has_evidence` /
      `supporting_documents` are first-class edge fields, not inside `supporting_text`.

## Verification

```bash
uv run pytest tests/unit/test_tablassert_configs.py -q
uv run pytest tests/integration/test_kgx_end_to_end.py -q   # needs tablassert + tiny fullmap
# Inspect generated tables/*.yaml: annotations named has_evidence / supporting_documents.
# Inspect a build's edges.ndjson: has_evidence / supporting_documents present as fields;
# clinical_approval_status / number_of_cases / approval_ids still in supporting_text (expected today).
```

**2026-08-17:** the verification line above is historical — `clinical_approval_status` and
`approval_ids` are first-class edge fields under Tablassert >= 12, and current builds are
verified by `tests/integration/test_kgx_end_to_end.py` (`has_evidence` = `dailymed:` set CURIEs,
`approval_ids` = JSON array).