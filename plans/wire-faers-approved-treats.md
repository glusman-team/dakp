# Wire FAERS into `shape_treatment_tables` (why approved-treats is ~200 rows)

## Context

The user noticed `shape_treatment_tables` produced a suspiciously small table (~300 rows in an
earlier run; **209 rows** in the last real Airflow run, 2026-08-01). Investigation confirmed two
compounding causes — the table is small **by construction of the current wiring**, not because
the source data is small.

### Diagnosis (numbers from `tmp/airflow-run/data`, 2026-08-01 run)

Evidence available in the run's own interim tables:

| Stage | Count |
|---|---|
| `spl_approvals.parquet` rows | 13,639 (→ 5,065 distinct NDAs) |
| `products.parquet` rows (Drugs@FDA) | 51,622 (→ 28,917 distinct NDAs) |
| DailyMed-approved NDAs that also map in Drugs@FDA | **4,902** |
| LOINC 34067-9 indication sections | 14,328 (across 14,988 SPL sets) |
| FAERS `cases.parquet` rows | 1,082,876 |
| FAERS distinct (NDA, indication) pairs | **54,560** |
| …of which pass both NDA gates (Drugs@FDA map + DailyMed approval) | **13,753** |
| **approved_treats_assertions rows actually written** | **209** (191 drugs, **5 diseases**) |

**Cause 1 (primary): FAERS is not wired into the stage.**
`dakp_build.py:shape_treatment_tables(dm_ext, drugsfda_ext)` passes only DailyMed + Drugs@FDA
refs, so `find_faers_cases()` returns `None` and the shaper falls back to its DailyMed
dictionary path (`_dailymed_candidates`). The module docstring says this explicitly:
*"The current pipeline wires this stage with DailyMed + Drugs@FDA only (FAERS joins this stage
in a later milestone)."* The FAERS candidate path (`_faers_candidates`) is **already implemented
and unit-tested** (`tests/unit/test_assertions_approved_treats.py`) — only the DAG wiring is
missing.

**Cause 2: the disease dictionary is a 5-entry test fixture.**
`runtime._load_disease_map` reads `<fixture_root>/ontology/disease_map.tsv`, and the CLI default
is `_DEFAULT_FIXTURE_ROOT = tests/fixtures/pipeline`. That file has exactly 5 entries
(asthma, headache, pain, hypercholesterolemia, peptic ulcer disease) — and the run's 209 output
rows contain exactly those 5 object texts. Even the fallback path is capped by this dictionary.

Contrast: `faers_applied_to_treat` got 83,800 rows (FAERS wired; dictionary only decorates
CURIEs) and `contraindications` got 11,133 (production GLiNER NER, no dictionary bottleneck).

### Expected impact of the fix

Candidate pool goes from "dictionary-disease mentions in DailyMed indication text" to the 54,560
FAERS (NDA, indication) pairs, 13,753 of which already pass the NDA gates; the DailyMed
indication-support gates (2)+(3) + ingredient resolution then apply. Realistic output is
**thousands of rows** (exact count only knowable by re-running).

## Approach

Wire the already-tested FAERS path into the stage — the milestone the code promised — plus the
two quality details that wiring exposes:

1. **DAG wiring**: `shape_treatment_tables` also receives `extract_faers` output.
2. **Placeholder-indication filter**: FAERS `indication` carries non-disease junk
   ("Product used for unknown indication", "Medication error", bare "Prophylaxis", …).
   `observed_uses` already filters these with `is_non_disease_indication` (~41% of case-weighted
   rows per its comment); `approved_treats._faers_candidates` must reuse it or that junk becomes
   bogus "approved treats Disease" rows.
3. **Column projection**: `find_faers_cases(inputs)` with no projection reads the full 17-column
   case table ("tens of millions of rows wide" in production). Call it with
   `columns=("nda", "nda_raw", "indication", "ingredient", "drugname")`, mirroring
   `observed_uses` (which projects to 3 columns for the same reason).

Out of scope (keep existing semantics):
- ANDA vs NDA: output already aggregates ANDA approvals as "approved"; keep.
- The 5-disease lexical map stays the baseline; on the FAERS path it only decorates object
  CURIEs (unmatched indications stay text-first with empty CURIE — documented design:
  "Canonical CURIE mapping is a later milestone").

## Files to modify

- `src/dakp_pipeline/dags/dakp_build.py` — task signature + wiring
- `src/dakp_pipeline/assertions/approved_treats.py` — junk filter in `_faers_candidates`,
  projected `find_faers_cases` call, docstring updates
- `tests/unit/test_dag.py` — upstream assertion for `shape_treatment_tables`
- `tests/integration/harness.py` — mirror the DAG wiring (`[*dm_ext, *drugsfda_ext, *faers_ext]`)
- `src/dakp_pipeline/assertions/evidence.py` — docstring-only: `find_faers_cases` note about the
  approved-treats stage "wiring without FAERS" goes stale

## Reuse

- `_faers_candidates` + `build_approved_treats_rows` FAERS path — already implemented and tested
  (`src/dakp_pipeline/assertions/approved_treats.py:144`, unit tests in
  `tests/unit/test_assertions_approved_treats.py:25-102`).
- `is_non_disease_indication` (`src/dakp_pipeline/assertions/observed_uses.py:55`) — the legacy
  DAKP stop-list regex; import it into `approved_treats` (keep canonical definition in place).
- `find_faers_cases(inputs, columns=...)` projection support
  (`src/dakp_pipeline/assertions/evidence.py`) — already used by `observed_uses`.
- Fixture FAERS data already carries NDA-bearing Examplestatin rows
  (`tests/fixtures/pipeline/faers/DRUG24Q3.txt` → NDA 012345), so
  `test_prod_smoke.py`'s "Examplestatin/hypercholesterolemia in approved" assertions hold
  (now via the FAERS-primary path).

## Steps

- [ ] `dakp_build.py`: `shape_treatment_tables(dm_ext, drugsfda_ext, faers_ext)`; merge
      `refs_from_xcom(faers_ext)` into the input refs; update the call site wiring.
- [ ] `approved_treats.py`: in `ApprovedTreatsShaper.transform`, call
      `find_faers_cases(inputs, columns=("nda", "nda_raw", "indication", "ingredient", "drugname"))`;
      in `_faers_candidates`, skip pairs where `is_non_disease_indication(indication)`; update the
      module docstring (FAERS is now the wired primary source; DailyMed fallback for FAERS-less runs).
- [ ] `evidence.py`: refresh the stale `find_faers_cases` docstring note.
- [ ] `tests/unit/test_dag.py`: `upstream("shape_treatment_tables") ==
      {"extract_dailymed", "extract_drugsfda", "extract_faers"}`.
- [ ] `tests/integration/harness.py`: pass `faers_ext` into `approved_treats.transform` (keep
      harness == DAG parity for `test_semantic_equivalence.py`).
- [ ] Unit test for the junk filter: a FAERS pair with a placeholder indication
      ("Product used for unknown indication") yields no row even when all gates pass.

## Verification

1. `uv run pytest tests/unit tests/integration` — full suite green (especially
   `test_dag.py`, `test_assertions_approved_treats*.py`, `test_prod_smoke.py`,
   `test_semantic_equivalence.py`, `test_kgx_end_to_end.py`).
2. Re-run the pipeline (`dakp up`), then re-run the funnel analysis on the new
   `build_summary.json`: `approved_treats_assertions` rows should jump from 209 to the
   low/mid thousands, with many more distinct object texts (no longer capped at 5).
3. Spot-check: no placeholder indications in `object_text`
   (`grep -i "unknown indication\|product used for" approved_treats_assertions.tsv` → empty).

## Follow-ups (not this change — user decision)

- **Real disease dictionary**: production runs read the lexical map from a 5-row *test fixture*
  (`_DEFAULT_FIXTURE_ROOT = tests/fixtures/pipeline`). FAERS wiring fixes row counts without it,
  but object CURIE coverage stays near zero until a MONDO/DOID-derived dictionary (or NER) is
  used for this stage too.
- Translator regression baseline (`families_seen`/row counts) will shift with the larger table;
  it's a relative invariant, but worth an eyeball on the first real run.
