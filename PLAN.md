# Dead Code Cleanup & Polish Plan

## Context

The DAKP pipeline rebuild is functionally complete through Milestone 8: ruff and pyright pass clean, coverage is 100%, and the full mocked pipeline runs end-to-end. Across milestones the codebase accumulated dead code, stale stubs, orphaned modules, duplicate logic, and stale docs. This plan removes the dead weight, deletes the orphaned modules (wiring in only the regression guardrails), and polishes docs for production readiness.

**User decisions (this revision):**
1. Orphaned modules → **delete** `benchmarks.py`, `release.py`, `ner/candidates.py`; **wire in** `translator/regression.py`.
2. Duplicate YAML emitters → **delete** `translator/rig.py` (the non-Tablassert RIG generator); keep only the `tablassert/configs.py` emitter.
3. Skip the `match_diseases` substring false-positive docstring note (the NER layer supersedes it).

## Findings

### A. Dead code (defined + tested, never called by production code)

| Symbol | Location | Note |
|---|---|---|
| `http_download()` | `io/downloads.py:37` | Milestone-1 stub raising `NotImplementedError`; real acquisition lives in `sources/{dailymed,faers,drugsfda}.py` |
| `require_mock()` | `sources/__init__.py:16` | Only called in tests; fetchers dispatch mock/non-mock themselves |
| `iter_quarter_sources()` | `sources/faers.py:158` | Identity passthrough (`yield from quarters`) |
| `_CONFIGURED` global | `logging_setup.py:29` | Set, never read |
| `LoggerLike` alias | `logging_setup.py:137` | Defined, not exported, never used |
| `default_workdir()` | `paths.py:85` | Only used in one edge test; CLI/pipeline pass explicit workdir |
| `pick()` | `assertions/evidence.py:78` | Exported + tested, no production shaper calls it |

### B. Orphaned modules → delete (per decision 1 & 2)

| Module | Tests to delete | Other references to fix |
|---|---|---|
| `benchmarks.py` | `test_benchmarks.py`, `test_benchmarks_edge.py` | none (undocumented; only its own tests import it) |
| `release.py` | `test_release.py`, `test_release_edge.py` | none (undocumented; only its own tests import it) |
| `ner/candidates.py` | `test_ner_candidates.py`, `test_ner_candidates_edge.py` | `ner/__init__.py` (import block + 7 `__all__` entries + docstring bullet); `docs/tabular-contracts.md:131-133` |
| `translator/rig.py` | `test_translator_rig.py`, `test_translator_rig_edge.py` | `translator/__init__.py` (import + 4 `__all__` entries + docstring bullet); `tests/integration/test_semantic_equivalence.py` (import + 2 tests); `docs/architecture.md:119-120,186`; `docs/semantic-equivalence.md:150` |

Notes:
- `tests/eval/benchmark_ner.py`'s `_candidates()` is a **local NER-predictor dict**, unrelated to `ner/candidates.py` — leave it untouched.
- In `test_semantic_equivalence.py`, **keep** `test_contract_categories_match_dingo_rig` and `test_contract_edge_families_match_dingo_predicates` (they use `contract.py` only); **remove** `test_generated_rig_edge_and_node_type_info_match_dingo` and `test_generated_rig_matches_live_dingo_file_when_available`. Remove any constant orphaned by the removal (e.g. `DINGO_NODE_TYPE_INFO` if unused elsewhere); `DINGO_SUBJECT_CATEGORIES`/`DINGO_OBJECT_CATEGORIES` stay (used by the kept test).
- `README.md:61,241` say RIG generation is *delegated to Tablassert* — that is correct and stays. `README.md:83` cites the DINGO `dakp_rig.yaml` as a reference contract — stays.

### C. Wire in `translator/regression.py` (per decision 1)

`regression.check_assertion_tables(refs)` is fully implemented and tested but runs only in one integration test. Wire it into `run_pipeline` after the translator contract check and surface the result in the build summary.

### D. Optimization opportunities

1. **Redundant `Workdir(ctx.workdir)` constructions** — `spl_xml.py` builds it 3×/method; `faers_ascii.py` 2×; `evidence.py` 2×. Build once, reuse.
2. **Repeated `list(inputs)`** — `evidence.build_dailymed_evidence` materializes the iterable 3× (lines 185/195/206). Materialize once.
3. **`find_table` linear scan** — called 3× in `build_dailymed_evidence`; build a `{uri.name: ref}` index once.
4. **Double parquet read** — `build_dailymed_evidence` reads `spl_ingredients.parquet`, then `contraindications._active_ingredients_by_set` reads it again. Share the frame.

(The duplicate-YAML-emitter item is resolved by deleting `rig.py` — no shared module needed.)

### E. Stale documentation

- **23 files** reference "Milestone N" as future work; all milestones are done.
- **3 `__init__.py`** files (`extract/`, `sources/`, `workers/`) still say "(stubs)".

### F. Worktree / branch hygiene

The AOE fleet ran 9 worker branches, all moved to locked `.aoe-trash/` worktrees (657 MB).

- All 9 worker branch tips are confirmed **ancestors of HEAD** (`rebuild/airflow-pipeline` @ `9615874`) — no work lost. Current branch is in sync with its remote (0 ahead / 0 behind).
- 3 worktrees show uncommitted changes, all gitignored artifacts (`.coverage`, `.tablassert/`) — no real work at risk.
- One unmerged remote branch `origin/dakp-scaffold` @ `815e930` is **fully superseded** by the rewrite (its DAG shadowing/`extract_medi`/`test_dag_build.py`/ruff-`.md` fixes all concern code that no longer exists; `ruff format --check` passes clean today).

## Approach

### Phase 0: Git / worktree hygiene

- [ ] Unlock + remove the 9 `.aoe-trash/` worktrees: `git worktree unlock <path> && git worktree remove --force <path>`
- [ ] Delete the 9 merged local branches: `git branch -d dakp-{tablassert-pypi,edge-infra,go-integration,airflow-downloads,edge-extract,ner-simplify,kgx-e2e,edge-ner,semantics-docs}`
- [ ] Delete the superseded remote branch: `git push origin --delete dakp-scaffold`
- [ ] Verify: `git worktree list` shows only the main worktree; `git branch` shows only `main` + `rebuild/airflow-pipeline`

### Phase 1: Remove dead code (§A)

- [ ] `io/downloads.py`: delete `http_download()`, trim `__all__`; delete its test in `test_io_edge.py`
- [ ] `sources/__init__.py`: delete `require_mock()`, trim `__all__`; delete its tests in `test_sources_edge.py`
- [ ] `sources/faers.py`: delete `iter_quarter_sources()`, trim `__all__`; delete its test
- [ ] `logging_setup.py`: delete `_CONFIGURED` global and `LoggerLike` alias
- [ ] `paths.py`: delete `default_workdir()`; delete/repurpose `test_paths_edge.py`
- [ ] `assertions/evidence.py`: delete `pick()`, trim `__all__`; delete its tests

### Phase 2: Delete orphaned modules + wire regression (§B, §C)

**Deletions:**
- [ ] Delete `benchmarks.py` + `test_benchmarks.py` + `test_benchmarks_edge.py`
- [ ] Delete `release.py` + `test_release.py` + `test_release_edge.py`
- [ ] Delete `ner/candidates.py` + its 2 tests; update `ner/__init__.py` (remove import block, 7 `__all__` entries, docstring bullet)
- [ ] Delete `translator/rig.py` + its 2 tests; update `translator/__init__.py` (remove import, 4 `__all__` entries, docstring bullet)
- [ ] `tests/integration/test_semantic_equivalence.py`: remove the `generate_rig` import + the 2 RIG tests; remove orphaned constants (keep the 2 contract-category/family tests)

**Wire regression:**
- [ ] `pipeline.py`: import `from dakp_pipeline.translator import regression`; after `report = translator_contract.validate(...)`, call `regression_report = regression.check_assertion_tables(assertion_refs)`; pass it to `_write_build_summary`
- [ ] `pipeline.py:_write_build_summary`: add a `regression` parameter; emit a `translator_regression` block (`ok`, `families_seen`, `row_count`, `violations`) in the summary JSON
- [ ] `dags/dakp_build.py`: add `"check_regression": regression.check_assertion_tables` to `STAGE_CALLABLES`; the `write_build_summary` task calls it and forwards to `_write_build_summary`; import `regression`
- [ ] Update tests for the new build-summary shape + STAGE_CALLABLES completeness (`test_mock_pipeline.py`, `test_dag.py`, `test_dag_edge.py`, any `_write_build_summary` callers); add coverage for the regression wiring to hold 100%

### Phase 3: Optimizations (§D)

- [ ] `extract/spl_xml.py`: construct `Workdir(ctx.workdir)` once per method, reuse
- [ ] `extract/faers_ascii.py`: same
- [ ] `assertions/evidence.py:build_dailymed_evidence`: materialize `list(inputs)` once; build a `{uri.name: ref}` index; inline/replace the 3 `find_table` scans
- [ ] `assertions/contraindications.py`: reuse the already-read ingredients frame from the evidence index instead of re-reading `spl_ingredients.parquet`

### Phase 4: Documentation polish (§E, §B docs)

- [ ] `docs/tabular-contracts.md`: remove the `mention_candidates.tsv` section (131-133)
- [ ] `docs/architecture.md`: remove the `translator/rig.py` paragraph (119-120) and the "RIG generation" table cell (186)
- [ ] `docs/semantic-equivalence.md`: update the RIG cross-check description (150) to reflect the contract-category/family checks that remain
- [ ] Scrub "Milestone N" future-tense references across the 23 files → present-tense descriptions
- [ ] Update `extract/__init__.py`, `sources/__init__.py`, `workers/__init__.py` docstrings: drop "(stubs)" language
- [ ] Update `io/downloads.py` module docstring (no longer ships a stub)

## Files to modify

**Delete (source):** `benchmarks.py`, `release.py`, `ner/candidates.py`, `translator/rig.py`
**Delete (tests):** `test_benchmarks{,_edge}.py`, `test_release{,_edge}.py`, `test_ner_candidates{,_edge}.py`, `test_translator_rig{,_edge}.py`

**Edit (dead code):** `io/downloads.py`, `sources/__init__.py`, `sources/faers.py`, `logging_setup.py`, `paths.py`, `assertions/evidence.py`, `tests/unit/test_io_edge.py`, `tests/unit/test_sources_edge.py`, `tests/unit/test_paths_edge.py`, `tests/unit/test_assertions_evidence.py`

**Edit (wiring + re-exports):** `pipeline.py`, `dags/dakp_build.py`, `ner/__init__.py`, `translator/__init__.py`, `tests/integration/test_semantic_equivalence.py`, `tests/integration/test_mock_pipeline.py`, `tests/unit/test_dag.py`, `tests/unit/test_dag_edge.py`

**Edit (optimizations):** `extract/spl_xml.py`, `extract/faers_ascii.py`, `assertions/evidence.py`, `assertions/contraindications.py`

**Edit (docs):** `docs/tabular-contracts.md`, `docs/architecture.md`, `docs/semantic-equivalence.md`, + the 23 Milestone-reference files, `extract/__init__.py`, `sources/__init__.py`, `workers/__init__.py`, `io/downloads.py`

## Reuse

- `regression.check_assertion_tables(list[ArtifactRef]) -> RegressionReport` is done and tested — a drop-in after `contract.validate`; `RegressionReport` is a dataclass that serializes cleanly into the build summary.
- The `tablassert/configs.py` YAML emitter stays as-is (its only duplicate, `rig.py`, is deleted).
- `_write_build_summary` already assembles the summary dict; the regression block is one more key.

## Verification

- [ ] `git worktree list` → only the main worktree; `git branch` → only `main` + `rebuild/airflow-pipeline`
- [ ] `uv run ruff check src tests` — clean
- [ ] `uv run ruff format --check` — clean
- [ ] `uv run pyright` — 0 errors
- [ ] `uv run pytest` — all pass, coverage stays 100%
- [ ] `uv run dakp run --profile mock --fixture-root tests/fixtures/pipeline --workdir tmp/polish-check` — runs end-to-end; `tmp/polish-check/data/reports/build_summary.json` contains a `translator_regression` block with `ok: true`
- [ ] `make check-go` — Go gate still passes (no Go changes; confirm no drift)
- [ ] `grep -rn "Milestone [0-9]" src/` — only historical/past-tense references remain (no "lands in Milestone N")
