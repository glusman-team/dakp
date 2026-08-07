# Airflow DAG optimization and modularization plan

## Context

The user asked whether the current `dakp_build` DAG is organized efficiently for speed and whether its steps are sufficiently modular, with Airflow best-practice research before any implementation.

Confirmed scope/constraints from the user:

- Only suggest DAG-structure and modularity changes; do not propose unrelated extractor, algorithm, data-model, or infrastructure rewrites.
- Tune recommendations for `wenceslaus`: 2x Xeon Gold 6230, 80 logical CPUs, 4x NVIDIA Tesla P100 16GB GPUs, large local XFS storage.
- Treat **50 GB RAM** as the usable pipeline memory budget even though the host has more physical RAM.
- The immediate deliverable is therefore an Airflow DAG/modularity optimization plan, not code changes during planning.

Current quick-scan baseline:

- The DAG is `src/dakp_pipeline/dags/dakp_build.py` and currently has 13 tasks.
- Pipeline shape is: source acquisition -> Go-backed extraction -> Python assertion shaping -> Tablassert config/run -> build summary.
- Heavy extraction is already delegated to Airflow Go SDK stubs on a dedicated `golang` queue and `dakp_extract` pool.
- Raw/interim payloads are passed as lightweight `ArtifactRef` JSON over XCom; large bytes stay in the content-addressed filesystem store.
- Acquisition tasks are independent and use a `dakp_download` pool.
- The DAG file currently defines all task callables inline inside one DAG function; there are no TaskGroups or reusable task-factory modules.
- Measured DAG import/parse from this checkout with `uv run python`: **~1.95 s** for the module, producing the expected 13 tasks. That is not catastrophic for one DAG, but it is high enough that Airflow's top-level-code guidance is relevant.
- Current extracted graph/resource facts:
  - 4 acquisition tasks in `dakp_download`.
  - 3 Go stub extraction tasks in `dakp_extract`, all currently one pool slot.
  - 3 Python shaping tasks plus Tablassert and summary in `default_pool`.
  - `shape_contraindication_tables` already starts as soon as DailyMed extraction + NER model acquisition complete; it does not wait for FAERS or Drugs@FDA.

## Airflow best-practice notes researched so far

From Apache Airflow documentation:

- Tasks should be transactional/idempotent and safe to retry; avoid producing incomplete results.
- XCom is intended for small messages; pass paths/handles to larger artifacts via external storage.
- Avoid heavy top-level DAG code, imports, database calls, networking, or `Variable.get()` during DAG parse; perform them inside task execution.
- Airflow Variables are OK inside task execution but should be avoided at top level.
- Pools limit parallelism for arbitrary task sets; `pool_slots` can represent heavier tasks.
- Dynamic task mapping lets runtime-discovered lists/dicts expand into multiple task instances, with limits via `max_map_length` and `max_active_tis_per_dag`.
- TaskGroups improve Graph view organization and reuse for repeated stage patterns without changing the underlying DAG semantics.

## Assessment

### What already looks strong

- The current graph is mostly efficient at the coarse stage level: independent sources acquire in parallel; each source extracts as soon as its own acquisition finishes; assertion shapers start as soon as their true inputs are ready.
- The design follows Airflow's small-XCom guidance: tasks pass `ArtifactRef` manifests, not dataframes or large payloads.
- Source acquisition and extraction are content-addressed/idempotent enough to align with Airflow retry best practices.
- Heavy extraction is correctly isolated into native Go stub tasks on the `golang` queue.
- The FAERS extractor already has bounded-memory streaming below the DAG layer, and contraindication mining already has internal multi-GPU dispatch; those should be reused rather than replaced with DAG-level microtasks.

### Concurrent indications-parser work observed

Plan-review feedback pointed to another agent's in-progress indication-section parsing/tests. I inspected the current worktree and found uncommitted changes in `src/dakp_pipeline/assertions/contraindications.py` and `tests/unit/test_assertions_contraindications_edge.py` that add a two-pass contraindication miner:

- Pass 1 still mines dedicated contraindication sections (`34070-3`).
- Pass 2 filters indication sections (`34067-9`) down to contraindication-context sentences, then mines those filtered sentences.
- Production dispatch is intended to split the 4 P100 GPUs across both passes (`_mine_two_passes_multi_gpu` / 2+2 split), falling back to all GPUs on Pass 1 when Pass 2 has no work.
- This does **not** require a new DAG edge: `shape_contraindication_tables` already receives `dm_ext`, and `build_dailymed_evidence()` already indexes both `indication_docs` and `contraindication_docs` from the DailyMed extract.

Implication for this DAG plan: the DAG refactor must preserve `shape_contraindication_tables(dm_ext, ner_models)` as a single all-GPU shaping task and must not introduce DAG-level mapping/splitting around contraindication mining. The in-task two-pass GPU scheduler is the right modular boundary for that work.

### Gaps in the DAG layer

- **Modularity/readability:** one large inline DAG function makes the product feel less engineered than the underlying stage modules. It is harder to reason about acquisition/extraction/shaping/Tablassert as composable stage groups.
- **Resource weighting:** all three extract stubs use one `dakp_extract` pool slot even though FAERS and DailyMed are heavy and Drugs@FDA is small. On a 50 GB RAM budget, letting FAERS and DailyMed extract concurrently is the main avoidable DAG-level memory risk.
- **Default-pool shaping:** the three Python shapers currently run in `default_pool`; this is acceptable but not explicit. If future runs show Python-side memory pressure, a dedicated shaping pool would be the next DAG-only lever.
- **Parse-time hygiene:** ~1.95 s DAG import is acceptable but not ideal. Airflow docs specifically recommend avoiding unnecessary top-level imports and `Variable.get()` calls; the DAG already avoids top-level Variables, but it imports all stage modules at parse time.
- **Dynamic task mapping is not recommended as an immediate DAG-only change:** mapping FAERS quarters/DailyMed releases would require new reducer semantics and changes to how the Go stub implementation receives inputs. That crosses the user's requested DAG/modularity-only boundary.

## Approach

Recommended DAG-only refactor:

1. **Keep the existing 13 logical tasks and preserve task IDs.** Use TaskGroups with `prefix_group_id=False` so the Airflow UI gains stage grouping without breaking task history, logs, or the current graph tests' task-id expectations.
2. **Extract stage construction into small task-factory helpers.** Keep the product DAG readable as a stage composition:
   - `build_acquire_stage()`
   - `build_extract_stage(raws)`
   - `build_shape_stage(extracts, ner_models)`
   - `build_tablassert_stage(assertions)`
   - `build_summary_stage(assertions, kgx)`
3. **Move non-Airflow stage imports carefully, after the indication-parser work lands.** Keep only Airflow decorators, constants, and lightweight typing at DAG parse time where practical, but preserve current monkeypatch/test seams such as `tablassert.run(...)` and `contraindications.transform(...)` being looked up at task runtime. This follows Airflow's top-level-code best practice without breaking the concurrent indication tests.
4. **Add weighted pool slots to the existing extract pool.** Airflow 3's `@task.stub(..., **kwargs)` accepts operator kwargs such as `pool_slots`; keep `dakp_extract` at its current 4-slot deployment size, but make heavy extractors consume more slots:
   - `extract_faers`: `pool_slots=3`
   - `extract_dailymed`: `pool_slots=3`
   - `extract_drugsfda`: `pool_slots=1`

   With a 4-slot pool, Airflow can run `extract_drugsfda` alongside either heavy extractor, but it will not run FAERS and DailyMed extraction at the same time. This is the best DAG-only speed/memory tradeoff for the 50 GB RAM budget.
5. **Do not introduce dynamic task mapping in this pass.** It is a valid future optimization idea, but in this codebase it is not just a DAG rewire: the Go stub currently reads upstream XCom by static task ID, mapped per-quarter/per-release extraction would need reducer contracts and Go input handling changes, and the in-progress contraindication indication-parser already owns its own 2-pass/4-GPU parallelism inside the shaper. Keep mapping out of scope.

## Files to modify

Recommended implementation scope after approval:

- `src/dakp_pipeline/dags/dakp_build.py` — compose TaskGroups, call stage factory helpers, add `pool_slots` to extract stubs, and keep the registered `dag_obj` unchanged.
- `src/dakp_pipeline/dags/stages.py` or `src/dakp_pipeline/dags/dakp_build.py` helpers — small reusable stage-builder functions. Prefer a separate `stages.py` only if it stays lightweight and does not import heavy stage modules at top level.
- `tests/unit/test_dag.py` — assert the same 13 task IDs, same dependencies, plus TaskGroup membership and extract `pool_slots`.
- `tests/unit/test_dag_downloads.py` and `tests/unit/test_dag_edge.py` — update pool/resource assertions as needed.

Explicitly out of scope per user request and concurrent-work safety:

- No Go extractor changes.
- No algorithm/data-contract changes.
- No new source acquisition behavior.
- No object-storage or distributed-executor migration.
- Do not edit `src/dakp_pipeline/assertions/contraindications.py` or its indication-parser tests as part of this DAG/modularity pass; treat that as another agent's ownership area and only adapt DAG tests around the final landed task behavior.

## Reuse

Existing reusable pieces found:

- `dakp_pipeline.io.xcom.refs_to_xcom` / `refs_from_xcom` — keep for Airflow/Go boundary manifests.
- `dakp_pipeline.io.contracts.ArtifactRef` and `TaskContext` — keep task interface lightweight.
- `dakp_pipeline.acquire.acquire_*` helpers — current acquisition delegation is already modular.
- `dakp_pipeline.runtime.build_context_from_config` — single runtime-config/context builder.
- `dakp_pipeline.assertions.*.transform` functions — pure stage entry points for TaskFlow tasks.
- `go/cmd/dakp-bundle/main.go` and `go/internal/airflow/extract_*.go` — existing native extraction implementations; keep untouched.
- Airflow `TaskGroup(prefix_group_id=False)` — use for visual/modular grouping while preserving existing task IDs.
- Existing pools `dakp_download` and `dakp_extract` — reuse; only add task-level slot weights for extraction.

## Steps

- [x] Before implementing this DAG refactor, re-check the current indication-parser worktree state and let the other agent's `contraindications.py` / `test_assertions_contraindications_edge.py` changes settle to avoid merge churn.
- [x] Confirm the final indication-parser behavior still requires no new DAG edge beyond `shape_contraindication_tables(dm_ext, ner_models)`.
- [x] Add lightweight stage-return structures if useful, e.g. `AcquireOutputs`, `ExtractOutputs`, and `AssertionOutputs`, so stage helper signatures are explicit and typed.
- [x] Wrap acquisition tasks in an `acquire` TaskGroup while preserving task IDs via `prefix_group_id=False`.
- [x] Wrap Go stub extraction tasks in an `extract` TaskGroup while preserving task IDs.
- [x] Add `pool_slots` weights to extraction stubs: FAERS=3, DailyMed=3, Drugs@FDA=1 on the existing 4-slot `dakp_extract` pool.
- [x] Wrap assertion shaping tasks in a `shape` TaskGroup while preserving task IDs.
- [x] Wrap `generate_tablassert_configs` and `run_tablassert` in a `tablassert` TaskGroup while preserving task IDs.
- [x] Keep `write_build_summary` as the terminal summary task, optionally under a small `summary` TaskGroup if the UI remains clear.
- [x] Move heavy/non-Airflow imports inside task bodies where this does not harm readability or monkeypatchability, explicitly preserving the `contraindications.transform` runtime lookup used by the shaping task.
- [x] Add DAG/task `doc_md` notes documenting the stage groups and the 50 GB memory-oriented pool-slot policy.
- [x] Update DAG tests for unchanged task IDs/dependencies, new TaskGroups, and extract `pool_slots`.
- [x] Re-measure DAG import time and compare to the current ~1.95 s baseline.

## Verification

Planned verification after implementation:

```bash
uv run pytest tests/unit/test_dag.py tests/unit/test_dag_downloads.py tests/unit/test_dag_edge.py -q
uv run pytest tests/unit/test_assertions_contraindications.py tests/unit/test_assertions_contraindications_edge.py -q
uv run pytest -q --cov
uv run ruff check
uv run ruff format --check
uv run pyright
uv run dakp up --small
```

Additional verification:

- Confirm DAG import time is no worse than the current ~1.95 s baseline, ideally lower after localizing imports.
- Confirm the graph still has the same 13 task IDs and same dependency semantics.
- In the Airflow UI, confirm TaskGroups make the pipeline easier to scan without hiding critical dependencies.
- Confirm `extract_faers` and `extract_dailymed` do not run concurrently under the weighted 4-slot `dakp_extract` pool, while `extract_drugsfda` can still run alongside either heavy extractor.
- Compare `--small` output row counts/artifact hashes before vs after; they should remain unchanged relative to the final indication-parser baseline because this is a DAG-only refactor.
- Confirm indication-section contraindication tests still pass after any import-localization, especially tests that monkeypatch `dakp_pipeline.assertions.contraindications` helpers.

## Assumptions baked into this plan

- Preserve existing task IDs for Airflow history/log continuity.
- Optimize for the current `LocalExecutor` + shared local filesystem deployment.
- Preserve output semantics; no fan-out/reduce rewrite in this pass.
- Treat the indication-section contraindication miner as an in-task implementation detail of `shape_contraindication_tables`, not as a separate DAG stage.
