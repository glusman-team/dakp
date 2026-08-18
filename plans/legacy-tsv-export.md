# Retrofit the Tablassert KGX into the legacy DAKP TSV pair

## Context

An internal service still consumes the **old TSV schema** of DAKP (the `drug_approvals_kg_*_v0.5.3`
pair produced by the pre-rewrite `druginfo` pipeline):

```
nodes: id  name  category
CHEBI:4875  Etanercept  biolink:ChemicalEntity

edges: id  subject  predicate  object  subject_name  object_name  object_modifier  knowledge_level  agent_type  approval  N_cases  supporting_spls
408826a1-…  CHEBI:4875  biolink:applied_to_treat  MONDO:0008383  Etanercept  rheumatoid arthritis  NA  observation  manual_validation_of_automated_agent  NA  269572  NA
```

DAKP now ends at the Tablassert handoff (`data/dakp_<version>.{nodes,edges}.ndjson`), so those TSVs
no longer exist. **New DAG stage**: convert the KGX ndjson pair produced by `run_tablassert` into a
`.nodes.tsv` + `.edges.tsv` pair in the old schema, so the old service keeps working off fresh builds.

### Legacy conversion semantics (recovered exactly)

The old TSVs were produced by `ref/legacy/bin/jsonlines2tsv.py` (deleted in c9444e3, readable via
`git show c9444e3^:ref/legacy/bin/jsonlines2tsv.py`). Its rules:

- **nodes**: `id`, `name`, `category` — when `category` is a list, take the **first** element.
- **edges**: one column per header; `NA` when absent; multi-valued fields **comma-joined** —
  `approval ← ",".join(approvals)` (via its `h+'s'` rule) and
  `supporting_spls ← ",".join(has_evidence)` (explicit special case).

### Field map (current KGX edge → legacy edge column)

| legacy column | KGX source | notes |
|---|---|---|
| `id` | `id` | Tablassert deterministic UUID |
| `subject` / `predicate` / `object` | same names | resolved CURIEs |
| `subject_name` / `object_name` | node `name` from nodes.ndjson; fallback `original_subject` / `original_object`; else `NA` | **canonical name preferred** (legacy parity — "Etanercept", not the raw mention) — confirmed |
| `object_modifier` | always `NA` | confirmed — legacy KG never populated it either |
| `knowledge_level` | `knowledge_level` | |
| `agent_type` | `agent_type` | |
| `approval` | `",".join(approval_ids)` else `NA` | treats edges only; `approval_ids` is the JSON array of `<type><number>` ids (e.g. `NDA012345`) |
| `N_cases` | `evidence_count` else `NA` | applied_to_treat edges only; arrives as int or numeric string — emit `str(value)` |
| `supporting_spls` | `",".join(has_evidence)` else `NA` | `dailymed:<spl_set_id>` CURIEs (treats/contra) |

Node category keeps the `biolink:` prefix (the legacy sample carries it).

### Output location / naming — confirmed

Same stem and directory as the KGX files, extension swapped: the ndjson pair
`<workdir>/data/dakp_<version>.nodes.ndjson` / `.edges.ndjson` (written by `tablassert build-kg`
with cwd = workdir root, `rig.artifact_base_path = "data"`) yields
**`<workdir>/data/dakp_<version>.nodes.tsv`** and **`<workdir>/data/dakp_<version>.edges.tsv`** —
plain uncompressed TSV, mirroring DAKP's uncompressed-TSV convention.

## Approach

1. New pure module `src/dakp_pipeline/legacy_tsv.py`:
   - `convert_nodes(nodes) -> pl.DataFrame` — `id` / `name` / `category` (first list element).
   - `convert_edges(edges, nodes) -> pl.DataFrame` — the 12 legacy columns, every cell pre-formatted
     to `str` (missing → `"NA"`, lists → comma-joined); name lookup uses a node-id → name index with
     the `original_subject` / `original_object` fallback.
   - `export(kgx_refs, ctx) -> list[ArtifactRef]` — the stage entry point:
     - locate the handoff report among `kgx_refs` (by `tablassert.REPORT_NAME`);
     - **deferred handoff ⇒ return `[]`** (never an error — mirrors the deferred-handoff convention);
     - glob `Workdir(ctx.workdir).root / "data"` for exactly one `*.nodes.ndjson` + one
       `*.edges.ndjson` (more/fewer → `RuntimeError`, loud — a real successful build must have them);
     - convert via `translator.read_kgx_jsonl`, write `<stem>.nodes.tsv` / `<stem>.edges.tsv`
       (`ndjson_path.with_suffix(".tsv")`), register both with `ArtifactStore`
       (`text/tab-separated-values`, inputs = ndjson blake3s, operation `export_legacy_tsv`), stats-log.
   - Converter is total: unexpected list/scalar shapes degrade (scalar evidence → kept as-is;
     empty list → `NA`), never crash on a weird record.
2. New DAG task `export_legacy_tsv` in a new `export` TaskGroup (6th stage; single-task group like
   `summary`), wired `run_tablassert → export_legacy_tsv → write_build_summary`:
   - body calls `legacy_tsv.export(...)`; **when it returns `[]` raise
     `airflow.sdk.exceptions.AirflowSkipException`** so a deferred run shows the task as skipped
     (not failed, not silently green). The stage itself stays Airflow-free; only the task body raises.
3. `runtime.write_build_summary(..., legacy_tsv_refs=None)` — additive optional param; adds a
   `legacy_tsv` section (name/path/rows/artifact_id per file) to `build_summary.json`.
4. `tests/integration/harness.py` mirrors the stage after the Tablassert handoff, passing the refs
   to `write_build_summary`.
5. Docs: DAG `doc_md` (five → six stages), README pipeline line (`… -> Tablassert handoff -> legacy
   TSV export`), brief note in the Output Tables area.

## Files to modify

- `src/dakp_pipeline/legacy_tsv.py` — NEW (converter + `export` stage)
- `src/dakp_pipeline/dags/dakp_build.py` — `export_legacy_tsv` task + `export` TaskGroup; extend
  `TablassertOutputs` with a `legacy` handle; rewire `write_build_summary` inputs; DAG doc_md
- `src/dakp_pipeline/runtime.py` — `write_build_summary` optional `legacy_tsv_refs`
- `tests/integration/harness.py` — call the new stage, pass refs through
- `tests/unit/test_legacy_tsv.py` — NEW
- `tests/unit/test_dag.py` — 13 → 14 task ids, `export` group membership, graph edges
  (`upstream(export_legacy_tsv) == {run_tablassert}`, summary upstream `+= {export_legacy_tsv}`)
- `tests/unit/test_runtime_edge.py` — cover the new summary section (incl. empty default)
- `tests/integration/test_kgx_end_to_end.py` — extend with the retrofit assertions over a real build
- `README.md` — pipeline diagram + one-line mention

## Reuse

- `dakp_pipeline.translator.read_kgx_jsonl` — ndjson reader
- `dakp_pipeline.io.schemas.write_tsv` (polars) — header'd uncompressed TSV writer
- `dakp_pipeline.io.artifact_store.ArtifactStore` — register outputs + provenance manifests
- `dakp_pipeline.tablassert.REPORT_NAME` — locate the handoff report ref
- `dakp_pipeline.paths.Workdir` — workdir-derived `data/` location
- DAG task conventions (`_ctx()`, `_refs_to_xcom`, `step`/`stats`, `# pragma: no cover` bodies)

## Steps

- [x] `legacy_tsv.py`: `NODES_HEADER` / `EDGES_HEADER` constants + `convert_nodes` / `convert_edges`
- [x] `legacy_tsv.py`: `export()` (report check → ndjson glob → convert → write → register)
- [x] DAG: `export` TaskGroup + `export_legacy_tsv` task (skip on `[]`), summary rewiring, doc_md
- [x] `runtime.write_build_summary`: `legacy_tsv_refs` param + `legacy_tsv` summary section
- [x] `harness.py`: mirror the stage
- [x] Unit tests: converter semantics (first-category, NA fills, comma joins, name fallback chain,
      int/str `evidence_count`, empty-list → NA, deferred → `[]`, real → files + contents,
      zero/two ndjson globs raise)
- [x] `test_dag.py` / `test_runtime_edge.py` updates; e2e legacy-TSV assertions
- [x] README note

## Implementation notes (post-execution)

* **Trigger rule:** `write_build_summary` carries `trigger_rule="none_failed"` — an
  `AirflowSkipException` in `export_legacy_tsv` (deferred handoff) would otherwise cascade the
  skip onto the terminal summary task under the default `all_success` rule; the skipped task's
  XCom then resolves to `None`, which `refs_from_xcom` already maps to `[]`.
* **Harness fidelity:** the two offline integration fakes now follow the REAL handoff contract —
  `test_mock_pipeline._fake_tablassert_run` writes a real-mode report + a `dakp_<version>` ndjson
  pair (so the export stage's real branch runs offline), and `test_prod_smoke._fake_tablassert_subprocess`
  writes the ndjson pair under `cwd/data/` (so the real runner + export both run on faked output).
* Verified: full suite green at the 100% branch-coverage gate; the real-tablassert e2e (tiny
  fullmap) passes including `test_legacy_tsv_pair_matches_the_old_schema`; `ruff check`,
  `ruff format --check`, and `pyright` all clean.

## Verification

```bash
uv run pytest tests/unit/test_legacy_tsv.py tests/unit/test_dag.py tests/unit/test_runtime_edge.py -q
uv run pytest tests/integration/test_kgx_end_to_end.py -q   # real tablassert + tiny fullmap -> retrofit
uv run pytest -q --cov          # 100% branch-coverage gate
uv run ruff check && uv run ruff format --check && uv run pyright
```

Manual: `dakp up --small --fullmap <path>` then inspect `data/dakp_*.nodes.tsv` / `.edges.tsv` —
header row exactly `id  name  category` / the 12-column edge header; `object_modifier` all `NA`;
treats rows carry comma-joined `approval`, applied rows carry `N_cases`, contra/treats rows carry
comma-joined `supporting_spls` CURIEs; deferred run (no `--fullmap`) shows `export_legacy_tsv`
skipped and the DAG green.
