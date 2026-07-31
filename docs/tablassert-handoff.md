# Tablassert handoff

How DAKP generates and consumes the Tablassert Graph + per-table configs, and how the
`provenance.override` conventions match the `../DINGO` translator-ingests reference.
Generation: [`src/dakp_pipeline/tablassert/configs.py`](../src/dakp_pipeline/tablassert/configs.py).
Consumption: [`src/dakp_pipeline/tablassert/run.py`](../src/dakp_pipeline/tablassert/run.py).

Recall the core rule ([`architecture.md`](./architecture.md#the-delegation-boundary)):
**DAKP shapes tables; Tablassert does the graph.** DAKP never reimplements fullmap resolution,
KGX compilation, dedup, deterministic IDs, or RIG generation.

## Tablassert is the installed PyPI package

KGX compilation uses the **installed `tablassert` CLI** (PyPI `8.0.0`, a core dependency) — not a
local checkout. Install with `uv sync`. The runner shells out to the venv
`tablassert` binary (falling back to `uv run tablassert`) and captures
stdout / exit code into the handoff report. An optional editable-checkout override (the
`tablassert_dir` ctx param, the `DAKP_TABLASERT_DIR` env var, or `TablassertRunner.tablassert_dir`)
switches to `uv run --with-editable <dir> tablassert` for dev against a local `../Tablassert`.

The two subcommands the production build uses (verified against `tablassert 8.0.0 --help`):

```bash
# 1. build the fullmap redb (canonical-resolution database) — wenceslaus-only (~120 GiB RAM):
uv run tablassert build-fullmap --output <fullmap.redb> --threads 64

# 2. build the KGX graph from the generated Graph config:
uv run tablassert build-kg tables/graph.yaml --fullmap <fullmap.redb> [--qc] [--release]
```

`--qc` (the embedding-based audit) is appended only when requested **and** the QC audit runtime
(sentence-transformers, part of the required `tablassert[qc]` install) is importable; `--release` is a boolean flag. See
[`wenceslaus-runbook.md`](./wenceslaus-runbook.md) for the full production sequence.

## What is generated

After the assertion shapers produce the three uncompressed TSVs, `generate(assertion_refs, ctx)`
writes a **Graph config** plus one **table config** per assertion table into `<workdir>/tables/`
(workdir-relative, so their references resolve when Tablassert runs from the workdir root) and
returns refs (graph first):

```
<workdir>/tables/
  graph.yaml                     # single Graph config
  approved_treats.yaml
  faers_applied_to_treat.yaml
  contraindications.yaml
```

The configs match the **actual Tablassert 8.x schema** (verified against `tablassert.models`):

- a table config is a `template:`-wrapped `Section` (`source` / `statement` / `provenance` /
  `annotations`). The loader only reads top-level `template` / `sections` keys, so the `template:`
  wrapper is mandatory;
- `source.kind: text` with a tab `delimiter` and the uncompressed assertion `.tsv` as `source.local`
  (a `url` is required by the model and recorded as provenance);
- column-encoded `statement.subject` / `statement.object` / `statement.predicate` with drug /
  disease `prioritize` categories;
- a `provenance.override` (`ManualProvenance`) block carrying the DAKP infores, the
  DINGO-conventional upstream infores chain, `knowledge_level`, and `agent_type`;
- column-encoded evidence `annotations`.

**Column letters are derived, not hardcoded.** `excel_column` / `column_letter` map each
assertion-table column position (from [`schemas.py`](../src/dakp_pipeline/io/schemas.py)) to its
Excel-style letter (`subject_text`→`A`, `object_text`→`F`, …). YAML is emitted by a tiny stdlib
emitter (no `pyyaml` runtime dependency) that round-trips through `yaml.safe_load` (asserted in the
unit tests). Every config is content-addressed and manifest-recorded like any other artifact.

## Graph config

Real output (`<workdir>/tables/graph.yaml`):

```yaml
name: dakp
version: "0.1.0"
description: >-
  Drug Approvals Knowledge Provider: FDA-approved treatment
  relationships, FAERS-observed applied-to-treat uses, and
  contraindications text-mined from DailyMed, modeled from DailyMed,
  Drugs@FDA, and FAERS.
infores: "infores:multiomics-drugapprovals"
fullmap: ".fullmap"
tables:
  - tables/approved_treats.yaml
  - tables/faers_applied_to_treat.yaml
  - tables/contraindications.yaml
```

`infores` is DAKP's own id (`infores:multiomics-drugapprovals`); `fullmap` points at the fullmap
redb Tablassert uses for canonical resolution (overridden with `--fullmap` in the real build).

## Per-table config

Real output for the `applied_to_treat` table (`<workdir>/tables/faers_applied_to_treat.yaml`):

```yaml
template:
  source:
    kind: text
    local: data/tabular/faers_applied_to_treat_assertions.tsv
    url: "https://example.invalid/dakp/generated/faers_applied_to_treat_assertions.tsv"
    delimiter: "\t"
  statement:
    subject:
      method: column
      encoding: A
      prioritize: [Drug, SmallMolecule, ChemicalEntity]
    predicate: applied_to_treat
    object:
      method: column
      encoding: F
      prioritize: [Disease, PhenotypicFeature]
  provenance:
    override:
      infores: "infores:multiomics-drugapprovals"
      upstream_resource_ids: ["infores:faers", "infores:dailymed"]
      knowledge_level: observation
      agent_type: manual_validation_of_automated_agent
  annotations:
    - annotation: number_of_cases
      method: column
      encoding: J
    - annotation: clinical_approval_status
      method: column
      encoding: K
```

### Evidence annotations per table

| Assertion table | predicate | annotations (column → name) |
| --- | --- | --- |
| `approved_treats_assertions` | `treats` | `approval_ids`, `supporting_spl_sets`, `clinical_approval_status` |
| `faers_applied_to_treat_assertions` | `applied_to_treat` | `case_count` → `number_of_cases`, `clinical_approval_status` |
| `contraindication_assertions` | `contraindicated_in` | `supporting_spl_sets`, `supporting_spl_documents`, `source_score` |

`case_count` maps to the Biolink `number_of_cases` edge slot (the DINGO convention);
`clinical_approval_status` is itself a Biolink slot written verbatim.

## Provenance conventions (matching `../DINGO`)

The `provenance.override` block encodes the source chain modeled in the DINGO translator ingest
(`../DINGO/src/translator_ingest/ingests/dakp/` and its
`../DINGO/tests/unit/ingests/dakp/test_dakp.py`). DAKP (`infores:multiomics-drugapprovals`) is the
owning KP; its **role** differs by family:

| Family | DAKP role | `upstream_resource_ids` (override) | override `knowledge_level` | override `agent_type` |
| --- | --- | --- | --- | --- |
| `treats` | **primary** knowledge source | `infores:dailymed`, `infores:faers` | `knowledge_assertion` | `manual_validation_of_automated_agent` |
| `applied_to_treat` | **aggregator** knowledge source | `infores:faers`, `infores:dailymed` | `observation` | `manual_validation_of_automated_agent` |
| `contraindicated_in` | **aggregator** knowledge source | `infores:dailymed` | `knowledge_assertion` | `text_mining_agent` |

In the DINGO ingest, the `sources` list becomes `RetrievalSource` objects: DAKP carries the
`upstream_resource_ids`; FAERS is `primary_knowledge_source` for the observation family it
originates; DailyMed is `supporting_data_source`. `contraindicated_in` is text-mined from DailyMed,
hence `agent_type = text_mining_agent` (matches the DAKP RIG).

> **Layering note.** The assertion tables *also* carry per-row `knowledge_level` / `agent_type` /
> `clinical_approval_status` columns ([`tabular-contracts.md`](./tabular-contracts.md)); these are
> the values the regression guardrail checks. For `applied_to_treat` the row records the
> finer-grained `statistical_association` while the family-level override Tablassert stamps is
> `observation` (the DINGO ingest's family value). See
> [`semantic-equivalence.md`](./semantic-equivalence.md#deliberate-refinements) for the
> `clinical_approval_status` reasoning.

## Consumption: `run_tablassert`

[`tablassert.run`](../src/dakp_pipeline/tablassert/run.py) takes the assertion refs + config refs
and either (a) runs the installed `tablassert` CLI (full build) or (b) writes a deferred-handoff
manifest (mock). The decision:

```python
run_real = bool(ctx.params.get("run_tablassert")) and ctx.profile != "mock"
runner = TablassertRunner() if run_real else MockTablassertRunner()
```

**Real mode** builds `tablassert build-kg <graph.yaml> --fullmap <path> [--qc] [--release]`, runs it
from the workdir root, and records `status` / `command` / `exit_code` / `stdout` / `stderr` in the
handoff report. A non-zero exit is captured as `status: failed` (logged loudly), not raised. It
raises `RuntimeError` when `tablassert` is unavailable and no editable override is configured
(reinstall with `uv sync`).

**Mock mode** writes `<workdir>/data/reports/tablassert_handoff.json` recording the assertion inputs
(table / `artifact_id` / rows) and the generated config paths, with `status: deferred`. It never
touches the network or Tablassert, and there is **no local fallback KGX compiler** — a full run that
reaches this point without `tablassert` installed fails loudly.

The integration test [`tests/integration/test_mock_pipeline.py`](../tests/integration/test_mock_pipeline.py)
monkeypatches `dakp_pipeline.tablassert.run` with a stand-in, proving the boundary is substitutable
without Tablassert installed.

## Expected KGX outputs

Names compatible with the DINGO ingest: `drug_approvals_kg_nodes.jsonl.gz` and
`drug_approvals_kg_edges.jsonl.gz`. Tablassert owns its own validation/QC; DAKP's
`write_build_summary` only records local manifests, reports, and output paths — it does not publish
artifacts or validate KGX.

## Related

- [`tabular-contracts.md`](./tabular-contracts.md) — the assertion tables these configs point at.
- [`architecture.md`](./architecture.md#the-delegation-boundary) — the Tablassert delegation boundary.
- [`wenceslaus-runbook.md`](./wenceslaus-runbook.md) — the full production build (fullmap + KGX).
- [`semantic-equivalence.md`](./semantic-equivalence.md) — preserved-vs-improved provenance semantics.
