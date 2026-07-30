# Tablassert handoff

How DAKP generates and consumes the Tablassert Graph + per-table configs, and how the
`provenance.override` conventions match the `../DINGO` translator-ingests reference.
Generation: [`src/dakp_pipeline/tablassert/configs.py`](../src/dakp_pipeline/tablassert/configs.py).
Consumption: [`src/dakp_pipeline/tablassert/run.py`](../src/dakp_pipeline/tablassert/run.py).

Recall the core rule ([`architecture.md`](./architecture.md#the-delegation-boundary)):
**DAKP shapes tables; Tablassert does the graph.** DAKP never reimplements fullmap
resolution, KGX compilation, dedup, deterministic IDs, or RIG generation.

## What is generated

After the assertion shapers produce the three uncompressed TSVs, `generate(assertion_refs, ctx)`
writes a **Graph config** plus one **table config** per assertion table into
`<workdir>/data/store/tablassert/tables/` and returns refs (graph first):

```
data/store/tablassert/tables/
  graph.yaml                              # single Graph config
  approved_treats_assertions.yaml
  faers_applied_to_treat_assertions.yaml
  contraindication_assertions.yaml
```

The per-table provenance is driven by `_TABLE_PROVENANCE` in `configs.py`:

| Assertion table | predicate | `upstream_resource_ids` | `knowledge_level` |
| --- | --- | --- | --- |
| `approved_treats_assertions` | `treats` | `infores:dailymed`, `infores:faers` | `knowledge_assertion` |
| `faers_applied_to_treat_assertions` | `applied_to_treat` | `infores:faers`, `infores:dailymed` | `observation` |
| `contraindication_assertions` | `contraindicated_in` | `infores:dailymed` | `knowledge_assertion` |

Configs are emitted by **string templating** (no `pyyaml` dependency in the base install).
Every config is itself content-addressed and manifest-recorded like any other artifact.

## Graph config

`tables/graph.yaml` (real output):

```yaml
name: dakp
version: "0.1.0"
description: >-
  Drug Approvals Knowledge Provider: FDA-approved treatment relationships,
  FAERS-observed applied-to-treat uses, and contraindications text-mined from
  DailyMed, modeled from DailyMed, Drugs@FDA, and FAERS.
infores: infores:multiomics-drugapprovals
fullmap: .fullmap
tables:
  - tables/approved_treats_assertions.yaml
  - tables/faers_applied_to_treat_assertions.yaml
  - tables/contraindication_assertions.yaml
```

`infores` is DAKP's own id (`infores:multiomics-drugapprovals`); `fullmap: .fullmap`
points at the fullmap redb Tablassert uses for canonical resolution.

## Per-table config

Each table config declares a `text` source over the generated TSV, a column-encoded
subject/object/predicate, and a `provenance.override` block. Real output for the
`applied_to_treat` table:

```yaml
source:
  kind: text
  local: data/tabular/faers_applied_to_treat_assertions.tsv
  url: https://example.invalid/dakp/generated/faers_applied_to_treat_assertions.tsv
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
    infores: infores:multiomics-drugapprovals
    upstream_resource_ids:
      - infores:faers
      - infores:dailymed
    knowledge_level: observation
    agent_type: manual_validation_of_automated_agent
```

### Source kind

`source.kind: text` with `delimiter: "\t"` — the current `../Tablassert` source model for
plain-text tables. DAKP does not point Tablassert at a PMC/PMID corpus, so it does not use
a `ManualSource` class; it uses `text` + `provenance.override` (`ManualProvenance`) per the
current Tablassert API (PLAN.md "Tablassert 8.x notes"). The `url` field is a placeholder
(`https://example.invalid/...`) used only as a stable identifier; `local` is the real path.

> **Column encodings are placeholders.** `encoding: A` (subject) and `encoding: F` (object)
> are template column letters; the exact column→letter mapping is finalized in Milestone 7
> when the assertion-table column order is locked against Tablassert's encoding scheme and
> the 8.0.0 API is confirmed. The assertion TSVs already carry the data; only the config
> column-mapping is pending.

> **Annotations are not yet emitted.** PLAN.md sketches per-annotation column encodings
> (`number_of_cases`, `clinical_approval_status`, `FDA_regulatory_approvals`). The
> assertion tables already carry these as columns; the `annotations:` block in the config
> is added in Milestone 7 when Tablassert's annotation API is confirmed.

## Provenance conventions (matching `../DINGO`)

The `provenance.override` block and the assertion-table columns encode the source chain
modeled in the DINGO translator ingest (local sibling repo:
`../DINGO/src/translator_ingest/ingests/dakp/` and its
`../DINGO/tests/unit/ingests/dakp/test_dakp.py` unit tests). DAKP
(`infores:multiomics-drugapprovals`) is the owning KP; its **role** differs by family:

| Family | DAKP `resource_role` | Primary upstream | Supporting | `upstream_resource_ids` |
| --- | --- | --- | --- | --- |
| `treats` | **primary** knowledge source | — | DailyMed, FAERS | `infores:dailymed`, `infores:faers` |
| `applied_to_treat` | **aggregator** knowledge source | FAERS (primary) | DailyMed | `infores:faers`, `infores:dailymed` |
| `contraindicated_in` | **aggregator** knowledge source | DailyMed (text-mined) | — | `infores:dailymed` |

In the DINGO ingest, the `sources` list becomes `RetrievalSource` objects: DAKP carries the
`upstream_resource_ids`; FAERS takes `resource_role = primary_knowledge_source` for the
observation family it originates; DailyMed is `supporting_data_source`. The treatment and
observed-use families use `agent_type = manual_validation_of_automated_agent`;
`contraindicated_in` is text-mined from DailyMed and uses `agent_type = text_mining_agent`.

> **Scaffold coverage.** The assertion tables carry `primary_knowledge_source` and
> `upstream_resource_ids` today. The full DINGO source chain fields
> (`aggregator_knowledge_source`, `supporting_data_sources`, `source_record_urls`) are
> finalized during Milestones 5–7 and emitted through the Tablassert provenance override /
> DINGO ingest — DAKP does **not** grow a parallel KGX postprocessor to encode them.
> If a needed Biolink slot is missing, it is upstreamed into `../Tablassert`.

## Consumption: `run_tablassert`

[`tablassert.run`](../src/dakp_pipeline/tablassert/run.py) takes the assertion refs +
config refs and either (a) delegates to `../Tablassert` (full build) or (b) writes a
deferred-handoff manifest (mock). The decision:

```python
run_real = bool(ctx.params.get("run_tablassert")) and ctx.profile != "mock"
if run_real:
    raise NotImplementedError("real ../Tablassert integration lands in Milestone 7; no local KGX fallback")
```

**Mock mode** writes `<workdir>/data/reports/tablassert_handoff.json` recording the
assertion inputs (table / `artifact_id` / rows) and the generated config paths, then
returns a ref to that manifest. It never touches the network or `../Tablassert`, and there
is **no local fallback KGX compiler** — a full run that reaches this point before
Milestone 7 fails loudly.

The integration test in
[`tests/integration/test_mock_pipeline.py`](../tests/integration/test_mock_pipeline.py)
monkeypatches `dakp_pipeline.tablassert.run` with a stand-in that writes a placeholder KGX
marker, proving the boundary is substitutable without `../Tablassert` installed.

## Target: running real Tablassert (Milestone 7)

Per PLAN.md, the live integration runs Tablassert from a local editable checkout without
adding it to `pyproject.toml`:

```bash
uv run --with-editable ../Tablassert dakp-tablassert ...   # or a config setting pointing at the checkout
```

Expected KGX outputs (names compatible with the DINGO ingest): `drug_approvals_kg_nodes.jsonl.gz`
and `drug_approvals_kg_edges.jsonl.gz` under `data/kgx/`. Tablassert owns its own
validation/QC; DAKP's `write_build_summary` only records local manifests, reports, and
output paths — it does not publish artifacts or validate KGX.

## Related

- [`tabular-contracts.md`](./tabular-contracts.md) — the assertion tables these configs point at.
- [`architecture.md`](./architecture.md#the-delegation-boundary) — the Tablassert delegation boundary.
- [`../PLAN.md`](../PLAN.md) "Tablassert modeling and NER/resolution" — the source design intent.
