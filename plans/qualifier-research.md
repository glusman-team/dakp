# Research: Biolink Qualifiers in DAKP Edges

## Context

Investigate which biolink qualifiers DAKP edges currently carry, research the full
qualifier landscape available in the installed Tablassert, and determine whether any
qualifiers must be added to avoid ambiguity in the output knowledge graph.

## Findings: Qualifiers Currently Used

**Only one qualifier** appears across all three DAKP edge tables:

| Table | Predicate | Qualifier(s) | Backing column |
|---|---|---|---|
| `approved_treats_assertions` | `treats` | *(none)* | — |
| `faers_applied_to_treat_assertions` | `applied_to_treat` | `disease_context_qualifier` | `object_text` (col F) |
| `contraindication_assertions` | `contraindicated_in` | *(none)* | — |

Defined in `_TABLE_QUALIFIERS` in `src/dakp_pipeline/tablassert.py:~480`.

## Findings: The disease_context_qualifier Is Tautological

The single qualifier in use points at the **same column as the object** (`object_text`).
After fullmap resolution, the qualifier CURIE always equals the object CURIE:

```
metformin --applied_to_treat--> MONDO:0005148  [disease_context_qualifier = MONDO:0005148]
```

This restates the object, adding no disambiguation. The code comment acknowledges this:

> *"the edge carries disease_context_qualifier encoded from the object column (the
> adverse event itself — FAERS `effects` — is aggregated away and not part of the
> assertion contract; re-point this qualifier at it if that ever lands)."*

The `disease_context_qualifier` is biolink-defined as *"a disease or condition in which
a relationship expressed in an association took place."* It is meaningful only when the
context disease **differs** from the object — e.g., "Drug X treats nausea
[disease_context_qualifier = chemotherapy]". DAKP currently has no column carrying a
distinct disease context (no comorbidity data, no treatment context, FAERS `effects`
is aggregated away).

## Findings: Full Qualifier Landscape on Our Edge Class

Every DAKP edge resolves to `biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation`
(drug subject → disease/phenotype object). Only **5 qualifier slots** are valid on that
class (verified against the installed Tablassert's bundled biolink-model 4.4.3):

| Qualifier slot | Range | Semantics | Usable for DAKP? |
|---|---|---|---|
| `disease_context_qualifier` | CURIE | Disease/condition context | **No** — no distinct context column exists; current use is tautological |
| `anatomical_context_qualifier` | CURIE | Anatomical site (tissue, cell, sub-cellular) | **No** — no anatomical data in any assertion column |
| `object_specialization_qualifier` | CURIE | More specific version of the object (cross-namespace, not a subclass) | **No** — no disease-subtype data |
| `subject_specialization_qualifier` | CURIE | More specific version of the subject (cross-namespace, not a subclass) | **No** — no drug-formulation/variant data |
| `qualifier` | CURIE | Generic grouping slot (rarely useful) | **No** — no meaningful generic qualifier entity |

**43 total qualifier slots** exist in the biolink model, but the other 38 are either:
- On **different association classes** (aspect/direction/form_or_variant/part/process →
  `ChemicalAffectsGeneAssociation` and regulatory classes; frequency/onset/sex/stage →
  other clinical classes) — Tablassert prunes them from our edges via `prune_to_class()`.
- **UNSATISFIABLE** — declared in the LinkML YAML but attached to no Pydantic class
  (e.g., `aspect_qualifier`, `direction_qualifier`, `context_qualifier`, `severity_qualifier`).
  Tablassert rejects these at config load.

Additionally:
- `qualified_predicate` — **NOT on this class** (belongs to `ChemicalAffectsGeneAssociation` etc.)
- `species_context_qualifier` — **auto-derived** by Tablassert from resolved taxon metadata;
  explicitly rejected if declared in a config.

## Critical Constraint: Tablassert Drops Edges on Unresolved Qualifiers

A Tablassert qualifier is a **node encoding** — it goes through fullmap entity
resolution alongside subject and object. The `join_matches` function in
`fullmap.py:476` explicitly drops rows where the resolved column is null:

```python
result = result.filter(pl.col(col).is_not_null())  # fullmap.py:514
```

This applies to **every** column resolved in `resolve_batch`, including qualifier
columns. So:

- **Every declared qualifier must resolve to a CURIE on every row, or the edge is dropped.**
- An empty qualifier cell → no fullmap match → null → edge dropped.
- An unresolved qualifier text → null → edge dropped.

This means the pipeline **cannot accommodate per-edge optional qualifiers** — you
cannot have some `contraindicated_in` edges with a `disease_context_qualifier` and
others without. A qualifier declared at the table level applies to ALL rows, and every
row must carry a resolvable value for it.

This is why DAKP's `_TABLE_QUALIFIERS` only declares qualifiers where the backing
column is **densely populated** (every row has a value). The current FAERS
`disease_context_qualifier` works only because it points at the object column (which
is always populated by definition).

## Analysis: Which Qualifiers SHOULD Be Used (If the Pipeline Allowed It)

### `contraindicated_in` — disease_context_qualifier IS pertinent

Many DailyMed contraindications are **conditional** on a disease context:

> *"Drug X is contraindicated in patients with severe hepatic impairment"*
> → object = hepatic impairment (no additional context needed)
>
> *"Drug X should not be used for the treatment of condition Y in patients with Z"*
> → object = Z (contraindicated condition), disease_context_qualifier = Y (the
> treatment context that triggers the contraindication)

The `disease_context_qualifier` (biolink: "a disease or condition in which a
relationship expressed in an association took place") fits this exactly. Adding it
would make contraindication edges more precise and prevent over-generalization —
without it, *"Drug X contraindicated_in Z"* is ambiguous about WHEN the
contraindication applies.

**But:** DAKP's contraindication extraction mines disease mentions from section text
and pairs them with ingredients as `contraindicated_in` objects. There is **no separate
"indication context" column** extracted — the treatment context that conditionally
triggers the contraindication is not surfaced as a distinct entity. And even if it
were, the nullable-qualifier constraint means edges without a context would be dropped.

### Comorbid diseases / subtypes — would help, same constraint

A `disease_context_qualifier` carrying a comorbid condition would disambiguate:
*"Drug X contraindicated_in renal impairment [disease_context_qualifier = diabetes]"*
means the contraindication is specific to diabetic patients, not all renal impairment.

Similarly, `object_specialization_qualifier` could narrow a generic object:
*"Drug X treats pain [object_specialization_qualifier = neuropathic pain]"*. But
DAKP has no disease-subtype data, and the nullable constraint still applies.

### `applied_to_treat` — the current qualifier is tautological

The FAERS `effects` column (adverse reaction PTs) IS extracted into `cases.parquet`
(`FAERS_CASES_COLUMNS` includes it) but is **not propagated** to the assertion
contract. Even if it were, FAERS effects are adverse events, not disease contexts —
using them as `disease_context_qualifier` would be semantically wrong. The current
qualifier pointing at the object adds no disambiguation.

## Recommendation

### Step 1 (immediate): Remove the tautological FAERS `disease_context_qualifier` — DONE

It encodes the object column, adding no information and potentially confusing
downstream consumers. The `applied_to_treat` predicate + `knowledge_level: observation`
express the semantics without it.

Landed: `_TABLE_QUALIFIERS` is now empty for every table, and no config emits a
`qualifiers` block. The generator still supports column-encoded qualifiers (covered by
`test_declared_qualifier_emits_a_column_encoding`) for Step 3.

**Files to modify:**
- `src/dakp_pipeline/tablassert.py`: set `_TABLE_QUALIFIERS["faers_applied_to_treat_assertions"] = ()`, update the docstring comment
- `tests/unit/test_tablassert_configs.py`: update `EXPECTED_QUALIFIERS` fixture
- Regenerate `tables/faers_applied_to_treat.yaml` (drop the `qualifiers` block)

### Step 2 (enabler, Tablassert-side): Support nullable qualifiers in Tablassert

The key blocker for meaningful `disease_context_qualifier` on contraindications
(and any other optional qualifier) is that Tablassert's `join_matches` drops edges
when a qualifier value is null/unresolved. A Tablassert enhancement to **skip the
filter for qualifier columns** (null qualifier → omit the qualifier, keep the edge)
would unblock per-edge optional qualifiers without edge loss.

This is a change in `../Tablassert/src/tablassert/fullmap.py` — the `join_matches`
function would need a `drop_unresolved: bool = True` parameter (default True for
subject/object, False for qualifiers).

### Step 3 (future, DAKP-side): Extract disease context for contraindications

Once nullable qualifiers are supported, extract a `disease_context` column from the
DailyMed contraindication/indication text and back a `disease_context_qualifier` on
the `contraindication_assertions` table. The indication section (LOINC 34067-9) for
the same SPL set provides the treatment context that conditions many contraindications.

## Verification

After Step 1:
```bash
uv run pytest tests/unit/test_tablassert_configs.py tests/unit/test_assertions_observed_uses.py -v
```
Confirm `disease_context_qualifier` disappears from `tables/faers_applied_to_treat.yaml`.
