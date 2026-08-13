# Plan: Use nullable qualifiers to preserve conditional drug semantics

## Context

Tablassert 10.0.0 now supports `nullable: true` on column-encoded qualifiers. A blank or
unresolvable qualifier keeps the subject/object edge and omits only that qualifier; subject and
object resolution remain strict. This directly enables sparse, per-edge context.

DAKP currently emits no qualifiers: the former FAERS `disease_context_qualifier` was tautological
(it repeated `object_text`) and was removed. Contraindication extraction currently runs disease /
phenotype NER over every dedicated contraindication section, filters indication sections by
contraindication keywords, then aggregates only by `(ingredient, condition)`. It loses the
mention's sentence, polarity, trigger, and any conditional context.

The scope is now **contraindications only**. The goal is precision-first semantic improvement:
retain the existing flat `drug --contraindicated_in--> condition` edge when the label supports it,
and add a qualifier only when the text explicitly supplies one.

## Research findings and semantic policy

- FDA 21 CFR 201.57 defines section 4 Contraindications as situations where the risk clearly
  outweighs benefit, including harm associated with age, sex, **concomitant therapy**, disease
  state, or another condition; it also says to list known hazards, not theoretical possibilities.
- The same regulation distinguishes section 5 Warnings and Precautions: it includes drug/drug
  interactions, avoiding concomitant therapy, and mitigations. Therefore `avoid`, `use with
  caution`, and `not recommended` in a warning context must not automatically become a hard
  `contraindicated_in` assertion.
- Biolink's `disease_context_qualifier` has range **Disease** and means the disease/condition in
  which the association took place. It is appropriate for an explicitly stated patient-condition
  context, not for a companion medication. On DAKP's chemical-to-disease association it is a
  scalar, so an `A and B` conjunction cannot be represented faithfully as two independent
  qualifier edges.
- Tablassert's own nullable-qualifier documentation uses conditional contraindications as the
  motivating sparse-column example. A populated qualifier is fullmap-resolved; an unresolved
  qualifier is retained as an unqualified edge and reported as an unmatched term.

Primary references consulted:

- [21 CFR 201.57](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-201/subpart-B/section-201.57), especially paragraphs (c)(5) and (c)(6), via the eCFR API.
- [Tablassert qualifier configuration](../Tablassert/docs/configuration/table.md#qualifiers) and
  [Tablassert 10.0 changelog](../Tablassert/CHANGELOG.md), nullable-qualifier behavior.
- The installed Biolink schema's `disease context qualifier` definition and
  `chemical entity to disease or phenotypic feature association` examples.

Quality rules derived from these findings:

1. Treat a dedicated `34070-3` section as the strongest source, but still reject explicit negation
   (`none known`, `not contraindicated`, `no contraindications`) and do not infer from mere disease
   co-occurrence.
2. Use only high-precision prohibition cues for `contraindicated_in` (for example, `contraindicated
   in/for`, `do not use in`, `must not use in`). Keep softer warning/precaution language out of
   this edge family unless it is explicitly under a contraindication statement.
3. Keep the contraindicated condition as `object_text`. Populate
   `disease_context_qualifier` only when a separate disease is explicitly linked by language such
   as `for treatment of A in patients with B` / `when used for A in patients with B`; do not attach
   every indication in the same SPL set to every contraindication.
4. Do not encode a co-medication as `disease_context_qualifier`. If the same sentence has an
   independently explicit disease target, retain the coarse disease edge with the exact sentence
   as evidence but leave the qualifier empty; a future drug-drug interaction table can model the
   medication relation with chemical-entity objects. If no disease target is independently stated,
   omit the hard disease edge rather than inventing one.
5. If the text has multiple conjunctive context conditions, ambiguous role assignment, negation,
   or unresolved context, do not manufacture a more specific qualified edge. Preserve the coarse
   base edge only when its disease target and positive prohibition are independently explicit;
   otherwise omit the candidate. Never split `A and B` into two edges that falsely mean either
   condition alone is sufficient.

## Approach

Add a conservative, deterministic context/polarity layer around the existing contraindication NER:

1. Preserve section/document provenance and mention offsets while creating sentence/clause work
   items. Run the existing NER unchanged for disease/phenotype span recall.
2. Classify each candidate mention against its local sentence using explicit contraindication
   trigger templates plus negation/heading guards. Dedicated contraindication sections and
   indication-section Pass 2 continue to be separate evidence paths; Pass 2 remains keyword
   filtered so ordinary indications are never mined as contraindications.
3. Extract at most one unambiguous **disease** context per assertion using explicit relational
   templates. Store blank context for ordinary contraindications and for representable co-medication
   cases where the disease target stands independently. Preserve the source sentence and trigger
   classification in an assertion evidence column so omissions are inspectable rather than silently
   inferred.
4. Change contraindication aggregation from `(subject, object)` to
   `(subject, object, context)`, so unconditional and separately qualified evidence cannot be
   merged. Union provenance and take the maximum score within each semantic key.
5. Add a nullable column-encoded `disease_context_qualifier` to the contraindication Tablassert
   config, constrained to `Disease` (not the object's broader Disease/PhenotypicFeature list), and
   keep subject/object category guards strict. Do not add a qualifier to approved-treats or model
   companion drugs in this change.

## Files to modify

- `src/dakp_pipeline/assertions/contraindications.py` — sentence/span-preserving candidate
  extraction, polarity/trigger/context rules, context-aware aggregation, and evidence text.
- `src/dakp_pipeline/io/schemas.py` — add `disease_context_text` and `evidence_text` to the
  contraindication public assertion contract in a stable position.
- `src/dakp_pipeline/tablassert.py` — declare the sparse `disease_context_qualifier`, set
  `nullable: true`, use a Disease-only qualifier category guard, and map the evidence text.
- `tables/contraindications.yaml` — regenerate from the config generator; do not hand-edit.
- `tests/unit/test_assertions_contraindications.py` — qualified, unqualified, and evidence-output
  behavior using DailyMed-like text.
- `tests/unit/test_assertions_contraindications_edge.py` — negation, soft-warning exclusion,
  conjunction/ambiguity, Pass 2 behavior, aggregation keys, and deterministic output.
- `tests/unit/test_tablassert_configs.py` — expected qualifier encoding, nullable flag, Disease-only
  guard, and new annotation column.
- `tests/integration/test_semantic_equivalence.py` and/or translator contract tests — update the
  uniqueness invariant to include qualifier context and assert the new context semantics.

## Reuse

- `DailyMedEvidence`, `build_dailymed_evidence`, and LOINC-indexed `indication_docs` /
  `contraindication_docs` in `src/dakp_pipeline/assertions/evidence.py`.
- `DiseaseNER.extract()` / `Mention` offsets and `normalize_text()` in
  `src/dakp_pipeline/ner/ner.py` and `src/dakp_pipeline/ner/dictionary.py`; no new NER backend.
- Existing `_split_sentences`, `_contraindication_sentences`, Pass 1/Pass 2 dispatch, `_accumulate`,
  `_finalize_row`, `sorted_pipe`, and deterministic ordering in
  `src/dakp_pipeline/assertions/contraindications.py`.
- `column_letter`, `category_avoid_list`, `_TABLE_QUALIFIERS`, `_TABLE_ANNOTATIONS`, and generated
  YAML parity tests in `src/dakp_pipeline/tablassert.py` / `tests/unit/test_tablassert_configs.py`.
- Tablassert's documented `nullable: true` behavior; no Tablassert-side change is required.

## Steps

- [x] Add a span-preserving sentence/clause representation and retain each mention's local evidence
      text through both mining passes.
- [x] Implement conservative positive-trigger and explicit-negation classification; distinguish
      section-4 contraindications from warning/precaution language.
- [x] Implement explicit disease-context templates and reject medication contexts, unresolved
      roles, multi-condition conjunctions, and inferred cross-section context.
- [x] Add `disease_context_text` and `evidence_text` columns while preserving the existing
      subject/object fields; map `evidence_text` to Biolink `supporting_text` as a multivalue
      annotation with a deterministic separator.
- [x] Aggregate by `(ingredient, condition, context)` and union support only within that key.
- [x] Add the nullable `disease_context_qualifier` config backed by the context column, with
      Disease-only fullmap constraints; map source text to a visible edge annotation.
- [x] Regenerate `tables/contraindications.yaml` and update schema/config/semantic tests.
- [x] Add a small gold-style fixture matrix: direct contraindication; explicit conditional disease;
      blank context; co-medication; `None known`; `not contraindicated`; warning-only `avoid`; and
      `A and B` context.
- [x] Run the fixture matrix through offline NER and the Tablassert smoke/fullmap path, checking
      both qualified and unqualified edges survive.

## Verification

- `uv run pytest tests/unit/test_assertions_contraindications.py tests/unit/test_assertions_contraindications_edge.py tests/unit/test_tablassert_configs.py -v`
- Run semantic/integration contract tests and confirm the public TSV schema fingerprint changes
  intentionally and only for the contraindication table.
- Build a minimal fullmap fixture with a resolvable context and a blank context: both edges must
  survive; only the first carries `disease_context_qualifier`.
- Assert negative and warning-only examples produce no hard contraindication edge, while a dedicated
  section with an explicit positive prohibition produces one.
- Assert co-medication context never appears in `disease_context_qualifier`; its source sentence is
  still available as evidence text.
- Assert `A and B` is not emitted as two independent qualified edges, and that repeated evidence
  for the same `(drug, condition, context)` remains deterministically aggregated.
- Run the existing offline semantic-equivalence and byte-determinism suites; update only the
  expected qualifier-aware key assertions.
