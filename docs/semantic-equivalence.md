# Semantic equivalence: NEW DAKP vs OLD DAKP

The rebuild replaces a collection of legacy Perl/Python scripts
([`ref/legacy/`](../ref/legacy)) with a reproducible pipeline. This document is the
explicit accounting of what is **preserved** (the knowledge semantics the old KP established
and the Translator program relies on) versus what is **deliberately improved** (better
coverage, provenance, and engineering — never a silent change in meaning).

The claims here are enforced, not aspirational: [`tests/integration/test_semantic_equivalence.py`](../tests/integration/test_semantic_equivalence.py)
runs the real (mocked-source) pipeline and asserts every "preserved" row below, and the
legacy-informed guardrail [`translator/regression.py`](../src/dakp_pipeline/translator/regression.py)
re-checks the family/provenance/label invariants on every build. The Translator contract is
cross-checked against the DINGO reference ingest
(`../DINGO/src/translator_ingest/ingests/dakp/dakp_rig.yaml`).

Legacy sources of truth referenced below:

- [`ref/legacy/bin/drug2indi2kg.py`](../ref/legacy/bin/drug2indi2kg.py) — NDA-backed `treats` + `applied_to_treat`
- [`ref/legacy/bin/uselist2kg.py`](../ref/legacy/bin/uselist2kg.py) — FAERS-use `applied_to_treat` + `treats`
- [`ref/legacy/bin/dakp-postprocess2jsonlBL.py`](../ref/legacy/bin/dakp-postprocess2jsonlBL.py) — Biolink/Translator postprocess (provenance, `clinical_approval_status`, edge ids)
- [`ref/legacy/matrix/bin/contraindications2kg.py`](../ref/legacy/matrix/bin/contraindications2kg.py) — MEDI/Matrix `contraindicated_in`

## Preserved semantics

### The three edge families

| Predicate | OLD DAKP | NEW DAKP | Status |
| --- | --- | --- | --- |
| `biolink:treats` | FDA-approved drug→condition (`drug2indi2kg` / `uselist2kg`, gated on SPL support) | `approved_treats_assertions` (NDA → Drugs@FDA ingredient → DailyMed SPL approval → indications-and-usage section `34067-9`) | **Preserved** — same three-part approval+support gate |
| `biolink:applied_to_treat` | FAERS-observed drug→condition use, no approval claim | `faers_applied_to_treat_assertions` (FAERS cases aggregated by drugname × indication) | **Preserved** |
| `biolink:contraindicated_in` | drug→condition contraindications | `contraindication_assertions` (mined from DailyMed SPL contraindication sections `34070-3`) | **Preserved family; improved source** (see [Improvements](#improvements)) |

### Subject and object categories

| | OLD DAKP | NEW DAKP | Status |
| --- | --- | --- | --- |
| Subject categories | `interventionCategories` = ChemicalEntity / SmallMolecule / Drug / MolecularMixture / ComplexMolecularMixture (emitted as `biolink:ChemicalEntity`) | `contract.CHEMICAL_DRUG_CATEGORIES` = ChemicalEntity / SmallMolecule / MolecularMixture / ComplexMolecularMixture / Drug; assertion rows carry `ChemicalEntity`, fullmap refines | **Preserved** (matches DINGO `edge_type_info.subject_categories`) |
| Object categories | `conditionCategories` = Disease / PhenotypicFeature | `contract.DISEASE_PHENOTYPE_CATEGORIES` = Disease / PhenotypicFeature / DiseaseOrPhenotypicFeature | **Preserved** (matches DINGO `edge_type_info.object_categories`) |
| Subject identifiers | UNII (from DailyMed active ingredients) | UNII where DailyMed provides it (source-provided, not DAKP-mapped) | **Preserved** |
| Object identifiers | MONDO / HP via BABEL | MONDO / HP via Tablassert/fullmap (DINGO `node_type_info`) | **Preserved**, mapping delegated (see [Improvements](#improvements)) |
| Edge categories | `EntityToDiseaseAssociation` / `EntityToPhenotypicFeatureAssociation` (from object category) | same derivation, emitted by Tablassert from the object category | **Preserved** |

### Provenance

Every family aggregates under `infores:multiomics-drugapprovals` as the owning KP, exactly as
the legacy `sources` blocks and the DINGO ingest require.

| Family | OLD DAKP upstream chain | NEW DAKP `upstream_resource_ids` | Status |
| --- | --- | --- | --- |
| `treats` | primary `multiomics-drugapprovals`; upstream `dailymed` + `faers` | `infores:dailymed\|infores:faers` | **Preserved** |
| `applied_to_treat` | aggregator `multiomics-drugapprovals`; primary `faers`; supporting `dailymed` | `infores:faers\|infores:dailymed` | **Preserved** |
| `contraindicated_in` | aggregator `multiomics-drugapprovals`; primary `medi`; supporting `dailymed` | `infores:dailymed` | **Changed for the better** — see [Improvements](#improvements) |

### `clinical_approval_status` logic

| Family | OLD DAKP | NEW DAKP | Status |
| --- | --- | --- | --- |
| `treats` | `approved_for_condition` (set in `postProcessEdges` when the subject is approved for the object) | `approved_for_condition` | **Preserved** |
| `applied_to_treat` | `off_label_use` (heuristic, unless `approved_for_condition` overrode it) | `observed_use` (the preserved FAERS label) | **Deliberate refinement** — see [Deliberate refinements](#deliberate-refinements) |
| `contraindicated_in` | unset (commented out in legacy postprocess) | unset at the assertion layer (not a treatment claim) | **Preserved** |

### Evidence fields

| Evidence | OLD DAKP field | NEW DAKP column | Status |
| --- | --- | --- | --- |
| FDA approval / NDA | `approval` / `approvals` (e.g. `NDA012345`) | `approval_ids` (on `treats`) | **Preserved** |
| FAERS case counts | `N_cases` (kept on `applied_to_treat`, dropped on `treats`) | `case_count` (distinct FAERS `primaryid`, on `applied_to_treat`) | **Preserved** |
| SPL support | `supporting_spls` → `has_evidence` (`dailymed:<setid>`) | `supporting_spl_sets` + `supporting_spl_documents` | **Preserved** (set + document granularity) |
| NER confidence | n/a | `source_score` (max NER span score, on `contraindicated_in`) | **Added** (see [Improvements](#improvements)) |

### Deterministic edge ids

The legacy build deduped edges on `(subject, predicate, object)` and derived a deterministic
UUIDv3 from that triple (`namespace_uuid('drug_approvals_kp', subj, pred, obj)` in
`dakp-postprocess2jsonlBL.py`). The rebuild keeps the **invariant** — assertion rows are unique
per `(subject, predicate, object)` and emitted in a deterministic sorted order, so the output is
byte-for-byte reproducible across runs — and delegates the actual UUID assignment to Tablassert's
deterministic id machinery at `build-kg`. Same stability guarantee; the id minting lives on the
Tablassert side of the delegation boundary. `test_semantic_equivalence.py` asserts both the unique
triple key and byte-identical output across two runs.

## Improvements

These are intentional, documented upgrades. Each preserves the contract above while improving
coverage, provenance, or maintainability.

- **Contraindications are NER-mined from DailyMed, not the MEDI/Matrix xlsx.** The legacy
  `contraindicated_in` edges came from an externally-sourced spreadsheet
  (`contraindicationList-<version>.xlsx`, `infores:medi`). The rebuild mines them **directly**
  from DailyMed SPL "Contraindications" sections (LOINC `34070-3`) with the NER backend, pairing
  each disease/phenotype mention with the SPL's active ingredients. This gives better coverage
  (every labeled contraindication, not a periodically-exported list) and DailyMed-grounded
  provenance (`upstream = infores:dailymed`; no third-party `infores:medi` hop). See
  [`assertions/contraindications.py`](../src/dakp_pipeline/assertions/contraindications.py).
- **One benchmarked NER backend (gazetteer + GLiNER composite).** The legacy build had no
  disease NER (it relied on BABEL lexical lookups). The rebuild ships a single composite
  `DiseaseNER` — a curated gazetteer anchoring high-precision spans plus GLiNER zero-shot filling
  out-of-gazetteer gaps — settled by a labeled benchmark (27 cases / 35 gold spans: composite
  precision 0.972 / recall 1.000 / F1 0.986). One backend, one entry point, an offline
  (deterministic, dep-free) vs production (`[ner]` extra) mode toggle — not a pluggable backend
  enum. See [`ner/README.md`](../src/dakp_pipeline/ner/README.md) and [`ner/BENCHMARK.md`](../src/dakp_pipeline/ner/BENCHMARK.md).
- **Ontology mapping delegated entirely to Tablassert/fullmap.** The legacy build resolved terms
  to CURIEs inline via a hardcoded BABEL.db (`/ssd2/sqlite/BABEL.db`). The rebuild emits mention
  **text** (plus source-provided UNII) and lets Tablassert/fullmap do canonical CURIE/name/category
  resolution at `build-kg`. This removes a hardcoded local path, a brittle dependency, and a whole
  class of mapping drift; contraindication objects carry empty `object_curie`/`object_category`
  *by design* for fullmap to fill.
- **Go-fast extraction with byte-parity.** The heavy DailyMed/FAERS/Drugs@FDA parsers have Go
  ports (`go/internal/{dailymed,faers,drugsfda}`) that are **byte-for-byte identical** to the
  Python extractors (golden-file parity tests in `go test ./...`), so production can run the
  native workers without changing a single output byte. See [`go/README.md`](../go/README.md).
- **100% branch coverage + zero dead code.** The Python suite runs at 100.00% branch coverage
  (`uv run pytest --cov`, `fail_under = 100`) with ruff lint + format and pyright all clean — the
  legacy scripts had no tests. There is no unreachable code path.
- **Airflow orchestration + real download tasks.** The legacy build was hand-driven shell/Perl.
  The rebuild has an import-safe TaskFlow DAG (`dags/dakp_build.py`) over the same pure-Python
  stage functions, with real stdlib-HTTP downloaders (content-addressed, idempotent) for
  DailyMed/Drugs@FDA/FAERS and a bounded `prod` smoke path exercised offline in CI.
- **Tablassert 8.0.0 from PyPI.** KGX compilation uses the published `tablassert` package (the
  `[kg]` / `[kg-qc]` extras) rather than a local editable checkout, so the build is reproducible
  from a pinned version. See [`tablassert-handoff.md`](./tablassert-handoff.md).

## Deliberate refinements

Two semantic values differ from the legacy build on purpose. Both are locked by tests and noted
here so the difference is never mistaken for a regression.

- **`applied_to_treat` uses the FAERS label `observed_use` / `statistical_association`, not the
  legacy heuristic `off_label_use` / `observation`.** The legacy postprocess *inferred*
  `off_label_use` for any non-approved use, which overstates the claim (off-label implies approved
  for *something else*; not-approved implies no approval at all — the legacy comment itself flags
  the ambiguity). The rebuild preserves the FAERS-derived label verbatim: these are real-world
  usage observations (`knowledge_level = statistical_association`,
  `clinical_approval_status = observed_use`), making no approval claim either way. The predicate,
  categories, provenance chain, and case-count evidence are unchanged.
- **`contraindicated_in` cites `infores:dailymed` (not `infores:medi`) and uses
  `agent_type = text_mining_agent`.** A direct consequence of mining contraindications from DailyMed
  (above): the upstream source *is* DailyMed, and the agent is a text-mining agent rather than the
  legacy `manual_validation_of_automated_agent` / `text_mining_assisted`. This matches the DINGO
  RIG, which lists `text_mining_agent` for `contraindicated_in`.

## How to verify

```bash
uv run pytest tests/integration/test_semantic_equivalence.py -q   # semantic-preservation suite
uv run pytest -q                                                  # full suite (incl. regression guardrail)
```

The DINGO contract checks keep local constants mirrored from `../DINGO/.../dakp_rig.yaml` and
assert the DAKP `contract.py` category tuples, predicates, and upstream infores chains match the
reference ingest. DAKP no longer ships its own RIG generator; RIG compilation stays on the
Tablassert side of the delegation boundary.

## Related

- [`architecture.md`](./architecture.md) — the layered pipeline and the Tablassert delegation boundary.
- [`sources.md`](./sources.md) — per-source acquisition/extraction, including DailyMed-NER contraindications.
- [`tablassert-handoff.md`](./tablassert-handoff.md) — the generated configs and provenance overrides.
- [`wenceslaus-runbook.md`](./wenceslaus-runbook.md) — the full production build (fullmap + prod KG).
