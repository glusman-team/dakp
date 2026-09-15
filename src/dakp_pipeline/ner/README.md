# NER layer — one composite backend, DailyMed mentions only

DAKP extracts disease/phenotype **mentions** (text spans + entity type) from DailyMed SPL
sections. FAERS observed-use rows bypass NER and pass raw drug names to downstream intervention
mapping. There is **one**
NER backend (`ner.py`, `DiseaseNER`) with **one** entry point — no pluggable backend selector.
DAKP never resolves terms to ontology CURIEs; ontology mapping is exclusively Tablassert's job
(fullmap/BABEL at `tablassert build-kg`). Assertion tables carry mention **text**; Tablassert
resolves the CURIEs.

## Consumers

- `assertions/contraindications.py` — three-pass mining: contraindication sections (LOINC
  `34070-3`), keyword-filtered indication sections (`34067-9`), and boxed-warning /
  warnings-and-precautions sections (`34066-1`, `43685-7`, `34071-1`, `42232-9`; hard-trigger-only
  acceptance).
- `assertions/approved_treats.py` — mines indication sections once per document and uses the
  mentions as a corroboration channel for `treats` candidates the dictionary matcher missed
  (recall for label prose that names a more specific condition than the FDA indication string).
- `assertions/observed_uses.py` — aggregates FAERS indication strings using the lexical disease
  baseline and passes raw FAERS drug names through for downstream intervention mapping; it does
  not invoke NER.
- `assertions/ner_dispatch.py` — the shared plumbing for these consumers: `default_ner`, GPU
  device resolution, and multi-pass multi-GPU dispatch (`mine_passes_multi_gpu`).

## The settled composite (see `BENCHMARK.md`)

Benchmarked on a hand-labeled fixture (34 cases / 42 gold spans, `tests/eval/`):

| approach  | precision | recall | F1    | notes                                   |
| --------- | --------- | ------ | ----- | --------------------------------------- |
| gazetteer | 0.949     | 0.881  | 0.914 | deterministic; no heavy deps; FN = 3 rare OOV + 2 qualified diseases |
| gliner    | 0.781     | 0.595  | 0.676 | zero-shot (`gliner_large-v2.5`); catches all 3 OOV exactly |
| composite | **1.000** | **1.000** | **1.000** | **settled backend** (gazetteer + GLiNER merge) |
| scispacy  | 0.571     | 0.457  | 0.508 | dropped: no phenotype label, coarse spans |

> These numbers were measured with older checkpoints (`gliner_large-v2.5`, then the
> `SkyeAv/drug-approvals-gliner-small-v2.1` v1 fine-tune). The production default is now the
> gliner2-native boundary checkpoint `fastino/gliner2.5-base-v1`, prompted with the
> Biolink-category label vocabulary in `MODEL_LABELS` (see below); the v1 fine-tune is not
> loadable by gliner2 (config schema and head layout differ) and re-fine-tuning it for gliner2 is
> a recorded follow-up. The gold fixture's `type` values are the canonical `Mention.type` strings
> (`Disease` / `PhenotypicFeature`) and only the **object** channel is scored — qualifier mentions
> have no gold span by policy. Re-run `tests/eval/benchmark_ner.py` to re-measure.

* **Offline mode (default):** curated gazetteer + deterministic lexical matcher. Zero heavy deps,
  fully deterministic. Used by tests + offline runs. Bounded by its fixed vocabulary: it returns
  the generic head for qualified diseases (`hypertension` for `pulmonary hypertension`).
* **Production mode (`offline=False`):** the same gazetteer anchors high-precision spans and
  a GLiNER2 boundary checkpoint (`fastino/gliner2.5-base-v1`, loaded via
  `gliner2.AutoExtractor`) fills out-of-gazetteer gaps when invoked on DailyMed
  sections. On overlap the **most specific span wins**: a model span that
  strictly contains a gazetteer span supersedes it (`pulmonary hypertension` over
  `hypertension`), taking the model's boundary and the gazetteer's type. Equal spans, partial
  overlaps, and spans covering several gazetteer terms (a conjunction) go to the gazetteer.
  Model spans whose normalized surface is a population descriptor (e.g. `women of childbearing
  potential`) are dropped, leading hedge tokens (`recent`, `a history of`) are trimmed, and spans
  a hard window split cuts across a phrase boundary are re-joined — **object channel only** (see
  below). GLiNER2 is natively **multi-entity** and schema-conditioned: ONE call per window
  carries the whole `MODEL_LABELS` vocabulary and every returned span is routed by its canonical
  type. The gazetteer remains
  the type authority whenever a span contests a gazetteer term, and Tablassert resolves the
  ontology concept downstream. GLiNER2 is a
  core, lazy-imported dependency (`gliner2`, loaded lazily via `AutoExtractor`). GLiNER2 silently truncates inputs past `config.max_len` word tokens (4096 on the
  shipped checkpoint), so long sections (some run to ~3000 words) are predicted in
  sentence-aware, exact-substring windows of ≤ that budget (`chunk_words` kwarg overrides it) and
  span offsets are remapped back into full-text coordinates — no mention past the truncation
  point is lost.

## Label vocabulary and the two channels

`MODEL_LABELS` is a `{label: description}` mapping handed to gliner2 verbatim as `entity_types`
(`MODEL_LABEL_NAMES` is the same vocabulary without descriptions). gliner2 renders each
description into the prompt as `[DESCRIPTION] <label>: <description>`, so the descriptions are
part of the model input, not documentation: they say *what span to draw* with inline positive
exemplars, are ≤220 chars, and are deterministic module constants (the mention-cache fingerprint
folds them). Labels are Biolink category IDs; measured on the shipped checkpoint they fix real
mistypes (plain `disease`/`phenotype` called "pregnancy" a disease at 0.82).

| channel   | model label                                | `Mention.type`                     | `notes`                       |
| --------- | ------------------------------------------ | ---------------------------------- | ----------------------------- |
| object    | `biolink:Disease`                          | `Disease`                          | `exact` / `gliner` / `gliner:extends` |
| object    | `biolink:PhenotypicFeature`                | `PhenotypicFeature`                | `exact` / `gliner` / `gliner:extends` |
| qualifier | `biolink:AnatomicalEntity`                 | `AnatomicalEntity`                 | `gliner:qualifier`            |
| qualifier | `biolink:BiologicalSex`                    | `BiologicalSex`                    | `gliner:qualifier`            |
| qualifier | `biolink:PopulationOfIndividualOrganisms`  | `PopulationOfIndividualOrganisms`  | `gliner:qualifier`            |
| qualifier | `biolink:OrganismTaxon`                    | `OrganismTaxon`                    | `gliner:qualifier`            |
| qualifier | `frequency_qualifier`                      | `frequency_qualifier`              | `gliner:qualifier`            |
| qualifier | `temporal_context_qualifier`               | `temporal_context_qualifier`       | `gliner:qualifier`            |
| qualifier | `temporal_interval_qualifier`              | `temporal_interval_qualifier`      | `gliner:qualifier`            |

* **Object channel** (`OBJECT_TYPES`) runs the full merge above and the use-specific acceptance
  floors.
* **Qualifier channel** (`QUALIFIER_TYPES`) is strip-aligned spans only, accepted at
  `QUALIFIER_ACCEPT_THRESHOLD` (0.5), deduped on `(start, end, type)`. No hedge trimming (it
  would strip the informative head off `history of hypertension`), no gazetteer merge, no
  population filter (it would delete exactly the surfaces the population/sex labels extract).
  Cross-label overlap is allowed, so a qualifier coexists with the maximal object span.
* `disease_context_qualifier` has **no label of its own**: it shares `biolink:Disease` and is
  derived from the object channel by the contraindication shaper.
* `MENTION_TYPES = OBJECT_TYPES + QUALIFIER_TYPES` is closed: a label canonicalizing outside it
  is dropped at the span adapter, so an unmodelled qualifier can never reach a shaper.
* `extract()` / `extract_batch()` therefore return **mixed channels**. Shapers narrow to the
  object channel with `dakp_pipeline.assertions.object_mentions`; qualifier mentions are not
  consumed as objects (attaching them to the objects they qualify is a separate step).

## Confidence and abstention

The backend separates a model-generation floor from use-specific acceptance profiles:

| setting | default | role |
| ---- | ------- | ---- |
| `GLINER_GENERATION_FLOOR` / `threshold` | `0.35` | candidate **generation**, passed to GLiNER |
| `INDICATION_ACCEPT_THRESHOLD` | `0.95` | precision-first acceptance for DailyMed indications |
| `CONTRAINDICATION_ACCEPT_THRESHOLD` | `0.35` | recall-first acceptance for contraindications |
| `STRICT_GAZETTEER_EXTENSION_THRESHOLD` | `0.95` | minimum model score to replace an exact gazetteer anchor with a longer span |

The `0.35` generation floor exists so specificity candidates remain visible to the merge. It is
not the DailyMed indication operating point. The current-model local sweep selected `0.95` for
DailyMed indications:
on the 34-case fixture, the full composite reached precision 1.000 with 39/42 recall. The lower
contraindication point preserves rare/OOV candidates because a missed contraindication is more
harmful; it is intentionally not the same production policy as indications. These values were
measured in a non-deployment checkout and should be revalidated on the deployment machine.

Use `DiseaseNER.for_indications(...)` and `DiseaseNER.for_contraindications(...)` rather than
passing two unexplained numeric defaults. Both profiles also use the strict gazetteer-extension
floor: weak model extensions are discarded and the exact gazetteer span remains, while
high-confidence qualified spans such as `pulmonary hypertension` can replace `hypertension`.
Candidates in `[threshold, accept_threshold)` remain
visible to the merge but are **abstained on** rather than asserted — `extract` returns fewer
mentions, or none at all, instead of emitting something the model is not confident about.

Abstention never downgrades. If a specific span supersedes a gazetteer span and then falls below
the floor, the generic term is **not** resurrected: emitting `hypertension` for text that reads
`pulmonary hypertension` would assert a broader contraindication than the label supports, so the
backend returns nothing. Callers must always handle an empty list. Abstentions are logged at
DEBUG as `ner_abstain` with the surface, score and reason (`below_accept_floor` /
`superseded_unresolved`) — turn DEBUG on to retune the floor.

## Modules

- `ner.py` — the single `DiseaseNER` backend + `extract_disease_mentions` /
  `extract_contraindication_diseases` + the curated `EMBEDDED_GAZETTEER`.
- `dictionary.py` — normalization (`normalize_text` / `normalize_with_map`) + the
  span-detection `Gazetteer` (term → type; **no** CURIE/name/category).
- `lexical.py` — the deterministic `LexicalMatcher` + `Mention` (text span + type only).
- `model_cache.py` — idempotent model download/cache (production mode weights).

## Usage

```python
from dakp_pipeline.ner.ner import DiseaseNER, extract_contraindication_diseases

ner = DiseaseNER()  # offline: deterministic embedded gazetteer
ner = DiseaseNER(offline=False)  # production: gazetteer + GLiNER (lazy-imported)
mentions = extract_contraindication_diseases(section_text, ner)
# Mention.text / .start / .end / .type / .score  — text span + type ONLY, no CURIE
```

`Mention` offsets are half-open: `mention.text == text[mention.start:mention.end]`. Output is
sorted by `(start, end, type, text)`.

## Core deps & lazy imports

The NER dependencies (`gliner2`, `huggingface_hub`) are **core DAKP dependencies** installed by the
single `uv sync` — there is no `[ner]` extra. They are still **lazy-imported**: `import
dakp_pipeline.ner.ner` never imports `gliner2` / `huggingface_hub`; those load only on a
production-mode `DiseaseNER`'s first `extract()`, so module import stays light (no torch at import
time) and the whole test suite runs offline. If a dep is somehow not importable, it raises
`NERDependencyError` (an `ImportError`):

> NER production mode requires the 'gliner2' package (a core DAKP dependency) but it is not importable. Install all dependencies with: uv sync

Reinstall the full runtime to use production mode:

```bash
uv sync
```

The NER deps are intentionally heavy (pull torch/transformers) but are part of the one required
`dependencies` set in `pyproject.toml`. GLiNER weights are fetched once and cached by
`model_cache.ensure_model` (BLAKE3-keyed, idempotent; `<workdir>/models` or
`$XDG_CACHE_HOME/dakp/models`).

## Conventions

- NER deps are core (installed by `uv sync`) but lazy-imported (no torch at module load).
- One backend / one entry point; offline (deterministic) vs production (model) is a mode toggle.
- Prefer the most specific span; abstain rather than assert a low-confidence or over-general one.
- Lazy imports for the model; weights cached once (the shipped fine-tune is a
  deberta-v3-small encoder, ≈ 0.8 GB fp32 — fits comfortably on any build GPU; CPU fallback
  works).
- `loguru` for logging; deterministic offline mode; no absolute paths.
- Mentions are text + type only; ontology CURIE resolution is Tablassert-only.
