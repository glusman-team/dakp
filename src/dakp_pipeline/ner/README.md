# NER layer — one composite backend, mentions only

DAKP extracts disease/phenotype **mentions** (text spans + entity type) from DailyMed SPL
"Contraindications" sections (LOINC `34070-3`) and FAERS indication strings. There is **one**
NER backend (`ner.py`, `DiseaseNER`) with **one** entry point — no pluggable backend selector.
DAKP never resolves terms to ontology CURIEs; ontology mapping is exclusively Tablassert's job
(fullmap/BABEL at `tablassert build-kg`). Assertion tables carry mention **text**; Tablassert
resolves the CURIEs.

## The settled composite (see `BENCHMARK.md`)

Benchmarked on a hand-labeled fixture (34 cases / 42 gold spans, `tests/eval/`):

| approach  | precision | recall | F1    | notes                                   |
| --------- | --------- | ------ | ----- | --------------------------------------- |
| gazetteer | 0.949     | 0.881  | 0.914 | deterministic; no heavy deps; FN = 3 rare OOV + 2 qualified diseases |
| gliner    | 0.781     | 0.595  | 0.676 | zero-shot (`gliner_large-v2.5`); catches all 3 OOV exactly |
| composite | **1.000** | **1.000** | **1.000** | **settled backend** (gazetteer + GLiNER merge) |
| scispacy  | 0.571     | 0.457  | 0.508 | dropped: no phenotype label, coarse spans |

* **Offline mode (default):** curated gazetteer + deterministic lexical matcher. Zero heavy deps,
  fully deterministic. Used by tests + offline runs. Bounded by its fixed vocabulary: it returns
  the generic head for qualified diseases (`hypertension` for `pulmonary hypertension`).
* **Production mode (`offline=False`):** the same gazetteer anchors high-precision spans and
  GLiNER zero-shot (`gliner-community/gliner_large-v2.5`) fills out-of-gazetteer gaps → perfect
  recall at perfect precision. On overlap the **most specific span wins**: a model span that
  strictly contains a gazetteer span supersedes it (`pulmonary hypertension` over
  `hypertension`), taking the model's boundary and the gazetteer's type. Equal spans, partial
  overlaps, and spans covering several gazetteer terms (a conjunction) go to the gazetteer.
  Model spans whose normalized surface is a population descriptor (e.g. `women of childbearing
  potential`) are dropped, leading hedge tokens (`recent`, `a history of`) are trimmed, and spans
  a hard window split cuts across a phrase boundary are re-joined. GLiNER is natively
  **multi-entity**: one call scores every requested label (disease + phenotype) and returns any
  number of spans per label (up to 30 labels per call in v2.5). GLiNER is a core, lazy-imported
  dependency. GLiNER silently truncates inputs past `config.max_len` word tokens (768 on the
  shipped v2.5 checkpoint), so long sections (some run to ~3000 words) are predicted in
  sentence-aware, exact-substring windows of ≤ that budget (`chunk_words` kwarg overrides it) and
  span offsets are remapped back into full-text coordinates — no mention past the truncation
  point is lost.

## Confidence and abstention

Two thresholds, deliberately split — generate wide, decide narrow:

| knob | default | role |
| ---- | ------- | ---- |
| `threshold` | `0.35` | candidate **generation**, passed to GLiNER |
| `accept_threshold` | `0.50` | DAKP-side **acceptance** floor |

Generating below the acceptance floor is what makes the specificity merge work: a specific span
often scores lower than its generic head, so generating at the floor hid exactly the spans worth
preferring. Candidates in the band between the two are visible to the merge but are **abstained
on** rather than asserted — `extract` returns fewer mentions, or none at all, instead of emitting
something the model is not confident about.

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

The NER dependencies (`gliner`, `huggingface_hub`) are **core DAKP dependencies** installed by the
single `uv sync` — there is no `[ner]` extra. They are still **lazy-imported**: `import
dakp_pipeline.ner.ner` never imports `gliner` / `huggingface_hub`; those load only on a
production-mode `DiseaseNER`'s first `extract()`, so module import stays light (no torch at import
time) and the whole test suite runs offline. If a dep is somehow not importable, it raises
`NERDependencyError` (an `ImportError`):

> NER production mode requires the 'gliner' package (a core DAKP dependency) but it is not importable. Install all dependencies with: uv sync

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
- Lazy imports for the model; weights cached once (`gliner_large-v2.5` ≈ 1.8 GB fp32 fits
  comfortably on a 12 GB GPU; CPU fallback works).
- `loguru` for logging; deterministic offline mode; no absolute paths.
- Mentions are text + type only; ontology CURIE resolution is Tablassert-only.
