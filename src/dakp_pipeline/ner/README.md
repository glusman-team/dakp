# NER layer — one composite backend, mentions only

DAKP extracts disease/phenotype **mentions** (text spans + entity type) from DailyMed SPL
"Contraindications" sections (LOINC `34070-3`) and FAERS indication strings. There is **one**
NER backend (`ner.py`, `DiseaseNER`) with **one** entry point — no pluggable backend selector.
DAKP never resolves terms to ontology CURIEs; ontology mapping is exclusively Tablassert's job
(fullmap/BABEL at `tablassert build-kg`). Assertion tables carry mention **text**; Tablassert
resolves the CURIEs.

## The settled composite (see `BENCHMARK.md`)

Benchmarked on a hand-labeled fixture (27 cases / 35 gold spans, `tests/eval/`):

| approach  | precision | recall | F1    | notes                                   |
| --------- | --------- | ------ | ----- | --------------------------------------- |
| gazetteer | **1.000** | 0.914  | **0.955** | deterministic; no heavy deps; FN = 3 rare OOV |
| gliner    | 0.864     | 0.543  | 0.667 | zero-shot; catches all 3 OOV exactly     |
| scispacy  | 0.571     | 0.457  | 0.508 | dropped: no phenotype label, coarse spans |

* **Offline mode (default):** curated gazetteer + deterministic lexical matcher. Precision
  1.000 / F1 0.955, zero heavy deps, fully deterministic. Used by tests + offline runs.
* **Production mode (`offline=False`):** the same gazetteer anchors high-precision spans and
  GLiNER zero-shot (`urchade/gliner_small-v2.1`) fills out-of-gazetteer gaps (gazetteer wins on
  overlap) → near-perfect recall at gazetteer precision. GLiNER is a core, lazy-imported dependency.

## Modules

- `ner.py` — the single `DiseaseNER` backend + `extract_disease_mentions` /
  `extract_contraindication_diseases` + the curated `EMBEDDED_GAZETTEER`.
- `dictionary.py` — normalization (`normalize_text` / `normalize_with_map`) + the
  span-detection `Gazetteer` (term → type; **no** CURIE/name/category).
- `lexical.py` — the deterministic `LexicalMatcher` + `Mention` (text span + type only).
- `candidates.py` — unique mention-string inventory emission (`mention_candidates.tsv`).
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
- Lazy imports for the model; laptop-safe (small model, cached).
- `loguru` for logging; deterministic offline mode; no absolute paths.
- Mentions are text + type only; ontology CURIE resolution is Tablassert-only.
