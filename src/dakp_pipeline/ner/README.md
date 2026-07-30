# NER backend layer

Pluggable biomedical NER for disease/phenotype mention extraction. This layer lets DAKP mine
contraindications **directly** from DailyMed SPL "Contraindications" sections (LOINC
`34070-3`) using state-of-the-art NER (PLAN.md "Phase 4", re-scoped to drop MEDI/Matrix). A
later worker feeds `extract_contraindication_diseases()` output into the contraindication
assertion builder.

## Modules

- `backends.py` — `EntitySpan`, the `NERBackend` protocol, four backends, the `get_backend`
  factory, and the `extract_contraindication_diseases` helper.
- `model_cache.py` — idempotent model download + cache with a BLAKE3 provenance manifest.
- `dictionary.py` / `lexical.py` / `candidates.py` / `mapping.py` — the deterministic
  dictionary baseline + mention-candidate emission (Milestone 4).

## Backends

Every backend satisfies:

```python
class NERBackend(Protocol):
    def extract(self, text: str, types: Sequence[str]) -> list[EntitySpan]: ...
```

`EntitySpan` has `text` / `start` / `end` / `type` / `score`; offsets are half-open so
`span.text == text[span.start:span.end]`. `types` are canonical labels
(`disease` / `phenotype` / `chemical` / `drug`; see `canonical_type`); a backend returns only
requested types (empty `types` = no filter for the filtering backends).

| Backend              | Config string  | Heavy deps | Notes                                                        |
| -------------------- | -------------- | ---------- | ------------------------------------------------------------ |
| `MockNERBackend`     | `"mock"`       | none       | Deterministic, fixture-driven vocabulary. For tests.         |
| `DictionaryNERBackend` | `"dictionary"` | none     | Wraps the MONDO/HPO `LexicalMatcher` baseline. Deterministic.|
| `GLiNERBackend`      | `"gliner"`     | `gliner`   | Zero-shot SOTA; small model; **lazy import**.                |
| `SciSpacyBackend`    | `"scispacy"`   | `spacy`/`scispacy` | BC5CDR biomedical (`DISEASE`/`CHEMICAL`); **lazy import**. |

Select via config string:

```python
from dakp_pipeline.ner.backends import get_backend, extract_contraindication_diseases

backend = get_backend("dictionary", dictionary=my_index)  # or "mock" | "gliner" | "scispacy"
spans = extract_contraindication_diseases(section_text, backend)
```

## Lazy imports & the `[ner]` extra (important)

The base install and the **entire test suite run WITHOUT any NER deps.** `import
dakp_pipeline.ner.backends` never imports `gliner` / `spacy` / `scispacy` / `huggingface_hub`.
Those are imported only on a real backend's first `extract()` (via `_load`). If a dep is
missing, the backend raises `NERDependencyError` (an `ImportError`) with:

> NER backend 'gliner' requires the optional [ner] extra (missing module: gliner). Install it with: uv sync --extra ner

Install the optional extra to use the real backends:

```bash
uv sync --extra ner
```

The extra is intentionally heavy (pulls torch/transformers/spacy). It is declared in
`pyproject.toml` under `[project.optional-dependencies] ner`; NER deps are **never** in the
base `dependencies`.

### SciSpacy model

`scispacy` needs a spaCy model package installed separately (not on PyPI as a normal dep):

```bash
uv run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

BC5CDR labels `DISEASE` and `CHEMICAL` only (no phenotype) — use GLiNER or the dictionary
backend for phenotype coverage.

### GLiNER model

Default model: `urchade/gliner_small-v2.1` (small, laptop-safe, CPU-feasible). Override with
`GLiNERBackend(model_id=...)` for a larger/biomedical-tuned checkpoint. Weights are fetched
once and cached by `model_cache.ensure_model`.

## Model cache (`model_cache.py`)

`ensure_model(model_id, *, cache_dir=None, workdir=None, downloader=None, force=False)` is
idempotent: it downloads a model exactly once and reuses it **by BLAKE3 tree hash**.

- Cache dir: `<workdir>/models` when a workdir is given, else `$XDG_CACHE_HOME/dakp/models`
  (`~/.cache/dakp/models`). No absolute paths in code.
- Layout: `<cache>/<source>/<model_id with '/' -> '--'>/{manifest.json, content/}`.
- `manifest.json` records `schema_version`, `model_id`, `source`, `b3` (BLAKE3 tree hash of
  `content/`), and `retrieved_at`.
- The downloader is dependency-injected; the default uses `huggingface_hub` (lazy import).
  Tests pass a fake downloader to exercise idempotency/manifests/drift with no network.
- `verify=True` (default) re-hashes `content/` on a would-be hit and re-downloads if it
  drifted; `force=True` re-downloads unconditionally.

## Conventions

- Minimal base deps; NER deps only in the optional `[ner]` extra.
- Lazy imports for heavy models; laptop-safe (small models, cached, bounded).
- `loguru` for logging; deterministic mock/dictionary backends; no absolute paths.
