"""Pluggable biomedical NER backends for disease/phenotype mention extraction.

This is the NER backend layer that lets DAKP mine contraindications **directly** from
DailyMed SPL "Contraindications" sections (LOINC ``34070-3``) using state-of-the-art
biomedical NER (PLAN.md "Phase 4: NER / entity resolution strategy", re-scoped to drop
MEDI/Matrix). A later worker feeds :func:`extract_contraindication_diseases` output into the
contraindication assertion builder.

Backends
--------
Every backend satisfies the :class:`NERBackend` protocol::

    extract(text: str, types: Sequence[str]) -> list[EntitySpan]

* :class:`MockNERBackend` — deterministic, fixture-driven (a fixed vocabulary of surface
  forms -> types). No heavy deps; the stand-in for pipeline/builder tests.
* :class:`DictionaryNERBackend` — wraps the deterministic MONDO/HPO dictionary baseline
  (:mod:`dakp_pipeline.ner.dictionary` + :mod:`dakp_pipeline.ner.lexical`). No heavy deps.
* :class:`GLiNERBackend` — zero-shot SOTA NER via the ``gliner`` library (small model).
  **Lazy:** ``gliner`` is imported only on first use.
* :class:`SciSpacyBackend` — classic biomedical NER via ``scispacy`` (``en_ner_bc5cdr_md``).
  **Lazy:** ``spacy``/``scispacy`` imported only on first use.

Select a backend by config string with :func:`get_backend`
(``"mock" | "dictionary" | "gliner" | "scispacy"``). **Constructing** a real backend never
imports its heavy deps; only the first :meth:`extract` (via ``_load``) does, and a missing
dep raises :class:`~dakp_pipeline.ner.model_cache.NERDependencyError` with a clear
"install the [ner] extra" message. So ``import dakp_pipeline.ner.backends`` — and the whole
base install + test suite — works with the ``[ner]`` extra NOT installed.

``types`` are canonical entity-type labels (see :func:`canonical_type`); a backend returns
only spans whose canonical type is requested. An empty ``types`` means "no filter" for the
filtering backends (mock/dictionary/scispacy); GLiNER, which needs explicit zero-shot labels,
falls back to :data:`CONTRAINDICATION_DISEASE_TYPES`.

All offsets are half-open into the original ``text``: ``span.text == text[span.start:span.end]``.
Output is sorted deterministically by ``(start, end, type, text)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl

from dakp_pipeline.ner.dictionary import DictionaryIndex, normalize_text, normalize_with_map
from dakp_pipeline.ner.lexical import LexicalMatcher
from dakp_pipeline.ner.model_cache import NERDependencyError, ensure_model

# --- canonical entity types ----------------------------------------------------

TYPE_DISEASE = "disease"
TYPE_PHENOTYPE = "phenotype"
TYPE_CHEMICAL = "chemical"
TYPE_DRUG = "drug"

# The object types a contraindication assertion targets (disease / phenotype mentions).
CONTRAINDICATION_DISEASE_TYPES: tuple[str, ...] = (TYPE_DISEASE, TYPE_PHENOTYPE)

# --- backend defaults (all configurable) ---------------------------------------

# Laptop-safe small zero-shot model; override for a larger / biomedical-tuned checkpoint.
DEFAULT_GLINER_MODEL = "urchade/gliner_small-v2.1"
DEFAULT_GLINER_THRESHOLD = 0.5
# SciSpacy BC5CDR biomedical model (labels: DISEASE, CHEMICAL).
DEFAULT_SCISPACY_MODEL = "en_ner_bc5cdr_md"

# Raw label -> canonical type. Unknown labels canonicalize to their lowercased form.
_TYPE_ALIASES: Mapping[str, str] = {
    "disease": TYPE_DISEASE,
    "diseases": TYPE_DISEASE,
    "phenotype": TYPE_PHENOTYPE,
    "phenotypes": TYPE_PHENOTYPE,
    "phenotypicfeature": TYPE_PHENOTYPE,
    "phenotypic_feature": TYPE_PHENOTYPE,
    "chemical": TYPE_CHEMICAL,
    "chemicalentity": TYPE_CHEMICAL,
    "drug": TYPE_DRUG,
    "smallmolecule": TYPE_DRUG,
}


def canonical_type(raw: str) -> str:
    """Canonicalize a raw entity label (e.g. ``"DISEASE"`` -> ``"disease"``)."""
    key = raw.strip().lower()
    return _TYPE_ALIASES.get(key, key)


# --- span + protocol -----------------------------------------------------------


@dataclass(frozen=True)
class EntitySpan:
    """One extracted entity mention.

    ``start``/``end`` are half-open offsets into the source text such that
    ``text == source[start:end]``; ``type`` is a canonical label; ``score`` is the backend's
    confidence (deterministic backends use a fixed score).
    """

    text: str
    start: int
    end: int
    type: str
    score: float


@runtime_checkable
class NERBackend(Protocol):
    """A disease/phenotype mention extractor.

    ``extract`` returns spans for the requested canonical ``types`` (empty = no filter for
    filtering backends), deterministically ordered by ``(start, end, type, text)``.
    """

    def extract(self, text: str, types: Sequence[str]) -> list[EntitySpan]: ...


# --- shared helpers ------------------------------------------------------------


def _wanted_types(types: Sequence[str]) -> frozenset[str]:
    return frozenset(canonical_type(t) for t in types)


def _finalize(spans: list[EntitySpan], types: Sequence[str]) -> list[EntitySpan]:
    """Filter spans to the requested canonical types (empty = keep all) and sort."""
    wanted = _wanted_types(types)
    filtered = [span for span in spans if not wanted or span.type in wanted]
    return sorted(filtered, key=_sort_key)


def _sort_key(span: EntitySpan) -> tuple[int, int, str, str]:
    return (span.start, span.end, span.type, span.text)


def _find_word_bounded(haystack: str, needle: str) -> list[int]:
    """All start positions where ``needle`` occurs in normalized ``haystack`` on word
    boundaries (start/end of string or an adjacent space). Normalized text is
    space-separated alphanumerics, so this is a plain substring scan with boundary checks.
    """
    if not needle:
        return []
    positions: list[int] = []
    n, m = len(haystack), len(needle)
    start = 0
    while True:
        at = haystack.find(needle, start)
        if at == -1:
            return positions
        before_ok = at == 0 or haystack[at - 1] == " "
        after_ok = at + m == n or haystack[at + m] == " "
        if before_ok and after_ok:
            positions.append(at)
        start = at + 1


def _overlaps_any(start: int, end: int, covered: list[tuple[int, int]]) -> bool:
    return any(start < cov_end and cov_start < end for cov_start, cov_end in covered)


def _install_message(backend: str, module: str) -> str:
    return f"NER backend {backend!r} requires the optional [ner] extra (missing module: {module}). Install it with: uv sync --extra ner"


# --- deterministic backends (no heavy deps) ------------------------------------


class MockNERBackend:
    """Deterministic, fixture-driven NER backend.

    The "model" is a fixed vocabulary mapping surface forms to entity types; extraction finds
    word-bounded, case/punctuation-insensitive occurrences and reports true offsets into the
    original text (reusing :func:`dakp_pipeline.ner.dictionary.normalize_with_map`). Greedy
    longest-phrase-first matching keeps a shorter term from matching inside a longer one.
    Fully deterministic: fixed scores, stable output order. No heavy deps.
    """

    def __init__(self, entities: Mapping[str, str] | None = None, *, score: float = 1.0) -> None:
        self._score = score
        by_norm: dict[str, str] = {}
        for surface, etype in (entities or {}).items():
            key = normalize_text(surface)
            if key:
                by_norm[key] = canonical_type(etype)
        self._types = by_norm
        # Longest-first then lexicographic for deterministic greedy matching.
        self._terms = sorted(by_norm, key=lambda term: (-len(term), term))

    @classmethod
    def from_tsv(cls, path: Path, *, text_col: str = "text", type_col: str = "type", score: float = 1.0) -> MockNERBackend:
        """Build a mock backend from a ``text``/``type`` TSV fixture."""
        frame = pl.read_csv(path, separator="\t")
        entities = {str(row.get(text_col) or ""): str(row.get(type_col) or "") for row in frame.iter_rows(named=True)}
        return cls(entities, score=score)

    def extract(self, text: str, types: Sequence[str]) -> list[EntitySpan]:
        if not text or not text.strip():
            return []
        normalized, index_map = normalize_with_map(text)
        spans: list[EntitySpan] = []
        covered: list[tuple[int, int]] = []
        for term in self._terms:
            for start in _find_word_bounded(normalized, term):
                end = start + len(term)
                if _overlaps_any(start, end, covered):
                    continue
                covered.append((start, end))
                orig_start = index_map[start]
                orig_end = index_map[end - 1] + 1
                spans.append(EntitySpan(text=text[orig_start:orig_end], start=orig_start, end=orig_end, type=self._types[term], score=self._score))
        return _finalize(spans, types)


class DictionaryNERBackend:
    """NER backend over the deterministic MONDO/HPO dictionary baseline.

    Wraps :class:`~dakp_pipeline.ner.lexical.LexicalMatcher` and projects each
    :class:`~dakp_pipeline.ner.lexical.Mention` into an :class:`EntitySpan` whose ``type`` is
    the mention's semantic group (``disease`` / ``phenotype`` / ``drug``) and whose ``score``
    is the lexical match score. Inherits the ignore-list and synonym behavior of the matcher.
    No heavy deps.
    """

    def __init__(self, dictionary: DictionaryIndex, *, ignore_terms: Sequence[str] | None = None, synonyms: Mapping[str, str] | None = None) -> None:
        self._matcher = LexicalMatcher(dictionary, ignore_terms=ignore_terms, synonyms=synonyms)

    @classmethod
    def from_tsv(cls, path: Path, **kwargs: Any) -> DictionaryNERBackend:
        """Build a backend from an ontology term TSV (see :meth:`DictionaryIndex.from_tsv`)."""
        return cls(DictionaryIndex.from_tsv(path), **kwargs)

    def extract(self, text: str, types: Sequence[str]) -> list[EntitySpan]:
        spans = [
            EntitySpan(
                text=mention.mention_text,
                start=mention.mention_start,
                end=mention.mention_end,
                type=canonical_type(mention.semantic_group),
                score=mention.score,
            )
            for mention in self._matcher.match(text)
        ]
        return _finalize(spans, types)


# --- real SOTA backends (lazy heavy imports) -----------------------------------


class GLiNERBackend:
    """Zero-shot SOTA NER via the ``gliner`` library (lazy import).

    GLiNER extracts arbitrary entity *labels* zero-shot, so the requested canonical ``types``
    are passed directly as labels (defaulting to :data:`CONTRAINDICATION_DISEASE_TYPES` when
    none are given). The default model is a small, laptop-safe checkpoint; pass ``model_id``
    to use a larger or biomedical-tuned model. Weights are fetched/cached idempotently via
    :func:`dakp_pipeline.ner.model_cache.ensure_model`.

    ``gliner`` (and its torch/transformers stack) is imported only on first :meth:`extract`;
    if it is missing, :class:`~dakp_pipeline.ner.model_cache.NERDependencyError` is raised
    with the install command. Construction is always import-free.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_GLINER_MODEL,
        *,
        threshold: float = DEFAULT_GLINER_THRESHOLD,
        cache_dir: Path | str | None = None,
        workdir: Path | str | None = None,
    ) -> None:
        self._model_id = model_id
        self._threshold = threshold
        self._cache_dir = cache_dir
        self._workdir = workdir
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from gliner import GLiNER  # lazy: only when the [ner] extra is installed  # type: ignore[import-not-found]
            except ImportError as exc:
                raise NERDependencyError(_install_message("gliner", "gliner")) from exc
            ref = ensure_model(self._model_id, cache_dir=self._cache_dir, workdir=self._workdir)
            self._model = GLiNER.from_pretrained(str(ref.path))
        return self._model

    def extract(self, text: str, types: Sequence[str]) -> list[EntitySpan]:
        labels = sorted(_wanted_types(types)) or list(CONTRAINDICATION_DISEASE_TYPES)
        model = self._load()
        raw = model.predict_entities(text, labels, threshold=self._threshold)
        spans = [
            EntitySpan(
                text=str(entity["text"]),
                start=int(entity["start"]),
                end=int(entity["end"]),
                type=canonical_type(str(entity["label"])),
                score=float(entity["score"]),
            )
            for entity in raw
        ]
        return _finalize(spans, types)


class SciSpacyBackend:
    """Biomedical NER via ``scispacy`` (lazy import), default ``en_ner_bc5cdr_md``.

    The BC5CDR model labels entities ``DISEASE`` and ``CHEMICAL``; labels are canonicalized
    (``DISEASE`` -> ``disease``). BC5CDR does not model phenotypes — use GLiNER or the
    dictionary backend for phenotype coverage. The model is an installed spaCy model package
    (see ``ner/README.md``); ``spacy``/``scispacy`` are imported only on first
    :meth:`extract`, raising :class:`~dakp_pipeline.ner.model_cache.NERDependencyError` if
    absent. Construction is always import-free.
    """

    def __init__(self, model_id: str = DEFAULT_SCISPACY_MODEL) -> None:
        self._model_id = model_id
        self._nlp: Any = None

    def _load(self) -> Any:
        if self._nlp is None:
            try:
                import spacy  # lazy: only when the [ner] extra is installed  # type: ignore[import-not-found]
            except ImportError as exc:
                raise NERDependencyError(_install_message("scispacy", "spacy")) from exc
            self._nlp = spacy.load(self._model_id)
        return self._nlp

    def extract(self, text: str, types: Sequence[str]) -> list[EntitySpan]:
        nlp = self._load()
        doc = nlp(text)
        spans = [EntitySpan(text=ent.text, start=ent.start_char, end=ent.end_char, type=canonical_type(ent.label_), score=1.0) for ent in doc.ents]
        return _finalize(spans, types)


# --- selection factory + high-level helper -------------------------------------

BACKEND_NAMES: tuple[str, ...] = ("mock", "dictionary", "gliner", "scispacy")


def get_backend(name: str, **kwargs: Any) -> NERBackend:
    """Build a backend by config string (``ner_backend``): mock|dictionary|gliner|scispacy.

    ``kwargs`` are forwarded to the backend constructor. Constructing a real backend does NOT
    import its heavy deps — only the first ``extract`` does. Raises :class:`ValueError` for an
    unknown name.
    """
    key = name.strip().lower()
    if key == "mock":
        return MockNERBackend(**kwargs)
    if key == "dictionary":
        return DictionaryNERBackend(**kwargs)
    if key == "gliner":
        return GLiNERBackend(**kwargs)
    if key == "scispacy":
        return SciSpacyBackend(**kwargs)
    msg = f"unknown ner_backend {name!r}; expected one of {', '.join(BACKEND_NAMES)}"
    raise ValueError(msg)


def extract_contraindication_diseases(section_text: str, backend: NERBackend) -> list[EntitySpan]:
    """Extract disease/phenotype mentions from a contraindication section.

    Runs ``backend`` for the canonical contraindication object types
    (:data:`CONTRAINDICATION_DISEASE_TYPES`). A later worker feeds these spans into the
    contraindication assertion builder (mined from DailyMed SPL LOINC ``34070-3`` text).
    """
    return backend.extract(section_text, CONTRAINDICATION_DISEASE_TYPES)


__all__ = [
    "BACKEND_NAMES",
    "CONTRAINDICATION_DISEASE_TYPES",
    "DEFAULT_GLINER_MODEL",
    "DEFAULT_GLINER_THRESHOLD",
    "DEFAULT_SCISPACY_MODEL",
    "TYPE_CHEMICAL",
    "TYPE_DISEASE",
    "TYPE_DRUG",
    "TYPE_PHENOTYPE",
    "DictionaryNERBackend",
    "EntitySpan",
    "GLiNERBackend",
    "MockNERBackend",
    "NERBackend",
    "NERDependencyError",
    "SciSpacyBackend",
    "canonical_type",
    "extract_contraindication_diseases",
    "get_backend",
]
