"""Normalized dictionary indexes for source-aware mention generation.

This is the *fast dictionary baseline* of PLAN.md "Phase 4: NER / entity resolution
strategy": normalized ontology/alias strings are indexed to candidate CURIE/name/category
records so the lexical matcher can find disease/phenotype/drug spans deterministically.

The index is deliberately source-agnostic. The tiny fixture
``tests/fixtures/pipeline/ontology/disease_map.tsv`` (``text/curie/name/category``) is the
current stand-in, but the same index accepts real MONDO/HPO/fullmap term lists later via
:py:meth:`DictionaryIndex.from_frame` (extra alias columns are indexed to the same CURIE).

Normalization is deterministic and shared with :mod:`dakp_pipeline.ner.lexical`:
lowercase, strip HTML tags, drop possessive ``'s``, fold non-alphanumerics to single
spaces. :func:`normalize_with_map` additionally returns a character-index map so the
lexical matcher can report mention offsets into the *original* (un-normalized) text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# --- normalization -------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]*>")


def normalize_text(text: str) -> str:
    """Canonical normalized form used as the dictionary key.

    Deterministic: lowercase, strip HTML tags, drop possessive ``'s``, replace every
    non-alphanumeric ASCII run with a single space, and trim. Mirrors the legacy
    ``cleanText`` + lowercase behavior without the lossy ligature folding (kept as a
    lexical matching fallback instead, so dictionary keys stay stable).
    """
    lowered = text.lower()
    lowered = _HTML_TAG.sub(" ", lowered)
    lowered = lowered.replace("'s", " ")
    folded = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(folded.split())


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` and return ``(normalized, index_map)``.

    ``index_map[i]`` is the offset in the *original* ``text`` of the character that
    produced ``normalized[i]``. The lexical matcher uses this to translate a match span
    in normalized space back to true offsets in the source text. Every normalized
    character maps to exactly one original character (folding is length-preserving per
    emitted character; dropped characters emit nothing), so spans map back cleanly.
    """
    out_chars: list[str] = []
    out_idx: list[int] = []
    i = 0
    n = len(text)
    in_tag = False
    tag_start = 0
    while i < n:
        ch = text[i]
        if in_tag:
            if ch == ">":
                in_tag = False
                # Emit a single boundary space for the whole tag (mapped to '<').
                out_chars.append(" ")
                out_idx.append(tag_start)
            i += 1
            continue
        if ch == "<":
            in_tag = True
            tag_start = i
            i += 1
            continue
        # Drop possessive 's (apostrophe + s/S): emit nothing.
        if ch == "'" and i + 1 < n and text[i + 1] in "sS":
            i += 2
            continue
        low = ch.lower()
        if low.isascii() and low.isalnum():
            out_chars.append(low)
            out_idx.append(i)
        else:
            out_chars.append(" ")
            out_idx.append(i)
        i += 1
    if in_tag:
        # Unterminated tag at EOF: close the boundary space at the '<'.
        out_chars.append(" ")
        out_idx.append(tag_start)

    # Collapse whitespace runs, keeping the index of each run's first character.
    collapsed_chars: list[str] = []
    collapsed_idx: list[int] = []
    prev_space = True  # trim leading whitespace
    for ch, idx in zip(out_chars, out_idx, strict=True):
        is_space = ch == " "
        if is_space:
            if not prev_space:
                collapsed_chars.append(" ")
                collapsed_idx.append(idx)
            prev_space = True
            continue
        collapsed_chars.append(ch)
        collapsed_idx.append(idx)
        prev_space = False
    # Trim a trailing space if present.
    if collapsed_chars and collapsed_chars[-1] == " ":
        collapsed_chars.pop()
        collapsed_idx.pop()
    return "".join(collapsed_chars), collapsed_idx


# --- source / category semantics ----------------------------------------------

# CURIE prefix -> candidate source label (PLAN.md candidate_source examples).
CURIE_PREFIX_TO_SOURCE: Mapping[str, str] = {
    "MONDO": "MONDO",
    "HP": "HPO",
    "HPO": "HPO",
    "NCIT": "NCIT",
    "UMLS": "UMLS",
    "DRUGBANK": "DRUGBANK",
    "CHEBI": "CHEBI",
    "UNII": "UNII",
    "MESH": "MESH",
    "OMIM": "OMIM",
}

# Ontology/biolink category -> PLAN.md semantic_group (drug | disease | phenotype).
CATEGORY_TO_SEMANTIC_GROUP: Mapping[str, str] = {
    "disease": "disease",
    "diseaseorphenotypicfeature": "disease",
    "phenotypicfeature": "phenotype",
    "phenotype": "phenotype",
    "drug": "drug",
    "chemicalentity": "drug",
    "smallmolecule": "drug",
    "molecularentity": "drug",
    "namedthing": "drug",
}


def infer_source(curie: str) -> str:
    """Infer a candidate source label from a CURIE prefix (default ``fullmap``)."""
    prefix = curie.split(":", 1)[0].upper() if ":" in curie else ""
    return CURIE_PREFIX_TO_SOURCE.get(prefix, "fullmap")


def semantic_group_for(category: str) -> str:
    """Map an ontology/biolink category to a PLAN.md semantic group.

    Unknown categories conservatively fall back to ``disease`` (the dominant DAKP object
    type); the three canonical groups are ``drug`` / ``disease`` / ``phenotype``.
    """
    return CATEGORY_TO_SEMANTIC_GROUP.get(category.strip().lower(), "disease")


# --- dictionary entries & index ------------------------------------------------


@dataclass(frozen=True)
class DictionaryEntry:
    """One normalized-string -> candidate binding.

    ``normalized`` is the dictionary key; ``original`` is the pre-normalization surface
    term (for provenance/notes); ``curie``/``name``/``category``/``source`` describe the
    candidate concept the term resolves to.
    """

    normalized: str
    curie: str
    name: str
    category: str
    source: str
    original: str

    @property
    def semantic_group(self) -> str:
        return semantic_group_for(self.category)


class DictionaryIndex:
    """Immutable, deterministic normalized-string -> candidate index.

    Lookup returns entries sorted by ``(source, category, curie, name)`` so repeated
    builds and lookups are byte-stable. Multiple CURIEs may share one normalized string
    (e.g. an ontology term and an alias); all are returned.
    """

    def __init__(self, entries: Iterable[DictionaryEntry]) -> None:
        by_normalized: dict[str, list[DictionaryEntry]] = {}
        for entry in entries:
            if not entry.normalized:
                continue  # never index an empty key
            by_normalized.setdefault(entry.normalized, []).append(entry)
        # Deterministic ordering per key (and de-duplicated).
        self._by_normalized: dict[str, tuple[DictionaryEntry, ...]] = {
            key: tuple(sorted(set(group), key=_entry_sort_key)) for key, group in by_normalized.items()
        }

    # -- builders --------------------------------------------------------------
    @classmethod
    def from_entries(cls, entries: Iterable[DictionaryEntry]) -> DictionaryIndex:
        return cls(entries)

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        text_col: str = "text",
        curie_col: str = "curie",
        name_col: str = "name",
        category_col: str = "category",
        source_col: str | None = None,
        alias_columns: Sequence[str] = (),
    ) -> DictionaryIndex:
        """Build an index from a term/alias table.

        Required columns: ``text_col`` (a term surface form), ``curie_col``, ``name_col``,
        ``category_col``. ``source_col`` overrides CURIE-prefix source inference. Each
        named column in ``alias_columns`` is indexed as an additional surface form for the
        same CURIE — this is how real MONDO/HPO alias lists (exact/related/narrow/broad)
        feed the index later without a schema change here.
        """
        entries: list[DictionaryEntry] = []
        for row in frame.iter_rows(named=True):
            curie = str(row.get(curie_col) or "").strip()
            name = str(row.get(name_col) or "").strip()
            category = str(row.get(category_col) or "").strip()
            source = str(row.get(source_col) or "").strip() if source_col else ""
            source = source or infer_source(curie)
            surface_forms = [str(row.get(text_col) or ""), *(str(row.get(col) or "") for col in alias_columns)]
            for surface in surface_forms:
                surface = surface.strip()
                if not surface:
                    continue
                entries.append(
                    DictionaryEntry(
                        normalized=normalize_text(surface), curie=curie, name=name or surface, category=category, source=source, original=surface
                    )
                )
        return cls(entries)

    @classmethod
    def from_tsv(cls, path: Path, **kwargs: object) -> DictionaryIndex:
        """Build an index from an uncompressed term TSV (see :meth:`from_frame`)."""
        return cls.from_frame(pl.read_csv(path, separator="\t"), **kwargs)  # type: ignore[arg-type]

    # -- queries ---------------------------------------------------------------
    def lookup(self, normalized: str) -> tuple[DictionaryEntry, ...]:
        """Return candidate entries for a normalized string (empty tuple if none)."""
        return self._by_normalized.get(normalized, ())

    def lookup_text(self, text: str) -> tuple[DictionaryEntry, ...]:
        """Normalize ``text`` then look it up (convenience for whole-string resolution)."""
        return self.lookup(normalize_text(text))

    def normalized_terms(self) -> tuple[str, ...]:
        """All indexed normalized strings, sorted (deterministic iteration)."""
        return tuple(sorted(self._by_normalized))

    def entries(self) -> Iterator[DictionaryEntry]:
        """Yield every entry in deterministic (normalized, sort-key) order."""
        for key in sorted(self._by_normalized):
            yield from self._by_normalized[key]

    def __contains__(self, normalized: str) -> bool:
        return normalized in self._by_normalized

    def __len__(self) -> int:
        """Number of distinct normalized keys (not total entries)."""
        return len(self._by_normalized)


def _entry_sort_key(entry: DictionaryEntry) -> tuple[str, str, str, str]:
    return (entry.source, entry.category, entry.curie, entry.name)


__all__ = [
    "CATEGORY_TO_SEMANTIC_GROUP",
    "CURIE_PREFIX_TO_SOURCE",
    "DictionaryEntry",
    "DictionaryIndex",
    "infer_source",
    "normalize_text",
    "normalize_with_map",
    "semantic_group_for",
]
