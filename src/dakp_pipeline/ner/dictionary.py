"""Normalized disease/phenotype gazetteer for mention **span detection only**.

This is the deterministic, high-precision anchor of DAKP's single composite NER backend
(see ``ner/BENCHMARK.md``). It indexes normalized term phrases to an entity **type**
(``disease`` / ``phenotype``) so the lexical matcher can locate mention spans. It does
**NOT** resolve terms to ontology CURIEs/names/categories — ontology mapping is exclusively
Tablassert's job (fullmap/BABEL at ``tablassert build-kg``). DAKP emits mention text spans +
type only.

Normalization is deterministic and shared with :mod:`dakp_pipeline.ner.lexical`: lowercase,
strip HTML tags, drop possessive ``'s``, fold non-alphanumerics to single spaces.
:func:`normalize_with_map` additionally returns a character-index map so the matcher can
report mention offsets into the *original* (un-normalized) text.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import polars as pl

# --- canonical entity types ----------------------------------------------------

TYPE_DISEASE = "disease"
TYPE_PHENOTYPE = "phenotype"

# The object types a contraindication assertion targets (disease / phenotype mentions).
CONTRAINDICATION_DISEASE_TYPES: tuple[str, ...] = (TYPE_DISEASE, TYPE_PHENOTYPE)

# Raw label -> canonical type. Unknown labels canonicalize to their lowercased form.
_TYPE_ALIASES: Mapping[str, str] = {
    "disease": TYPE_DISEASE,
    "diseases": TYPE_DISEASE,
    "phenotype": TYPE_PHENOTYPE,
    "phenotypes": TYPE_PHENOTYPE,
    "phenotypicfeature": TYPE_PHENOTYPE,
    "phenotypic_feature": TYPE_PHENOTYPE,
}


def canonical_type(raw: str) -> str:
    """Canonicalize a raw entity label (e.g. ``"PhenotypicFeature"`` -> ``"phenotype"``)."""
    key = raw.strip().lower()
    return _TYPE_ALIASES.get(key, key)


# --- normalization -------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]*>")


def normalize_text(text: str) -> str:
    """Canonical normalized form used as the gazetteer key.

    Deterministic: lowercase, strip HTML tags, drop possessive ``'s``, replace every
    non-alphanumeric ASCII run with a single space, and trim.
    """
    lowered = text.lower()
    lowered = _HTML_TAG.sub(" ", lowered)
    lowered = lowered.replace("'s", " ")
    folded = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(folded.split())


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` and return ``(normalized, index_map)``.

    ``index_map[i]`` is the offset in the *original* ``text`` of the character that produced
    ``normalized[i]``. The matcher uses this to translate a match span in normalized space
    back to true offsets in the source text, so ``mention.text == text[mention.start:mention.end]``.
    """
    out_chars: list[str] = []
    out_idx: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "<":
            close = text.find(">", i + 1)
            # Well-formed tag (or a lone '<'): emit one boundary space mapped to the '<' and
            # skip the tag, mirroring normalize_text's `<[^>]*>` regex (a lone '<' falls through
            # to the punctuation rule there, which also yields a boundary space).
            out_chars.append(" ")
            out_idx.append(i)
            i = close + 1 if close != -1 else i + 1
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

    # Collapse whitespace runs, keeping the index of each run's first character; trim ends.
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
    if collapsed_chars and collapsed_chars[-1] == " ":
        collapsed_chars.pop()
        collapsed_idx.pop()
    return "".join(collapsed_chars), collapsed_idx


# --- gazetteer -----------------------------------------------------------------


class Gazetteer:
    """Immutable, deterministic normalized-phrase -> entity-type index (span detection only).

    Keys are normalized term phrases; values are canonical entity types
    (:data:`TYPE_DISEASE` / :data:`TYPE_PHENOTYPE`). No CURIE/name/category is stored or
    assigned — the gazetteer only answers "is this phrase a disease/phenotype mention, and
    which type?". Multiple surface forms may normalize to the same key; the type is whichever
    was inserted last for that key (deterministic for a given input mapping).
    """

    def __init__(self, terms: Mapping[str, str]) -> None:
        by_normalized: dict[str, str] = {}
        for surface, etype in terms.items():
            key = normalize_text(surface)
            if key:
                by_normalized[key] = canonical_type(etype)
        self._by_normalized = by_normalized
        self._terms = tuple(sorted(by_normalized))

    # -- builders --------------------------------------------------------------
    @classmethod
    def from_frame(cls, frame: pl.DataFrame, *, text_col: str = "text", type_col: str = "type") -> Gazetteer:
        """Build a gazetteer from a term table with a surface-form column and a type column.

        ``type_col`` values are canonicalized (:func:`canonical_type`), so biolink categories
        like ``"Disease"`` / ``"PhenotypicFeature"`` are accepted as well as ``"disease"`` /
        ``"phenotype"``. Only ``text_col`` and ``type_col`` are read — any CURIE/name columns
        in the frame are ignored (DAKP does not map terms to ontology concepts).
        """
        terms: dict[str, str] = {}
        for row in frame.iter_rows(named=True):
            surface = str(row.get(text_col) or "").strip()
            etype = str(row.get(type_col) or "").strip()
            if surface and etype:
                terms[surface] = etype
        return cls(terms)

    @classmethod
    def from_tsv(cls, path: Path, **kwargs: str) -> Gazetteer:
        """Build a gazetteer from an uncompressed term TSV (see :meth:`from_frame`)."""
        return cls.from_frame(pl.read_csv(path, separator="\t"), **kwargs)  # type: ignore[arg-type]

    # -- queries ---------------------------------------------------------------
    def type_for(self, normalized: str) -> str | None:
        """Entity type for a normalized phrase (``None`` if not in the gazetteer)."""
        return self._by_normalized.get(normalized)

    def normalized_terms(self) -> tuple[str, ...]:
        """All indexed normalized phrases, sorted (deterministic iteration)."""
        return self._terms

    def items(self) -> Iterator[tuple[str, str]]:
        """Yield ``(normalized, type)`` pairs in deterministic (sorted-key) order."""
        yield from sorted(self._by_normalized.items())

    def __contains__(self, normalized: object) -> bool:
        return normalized in self._by_normalized

    def __len__(self) -> int:
        """Number of distinct normalized phrases."""
        return len(self._by_normalized)


__all__ = ["CONTRAINDICATION_DISEASE_TYPES", "TYPE_DISEASE", "TYPE_PHENOTYPE", "Gazetteer", "canonical_type", "normalize_text", "normalize_with_map"]
