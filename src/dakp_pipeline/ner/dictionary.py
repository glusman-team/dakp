"""Normalized disease/phenotype gazetteer + the canonical mention-type vocabulary.

This is the deterministic, high-precision anchor of DAKP's single composite NER backend
(see ``ner/BENCHMARK.md``). It indexes normalized term phrases to an entity **type**
(:data:`TYPE_DISEASE` / :data:`TYPE_PHENOTYPE`) so the lexical matcher can locate mention
spans. It does **NOT** resolve terms to ontology CURIEs/names/categories — ontology mapping is
exclusively Tablassert's job (fullmap/BABEL at ``tablassert build-kg``). DAKP emits mention
text spans + type only.

Type vocabulary
---------------
:data:`MENTION_TYPES` is the closed set of ``Mention.type`` values DAKP produces. It splits
into two channels: :data:`OBJECT_TYPES` (``Disease`` / ``PhenotypicFeature``) may become
assertion **objects**, while :data:`QUALIFIER_TYPES` describe an object (anatomical site, sex,
population, taxon, frequency, temporal context/interval) and are consumed as qualifiers only.
Object types are the exact ``tablassert.biolink.Categories`` values the assertion tables already
prioritize, so a DAKP mention type needs no second vocabulary translation downstream.
:func:`canonical_type` folds every label dialect a model or fixture may emit (``biolink:Disease``,
legacy ``disease``, ``PhenotypicFeature``) onto one canonical value; anything outside
:data:`MENTION_TYPES` is dropped by the span adapter.

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

# Object channel: the types a DAKP assertion may target. These are the exact
# ``tablassert.biolink.Categories`` values (``Categories.DISEASE.value`` == "Disease",
# ``Categories.PHENOTYPIC_FEATURE.value`` == "PhenotypicFeature") that the assertion tables
# already list under ``prioritize:`` / ``object_category_override:``, so no second vocabulary
# translation is needed between mining and ``tablassert build-kg``.
TYPE_DISEASE = "Disease"
TYPE_PHENOTYPE = "PhenotypicFeature"

# Qualifier channel: Biolink categories that describe an object rather than being one.
TYPE_ANATOMICAL_ENTITY = "AnatomicalEntity"
TYPE_BIOLOGICAL_SEX = "BiologicalSex"
TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS = "PopulationOfIndividualOrganisms"
TYPE_ORGANISM_TAXON = "OrganismTaxon"
# Qualifier channel: DAKP field-named qualifiers (no Biolink category of that name exists).
TYPE_FREQUENCY_QUALIFIER = "frequency_qualifier"
TYPE_TEMPORAL_CONTEXT_QUALIFIER = "temporal_context_qualifier"
TYPE_TEMPORAL_INTERVAL_QUALIFIER = "temporal_interval_qualifier"

#: Types that may become an assertion **object** (a contraindication/treatment target).
OBJECT_TYPES: tuple[str, ...] = (TYPE_DISEASE, TYPE_PHENOTYPE)
#: Types that only ever describe an object; shapers must never treat them as objects.
#: ``disease_context_qualifier`` is deliberately absent: it shares :data:`TYPE_DISEASE`.
QUALIFIER_TYPES: tuple[str, ...] = (
    TYPE_ANATOMICAL_ENTITY,
    TYPE_BIOLOGICAL_SEX,
    TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS,
    TYPE_ORGANISM_TAXON,
    TYPE_FREQUENCY_QUALIFIER,
    TYPE_TEMPORAL_CONTEXT_QUALIFIER,
    TYPE_TEMPORAL_INTERVAL_QUALIFIER,
)
#: The closed set of ``Mention.type`` values DAKP emits. A label canonicalizing outside it is
#: dropped at the span adapter, so a checkpoint emitting a qualifier DAKP has no column for can
#: never reach a shaper.
MENTION_TYPES: tuple[str, ...] = OBJECT_TYPES + QUALIFIER_TYPES

#: Biolink CURIE prefix carried by model labels (``biolink:Disease``).
_BIOLINK_PREFIX = "biolink:"

# Raw label -> canonical type. Every canonical name is its own alias case-insensitively; the
# explicit legacy entries cover the pre-Biolink dialect still present in fixtures and older
# caches. Unknown labels canonicalize to their lowercased, prefix-stripped form (and are then
# dropped as outside MENTION_TYPES).
_TYPE_ALIASES: Mapping[str, str] = {
    **{name.lower(): name for name in MENTION_TYPES},
    "diseases": TYPE_DISEASE,
    "phenotype": TYPE_PHENOTYPE,
    "phenotypes": TYPE_PHENOTYPE,
    "phenotypic_feature": TYPE_PHENOTYPE,
}


def canonical_type(raw: str) -> str:
    """Canonicalize a raw entity label (``"biolink:Disease"``/``"disease"`` -> ``"Disease"``).

    The ``biolink:`` prefix is stripped and the remainder matched case-insensitively against
    :data:`MENTION_TYPES` plus the legacy aliases; an unrecognized label falls back to that
    lowercased form, which is outside :data:`MENTION_TYPES` and therefore dropped downstream.
    """
    key = raw.strip().lower().removeprefix(_BIOLINK_PREFIX)
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
    (:data:`OBJECT_TYPES`). No CURIE/name/category is stored or
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


__all__ = [
    "MENTION_TYPES",
    "OBJECT_TYPES",
    "QUALIFIER_TYPES",
    "TYPE_ANATOMICAL_ENTITY",
    "TYPE_BIOLOGICAL_SEX",
    "TYPE_DISEASE",
    "TYPE_FREQUENCY_QUALIFIER",
    "TYPE_ORGANISM_TAXON",
    "TYPE_PHENOTYPE",
    "TYPE_POPULATION_OF_INDIVIDUAL_ORGANISMS",
    "TYPE_TEMPORAL_CONTEXT_QUALIFIER",
    "TYPE_TEMPORAL_INTERVAL_QUALIFIER",
    "Gazetteer",
    "canonical_type",
    "normalize_text",
    "normalize_with_map",
]
