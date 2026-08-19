"""Deterministic lexical mention matcher (the gazetteer span-detector).

Extracts disease/phenotype mention **spans** from source text — DailyMed section text and
FAERS indication strings — by matching normalized gazetteer phrases with word boundaries.
Fully deterministic: no randomness, stable scoring, and stable output order. A mention is a
text span + entity type ONLY; no ontology CURIE/name/category is attached (that resolution is
Tablassert's job).

Design notes:

* Matching happens in *normalized* space (see :func:`dakp_pipeline.ner.dictionary.normalize_with_map`)
  and match spans are mapped back to **true offsets in the original text**, so
  ``mention.text == text[mention.start:mention.end]`` always holds.
* The legacy ignore-term concept (``ref/legacy/FAERS/bin/drug2indi.pl`` / ``listCases.pl``) is ported
  as :data:`DEFAULT_IGNORE_TERMS`: a whole indication string in the ignore set suppresses the
  entire record, and an individual mention equal to an ignore term is dropped.
* Legacy synonym fallbacks (``cardiac`` -> ``heart`` etc., from
  ``ref/legacy/DailyMed/bin/findTermsInIndications.pl``) are available opt-in via ``synonyms=``
  and keep true offsets (the reported surface form stays the original word).
* Greedy longest-phrase-first matching keeps a shorter term from matching inside an
  already-matched longer phrase (e.g. ``pain`` inside ``peptic ulcer disease``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dakp_pipeline.ner.dictionary import Gazetteer, normalize_text, normalize_with_map

# Direct normalized-phrase match confidence (the gazetteer baseline is high-confidence).
DIRECT_SCORE = 1.0
# Confidence for a match that required a synonym substitution fallback.
SYNONYM_SCORE = 0.9

# Legacy FAERS indication strings that are not real indications (union of the active,
# uncommented entries in ref/legacy/FAERS/bin/drug2indi.pl and ref/legacy/FAERS/bin/listCases.pl).
# Stored normalized; used for whole-field suppression and per-mention filtering.
_LEGACY_IGNORE_RAW = (
    "product used for unknown indication",
    "drug use unknown indication",
    "off label use",
    "prophylaxis",
    "ill-defined disorder",
    "premedication",
    "product use in unapproved indication",
    "intentional product misuse",
    "exposure during pregnancy",
    "foetal exposure during pregnancy",
    "product origin unknown",
    "accidental exposure to product",
    "product use issue",
    "unknown indication",
    "unknown indcation",
    "product administration",
)

DEFAULT_IGNORE_TERMS: frozenset[str] = frozenset(normalize_text(term) for term in _LEGACY_IGNORE_RAW)

# Legacy synonym fallbacks (ref/legacy/DailyMed/bin/findTermsInIndications.pl): whole-word
# substitutions tried so e.g. "cardiac" text resolves a "heart ..." term. Opt-in.
LEGACY_SYNONYMS: Mapping[str, str] = {"cardiac": "heart", "renal": "kidney", "hepatic": "liver"}


@dataclass(frozen=True)
class Mention:
    """One disease/phenotype mention span — text + type ONLY (no ontology CURIE).

    ``text`` is the original surface form; ``start``/``end`` are half-open offsets into the
    source text such that ``text == source[start:end]``. ``type`` is the canonical entity type
    (``disease`` / ``phenotype``); ``score`` is the match confidence. ``normalized`` is the
    normalized surface form (the unique-string key used downstream); ``notes`` records how the
    span matched (``exact`` / ``synonym:...`` / ``gliner``); ``section`` is caller context.
    """

    text: str
    start: int
    end: int
    type: str
    score: float
    normalized: str = ""
    notes: str = ""
    section: str = ""

    def to_dict(self) -> dict[str, str | int | float]:
        """Lossless JSON-serializable form (all fields; used by the persistent mention cache)."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "score": self.score,
            "normalized": self.normalized,
            "notes": self.notes,
            "section": self.section,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Mention:
        """Inverse of :meth:`to_dict` — round-trips a cached mention exactly."""
        return cls(
            text=str(data["text"]),
            start=int(data["start"]),
            end=int(data["end"]),
            type=str(data["type"]),
            score=float(data["score"]),
            normalized=str(data.get("normalized", "")),
            notes=str(data.get("notes", "")),
            section=str(data.get("section", "")),
        )


class LexicalMatcher:
    """Deterministic normalized-phrase mention matcher over a :class:`Gazetteer`."""

    def __init__(self, gazetteer: Gazetteer, *, ignore_terms: Sequence[str] | None = None, synonyms: Mapping[str, str] | None = None) -> None:
        # Ignore terms are normalized on ingest so callers may pass raw strings.
        raw_ignore = DEFAULT_IGNORE_TERMS if ignore_terms is None else ignore_terms
        self._ignore = frozenset(normalize_text(term) for term in raw_ignore)
        self._synonyms: dict[str, str] = {normalize_text(k): normalize_text(v) for k, v in (synonyms or {}).items()}
        # Longest-first (then lexicographic) (term, type) pairs for deterministic greedy matching.
        self._terms = sorted(gazetteer.items(), key=lambda item: (-len(item[0]), item[0]))

    # -- ignore handling -------------------------------------------------------
    def is_ignored_text(self, text: str) -> bool:
        """True if the whole normalized field is an ignore term (legacy FAERS behavior)."""
        return normalize_text(text) in self._ignore

    # -- matching --------------------------------------------------------------
    def match(self, text: str, *, section: str = "") -> list[Mention]:
        """Return all gazetteer mentions in ``text``, deterministically ordered.

        Whole-field ignore terms yield no mentions. Output is sorted by
        ``(start, end, type, text)``.
        """
        if not text or not text.strip() or self.is_ignored_text(text):
            return []

        normalized, index_map = normalize_with_map(text)
        synonym_notes: list[str] = [""] * len(normalized)
        if self._synonyms:
            normalized, index_map, synonym_notes = self._apply_synonyms(normalized, index_map)

        mentions: list[Mention] = []
        covered: list[tuple[int, int]] = []  # accepted normalized-space spans
        for term, etype in self._terms:
            for start in _find_word_bounded(normalized, term):
                end = start + len(term)
                if _overlaps_any(start, end, covered):
                    continue
                covered.append((start, end))
                mention = self._mention_for_span(text, term, etype, start, end, index_map, synonym_notes, section)
                if mention is not None:
                    mentions.append(mention)

        return sorted(mentions, key=_mention_sort_key)

    # -- internals -------------------------------------------------------------
    def _mention_for_span(
        self, text: str, term: str, etype: str, start: int, end: int, index_map: list[int], synonym_notes: list[str], section: str
    ) -> Mention | None:
        """Build the :class:`Mention` for an accepted match span (``None`` if per-mention ignored)."""
        orig_start = index_map[start]
        orig_end = index_map[end - 1] + 1
        mention_text = text[orig_start:orig_end]
        normalized_surface = normalize_text(mention_text)
        if normalized_surface in self._ignore:
            return None  # per-mention ignore filter

        span_notes = sorted({note for note in synonym_notes[start:end] if note})
        via_synonym = bool(span_notes)
        score = SYNONYM_SCORE if via_synonym else DIRECT_SCORE
        notes = ";".join(span_notes) if via_synonym else "exact"
        return Mention(
            text=mention_text, start=orig_start, end=orig_end, type=etype, score=score, normalized=normalized_surface, notes=notes, section=section
        )

    def _apply_synonyms(self, normalized: str, index_map: list[int]) -> tuple[str, list[int], list[str]]:
        """Expand whole-word synonyms, maintaining the offset map and per-position notes.

        A substituted word's replacement characters all map back to the original word's first
        character, so the reported surface form/offset stays the original word (e.g. ``cardiac``)
        while matching the synonymized concept (``heart``).
        """
        out_chars: list[str] = []
        out_idx: list[int] = []
        out_notes: list[str] = []
        i = 0
        n = len(normalized)
        while i < n:
            if normalized[i] == " ":
                out_chars.append(" ")
                out_idx.append(index_map[i])
                out_notes.append("")
                i += 1
                continue
            j = i
            while j < n and normalized[j] != " ":
                j += 1
            token = normalized[i:j]
            replacement = self._synonyms.get(token)
            if replacement is not None and replacement != token:
                note = f"synonym:{token}>{replacement}"
                anchor = index_map[i]
                for ch in replacement:
                    out_chars.append(ch)
                    out_idx.append(anchor)
                    out_notes.append(note)
            else:
                for k in range(i, j):
                    out_chars.append(normalized[k])
                    out_idx.append(index_map[k])
                    out_notes.append("")
            i = j
        return "".join(out_chars), out_idx, out_notes


# --- helpers -------------------------------------------------------------------


def _find_word_bounded(haystack: str, needle: str) -> list[int]:
    """All start positions where ``needle`` occurs in ``haystack`` on word boundaries.

    Normalized text is space-separated alphanumerics, so a boundary is start-of-string or a
    space immediately before/after the occurrence.
    """
    if not needle:
        return []
    positions: list[int] = []
    n = len(haystack)
    m = len(needle)
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


def _mention_sort_key(mention: Mention) -> tuple[int, int, str, str]:
    return (mention.start, mention.end, mention.type, mention.text)


__all__ = ["DEFAULT_IGNORE_TERMS", "DIRECT_SCORE", "LEGACY_SYNONYMS", "SYNONYM_SCORE", "LexicalMatcher", "Mention"]
