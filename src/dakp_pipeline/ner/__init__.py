"""NER / mention + canonical-mapping layer (PLAN.md Milestone 4).

Source-aware mention generation decoupled from final canonical mapping:

* :mod:`~dakp_pipeline.ner.dictionary` — normalized dictionary indexes (fast baseline).
* :mod:`~dakp_pipeline.ner.lexical` — deterministic lexical mention matcher (spans,
  ignore-list, synonyms, scoring).
* :mod:`~dakp_pipeline.ner.mapping` — :class:`MappingBackend` protocol + a deterministic
  mocked fullmap backend (the canonical-resolver seam; real Tablassert/fullmap later).
* :mod:`~dakp_pipeline.ner.candidates` — ``mention_candidates.tsv`` emission with
  unique-string resolution.
"""

from __future__ import annotations

from dakp_pipeline.ner.candidates import (
    MENTION_CANDIDATES_COLUMNS,
    MentionCandidateTransformer,
    TextRecord,
    resolve_mention_candidates,
    text_records_from_dailymed_sections,
    text_records_from_faers_cases,
    write_mention_candidates,
)
from dakp_pipeline.ner.dictionary import DictionaryEntry, DictionaryIndex, normalize_text, normalize_with_map, semantic_group_for
from dakp_pipeline.ner.lexical import DEFAULT_IGNORE_TERMS, LEGACY_SYNONYMS, LexicalMatcher, Mention
from dakp_pipeline.ner.mapping import DEFAULT_TAXON, MappedTerm, MappingBackend, MockFullmapBackend

__all__ = [
    "DEFAULT_IGNORE_TERMS",
    "DEFAULT_TAXON",
    "LEGACY_SYNONYMS",
    "MENTION_CANDIDATES_COLUMNS",
    "DictionaryEntry",
    "DictionaryIndex",
    "LexicalMatcher",
    "MappedTerm",
    "MappingBackend",
    "Mention",
    "MentionCandidateTransformer",
    "MockFullmapBackend",
    "TextRecord",
    "normalize_text",
    "normalize_with_map",
    "resolve_mention_candidates",
    "semantic_group_for",
    "text_records_from_dailymed_sections",
    "text_records_from_faers_cases",
    "write_mention_candidates",
]
