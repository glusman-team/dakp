"""NER mention-extraction layer (PLAN.md Milestone 4) — ONE composite backend, mentions only.

DAKP extracts disease/phenotype **mentions** (text spans + entity type). It never resolves
terms to ontology CURIEs — ontology mapping is exclusively Tablassert's job (fullmap/BABEL at
``tablassert build-kg``). There is no pluggable backend selector; :mod:`~dakp_pipeline.ner.ner`
is the single composite backend (gazetteer-first, GLiNER-augmented; see ``ner/BENCHMARK.md``).

* :mod:`~dakp_pipeline.ner.dictionary` — normalization + the span-detection :class:`Gazetteer`
  (term -> type; no CURIE/name/category).
* :mod:`~dakp_pipeline.ner.lexical` — deterministic lexical :class:`Mention` matcher.
* :mod:`~dakp_pipeline.ner.ner` — the single :class:`DiseaseNER` backend + entry points.
* :mod:`~dakp_pipeline.ner.candidates` — unique mention-string inventory emission.
* :mod:`~dakp_pipeline.ner.model_cache` — idempotent model download/cache (production mode).
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
from dakp_pipeline.ner.dictionary import (
    CONTRAINDICATION_DISEASE_TYPES,
    TYPE_DISEASE,
    TYPE_PHENOTYPE,
    Gazetteer,
    canonical_type,
    normalize_text,
    normalize_with_map,
)
from dakp_pipeline.ner.lexical import DEFAULT_IGNORE_TERMS, LEGACY_SYNONYMS, LexicalMatcher, Mention
from dakp_pipeline.ner.ner import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    EMBEDDED_GAZETTEER,
    DiseaseNER,
    extract_contraindication_diseases,
    extract_disease_mentions,
)

__all__ = [
    "CONTRAINDICATION_DISEASE_TYPES",
    "DEFAULT_IGNORE_TERMS",
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
    "EMBEDDED_GAZETTEER",
    "LEGACY_SYNONYMS",
    "MENTION_CANDIDATES_COLUMNS",
    "TYPE_DISEASE",
    "TYPE_PHENOTYPE",
    "DiseaseNER",
    "Gazetteer",
    "LexicalMatcher",
    "Mention",
    "MentionCandidateTransformer",
    "TextRecord",
    "canonical_type",
    "extract_contraindication_diseases",
    "extract_disease_mentions",
    "normalize_text",
    "normalize_with_map",
    "resolve_mention_candidates",
    "text_records_from_dailymed_sections",
    "text_records_from_faers_cases",
    "write_mention_candidates",
]
