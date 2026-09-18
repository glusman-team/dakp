"""Assertion-shaping stage: join extracted tables into evidence-rich assertion TSVs.

Shared helpers (disease-map lookup, provenance constants, row builder, the mention-channel
filter) live here so each shaper stays a thin, auditable join. Disease mapping is a fast
exact-match *dictionary baseline*; canonical fullmap/Tablassert resolution is delegated to
Tablassert.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from dakp_pipeline.io import schemas
from dakp_pipeline.ner.dictionary import OBJECT_TYPES
from dakp_pipeline.ner.lexical import Mention

# Translator provenance constants (Translator provenance conventions).
INFORES_DAKP = "infores:multiomics-drugapprovals"
INFORES_DAILYMED = "infores:dailymed"
INFORES_EMA = "infores:ema"
INFORES_EPAR = "infores:epar"
INFORES_FAERS = "infores:faers"

KL_ASSERTION = "knowledge_assertion"
AT_MANUAL = "manual_validation_of_automated_agent"


def match_diseases(text: str, disease_map: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    """Substring-match disease-map keys against ``text`` (case-insensitive).

    Returns one match dict per disease found (``text``, ``curie``, ``name``,
    ``category``). This is the lexical baseline; fullmap/Tablassert resolution replaces
    the Tablassert/fullmap resolution layer.
    """
    lowered = (text or "").lower()
    matches: list[dict[str, str]] = []
    for key, info in disease_map.items():
        if key and key.lower() in lowered:
            matches.append({"text": key, "curie": info.get("curie", ""), "name": info.get("name", key), "category": info.get("category", "Disease")})
    return matches


def row_for(table: str, **values: Any) -> dict[str, str]:
    """Build a full assertion-table row, filling unspecified contract columns with ""."""
    columns = schemas.columns_for(table)
    return {col: str(values[col]) if col in values and values[col] is not None else "" for col in columns}


def join_pipe(*parts: str) -> str:
    """Join non-empty parts with ``|`` (Translator list-encoding convention)."""
    return "|".join(p for p in parts if p)


def object_mentions(mentions: Iterable[Mention]) -> list[Mention]:
    """The object-channel subset of ``mentions`` — the only ones that may become an assertion object.

    :meth:`~dakp_pipeline.ner.ner.DiseaseNER.extract` returns MIXED channels: alongside
    ``Disease`` / ``PhenotypicFeature`` objects it emits qualifier mentions (anatomical site, sex,
    population, taxon, frequency, temporal) that *describe* an object. Asserting a qualifier as an
    object would emit nonsense edges such as ``contraindicated_in women``, so every shaper narrows
    its mention iteration through here. Qualifier mentions are dropped, not reshaped: attaching
    them to the objects they qualify is a separate concern.
    """
    return [mention for mention in mentions if mention.type in OBJECT_TYPES]


__all__ = [
    "AT_MANUAL",
    "INFORES_DAILYMED",
    "INFORES_DAKP",
    "INFORES_EMA",
    "INFORES_EPAR",
    "INFORES_FAERS",
    "KL_ASSERTION",
    "join_pipe",
    "match_diseases",
    "object_mentions",
    "row_for",
]

# Shared evidence helpers (NDA normalization, SPL-support joining, provenance assembly) live in
# :mod:`dakp_pipeline.assertions.evidence`; import them from there directly.
