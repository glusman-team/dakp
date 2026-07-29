"""Assertion-shaping stage: join extracted tables into evidence-rich assertion TSVs.

Shared helpers (disease-map lookup, provenance constants, row builder) live here so each
shaper stays a thin, auditable join. Disease mapping is a fast exact-match *dictionary
baseline* for Milestone 1; canonical fullmap/Tablassert resolution lands in Milestone 4.
"""

from __future__ import annotations

from typing import Any

from dakp_pipeline.io import schemas

# Translator provenance constants (PLAN.md "Translator provenance conventions").
INFORES_DAKP = "infores:multiomics-drugapprovals"
INFORES_DAILYMED = "infores:dailymed"
INFORES_FAERS = "infores:faers"
INFORES_MEDI = "infores:medi"

KL_ASSERTION = "knowledge_assertion"
AT_MANUAL = "manual_validation_of_automated_agent"


def match_diseases(text: str, disease_map: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Substring-match disease-map keys against ``text`` (case-insensitive).

    Returns one match dict per disease found (``text``, ``curie``, ``name``,
    ``category``). This is the lexical baseline; fullmap/Tablassert resolution replaces
    it in Milestone 4.
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


__all__ = ["AT_MANUAL", "INFORES_DAILYMED", "INFORES_DAKP", "INFORES_FAERS", "INFORES_MEDI", "KL_ASSERTION", "join_pipe", "match_diseases", "row_for"]
