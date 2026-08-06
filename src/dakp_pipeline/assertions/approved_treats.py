"""Approved-treatment assertion aggregation.

Builds ``approved_treats_assertions.tsv``: FDA-approved drug→condition assertions.

Aggregation rule (explicit and tested)
---------------------------------------
An approved-treats row requires an **NDA-bearing drug-indication pair** that satisfies all of:

1. the pair's NDA maps (Drugs@FDA ``products``) to an ingredient — confirming a real FDA
   application;
2. the NDA has a **DailyMed SPL approval** (``spl_approvals``); and
3. that approved SPL set has an **indications-and-usage section** (LOINC ``34067-9``) — the
   SPL indication support.

Candidate drug-indication pairs come from **FAERS** (``cases.parquet`` rows carrying an NDA +
indication) — the primary source, wired into this stage by the DAG. Placeholder/usage-context
FAERS indications ("Product used for unknown indication", bare "Prophylaxis", ...) are dropped
via the shared :func:`~dakp_pipeline.assertions.observed_uses.is_non_disease_indication`
stop-list. When no FAERS case table is present the candidates fall back to DailyMed SPL
indication sections whose text names a dictionary condition. Both paths apply the *same*
three-part filter above.

Provenance (``approval_ids``, ``supporting_spl_sets``, ``supporting_spl_documents``) is
aggregated per ``(subject, object)`` as deduplicated, sorted, pipe-joined lists. Subject CURIEs
are populated only where DailyMed already gives a UNII; object CURIEs come from the lexical
disease baseline. Canonical CURIE mapping is a later milestone (text-first).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import polars as pl

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_FAERS, KL_ASSERTION, join_pipe, match_diseases, row_for
from dakp_pipeline.assertions.evidence import (
    DailyMedEvidence,
    build_dailymed_evidence,
    build_drugsfda_ingredient_map,
    find_faers_cases,
    normalize_nda,
    sorted_pipe,
    write_assertion_table,
)
from dakp_pipeline.assertions.observed_uses import is_non_disease_indication
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext

_TABLE = "approved_treats_assertions"
_PREDICATE = "biolink:treats"
_STATUS = "approved_for_condition"

#: Case-table columns the FAERS candidate path reads (projection keeps production-scale reads cheap).
_FAERS_CASE_COLUMNS = ("nda", "nda_raw", "indication", "ingredient", "drugname")


class ApprovedTreatsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
        dailymed = build_dailymed_evidence(inputs)
        drugsfda_map = build_drugsfda_ingredient_map(inputs)
        faers_cases = find_faers_cases(inputs, columns=_FAERS_CASE_COLUMNS)
        rows = build_approved_treats_rows(faers_cases, dailymed, drugsfda_map, disease_map)
        return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_approved_treats")


def build_approved_treats_rows(
    faers_cases: pl.DataFrame | None, dailymed: DailyMedEvidence, drugsfda_map: Mapping[str, set[str]], disease_map: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    """Aggregate approved-treats assertion rows (pure; deterministic ordering)."""
    candidates = _faers_candidates(faers_cases, disease_map) if faers_cases is not None else _dailymed_candidates(dailymed, disease_map)

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for cand in candidates:
        norm = cand["norm_nda"]
        if norm not in drugsfda_map:  # (1) NDA must map (Drugs@FDA) to an ingredient
            continue
        sets, docs = dailymed.indication_support(norm)
        if not sets:  # (2)+(3) DailyMed approval AND SPL indication-section support
            continue
        subject_text, subject_curie = _subject_for_sets(dailymed, sets, cand["fallback_subject"])
        if not subject_text:
            continue
        key = (subject_text, cand["object_text"])
        agg = aggregated.setdefault(
            key,
            {
                "subject_text": subject_text,
                "subject_curie": subject_curie,
                "object_text": cand["object_text"],
                "object_curie": cand["object_curie"],
                "object_name": cand["object_name"],
                "object_category": cand["object_category"],
                "approval_ids": [],
                "sets": [],
                "docs": [],
            },
        )
        agg["approval_ids"].append(dailymed.approval_display.get(norm) or norm)
        agg["sets"].extend(sets)
        agg["docs"].extend(docs)
        if not agg["subject_curie"] and subject_curie:
            agg["subject_curie"] = subject_curie
        # object_curie needs no back-fill: it is a deterministic function of object_text (the
        # aggregation key), so every candidate for a key carries the same value already set above.

    return [_finalize_row(agg) for _key, agg in sorted(aggregated.items())]


def _finalize_row(agg: dict[str, Any]) -> dict[str, str]:
    return row_for(
        _TABLE,
        subject_text=agg["subject_text"],
        subject_curie=agg["subject_curie"],
        subject_name=agg["subject_text"],
        subject_category="ChemicalEntity",
        predicate=_PREDICATE,
        object_text=agg["object_text"],
        object_curie=agg["object_curie"],
        object_name=agg["object_name"],
        object_category=agg["object_category"],
        approval_ids=sorted_pipe(agg["approval_ids"]),
        supporting_spl_sets=sorted_pipe(agg["sets"]),
        supporting_spl_documents=sorted_pipe(agg["docs"]),
        clinical_approval_status=_STATUS,
        knowledge_level=KL_ASSERTION,
        agent_type=AT_MANUAL,
        primary_knowledge_source=INFORES_DAKP,
        upstream_resource_ids=join_pipe(INFORES_DAILYMED, INFORES_FAERS),
    )


def _subject_for_sets(dailymed: DailyMedEvidence, sets: list[str], fallback: str) -> tuple[str, str]:
    """Subject ingredient (name, UNII) from the first supporting SPL set, else the FAERS fallback."""
    for set_id in sets:  # already sorted deterministically
        if set_id in dailymed.set_ingredient:
            return dailymed.set_ingredient[set_id]
    return fallback.strip(), ""


def _object_attrs(text: str, disease_map: Mapping[str, Mapping[str, str]]) -> tuple[str, str, str]:
    """Resolve ``(curie, name, category)`` for an object text via the lexical disease baseline."""
    matches = match_diseases(text, disease_map)
    if matches:
        match = matches[0]
        return match["curie"], match["name"], match["category"]
    return "", text, "Disease"


def _faers_candidates(faers_cases: pl.DataFrame, disease_map: Mapping[str, Mapping[str, str]]) -> Iterator[dict[str, str]]:
    """NDA-bearing FAERS drug-indication pairs (deduplicated by NDA + indication)."""
    seen: set[tuple[str, str]] = set()
    for rec in faers_cases.iter_rows(named=True):
        norm = normalize_nda(rec.get("nda") or rec.get("nda_raw"))
        indication = str(rec.get("indication") or "").strip()
        if not norm or not indication:
            continue
        if is_non_disease_indication(indication):
            continue  # FAERS placeholder/usage-context indication, not a drug->condition approval claim
        key = (norm, indication)
        if key in seen:
            continue
        seen.add(key)
        curie, name, category = _object_attrs(indication, disease_map)
        yield {
            "norm_nda": norm,
            "object_text": indication,
            "object_curie": curie,
            "object_name": name,
            "object_category": category,
            "fallback_subject": str(rec.get("ingredient") or rec.get("drugname") or "").strip(),
        }


def _dailymed_candidates(dailymed: DailyMedEvidence, disease_map: Mapping[str, Mapping[str, str]]) -> Iterator[dict[str, str]]:
    """Fallback candidates: dictionary conditions named in approved SPL indication sections."""
    set_to_ndas: dict[str, set[str]] = {}
    for norm, sets in dailymed.approval_sets.items():
        for set_id in sets:
            set_to_ndas.setdefault(set_id, set()).add(norm)

    seen: set[tuple[str, str]] = set()
    for set_id in sorted(dailymed.indication_docs):
        ndas = sorted(set_to_ndas.get(set_id, ()))
        if not ndas:
            continue
        for _doc_id, text in dailymed.indication_docs[set_id]:
            for match in match_diseases(text, disease_map):
                for norm in ndas:
                    key = (norm, match["text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "norm_nda": norm,
                        "object_text": match["text"],
                        "object_curie": match["curie"],
                        "object_name": match["name"],
                        "object_category": match["category"],
                        "fallback_subject": "",
                    }


transform = ApprovedTreatsShaper().transform

__all__ = ["ApprovedTreatsShaper", "build_approved_treats_rows", "transform"]
