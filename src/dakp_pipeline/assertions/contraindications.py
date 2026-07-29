"""Contraindication assertion aggregation (MEDI + DailyMed support, Milestone 5).

Builds ``contraindication_assertions.tsv``: drug→condition contraindication assertions.

Aggregation rule (explicit and tested)
---------------------------------------
Each MEDI contraindication row becomes a ``biolink:contraindicated_in`` assertion. DailyMed
contraindication-section support is attached **first-scope**: the SPL sets whose active ingredient
matches the MEDI drug (case-insensitive) and that carry a contraindication section (LOINC
``34070-3``) are collected into ``supporting_spl_sets``. Rows are deduplicated/aggregated by
``(subject, object)`` — ``supporting_spl_sets`` unioned, ``source_score`` taking the max.

The shaper reads either the canonical MEDI extractor columns (``normalized_drug_id`` …) or the
legacy fixture/shim columns (``final_normalized_drug_id`` …) via column aliases. Subject/object
CURIEs are populated only where MEDI already provides ids; the lexical disease baseline fills any
remaining object gap. ``medi_version`` comes from the row, then context, then a profile default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_MEDI, KL_ASSERTION, join_pipe, match_diseases, row_for
from dakp_pipeline.assertions.evidence import DailyMedEvidence, build_dailymed_evidence, find_table, pick, sorted_pipe, write_assertion_table
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext

_TABLE = "contraindication_assertions"
_PREDICATE = "biolink:contraindicated_in"
_MEDI_FILE = "contraindications.parquet"
_MOCK_DEFAULT_VERSION = "MEDI-0.0-mock"


class ContraindicationsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
        dailymed = build_dailymed_evidence(inputs)
        medi_frame = find_table(inputs, _MEDI_FILE)
        default_version = str(ctx.params.get("medi_version") or (_MOCK_DEFAULT_VERSION if ctx.profile == "mock" else "unknown"))
        rows = build_contraindication_rows(medi_frame, dailymed, disease_map, default_version)
        return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_contraindications")


def build_contraindication_rows(
    medi_frame: pl.DataFrame | None, dailymed: DailyMedEvidence, disease_map: Mapping[str, Mapping[str, str]], default_version: str
) -> list[dict[str, str]]:
    """Aggregate MEDI contraindication rows + DailyMed support into assertion rows (deterministic)."""
    if medi_frame is None:
        return []

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in medi_frame.iter_rows(named=True):
        subject_text = pick(rec, "active_ingredient", "normalized_drug_label", "drug_name")
        object_text = pick(rec, "disease_contraindicated", "contraindicated_condition", "contraindication_text")
        if not subject_text or not object_text:
            continue

        subject_curie = pick(rec, "normalized_drug_id", "final_normalized_drug_id")
        subject_name = pick(rec, "normalized_drug_label", "final_normalized_drug_label", "drug_name", "active_ingredient") or subject_text
        object_curie = pick(rec, "normalized_disease_id", "final_normalized_disease_id")
        object_name = pick(rec, "normalized_disease_label", "final_normalized_disease_label")
        object_category = "Disease"
        if not object_curie or not object_name:
            matches = match_diseases(object_text, disease_map)
            if matches:
                object_curie = object_curie or matches[0]["curie"]
                object_name = object_name or matches[0]["name"]
                object_category = matches[0]["category"]
        object_name = object_name or object_text

        key = (subject_text, object_text)
        agg = aggregated.setdefault(
            key,
            {
                "subject_text": subject_text,
                "subject_curie": subject_curie,
                "subject_name": subject_name,
                "object_text": object_text,
                "object_curie": object_curie,
                "object_name": object_name,
                "object_category": object_category,
                "sets": [],
                "scores": [],
                "medi_version": pick(rec, "medi_version") or default_version,
            },
        )
        agg["sets"].extend(dailymed.contraindication_sets_for_drug(subject_text))
        score = pick(rec, "source_score")
        if score:
            agg["scores"].append(score)
        if not agg["subject_curie"] and subject_curie:
            agg["subject_curie"] = subject_curie
        if not agg["object_curie"] and object_curie:
            agg["object_curie"] = object_curie

    return [_finalize_row(agg) for _key, agg in sorted(aggregated.items())]


def _finalize_row(agg: dict[str, Any]) -> dict[str, str]:
    return row_for(
        _TABLE,
        subject_text=agg["subject_text"],
        subject_curie=agg["subject_curie"],
        subject_name=agg["subject_name"],
        subject_category="ChemicalEntity",
        predicate=_PREDICATE,
        object_text=agg["object_text"],
        object_curie=agg["object_curie"],
        object_name=agg["object_name"],
        object_category=agg["object_category"],
        supporting_spl_sets=sorted_pipe(agg["sets"]),
        medi_version=agg["medi_version"],
        source_score=_max_score(agg["scores"]),
        knowledge_level=KL_ASSERTION,
        agent_type=AT_MANUAL,
        primary_knowledge_source=INFORES_DAKP,
        upstream_resource_ids=join_pipe(INFORES_MEDI, INFORES_DAILYMED),
    )


def _max_score(scores: list[str]) -> str:
    """Highest source score, preserving the source's own string formatting ("" if none)."""
    if not scores:
        return ""

    def numeric(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return float("-inf")

    return max(scores, key=numeric)


transform = ContraindicationsShaper().transform

__all__ = ["ContraindicationsShaper", "build_contraindication_rows", "transform"]
