"""FAERS observed-use (applied_to_treat) assertion aggregation.

Builds ``faers_applied_to_treat_assertions.tsv``: real-world drug→condition uses observed in
FAERS adverse-event reports, without any approval claim.

Aggregation rule (explicit and tested)
---------------------------------------
FAERS case rows (``cases.parquet``) are aggregated by ``(drugname, indication)``; ``case_count``
is the number of **distinct cases** (``primaryid``) reporting that pair (falls back to row count
when ``primaryid`` is absent). The FAERS ``clinical_approval_status``/``knowledge_level`` labels
are preserved from the first rebuild (``observed_use`` / ``statistical_association``).

Provenance: DAKP aggregates FAERS primary observations with DailyMed support; FAERS is the
primary upstream source, DailyMed the supporting one. Object CURIEs come from the lexical disease
baseline; subjects carry no CURIE (FAERS gives no drug id here). Canonical mapping is later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import polars as pl

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_FAERS, join_pipe, match_diseases, row_for
from dakp_pipeline.assertions.evidence import find_faers_cases, write_assertion_table
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext

_TABLE = "faers_applied_to_treat_assertions"
_PREDICATE = "biolink:applied_to_treat"
# Stable FAERS applied-to-treat labels (resolved planning decision: keep current behavior).
_STATUS = "observed_use"
_KNOWLEDGE_LEVEL = "statistical_association"

# FAERS ``indi_pt`` is free text and carries non-disease usage-context values that name no real
# condition (placeholders like "Product used for unknown indication", generic procedures like a
# bare "Prophylaxis"/"Chemotherapy", and reporting artifacts like "Medication error"). These are
# not drug->condition observations and would otherwise default to bogus "Disease" objects (~41% of
# the case-weighted rows in a real quarter). The list is the union of the legacy DAKP FAERS
# stop-lists (ref/legacy FAERS/bin/drug2indi.pl + listCases.pl + caseList2uses.pl). Specific
# conditions are untouched: "Migraine prophylaxis" or "Hormone receptor positive HER2 negative
# breast cancer" do NOT match (the anchored generic terms require the whole string; the phrase
# terms target the placeholder wording only).
_NON_DISEASE_INDICATION_RE = re.compile(
    r"unknown indication|unapproved indication|off[- ]label|ill-defined|adverse drug reaction"
    r"|evidence based treatment|medication error|not applicable"
    r"|product used for|product use in|product use issue|drug use in|product dose|product prescribing"
    r"|product storage|product availab|product quality|product misuse|product origin unknown"
    r"|product administration|accidental exposure|exposure during pregnancy"
    r"|\Aprophylaxis\Z|\Apremedication\Z|\Achemotherapy\Z|\Adrug therapy\Z|\Asupplementation therapy\Z",
    re.IGNORECASE,
)


def is_non_disease_indication(indication: str) -> bool:
    """True when a FAERS indication is a placeholder/usage-context value naming no real condition."""
    return bool(_NON_DISEASE_INDICATION_RE.search(indication.strip()))


class ObservedUsesShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
        faers_cases = find_faers_cases(inputs)
        rows = build_observed_use_rows(faers_cases, disease_map)
        return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_faers_applied_to_treat")


def build_observed_use_rows(faers_cases: pl.DataFrame | None, disease_map: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    """Aggregate FAERS drug-indication case counts into applied-to-treat rows (deterministic)."""
    if faers_cases is None:
        return []

    has_primaryid = "primaryid" in faers_cases.columns
    pair_cases: dict[tuple[str, str], set[str]] = {}
    for index, rec in enumerate(faers_cases.iter_rows(named=True)):
        drug = str(rec.get("drugname") or "").strip()
        indication = str(rec.get("indication") or "").strip()
        if not drug or not indication:
            continue
        case_id = str(rec.get("primaryid") or "").strip() if has_primaryid else ""
        if not case_id:  # no case id -> count rows (each row is its own observation)
            case_id = f"_row{index}"
        pair_cases.setdefault((drug, indication), set()).add(case_id)

    rows: list[dict[str, str]] = []
    for drug, indication in sorted(pair_cases):
        if is_non_disease_indication(indication):
            continue  # FAERS placeholder/usage-context indication, not a drug->condition observation
        matches = match_diseases(indication, disease_map)
        obj = matches[0] if matches else {"text": indication, "curie": "", "name": indication, "category": "Disease"}
        rows.append(
            row_for(
                _TABLE,
                subject_text=drug,
                subject_curie="",
                subject_name=drug,
                subject_category="ChemicalEntity",
                predicate=_PREDICATE,
                object_text=obj["text"],
                object_curie=obj["curie"],
                object_name=obj["name"],
                object_category=obj["category"],
                case_count=len(pair_cases[(drug, indication)]),
                clinical_approval_status=_STATUS,
                knowledge_level=_KNOWLEDGE_LEVEL,
                agent_type=AT_MANUAL,
                primary_knowledge_source=INFORES_DAKP,
                upstream_resource_ids=join_pipe(INFORES_FAERS, INFORES_DAILYMED),
            )
        )
    return rows


transform = ObservedUsesShaper().transform

__all__ = ["ObservedUsesShaper", "build_observed_use_rows", "is_non_disease_indication", "transform"]
