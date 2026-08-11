"""FAERS observed-use (applied_to_treat) assertion aggregation.

Builds ``faers_applied_to_treat_assertions.tsv``: real-world drug→condition uses observed in
FAERS adverse-event reports, without any approval claim.

Aggregation rule (explicit and tested)
---------------------------------------
FAERS case rows (``cases.parquet``) are aggregated by ``(drugname, indication)``; ``case_count``
is the number of **distinct cases** (``primaryid``) reporting that pair (falls back to row count
when ``primaryid`` is absent). The FAERS ``knowledge_level`` label is preserved from the first
rebuild (``statistical_association``); ``clinical_approval_status`` is the biolink-valid
``not_provided`` (see below).

Provenance: DAKP aggregates FAERS primary observations with DailyMed support; FAERS is the
primary upstream source, DailyMed the supporting one. Object CURIEs come from the lexical disease
baseline; subjects carry no CURIE (FAERS gives no drug id here). Canonical mapping is later.

``clinical_approval_status`` is ``not_provided``: a FAERS observed use makes no approval claim,
and under Tablassert >= 8.2 the field is a first-class ``ClinicalApprovalStatusEnum`` edge field,
so the legacy ``observed_use`` label (never an enum member — the DINGO ingest already coerced it
to ``not_provided``) would emit biolink-invalid edges. The observed-use meaning stays on the edge
via ``predicate = applied_to_treat`` + ``knowledge_level = observation`` (config override).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import polars as pl

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_FAERS, join_pipe, match_diseases, row_for
from dakp_pipeline.assertions.evidence import find_faers_cases, write_assertion_table
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, stats, step

_TABLE = "faers_applied_to_treat_assertions"
_PREDICATE = "biolink:applied_to_treat"
#: FAERS carries no approval claim; ``observed_use`` is not a ClinicalApprovalStatusEnum member
#: and would fail biolink validation now that Tablassert >= 8.2 emits the field first-class.
_STATUS = "not_provided"
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
        with step(logger, "shape_faers_applied_to_treat"):
            disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
            stats(logger, "shape_faers_applied_to_treat", inputs=len(inputs), disease_map_terms=len(disease_map))
            # Projection: only the three columns the aggregation needs (the production case table
            # is tens of millions of rows wide; reading all 17 columns wastes gigabytes).
            faers_cases = find_faers_cases(inputs, columns=("drugname", "indication", "primaryid"))
            rows = build_observed_use_rows(faers_cases, disease_map)
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_faers_applied_to_treat")


def build_observed_use_rows(faers_cases: pl.DataFrame | None, disease_map: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    """Aggregate FAERS drug-indication case counts into applied-to-treat rows (deterministic).

    The distinct-case counting runs as a Polars group-by (never Python row iteration), so the
    full-production case table (tens of millions of rows) aggregates in a few seconds with
    bounded memory. Semantics are unchanged: per (drugname, indication) pair the count is the
    number of distinct non-empty primaryids plus one per anonymous row (the legacy
    ``_row{index}`` fallback made every primaryid-less row its own observation), and row-count
    when the frame has no primaryid column at all.
    """
    if faers_cases is None:
        return []

    has_primaryid = "primaryid" in faers_cases.columns
    primaryid = pl.col("primaryid").fill_null("").cast(pl.Utf8).str.strip_chars() if has_primaryid else pl.lit("")
    pairs = (
        faers_cases.lazy()
        .select(
            pl.col("drugname").fill_null("").cast(pl.Utf8).str.strip_chars().alias("drugname"),
            pl.col("indication").fill_null("").cast(pl.Utf8).str.strip_chars().alias("indication"),
            primaryid.alias("primaryid"),
        )
        .filter((pl.col("drugname") != "") & (pl.col("indication") != ""))
        .group_by("drugname", "indication")
        .agg(
            pl.col("primaryid").filter(pl.col("primaryid") != "").n_unique().alias("distinct_cases"),
            pl.col("primaryid").filter(pl.col("primaryid") == "").len().alias("anon_rows"),
        )
        .collect()
        .sort("drugname", "indication")
    )

    rows: list[dict[str, str]] = []
    pairs_seen = 0
    stoplist_drops = 0
    for rec in pairs.iter_rows(named=True):
        pairs_seen += 1
        drug = str(rec["drugname"])
        indication = str(rec["indication"])
        if is_non_disease_indication(indication):
            stoplist_drops += 1
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
                case_count=int(rec["distinct_cases"]) + int(rec["anon_rows"]),
                clinical_approval_status=_STATUS,
                knowledge_level=_KNOWLEDGE_LEVEL,
                agent_type=AT_MANUAL,
                primary_knowledge_source=INFORES_DAKP,
                upstream_resource_ids=join_pipe(INFORES_FAERS, INFORES_DAILYMED),
            )
        )
    stats(logger, "shape_faers_applied_to_treat", pairs=pairs_seen, stoplist_drops=stoplist_drops, assertions=len(rows))
    return rows


transform = ObservedUsesShaper().transform

__all__ = ["ObservedUsesShaper", "build_observed_use_rows", "is_non_disease_indication", "transform"]
