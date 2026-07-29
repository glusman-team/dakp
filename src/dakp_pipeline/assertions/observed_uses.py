"""FAERS observed-use (applied_to_treat) assertion shaper.

Aggregates FAERS case-level drug/indication rows into ``faers_applied_to_treat_assertions.tsv``
with case counts. The FAERS ``clinical_approval_status`` label behavior is intentionally
kept stable for the first rebuild; its exact legacy value is confirmed during the
Milestone-5 assertion-aggregation audit.
"""

from __future__ import annotations

from collections import Counter

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_FAERS, match_diseases, row_for
from dakp_pipeline.assertions.approved_treats import _first_parquet, _write_assertion
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext

_TABLE = "faers_applied_to_treat_assertions"


class ObservedUsesShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
        cases = _first_parquet(inputs, "faers")
        if cases is None:
            return []
        frame = schemas.read_table(cases.uri)

        # Aggregate (drugname, indication) -> case_count.
        pair_counts: Counter[tuple[str, str]] = Counter()
        for rec in frame.iter_rows(named=True):
            drug = str(rec.get("drugname", "") or "")
            indication = str(rec.get("indication", "") or "")
            if drug and indication:
                pair_counts[(drug, indication)] += 1

        rows: list[dict[str, str]] = []
        for (drug, indication), count in pair_counts.items():
            disease = match_diseases(indication, disease_map)
            obj = disease[0] if disease else {"text": indication, "curie": "", "name": indication, "category": "Disease"}
            rows.append(
                row_for(
                    _TABLE,
                    subject_text=drug,
                    subject_category="ChemicalEntity",
                    predicate="biolink:applied_to_treat",
                    object_text=obj["text"],
                    object_curie=obj["curie"],
                    object_name=obj["name"],
                    object_category=obj["category"],
                    case_count=count,
                    # Stable FAERS applied-to-treat status; exact legacy label confirmed in Milestone 5.
                    clinical_approval_status="observed_use",
                    knowledge_level="statistical_association",
                    agent_type=AT_MANUAL,
                    primary_knowledge_source=INFORES_DAKP,
                    upstream_resource_ids=f"{INFORES_FAERS}|{INFORES_DAILYMED}",
                )
            )

        return _write_assertion(_TABLE, rows, inputs, ctx)


transform = ObservedUsesShaper().transform

__all__ = ["ObservedUsesShaper", "transform"]
