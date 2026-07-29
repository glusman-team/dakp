"""Contraindication assertion shaper (MEDI + DailyMed support).

Builds ``contraindication_assertions.tsv`` from MEDI contraindication rows, mapping the
contraindicated-condition text via the lexical dictionary baseline and attaching DailyMed
SPL support where available (first-scope per PLAN.md "Resolved planning decisions").
"""

from __future__ import annotations

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_MEDI, match_diseases, row_for
from dakp_pipeline.assertions.approved_treats import _first_parquet, _write_assertion
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext

_TABLE = "contraindication_assertions"
_MEDI_VERSION = "MEDI-0.x-mock"


class ContraindicationsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
        medi = _first_parquet(inputs, "medi")
        if medi is None:
            return []
        medi_frame = schemas.read_table(medi.uri)

        rows: list[dict[str, str]] = []
        for rec in medi_frame.iter_rows(named=True):
            drug = str(rec.get("drug_name", "") or "")
            condition = str(rec.get("contraindicated_condition", "") or rec.get("contraindicated_disease", "") or "")
            score = str(rec.get("source_score", "") or "")
            if not drug or not condition:
                continue
            disease = match_diseases(condition, disease_map)
            obj = disease[0] if disease else {"text": condition, "curie": "", "name": condition, "category": "Disease"}
            rows.append(
                row_for(
                    _TABLE,
                    subject_text=drug,
                    subject_category="ChemicalEntity",
                    predicate="biolink:contraindicated_in",
                    object_text=obj["text"],
                    object_curie=obj["curie"],
                    object_name=obj["name"],
                    object_category=obj["category"],
                    supporting_spl_sets="",
                    medi_version=_MEDI_VERSION,
                    source_score=score,
                    knowledge_level="knowledge_assertion",
                    agent_type=AT_MANUAL,
                    primary_knowledge_source=INFORES_DAKP,
                    upstream_resource_ids=f"{INFORES_MEDI}|{INFORES_DAILYMED}",
                )
            )

        return _write_assertion(_TABLE, rows, inputs, ctx)


transform = ContraindicationsShaper().transform

__all__ = ["ContraindicationsShaper", "transform"]
