"""Approved-treatment assertion shaper.

Builds ``approved_treats_assertions.tsv`` from DailyMed SPL indications joined to
Drugs@FDA approvals, with disease objects mapped via the lexical dictionary baseline.
Subject/object CURIEs are populated where the dictionary resolves them; everything else
is left for Tablassert/fullmap canonical resolution (Milestone 4+).
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_FAERS, KL_ASSERTION, join_pipe, match_diseases, row_for
from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.paths import Workdir

_TABLE = "approved_treats_assertions"


class ApprovedTreatsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
        spl = _first_parquet(inputs, "dailymed")
        if spl is None:
            return []
        spl_frame = schemas.read_table(spl.uri)
        drugsfda_ndas = _drugsfda_appl_nos(inputs)

        rows: list[dict[str, str]] = []
        for rec in spl_frame.iter_rows(named=True):
            ingredient = str(rec.get("active_ingredient_name", "") or "")
            section_text = str(rec.get("section_text", "") or "")
            set_id = str(rec.get("spl_set_id", "") or "")
            approval_code = str(rec.get("approval_code", "") or "")
            for disease in match_diseases(section_text, disease_map):
                approval_ids = join_pipe(approval_code, *(f"NDA:{n}" for n in drugsfda_ndas))
                rows.append(
                    row_for(
                        _TABLE,
                        subject_text=ingredient,
                        subject_category="ChemicalEntity",
                        predicate="biolink:treats",
                        object_text=disease["text"],
                        object_curie=disease["curie"],
                        object_name=disease["name"],
                        object_category=disease["category"],
                        approval_ids=approval_ids,
                        supporting_spl_sets=set_id,
                        supporting_spl_documents=str(rec.get("spl_document_id", "") or ""),
                        clinical_approval_status="approved_for_condition",
                        knowledge_level=KL_ASSERTION,
                        agent_type=AT_MANUAL,
                        primary_knowledge_source=INFORES_DAKP,
                        upstream_resource_ids=join_pipe(INFORES_DAILYMED, INFORES_FAERS),
                    )
                )

        return _write_assertion(_TABLE, rows, inputs, ctx)


def _first_parquet(inputs: list[ArtifactRef], name_part: str) -> ArtifactRef | None:
    for ref in inputs:
        if name_part in str(ref.uri) and ref.uri.suffix == ".parquet":
            return ref
    return None


def _drugsfda_appl_nos(inputs: list[ArtifactRef]) -> list[str]:
    ref = _first_parquet(inputs, "drugsfda")
    if ref is None:
        return []
    frame = schemas.read_table(ref.uri)
    if "appl_no" not in frame.columns:
        return []
    return [str(v) for v in frame.get_column("appl_no").to_list() if str(v).strip()]


def _write_assertion(table: str, rows: list[dict[str, str]], inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    frame = pl.DataFrame(rows, schema=schemas.columns_for(table)) if rows else pl.DataFrame(schema=schemas.columns_for(table))
    out = Workdir(ctx.workdir).tabular / f"{table}.tsv"
    rows_written = schemas.write_tsv(frame, out)
    store = ArtifactStore(Workdir(ctx.workdir))
    input_ids = [ref.blake3 for ref in inputs]
    ref = store.register(
        out,
        media_type=schemas.TSV_MEDIA_TYPE,
        rows=rows_written,
        schema_fingerprint=schemas.schema_fingerprint(schemas.columns_for(table)),
        inputs=input_ids,
        operation=OperationBlock(name=f"shape_{table}"),
        table=TableBlock(rows=rows_written, schema_fingerprint=schemas.schema_fingerprint(schemas.columns_for(table))),
    )
    return [ref]


transform = ApprovedTreatsShaper().transform

__all__ = ["ApprovedTreatsShaper", "transform"]
