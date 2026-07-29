"""DailyMed SPL XML extractor (stub).

Parses the tiny mock SPL fixture (a gzipped XML of ``<document>`` elements) into the
``spl_documents`` interim table with the PLAN.md column contract. Real streaming SPL
extraction (document/set/approval/ingredient/section/evidence tables, partitioned by
release/bin) lands in **Milestone 3**.

The mock SPL schema is a simplified, namespace-free analog of HL7 v3 SPL carrying exactly
the fields DAKP needs (set id, active ingredient + UNII, approval, LOINC-coded sections).
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.paths import Workdir

SECTION_CODE_NAMES = {
    "34067-9": "indications_and_usage",
    "34070-3": "contraindications",
    "34066-1": "boxed_warning",
    "42229-5": "warnings_and_precautions",
}


class SPLXMLExtractor:
    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        store = ArtifactStore(Workdir(ctx.workdir))
        refs: list[ArtifactRef] = []
        for ref in inputs:
            if not _looks_like_spl(ref.uri):
                continue
            rows = list(_iter_spl_documents(ref.uri))
            frame = (
                pl.DataFrame(rows, schema=schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS)
                if rows
                else pl.DataFrame(schema=schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS)
            )
            out = Workdir(ctx.workdir).interim / "dailymed" / "spl_documents.parquet"
            rows_written = schemas.write_parquet(frame, out)
            registered = store.register(
                out,
                media_type=schemas.PARQUET_MEDIA_TYPE,
                rows=rows_written,
                schema_fingerprint=schemas.schema_fingerprint(schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS),
                inputs=[ref.blake3],
                operation=OperationBlock(name="extract_dailymed_spl"),
                table=TableBlock(rows=rows_written, schema_fingerprint=schemas.schema_fingerprint(schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS)),
            )
            refs.append(registered)
        return refs


def _looks_like_spl(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".xml.gz", ".xml"))


def _iter_spl_documents(gz_path: Path) -> list[dict[str, str]]:
    """Yield one row per SPL section in the gzipped mock SPL batch."""
    with gzip.open(gz_path, "rb") as handle:
        root = ET.parse(handle).getroot()

    rows: list[dict[str, str]] = []
    for doc in root.iter("document"):
        rows.extend(_parse_document(doc, source_name=gz_path.name))
    return rows


def _parse_document(doc: ET.Element, *, source_name: str) -> list[dict[str, str]]:
    set_id = (doc.findtext("setId") or "").strip()
    ingredient = doc.find("activeIngredient")
    ing_name = ingredient.get("name", "") if ingredient is not None else ""
    ing_unii = ingredient.get("unii", "") if ingredient is not None else ""
    approval = doc.find("approval")
    ap_code = approval.get("code", "") if approval is not None else ""
    ap_type = approval.get("type", "") if approval is not None else ""

    out: list[dict[str, str]] = []
    for section in doc.findall("section"):
        loinc = section.get("loinc", "")
        name = section.get("name", "") or SECTION_CODE_NAMES.get(loinc, "")
        text = " ".join((section.text or "").split())  # collapse whitespace
        out.append(
            {
                "spl_document_id": f"{set_id}#{loinc}" if loinc else set_id,
                "spl_set_id": set_id,
                "xml_path": source_name,
                "release_file": source_name,
                "approval_code": ap_code,
                "approval_type": ap_type,
                "loinc_code": loinc,
                "section_name": name,
                "section_text": text,
                "active_ingredient_name": ing_name,
                "active_ingredient_unii": ing_unii,
            }
        )
    return out


extract = SPLXMLExtractor().extract

__all__ = ["SECTION_CODE_NAMES", "SPLXMLExtractor", "extract"]
