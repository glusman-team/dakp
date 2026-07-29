"""FAERS ASCII extractor (stub).

Parses the tiny mock FAERS ASCII fixture (``$``-delimited DEMO/DRUG/INDI files for one
quarter) into a joined case-level interim table with the PLAN.md ``faers_cases`` column
contract. Real partitioned FAERS extraction (per-quarter DEMO/DRUG/INDI/REAC/RPSR/DELETE
parquet + case-level joins + dedup/delete audit) lands in **Milestone 3**.

Modern FAERS ASCII files are ``$``-delimited; the mock fixture mirrors that.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.paths import Workdir

_QUARTER = "24Q3"
_DELIMITER = "$"


class FAERSASCIIExtractor:
    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        by_family = _partition_by_family(inputs)
        if not by_family:
            return []

        demo = _read_records(by_family.get("DEMO", []))
        drug = _read_records(by_family.get("DRUG", []))
        indi = _read_records(by_family.get("INDI", []))

        rows = _join_cases(demo, drug, indi)
        frame = pl.DataFrame(rows, schema=schemas.FAERS_CASES_COLUMNS) if rows else pl.DataFrame(schema=schemas.FAERS_CASES_COLUMNS)
        out = Workdir(ctx.workdir).interim / "faers" / "cases.parquet"
        rows_written = schemas.write_parquet(frame, out)

        input_ids = [ref.blake3 for refs in by_family.values() for ref in refs]
        store = ArtifactStore(Workdir(ctx.workdir))
        ref = store.register(
            out,
            media_type=schemas.PARQUET_MEDIA_TYPE,
            rows=rows_written,
            schema_fingerprint=schemas.schema_fingerprint(schemas.FAERS_CASES_COLUMNS),
            inputs=input_ids,
            operation=OperationBlock(name="extract_faers_cases"),
            table=TableBlock(rows=rows_written, schema_fingerprint=schemas.schema_fingerprint(schemas.FAERS_CASES_COLUMNS)),
        )
        return [ref]


def _partition_by_family(inputs: list[ArtifactRef]) -> dict[str, list[ArtifactRef]]:
    grouped: dict[str, list[ArtifactRef]] = defaultdict(list)
    for ref in inputs:
        family = _family_of(ref.uri.name)
        if family is not None:
            grouped[family].append(ref)
    return grouped


def _family_of(filename: str) -> str | None:
    upper = filename.upper()
    for family in ("DEMO", "DRUG", "INDI", "REAC", "RPSR", "DELETE"):
        if family in upper:
            return family
    return None


def _read_records(refs: list[ArtifactRef]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for ref in refs:
        records.extend(_parse_dollar_tsv(ref.uri))
    return records


def _parse_dollar_tsv(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split(_DELIMITER)
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        parts = line.split(_DELIMITER)
        rows.append({col: parts[i] if i < len(parts) else "" for i, col in enumerate(header)})
    return rows


def _join_cases(demo: list[dict[str, str]], drug: list[dict[str, str]], indi: list[dict[str, str]]) -> list[dict[str, str]]:
    # Index indications by primaryid (FAERS joins DRUG/INDI within a case on primaryid).
    indications_by_case: dict[str, list[str]] = defaultdict(list)
    for rec in indi:
        pid = rec.get("primaryid", "")
        indications_by_case[pid].append(rec.get("indi_pt", ""))

    rows: list[dict[str, str]] = []
    demo_by_pid = {rec.get("primaryid", ""): rec for rec in demo}
    for drec in drug:
        pid = drec.get("primaryid", "")
        case = demo_by_pid.get(pid, {})
        indications = indications_by_case.get(pid, [])
        indication_text = "; ".join(filter(None, indications))
        rows.append(
            {
                "quarter": _QUARTER,
                "primaryid": pid,
                "caseid": case.get("caseid", ""),
                "source": "faers",
                "occp_cod": case.get("occp_cod", ""),
                "reporter_country": case.get("reporter_country", ""),
                "drugname": drec.get("drugname", ""),
                "ingredient": drec.get("ingredient", drec.get("drugname", "")),
                "nda": _normalize_nda(drec.get("nda", "")),
                "indication": indication_text,
                "effects": "",
            }
        )
    return rows


def _normalize_nda(nda: str) -> str:
    """Strip non-digits so NDA numbers join consistently with Drugs@FDA ApplNo."""
    return "".join(ch for ch in nda if ch.isdigit())


extract = FAERSASCIIExtractor().extract

__all__ = ["FAERSASCIIExtractor", "extract"]
