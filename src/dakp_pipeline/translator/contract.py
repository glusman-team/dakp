"""Translator-readiness contract (stub).

Minimal, dependency-free checks that the three assertion tables exist with their
expected column contracts. Full Biolink/Translator validation (predicate/category
compatibility, required provenance fields, no dangling node references) lands in
Milestone 6+ and is largely delegated to Tablassert's QC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef


@dataclass
class ContractReport:
    ok: bool
    tables: dict[str, dict[str, object]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def validate(assertion_refs: list[ArtifactRef]) -> ContractReport:
    """Check each assertion table is present with its declared column contract."""
    report = ContractReport(ok=True)
    found = {ref.uri.stem: ref for ref in assertion_refs}
    for table in schemas.ASSERTION_TABLES:
        expected = schemas.columns_for(table)
        if table not in found:
            report.ok = False
            report.problems.append(f"missing assertion table: {table}")
            continue
        ref = found[table]
        try:
            frame = schemas.read_table(ref.uri)
        except Exception as exc:
            report.ok = False
            report.problems.append(f"unreadable table {table}: {exc}")
            continue
        missing = [c for c in expected if c not in frame.columns]
        report.tables[table] = {"rows": frame.height, "missing_columns": missing}
        if missing:
            report.ok = False
            report.problems.append(f"{table} missing columns: {missing}")
    return report


__all__ = ["ContractReport", "validate"]
