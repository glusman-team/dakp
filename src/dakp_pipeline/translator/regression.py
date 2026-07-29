"""Legacy-informed regression guardrails (Milestone 8).

Asserts the produced assertion/edge set preserves the three DAKP edge families and their
Translator provenance semantics — without requiring edge-for-edge equality with the legacy
build (PLAN.md: "allowing improved coverage and improved mappings"). The invariants are the
family/provenance/label contracts the legacy DAKP established and the rebuild locks in:

* ``biolink:treats`` — FDA-approved condition assertions: ``clinical_approval_status`` is
  ``approved_for_condition`` with DailyMed **and** FAERS upstream.
* ``biolink:applied_to_treat`` — FAERS-observed use without approval: keeps the current FAERS
  label/status (``observed_use`` / ``statistical_association``) with FAERS as the primary
  upstream source.
* ``biolink:contraindicated_in`` — MEDI/DailyMed contraindication assertions with MEDI **and**
  DailyMed upstream.

Every family aggregates under the DAKP knowledge provider (``infores:multiomics-drugapprovals``)
as ``primary_knowledge_source``. A family being *absent* is a coverage concern, not a regression
violation; :attr:`RegressionReport.families_seen` records which families appeared.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.translator.contract import (
    INFORES_DAILYMED,
    INFORES_DAKP,
    INFORES_FAERS,
    INFORES_MEDI,
    PREDICATE_APPLIED_TO_TREAT,
    PREDICATE_CONTRAINDICATED_IN,
    PREDICATE_TREATS,
)


@dataclass(frozen=True)
class FamilyInvariant:
    """The legacy provenance/label contract one edge family must preserve."""

    predicate: str
    required_upstream: frozenset[str]
    clinical_approval_status: str | None  # required value, or None when unconstrained
    knowledge_level: str | None


# Insertion order is the canonical family order.
FAMILY_INVARIANTS: dict[str, FamilyInvariant] = {
    PREDICATE_TREATS: FamilyInvariant(
        PREDICATE_TREATS, frozenset({INFORES_DAILYMED, INFORES_FAERS}), "approved_for_condition", "knowledge_assertion"
    ),
    PREDICATE_APPLIED_TO_TREAT: FamilyInvariant(PREDICATE_APPLIED_TO_TREAT, frozenset({INFORES_FAERS}), "observed_use", "statistical_association"),
    PREDICATE_CONTRAINDICATED_IN: FamilyInvariant(
        PREDICATE_CONTRAINDICATED_IN, frozenset({INFORES_MEDI, INFORES_DAILYMED}), None, "knowledge_assertion"
    ),
}

EXPECTED_FAMILIES: tuple[str, ...] = tuple(FAMILY_INVARIANTS)


@dataclass(frozen=True)
class RegressionViolation:
    """One invariant broken by one or more rows of a family."""

    family: str
    invariant: str
    message: str


@dataclass
class RegressionReport:
    ok: bool
    families_seen: list[str] = field(default_factory=list)
    row_count: int = 0
    violations: list[RegressionViolation] = field(default_factory=list)


@dataclass
class _Offender:
    """Mutable per-(family, invariant) aggregation bucket."""

    count: int
    example: str


def _record(offenders: dict[tuple[str, str], _Offender], family: str, invariant: str, message: str) -> None:
    """Aggregate one offending row into a per-(family, invariant) count + first example."""
    key = (family, invariant)
    bucket = offenders.get(key)
    if bucket is None:
        offenders[key] = _Offender(count=1, example=message)
    else:
        bucket.count += 1


def check_rows(rows: Iterable[Mapping[str, object]]) -> RegressionReport:
    """Check family/provenance/label invariants over normalized assertion rows (pure).

    Rows whose predicate is not a DAKP family are ignored (out of scope). Violations are
    aggregated per ``(family, invariant)`` with an offending-row count and one example.
    """
    families_seen: set[str] = set()
    offenders: dict[tuple[str, str], _Offender] = {}
    row_count = 0

    for row in rows:
        row_count += 1
        predicate = str(row.get("predicate") or "").strip()
        invariant = FAMILY_INVARIANTS.get(predicate)
        if invariant is None:
            continue
        families_seen.add(predicate)

        primary = str(row.get("primary_knowledge_source") or "").strip()
        if primary != INFORES_DAKP:
            _record(offenders, predicate, "primary_knowledge_source", f"expected {INFORES_DAKP!r}, got {primary!r}")

        upstream = {token for token in str(row.get("upstream_resource_ids") or "").split("|") if token}
        missing = sorted(invariant.required_upstream - upstream)
        if missing:
            _record(offenders, predicate, "upstream_provenance", f"missing upstream infores {missing}")

        if invariant.clinical_approval_status is not None:
            status = str(row.get("clinical_approval_status") or "").strip()
            if status != invariant.clinical_approval_status:
                _record(offenders, predicate, "clinical_approval_status", f"expected {invariant.clinical_approval_status!r}, got {status!r}")

        if invariant.knowledge_level is not None:
            knowledge_level = str(row.get("knowledge_level") or "").strip()
            if knowledge_level != invariant.knowledge_level:
                _record(offenders, predicate, "knowledge_level", f"expected {invariant.knowledge_level!r}, got {knowledge_level!r}")

    violations = [
        RegressionViolation(family, invariant, f"{bucket.count} row(s): {bucket.example}")
        for (family, invariant), bucket in sorted(offenders.items())
    ]
    return RegressionReport(ok=not violations, families_seen=sorted(families_seen), row_count=row_count, violations=violations)


def check_assertion_tables(refs: list[ArtifactRef]) -> RegressionReport:
    """Run :func:`check_rows` over every DAKP assertion table found among ``refs``."""
    rows: list[Mapping[str, object]] = []
    for ref in refs:
        if ref.uri.stem not in schemas.ASSERTION_TABLES:
            continue
        rows.extend(schemas.read_table(ref.uri).iter_rows(named=True))
    return check_rows(rows)


__all__ = [
    "EXPECTED_FAMILIES",
    "FAMILY_INVARIANTS",
    "FamilyInvariant",
    "RegressionReport",
    "RegressionViolation",
    "check_assertion_tables",
    "check_rows",
]
