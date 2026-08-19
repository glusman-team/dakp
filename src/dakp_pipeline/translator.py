"""Translator-readiness layer: KGX/Translator contract + legacy-informed regression guardrails.

Flattened from the former ``translator/contract.py`` + ``translator/regression.py`` (US-004);
import-safe and monkeypatchable. Three public entry points:

* :func:`validate` — the lightweight assertion-table contract used by the runner/DAG:
  every public assertion table exists with its declared column contract.
* :func:`validate_kgx` — full KGX/Translator contract validation over the
  node/edge JSONL that Tablassert emits. Since Tablassert is not run in tests, callers
  validate small fixture KGX that mirrors its output. Checks, in order per entity:

  1. **node coverage** — every edge ``subject``/``object`` id exists in the node set;
  2. **required Biolink node fields** — ``id``, ``name``, and a ``category`` list whose
     entries all carry the ``biolink:`` prefix;
  3. **required edge fields** — ``id``, ``subject``, ``predicate``, ``object``, ``category``,
     ``knowledge_level``, ``agent_type``, and provenance (``primary_knowledge_source`` and/or
     ``sources``);
  4. **category/predicate compatibility** — the three DAKP edge families (``treats`` /
     ``applied_to_treat`` / ``contraindicated_in``) with chemical/drug subject categories and
     Disease/PhenotypicFeature object categories, matching ``../DINGO`` ``dakp_rig.yaml``
     ``edge_type_info``;
  5. **source provenance** — the ``infores:multiomics-drugapprovals`` chain plus the upstream
     infores required per family (dailymed/faers).

  Problems are returned both as structured :class:`ContractProblem` records (``kgx_problems``)
  and as rendered strings (``problems``) so the existing build summary keeps working.
* :func:`check_rows` / :func:`check_assertion_tables` — regression guardrails asserting the
  produced assertion/edge set preserves the three DAKP edge families and their Translator
  provenance semantics — without requiring edge-for-edge equality with the legacy build
  (allowing improved coverage and improved mappings). The invariants are the
  family/provenance/label contracts the legacy DAKP established and the rebuild locks in:

  * ``biolink:treats`` — FDA-approved condition assertions: ``clinical_approval_status`` is
    ``approved_for_condition`` with DailyMed **and** FAERS upstream.
  * ``biolink:applied_to_treat`` — FAERS-observed use cross-referenced with the approved-treats
    table: ``clinical_approval_status`` is ``approved_for_condition`` when the pair has a treats
    counterpart, else ``off_label_use`` (``not_provided`` only in the degraded no-approved-table
    mode) — all biolink-valid ``ClinicalApprovalStatusEnum`` members (the legacy ``observed_use``
    label is not one) — with FAERS as the primary upstream source.
  * ``biolink:contraindicated_in`` — contraindication assertions text-mined from DailyMed SPL
    contraindication sections, with DailyMed upstream.

  Every family aggregates under the DAKP knowledge provider (``infores:multiomics-drugapprovals``)
  as ``primary_knowledge_source``. A family being *absent* is a coverage concern, not a regression
  violation; :attr:`RegressionReport.families_seen` records which families appeared.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.logging_setup import logger, stats

# --- Translator provenance constants (match dakp_pipeline.assertions + ../DINGO) ----

INFORES_DAKP = "infores:multiomics-drugapprovals"
INFORES_DAILYMED = "infores:dailymed"
INFORES_FAERS = "infores:faers"

BIOLINK_PREFIX = "biolink:"

# --- DAKP edge-family + category contract (matches ../DINGO dakp_rig.yaml) ----------

# Ordered tuples (stable emission order for RIG/messages); frozensets for membership.
CHEMICAL_DRUG_CATEGORIES: tuple[str, ...] = (
    "biolink:ChemicalEntity",
    "biolink:SmallMolecule",
    "biolink:MolecularMixture",
    "biolink:ComplexMolecularMixture",
    "biolink:Drug",
)
DISEASE_PHENOTYPE_CATEGORIES: tuple[str, ...] = ("biolink:Disease", "biolink:PhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature")
_CHEMICAL_SET = frozenset(CHEMICAL_DRUG_CATEGORIES)
_DISEASE_SET = frozenset(DISEASE_PHENOTYPE_CATEGORIES)

PREDICATE_TREATS = "biolink:treats"
PREDICATE_APPLIED_TO_TREAT = "biolink:applied_to_treat"
PREDICATE_CONTRAINDICATED_IN = "biolink:contraindicated_in"


@dataclass(frozen=True)
class EdgeFamily:
    """One DAKP edge family: its predicate and the upstream infores it must carry."""

    predicate: str
    required_upstream: frozenset[str]


# Insertion order is the canonical family order (treats, applied_to_treat, contraindicated_in).
EDGE_FAMILIES: dict[str, EdgeFamily] = {
    PREDICATE_TREATS: EdgeFamily(PREDICATE_TREATS, frozenset({INFORES_DAILYMED, INFORES_FAERS})),
    PREDICATE_APPLIED_TO_TREAT: EdgeFamily(PREDICATE_APPLIED_TO_TREAT, frozenset({INFORES_FAERS, INFORES_DAILYMED})),
    PREDICATE_CONTRAINDICATED_IN: EdgeFamily(PREDICATE_CONTRAINDICATED_IN, frozenset({INFORES_DAILYMED})),
}

# --- required KGX fields -----------------------------------------------------------

REQUIRED_NODE_FIELDS: tuple[str, ...] = ("id", "name", "category")
REQUIRED_EDGE_FIELDS: tuple[str, ...] = ("id", "subject", "predicate", "object", "category", "knowledge_level", "agent_type")

# --- structured problem codes ------------------------------------------------------

MISSING_NODE_REFERENCE = "missing_node_reference"
MISSING_NODE_FIELD = "missing_node_field"
INVALID_NODE_CATEGORY = "invalid_node_category"
DUPLICATE_NODE_ID = "duplicate_node_id"
MISSING_EDGE_FIELD = "missing_edge_field"
DUPLICATE_EDGE_ID = "duplicate_edge_id"
INVALID_PREDICATE = "invalid_predicate"
INCOMPATIBLE_SUBJECT_CATEGORY = "incompatible_subject_category"
INCOMPATIBLE_OBJECT_CATEGORY = "incompatible_object_category"
MISSING_PROVENANCE = "missing_provenance"


@dataclass(frozen=True)
class ContractProblem:
    """One structured KGX contract violation."""

    code: str
    entity_kind: str  # "node" | "edge"
    entity_id: str
    field: str
    message: str

    def render(self) -> str:
        """Human-readable one-liner (used for the legacy ``problems`` string list)."""
        return f"[{self.code}] {self.message}"


@dataclass
class ContractReport:
    ok: bool
    tables: dict[str, dict[str, object]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    kgx_problems: list[ContractProblem] = field(default_factory=list)


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
        stats(logger, "translator_contract", table=table, rows=frame.height, missing_columns=",".join(missing) if missing else "-")
        if missing:
            report.ok = False
            report.problems.append(f"{table} missing columns: {missing}")
    for problem in report.problems:
        logger.warning("translator_contract: problem = {}", problem)
    stats(logger, "translator_contract", ok=report.ok, problems=len(report.problems))
    return report


# --- KGX loading -------------------------------------------------------------------


def _jsonl_records(handle: Iterable[str]) -> list[dict[str, Any]]:
    """Parse non-empty JSONL lines into records."""
    return [json.loads(line) for line in handle if line.strip()]


def read_kgx_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a KGX JSONL file (plain ``.jsonl`` or gzipped ``.jsonl.gz``) into records."""
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return _jsonl_records(handle)
    with path.open(encoding="utf-8") as handle:
        return _jsonl_records(handle)


# --- KGX validation ----------------------------------------------------------------


def _as_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _category_set(record: Mapping[str, Any]) -> set[str]:
    """The record's ``category`` entries as a set of strings (empty if malformed)."""
    category = record.get("category")
    if not isinstance(category, list):
        return set()
    return {entry.strip() for entry in category if isinstance(entry, str) and entry.strip()}


def _edge_infores(edge: Mapping[str, Any]) -> set[str]:
    """All infores ids referenced by an edge (primary + sources + upstream chains)."""
    found: set[str] = set()
    primary = _as_str(edge.get("primary_knowledge_source"))
    if primary:
        found.add(primary)
    sources = edge.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for key in ("resource_id", "id"):
                resource = _as_str(source.get(key))
                if resource:
                    found.add(resource)
            upstream = source.get("upstream_resource_ids")
            if isinstance(upstream, list):
                found.update(_as_str(entry) for entry in upstream if _as_str(entry))
    return found


def _check_node(node: Mapping[str, Any], index: int, seen_ids: set[str], problems: list[ContractProblem]) -> None:
    entity_id = _as_str(node.get("id")) or f"<node#{index}>"
    node_id = _as_str(node.get("id"))
    if not node_id:
        problems.append(ContractProblem(MISSING_NODE_FIELD, "node", entity_id, "id", f"node {entity_id} missing required field: id"))
    elif node_id in seen_ids:
        problems.append(ContractProblem(DUPLICATE_NODE_ID, "node", entity_id, "id", f"duplicate node id: {node_id}"))
    else:
        seen_ids.add(node_id)

    if not _as_str(node.get("name")):
        problems.append(ContractProblem(MISSING_NODE_FIELD, "node", entity_id, "name", f"node {entity_id} missing required field: name"))

    category = node.get("category")
    if not isinstance(category, list) or not category:
        problems.append(ContractProblem(INVALID_NODE_CATEGORY, "node", entity_id, "category", f"node {entity_id} category must be a non-empty list"))
        return
    bad = [entry for entry in category if not (isinstance(entry, str) and entry.startswith(BIOLINK_PREFIX))]
    if bad:
        problems.append(
            ContractProblem(
                INVALID_NODE_CATEGORY, "node", entity_id, "category", f"node {entity_id} category entries must use the biolink: prefix: {bad}"
            )
        )


def _check_edge(
    edge: Mapping[str, Any], index: int, node_index: Mapping[str, Mapping[str, Any]], seen_ids: set[str], problems: list[ContractProblem]
) -> None:
    entity_id = _as_str(edge.get("id")) or f"<edge#{index}>"
    edge_id = _as_str(edge.get("id"))
    if not edge_id:
        problems.append(ContractProblem(MISSING_EDGE_FIELD, "edge", entity_id, "id", f"edge {entity_id} missing required field: id"))
    elif edge_id in seen_ids:
        problems.append(ContractProblem(DUPLICATE_EDGE_ID, "edge", entity_id, "id", f"duplicate edge id: {edge_id}"))
    else:
        seen_ids.add(edge_id)

    # Required scalar fields (id already handled above).
    for field_name in ("subject", "predicate", "object", "knowledge_level", "agent_type"):
        if not _as_str(edge.get(field_name)):
            problems.append(
                ContractProblem(MISSING_EDGE_FIELD, "edge", entity_id, field_name, f"edge {entity_id} missing required field: {field_name}")
            )
    if not _category_set(edge):
        problems.append(ContractProblem(MISSING_EDGE_FIELD, "edge", entity_id, "category", f"edge {entity_id} category must be a non-empty list"))

    predicate = _as_str(edge.get("predicate"))
    family = EDGE_FAMILIES.get(predicate)
    if predicate and family is None:
        problems.append(
            ContractProblem(INVALID_PREDICATE, "edge", entity_id, "predicate", f"edge {entity_id} predicate {predicate!r} is not a DAKP edge family")
        )

    # (1) node coverage + (4) category/predicate compatibility.
    subject = _as_str(edge.get("subject"))
    obj = _as_str(edge.get("object"))
    subject_node = node_index.get(subject) if subject else None
    object_node = node_index.get(obj) if obj else None
    if subject and subject_node is None:
        problems.append(
            ContractProblem(MISSING_NODE_REFERENCE, "edge", entity_id, "subject", f"edge {entity_id} subject {subject!r} not found in nodes")
        )
    if obj and object_node is None:
        problems.append(ContractProblem(MISSING_NODE_REFERENCE, "edge", entity_id, "object", f"edge {entity_id} object {obj!r} not found in nodes"))
    if family is not None:
        if subject_node is not None and not (_category_set(subject_node) & _CHEMICAL_SET):
            problems.append(
                ContractProblem(
                    INCOMPATIBLE_SUBJECT_CATEGORY,
                    "edge",
                    entity_id,
                    "subject",
                    f"edge {entity_id} subject {subject!r} categories are not chemical/drug for {predicate}",
                )
            )
        if object_node is not None and not (_category_set(object_node) & _DISEASE_SET):
            problems.append(
                ContractProblem(
                    INCOMPATIBLE_OBJECT_CATEGORY,
                    "edge",
                    entity_id,
                    "object",
                    f"edge {entity_id} object {obj!r} categories are not disease/phenotype for {predicate}",
                )
            )

    # (5) source provenance: presence, then the per-family infores chain.
    has_primary = bool(_as_str(edge.get("primary_knowledge_source")))
    sources = edge.get("sources")
    has_sources = isinstance(sources, list) and bool(sources)
    if not has_primary and not has_sources:
        problems.append(
            ContractProblem(
                MISSING_PROVENANCE,
                "edge",
                entity_id,
                "primary_knowledge_source/sources",
                f"edge {entity_id} has neither primary_knowledge_source nor sources",
            )
        )
    else:
        infores = _edge_infores(edge)
        if family is not None:
            required = family.required_upstream | {INFORES_DAKP}
            missing = sorted(required - infores)
            if missing:
                problems.append(
                    ContractProblem(
                        MISSING_PROVENANCE, "edge", entity_id, "sources", f"edge {entity_id} missing provenance infores for {predicate}: {missing}"
                    )
                )
        elif INFORES_DAKP not in infores:
            problems.append(
                ContractProblem(
                    MISSING_PROVENANCE, "edge", entity_id, "primary_knowledge_source", f"edge {entity_id} missing DAKP infores {INFORES_DAKP}"
                )
            )


def validate_kgx(nodes: Iterable[Mapping[str, Any]], edges: Iterable[Mapping[str, Any]]) -> ContractReport:
    """Validate KGX node/edge records against the DAKP Translator contract.

    Returns a :class:`ContractReport` whose ``kgx_problems`` carries structured
    :class:`ContractProblem` records and whose ``problems`` carries rendered one-liners.
    """
    node_list = list(nodes)
    edge_list = list(edges)
    problems: list[ContractProblem] = []

    node_index: dict[str, Mapping[str, Any]] = {}
    seen_node_ids: set[str] = set()
    for index, node in enumerate(node_list):
        _check_node(node, index, seen_node_ids, problems)
        node_id = _as_str(node.get("id"))
        if node_id:
            node_index.setdefault(node_id, node)

    seen_edge_ids: set[str] = set()
    for index, edge in enumerate(edge_list):
        _check_edge(edge, index, node_index, seen_edge_ids, problems)

    report = ContractReport(ok=not problems, kgx_problems=problems, problems=[problem.render() for problem in problems])
    stats(logger, "kgx_contract", level="DEBUG", nodes=len(node_list), edges=len(edge_list), problems=len(problems), ok=report.ok)
    return report


# --- legacy-informed regression guardrails -----------------------------------------


@dataclass(frozen=True)
class FamilyInvariant:
    """The legacy provenance/label contract one edge family must preserve."""

    predicate: str
    required_upstream: frozenset[str]
    clinical_approval_status: frozenset[str] | None  # allowed values, or None when unconstrained
    knowledge_level: str | None


# Insertion order is the canonical family order.
FAMILY_INVARIANTS: dict[str, FamilyInvariant] = {
    PREDICATE_TREATS: FamilyInvariant(
        PREDICATE_TREATS, frozenset({INFORES_DAILYMED, INFORES_FAERS}), frozenset({"approved_for_condition"}), "knowledge_assertion"
    ),
    # applied_to_treat: approved_for_condition (treats counterpart) / off_label_use (none) /
    # not_provided (no approved-treats table to check against — degraded mode).
    PREDICATE_APPLIED_TO_TREAT: FamilyInvariant(
        PREDICATE_APPLIED_TO_TREAT,
        frozenset({INFORES_FAERS}),
        frozenset({"approved_for_condition", "off_label_use", "not_provided"}),
        "statistical_association",
    ),
    PREDICATE_CONTRAINDICATED_IN: FamilyInvariant(PREDICATE_CONTRAINDICATED_IN, frozenset({INFORES_DAILYMED}), None, "knowledge_assertion"),
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
            if status not in invariant.clinical_approval_status:
                _record(
                    offenders, predicate, "clinical_approval_status", f"expected one of {sorted(invariant.clinical_approval_status)}, got {status!r}"
                )

        if invariant.knowledge_level is not None:
            knowledge_level = str(row.get("knowledge_level") or "").strip()
            if knowledge_level != invariant.knowledge_level:
                _record(offenders, predicate, "knowledge_level", f"expected {invariant.knowledge_level!r}, got {knowledge_level!r}")

    violations = [
        RegressionViolation(family, invariant, f"{bucket.count} row(s): {bucket.example}")
        for (family, invariant), bucket in sorted(offenders.items())
    ]
    return RegressionReport(ok=not violations, families_seen=sorted(families_seen), row_count=row_count, violations=violations)


def _iter_assertion_rows(path: Path) -> Iterator[Mapping[str, object]]:
    """Stream one assertion table's rows as dicts in bounded memory.

    The table (TSV — the on-disk form of every DAKP assertion table — or a parquet interim) is
    scanned lazily and collected in batches via the streaming engine, so row values are
    identical to :func:`~dakp_pipeline.io.schemas.read_table` — only the materialization
    strategy differs.
    """
    scan = pl.scan_parquet(path) if path.suffix == ".parquet" else pl.scan_csv(path, separator="\t")
    for batch in scan.collect_batches():
        yield from batch.iter_rows(named=True)


def check_assertion_tables(refs: list[ArtifactRef]) -> RegressionReport:
    """Run :func:`check_rows` over every DAKP assertion table found among ``refs``.

    Tables stream one at a time (:func:`_iter_assertion_rows`) and the per-table row streams
    chain lazily into ``check_rows``, so the full row set is never materialized as one Python
    list — the production FAERS-derived tables can carry millions of rows. The checks are
    row-local (only the per-(family, invariant) offender aggregation is global state), so
    per-table streaming is semantically identical to the former whole-list read.
    """

    def rows() -> Iterator[Mapping[str, object]]:
        for ref in refs:
            if ref.uri.stem not in schemas.ASSERTION_TABLES:
                continue
            yield from _iter_assertion_rows(ref.uri)

    report = check_rows(rows())
    stats(
        logger,
        "translator_regression",
        rows_checked=report.row_count,
        families_seen=",".join(report.families_seen) if report.families_seen else "-",
        violations=len(report.violations),
        ok=report.ok,
    )
    for violation in report.violations:
        logger.warning("translator_regression: violation = {} / {}", violation.family, violation.invariant)
    return report


__all__ = [
    "BIOLINK_PREFIX",
    "CHEMICAL_DRUG_CATEGORIES",
    "DISEASE_PHENOTYPE_CATEGORIES",
    "DUPLICATE_EDGE_ID",
    "DUPLICATE_NODE_ID",
    "EDGE_FAMILIES",
    "EXPECTED_FAMILIES",
    "FAMILY_INVARIANTS",
    "INCOMPATIBLE_OBJECT_CATEGORY",
    "INCOMPATIBLE_SUBJECT_CATEGORY",
    "INFORES_DAILYMED",
    "INFORES_DAKP",
    "INFORES_FAERS",
    "INVALID_NODE_CATEGORY",
    "INVALID_PREDICATE",
    "MISSING_EDGE_FIELD",
    "MISSING_NODE_FIELD",
    "MISSING_NODE_REFERENCE",
    "MISSING_PROVENANCE",
    "PREDICATE_APPLIED_TO_TREAT",
    "PREDICATE_CONTRAINDICATED_IN",
    "PREDICATE_TREATS",
    "REQUIRED_EDGE_FIELDS",
    "REQUIRED_NODE_FIELDS",
    "ContractProblem",
    "ContractReport",
    "EdgeFamily",
    "FamilyInvariant",
    "RegressionReport",
    "RegressionViolation",
    "check_assertion_tables",
    "check_rows",
    "read_kgx_jsonl",
    "validate",
    "validate_kgx",
]
