"""Generate Tablassert Graph + per-table config YAMLs (Milestone 7).

DAKP does everything up to the shape Tablassert consumes, then emits ONE Graph config
(``tables/graph.yaml``) plus one table config per assertion table. The configs match the
ACTUAL current Tablassert 8.x schema (verified against ``../Tablassert/src/tablassert/
models.py`` and its ``ingests.to_sections`` loader):

* a table config is a ``template:``-wrapped :class:`~tablassert.models.Section`
  (``source`` / ``statement`` / ``provenance`` / ``annotations``). The loader only reads
  top-level ``template`` / ``sections`` keys — a bare ``source:``/``statement:`` shape is
  silently dropped, so the ``template:`` wrapper is mandatory;
* ``source.kind: text`` with a tab ``delimiter`` and the uncompressed assertion ``.tsv``
  as ``source.local`` (a ``url`` is required by the model and recorded as provenance);
* column-encoded ``statement.subject`` / ``statement.object`` / ``statement.predicate``
  with drug / disease ``prioritize`` categories;
* a ``provenance.override`` (:class:`~tablassert.models.ManualProvenance`) block carrying
  the DAKP infores, the DINGO-conventional upstream infores chain, ``knowledge_level`` and
  ``agent_type`` (no ``publication`` — the override replaces repo/publication provenance);
* column-encoded evidence ``annotations``.

Column letters are DERIVED from the assertion-table column contracts in
:mod:`dakp_pipeline.io.schemas` (never hardcoded). YAML is emitted by a tiny stdlib
emitter (no ``pyyaml`` runtime dependency) that round-trips through ``yaml.safe_load``
(asserted in the unit tests).
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from dakp_pipeline import __version__
from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock
from dakp_pipeline.paths import Workdir

# --- Translator provenance constants (match dakp_pipeline.assertions + ../DINGO) ----

INFORES_DAKP = "infores:multiomics-drugapprovals"
AGENT_TYPE = "manual_validation_of_automated_agent"

GRAPH_NAME = "dakp"
FULLMAP_DEFAULT = ".fullmap"
SOURCE_URL_BASE = "https://example.invalid/dakp/generated"
GRAPH_DESCRIPTION = (
    "Drug Approvals Knowledge Provider: FDA-approved treatment relationships, "
    "FAERS-observed applied-to-treat uses, and contraindications, modeled from "
    "DailyMed, Drugs@FDA, FAERS, and MEDI."
)

# Canonical emission order for the three assertion tables.
_TABLE_ORDER = ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions")

# assertion table -> (config file basename, predicate, upstream infores chain, knowledge_level).
# Upstream order + knowledge_level match the DINGO translator-ingest provenance contract
# (../DINGO/tests/unit/ingests/dakp/test_dakp.py): treats = knowledge_assertion over
# dailymed|faers; applied_to_treat = observation over faers|dailymed (current FAERS
# label/status behavior); contraindicated_in = knowledge_assertion over medi|dailymed.
_TABLE_SPECS: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "approved_treats_assertions": ("approved_treats", "treats", ("infores:dailymed", "infores:faers"), "knowledge_assertion"),
    "faers_applied_to_treat_assertions": ("faers_applied_to_treat", "applied_to_treat", ("infores:faers", "infores:dailymed"), "observation"),
    "contraindication_assertions": ("contraindications", "contraindicated_in", ("infores:medi", "infores:dailymed"), "knowledge_assertion"),
}

# assertion column -> annotation name, per table. Names equal the assertion column except
# where DINGO establishes a Translator slot: ``case_count`` -> ``number_of_cases`` (DINGO
# maps FAERS case counts to the Biolink ``number_of_cases`` edge slot). ``clinical_approval_status``
# is itself a Biolink slot (written verbatim); the remaining names fold into ``supporting_text``.
_TABLE_ANNOTATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "approved_treats_assertions": (
        ("approval_ids", "approval_ids"),
        ("supporting_spl_sets", "supporting_spl_sets"),
        ("clinical_approval_status", "clinical_approval_status"),
    ),
    "faers_applied_to_treat_assertions": (("case_count", "number_of_cases"), ("clinical_approval_status", "clinical_approval_status")),
    "contraindication_assertions": (
        ("supporting_spl_sets", "supporting_spl_sets"),
        ("medi_version", "medi_version"),
        ("source_score", "source_score"),
    ),
}

SUBJECT_COLUMN = "subject_text"
OBJECT_COLUMN = "object_text"
SUBJECT_PRIORITIZE = ("Drug", "SmallMolecule", "ChemicalEntity")
OBJECT_PRIORITIZE = ("Disease", "PhenotypicFeature")

_OPERATION = "generate_tablassert_configs"


# --- Excel-style column letters ---------------------------------------------------


def excel_column(index: int) -> str:
    """0-based column index -> Excel-style letters (``0->A``, ``25->Z``, ``26->AA``).

    Tablassert reads source files headerless and addresses columns by Excel-style letters
    (``EncodingMethods.COLUMN``); this maps an assertion-table column position to its letter.
    """
    if index < 0:
        msg = f"column index must be >= 0, got {index}"
        raise ValueError(msg)
    letters = ""
    n = index + 1
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def column_letter(table: str, column: str) -> str:
    """Excel-style letter for ``column`` in ``table``'s ordered contract (KeyError if absent)."""
    columns = schemas.columns_for(table)
    if column not in columns:
        msg = f"column {column!r} not in {table} contract: {columns}"
        raise KeyError(msg)
    return excel_column(columns.index(column))


# --- config dict builders (single source of truth) --------------------------------


def table_config(table: str) -> dict[str, Any]:
    """Build the Tablassert ``Section`` body (the ``template:`` value) for an assertion table.

    Shape verified against ``tablassert.models.Section``: ``source`` (text), ``statement``
    (column-encoded subject/predicate/object), ``provenance.override`` (ManualProvenance),
    and column-encoded ``annotations`` for the table's evidence columns.
    """
    _basename, predicate, upstream, knowledge_level = _TABLE_SPECS[table]  # KeyError for unknown tables
    annotations = [
        {"annotation": annotation, "method": "column", "encoding": column_letter(table, column)} for column, annotation in _TABLE_ANNOTATIONS[table]
    ]
    return {
        "source": {"kind": "text", "local": f"data/tabular/{table}.tsv", "url": f"{SOURCE_URL_BASE}/{table}.tsv", "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": column_letter(table, SUBJECT_COLUMN), "prioritize": list(SUBJECT_PRIORITIZE)},
            "predicate": predicate,
            "object": {"method": "column", "encoding": column_letter(table, OBJECT_COLUMN), "prioritize": list(OBJECT_PRIORITIZE)},
        },
        "provenance": {
            "override": {
                "infores": INFORES_DAKP,
                "upstream_resource_ids": list(upstream),
                "knowledge_level": knowledge_level,
                "agent_type": AGENT_TYPE,
            }
        },
        "annotations": annotations,
    }


def graph_config(tables: list[str] | None = None, version: str | None = None) -> dict[str, Any]:
    """Build the Tablassert ``Graph`` config dict (verified against ``tablassert.models.Graph``).

    ``tables`` defaults to the three committed table configs (``tables/<basename>.yaml``);
    ``version`` defaults to the DAKP package version.
    """
    if tables is None:
        tables = [f"tables/{_TABLE_SPECS[table][0]}.yaml" for table in _TABLE_ORDER]
    return {
        "name": GRAPH_NAME,
        "version": version if version is not None else __version__,
        "description": GRAPH_DESCRIPTION,
        "infores": INFORES_DAKP,
        "fullmap": FULLMAP_DEFAULT,
        "tables": list(tables),
    }


# --- minimal stdlib YAML emitter (no pyyaml runtime dep; round-trips) -------------

_PLAIN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")
_RESERVED = {"", "~", "true", "false", "null", "yes", "no", "on", "off", "True", "False", "Null", "None"}
_FOLD_WIDTH = 70
_FOLD_THRESHOLD = 60


def _is_safe_plain(text: str) -> bool:
    """True when ``text`` may be emitted as an unquoted YAML plain scalar.

    Excludes YAML reserved words and anything with characters outside a conservative
    identifier-ish set (so infores CURIEs, URLs, and the tab delimiter are always quoted).
    A leading letter/underscore is required, which also keeps bare numbers quoted.
    """
    return text not in _RESERVED and bool(_PLAIN_RE.match(text))


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if _is_safe_plain(text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t").replace("\n", "\\n")
    return f'"{escaped}"'


def _should_fold(text: str) -> bool:
    """Long, space-bearing strings are emitted as ``>-`` folded blocks (readable committed configs)."""
    return len(text) > _FOLD_THRESHOLD and " " in text


def _wrap(text: str, width: int = _FOLD_WIDTH) -> list[str]:
    """Greedy word-wrap; ``>-`` folding joins the lines back with single spaces (lossless)."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


def _indent(level: int) -> str:
    return "  " * level


def _emit_mapping(mapping: dict[str, Any], level: int, lines: list[str]) -> None:
    pad = _indent(level)
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            _emit_mapping(value, level + 1, lines)
        elif isinstance(value, list):
            lines.append(f"{pad}{key}:")
            _emit_list(value, level + 1, lines)
        elif isinstance(value, str) and _should_fold(value):
            lines.append(f"{pad}{key}: >-")
            lines.extend(f"{_indent(level + 1)}{folded}" for folded in _wrap(value))
        else:
            lines.append(f"{pad}{key}: {_yaml_scalar(value)}")


def _emit_list(items: list[Any], level: int, lines: list[str]) -> None:
    pad = _indent(level)
    for item in items:
        if isinstance(item, dict):
            # Render as a mapping one level deeper, then rewrite the first line's indent into
            # a "- " dash (same width: one indent level == two spaces == "- ").
            sub: list[str] = []
            _emit_mapping(item, level + 1, sub)
            prefix = _indent(level + 1)
            sub[0] = f"{pad}- {sub[0][len(prefix) :]}"
            lines.extend(sub)
        else:
            lines.append(f"{pad}- {_yaml_scalar(item)}")


def _dump_yaml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _emit_mapping(data, 0, lines)
    return "\n".join(lines) + "\n"


# --- serialized config strings ----------------------------------------------------


def table_yaml(table: str) -> str:
    """Serialized table config YAML: the ``Section`` body wrapped in a top-level ``template:``."""
    return _dump_yaml({"template": table_config(table)})


def graph_yaml(tables: list[str] | None = None, version: str | None = None) -> str:
    """Serialized Graph config YAML."""
    return _dump_yaml(graph_config(tables=tables, version=version))


# --- runtime generation into the workdir ------------------------------------------


def generate(assertion_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Write ``tables/graph.yaml`` plus one table config per assertion table into the workdir.

    Configs land in ``<workdir>/tables/`` so their workdir-relative references
    (``tables/<name>.yaml``, ``data/tabular/<table>.tsv``, ``.fullmap``) resolve when
    Tablassert runs from the workdir root. Returns ``[graph_ref, *table_refs]`` in the
    canonical table order; assertion refs are linked as input provenance by table stem.
    """
    workdir = Workdir(ctx.workdir)
    store = ArtifactStore(workdir)
    tables_dir = workdir.root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    input_ids = {ref.uri.stem: ref.blake3 for ref in assertion_refs}
    operation = OperationBlock(name=_OPERATION)

    table_refs: list[ArtifactRef] = []
    table_paths: list[str] = []
    for table in _TABLE_ORDER:
        basename = _TABLE_SPECS[table][0]
        config_path = tables_dir / f"{basename}.yaml"
        config_path.write_text(table_yaml(table), encoding="utf-8")
        table_paths.append(f"tables/{basename}.yaml")
        inputs = [input_ids[table]] if table in input_ids else []
        table_refs.append(store.register(config_path, media_type="application/yaml", inputs=inputs, operation=operation))

    graph_path = tables_dir / "graph.yaml"
    graph_path.write_text(graph_yaml(table_paths), encoding="utf-8")
    graph_ref = store.register(graph_path, media_type="application/yaml", inputs=[ref.blake3 for ref in table_refs], operation=operation)

    logger.info("generated Tablassert configs: graph + {} tables -> {}", len(table_refs), tables_dir)
    return [graph_ref, *table_refs]


__all__ = [
    "AGENT_TYPE",
    "FULLMAP_DEFAULT",
    "GRAPH_DESCRIPTION",
    "GRAPH_NAME",
    "INFORES_DAKP",
    "column_letter",
    "excel_column",
    "generate",
    "graph_config",
    "graph_yaml",
    "table_config",
    "table_yaml",
]
