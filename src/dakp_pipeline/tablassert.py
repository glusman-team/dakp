"""Tablassert handoff: generate Graph/table configs and (optionally) run Tablassert.

Flattened from the former ``tablassert/configs.py`` + ``tablassert/run.py`` (US-004).

DAKP does everything up to the shape Tablassert consumes, then emits ONE Graph config
(``tables/graph.yaml``) plus one table config per assertion table (:func:`generate`).
Canonical entity resolution, KGX compilation, dedup, deterministic IDs, and RIG generation
are delegated to ``../Tablassert`` — DAKP ships **no** local fallback KGX compiler.

The configs match the ACTUAL current Tablassert 8.x schema (verified against
``../Tablassert/src/tablassert/models.py`` and its ``ingests.to_sections`` loader):

* a table config is a ``template:``-wrapped :class:`~tablassert.models.Section`
  (``source`` / ``statement`` / ``provenance`` / ``annotations``). The loader only reads
  top-level ``template`` / ``sections`` keys — a bare ``source:``/``statement:`` shape is
  silently dropped, so the ``template:`` wrapper is mandatory;
* ``source.kind: text`` with a tab ``delimiter`` and the uncompressed assertion ``.tsv``
  as ``source.local`` (a ``url`` is required by the model and recorded as provenance);
* column-encoded ``statement.subject`` / ``statement.object`` / ``statement.predicate``
  with drug / disease ``prioritize`` categories;
* a ``provenance.override`` (:class:`~tablassert.models.ManualProvenance`) block carrying
  the DINGO-conventional upstream infores chain, ``knowledge_level`` and ``agent_type``
  (the DAKP ``infores`` is graph-level only since Tablassert 8.0.1 forbids it in the override;
  no ``publication`` — the override replaces repo/publication provenance);
* column-encoded evidence ``annotations``.

Column letters are DERIVED from the assertion-table column contracts in
:mod:`dakp_pipeline.io.schemas` (never hardcoded). YAML is emitted by a tiny stdlib
emitter (no ``pyyaml`` runtime dependency) that round-trips through ``yaml.safe_load``
(asserted in the unit tests).

The real runner (:class:`TablassertRunner`) shells out to the installed ``tablassert`` CLI
(a CORE dependency installed by the single ``uv sync``) and captures stdout / exit code into
a handoff report; the deferred runner (:class:`DeferredTablassertRunner`) writes a
deferred-handoff report without ever touching Tablassert (used when no fullmap triggers the
real handoff, and in tests).

The DEFAULT invocation runs the installed package — the venv ``tablassert`` binary when it is
on ``PATH``, otherwise ``uv run tablassert``. An OPTIONAL editable-checkout override (the
``tablassert_dir`` ctx param, the ``DAKP_TABLASERT_DIR`` env var, or the
``TablassertRunner.tablassert_dir`` field; conventionally ``DEFAULT_TABLASERT_DIR``) switches
to ``uv run --with-editable <dir> tablassert`` for dev against a local ``../Tablassert``
checkout. ``--qc`` is appended only when requested AND the QC audit runtime
(sentence-transformers, part of the required ``tablassert[qc]`` install) is importable;
``--release`` is a boolean flag.

The module-level :func:`run` is the entry point the stage harness and ``dags.dakp_build``
invoke as a MODULE ATTRIBUTE at call time (``tablassert.run(...)``), so
``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` replaces the callable they see.
It dispatches to the real runner when ``ctx.params["run_tablassert"]`` is truthy, else the
deferred runner. Tests monkeypatch the runner's subprocess hook (:func:`run_subprocess`) and
the availability probes (:func:`tablassert_available` / :func:`qc_runtime_available`) — no
real Tablassert required.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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
#: Fallback ``fullmap`` written into ``graph.yaml`` when no real fullmap is configured (deferred
#: runs, which never invoke ``build-kg``). Tablassert's ``Graph`` model REQUIRES a ``fullmap``
#: field, and for a graph build Tablassert reads the fullmap path FROM this field (it ignores the
#: ``--fullmap`` CLI arg unless ``--table-config`` wraps a single table) — so :func:`generate`
#: writes the real ``ctx.params["fullmap"]`` here for real runs. DAKP never downloads a fullmap.
FULLMAP_DEFAULT = ".fullmap"
SOURCE_URL_BASE = "https://example.invalid/dakp/generated"
GRAPH_DESCRIPTION = (
    "Drug Approvals Knowledge Provider: FDA-approved treatment relationships, "
    "FAERS-observed applied-to-treat uses, and contraindications text-mined from "
    "DailyMed, modeled from DailyMed, Drugs@FDA, and FAERS."
)

# Canonical emission order for the three assertion tables.
_TABLE_ORDER = ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions")

# assertion table -> (config basename, predicate, upstream infores chain, knowledge_level,
# agent_type). Upstream order + knowledge_level match the DINGO translator-ingest provenance
# contract (../DINGO/tests/unit/ingests/dakp/test_dakp.py): treats = knowledge_assertion over
# dailymed|faers; applied_to_treat = observation over faers|dailymed (current FAERS
# label/status behavior); contraindicated_in = knowledge_assertion text-mined from DailyMed
# (dailymed upstream, text_mining_agent — matches the DAKP RIG).
_TABLE_SPECS: dict[str, tuple[str, str, tuple[str, ...], str, str]] = {
    "approved_treats_assertions": ("approved_treats", "treats", ("infores:dailymed", "infores:faers"), "knowledge_assertion", AGENT_TYPE),
    "faers_applied_to_treat_assertions": (
        "faers_applied_to_treat",
        "applied_to_treat",
        ("infores:faers", "infores:dailymed"),
        "observation",
        AGENT_TYPE,
    ),
    "contraindication_assertions": ("contraindications", "contraindicated_in", ("infores:dailymed",), "knowledge_assertion", "text_mining_agent"),
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
        ("supporting_spl_documents", "supporting_spl_documents"),
        ("source_score", "source_score"),
    ),
}

SUBJECT_COLUMN = "subject_text"
OBJECT_COLUMN = "object_text"
SUBJECT_PRIORITIZE = ("Drug", "SmallMolecule", "ChemicalEntity")
OBJECT_PRIORITIZE = ("Disease", "PhenotypicFeature")

_GENERATE_OPERATION = "generate_tablassert_configs"


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
    _basename, predicate, upstream, knowledge_level, agent_type = _TABLE_SPECS[table]  # KeyError for unknown tables
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
        "provenance": {"override": {"upstream_resource_ids": list(upstream), "knowledge_level": knowledge_level, "agent_type": agent_type}},
        "annotations": annotations,
    }


def graph_config(tables: list[str] | None = None, version: str | None = None, fullmap: str = FULLMAP_DEFAULT) -> dict[str, Any]:
    """Build the Tablassert ``Graph`` config dict (verified against ``tablassert.models.Graph``).

    ``tables`` defaults to the three committed table configs (``tables/<basename>.yaml``);
    ``version`` defaults to the DAKP package version. ``fullmap`` is the fullmap redb path the
    ``build-kg`` resolve step reads from the Graph config (Tablassert ignores the ``--fullmap`` CLI
    arg for graph builds); it defaults to :data:`FULLMAP_DEFAULT` for deferred runs that never
    invoke ``build-kg``.
    """
    if tables is None:
        tables = [f"tables/{_TABLE_SPECS[table][0]}.yaml" for table in _TABLE_ORDER]
    return {
        "name": GRAPH_NAME,
        "version": version if version is not None else __version__,
        "description": GRAPH_DESCRIPTION,
        "infores": INFORES_DAKP,
        "fullmap": fullmap,
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


def graph_yaml(tables: list[str] | None = None, version: str | None = None, fullmap: str = FULLMAP_DEFAULT) -> str:
    """Serialized Graph config YAML."""
    return _dump_yaml(graph_config(tables=tables, version=version, fullmap=fullmap))


# --- runtime generation into the workdir ------------------------------------------


def generate(assertion_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Write ``tables/graph.yaml`` plus one table config per assertion table into the workdir.

    Configs land in ``<workdir>/tables/`` so their workdir-relative references
    (``tables/<name>.yaml``, ``data/tabular/<table>.tsv``) resolve when Tablassert runs from the
    workdir root. ``graph.yaml`` carries the fullmap redb path the ``build-kg`` resolve step reads
    (Tablassert reads it from the Graph config, ignoring the ``--fullmap`` CLI arg for graph
    builds): the real ``ctx.params["fullmap"]`` when present, else the :data:`FULLMAP_DEFAULT`
    placeholder (deferred runs never invoke ``build-kg``). Returns ``[graph_ref, *table_refs]`` in
    the canonical table order; assertion refs are linked as input provenance by table stem.
    """
    workdir = Workdir(ctx.workdir)
    store = ArtifactStore(workdir)
    tables_dir = workdir.root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    input_ids = {ref.uri.stem: ref.blake3 for ref in assertion_refs}
    operation = OperationBlock(name=_GENERATE_OPERATION)

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
    fullmap = str(ctx.params["fullmap"]) if ctx.params.get("fullmap") else FULLMAP_DEFAULT
    graph_path.write_text(graph_yaml(table_paths, fullmap=fullmap), encoding="utf-8")
    graph_ref = store.register(graph_path, media_type="application/yaml", inputs=[ref.blake3 for ref in table_refs], operation=operation)

    logger.info("generated Tablassert configs: graph + {} tables -> {}", len(table_refs), tables_dir)
    return [graph_ref, *table_refs]


# --- runner ------------------------------------------------------------------------

DEFAULT_TABLASERT_DIR = "../Tablassert"
TABLASERT_DIR_ENV = "DAKP_TABLASERT_DIR"
REPORT_NAME = "tablassert_handoff.json"
_REPORT_SCHEMA = "dakp.tablassert_handoff.v1"
_RUN_OPERATION = "run_tablassert"


def run_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Execute ``command`` capturing stdout/stderr, never raising on non-zero exit.

    This is the monkeypatch point for tests: patch the ``run_subprocess`` attribute on THIS
    module (``dakp_pipeline.tablassert``) and no real process is spawned.
    """
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def tablassert_available() -> bool:
    """True when the ``tablassert`` package (a core DAKP dependency) is importable here."""
    return importlib.util.find_spec("tablassert") is not None


def qc_runtime_available() -> bool:
    """True when the QC audit runtime (sentence-transformers, via ``tablassert[qc]``) is importable."""
    return importlib.util.find_spec("sentence_transformers") is not None


def _command_prefix(tablassert_dir: str | None) -> list[str]:
    """argv prefix that launches the ``tablassert`` CLI.

    Editable override (dev against a local checkout): ``uv run --with-editable <dir> tablassert``;
    installed package: the venv ``tablassert`` binary when it is on ``PATH``, otherwise
    ``uv run tablassert`` (uv materializes the console script for the rare importable-but-no-PATH
    case; the availability guard in ``TablassertRunner.run`` has already confirmed ``tablassert``
    is importable before this fallback is reachable).
    """
    if tablassert_dir:
        return ["uv", "run", "--with-editable", tablassert_dir, "tablassert"]
    binary = shutil.which("tablassert")
    if binary is not None:
        return [binary]
    return ["uv", "run", "tablassert"]


def _resolve_tablassert_dir(runner_dir: str | None, params_dir: str | None) -> str | None:
    """Editable-checkout override precedence: ctx param > ``DAKP_TABLASERT_DIR`` env > runner default.

    ``None`` (the default everywhere) means "use the installed PyPI package".
    """
    return params_dir or os.environ.get(TABLASERT_DIR_ENV) or runner_dir


def _base_report(mode: str, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef]) -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "stage": "tablassert_handoff",
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "assertion_inputs": [{"table": ref.uri.stem, "artifact_id": ref.blake3, "rows": ref.rows} for ref in assertion_refs],
        "config_inputs": [str(ref.uri) for ref in config_refs],
    }


def _write_report(report: dict[str, Any], assertion_refs: list[ArtifactRef], ctx: TaskContext) -> ArtifactRef:
    workdir = Workdir(ctx.workdir)
    path = workdir.reports / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    store = ArtifactStore(workdir)
    return store.register(
        path, media_type="application/json", inputs=[ref.blake3 for ref in assertion_refs], operation=OperationBlock(name=_RUN_OPERATION)
    )


def _find_graph(config_refs: list[ArtifactRef], ctx: TaskContext) -> Path:
    """Locate ``graph.yaml`` among the generated config refs (fall back to the conventional path)."""
    for ref in config_refs:
        if ref.uri.name == "graph.yaml":
            return ref.uri
    return Workdir(ctx.workdir).root / "tables" / "graph.yaml"


@dataclass(frozen=True)
class TablassertRunner:
    """Run the INSTALLED ``tablassert`` CLI (a core DAKP dependency) as a subprocess.

    Builds ``tablassert build-kg <graph.yaml> --fullmap <path> [--qc] [--release]`` and records
    stdout / stderr / exit code in the handoff report. A non-zero exit is captured as
    ``status: failed`` (logged loudly), not raised — the report is the artifact the pipeline
    surfaces. Raises ``RuntimeError`` when ``tablassert`` is unavailable and no editable-checkout
    override is configured (reinstall with ``uv sync``).
    """

    tablassert_dir: str | None = None

    def build_command(
        self, graph_yaml: Path, fullmap: str, *, tablassert_dir: str | None = None, qc: bool = False, release: bool = False
    ) -> list[str]:
        """The exact Tablassert invocation (pure; testable without spawning a process)."""
        command = [*_command_prefix(tablassert_dir), "build-kg", str(graph_yaml), "--fullmap", fullmap]
        if qc:
            command.append("--qc")
        if release:
            command.append("--release")
        return command

    def run(self, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        graph_yaml = _find_graph(config_refs, ctx)
        fullmap_value = ctx.params.get("fullmap")
        if not fullmap_value:
            msg = (
                "a fullmap redb path is required for a real Tablassert handoff but none was provided: pass "
                "`--fullmap <path>` to `dakp up` (DAKP no longer downloads a fullmap; build one with "
                "`tablassert build-fullmap`)"
            )
            raise RuntimeError(msg)
        fullmap = str(fullmap_value)
        tablassert_dir = _resolve_tablassert_dir(self.tablassert_dir, ctx.params.get("tablassert_dir"))
        if tablassert_dir is None and not tablassert_available():
            msg = (
                "tablassert is not available: it is a core DAKP dependency, so reinstall with `uv sync`, or point at a "
                f"local editable checkout via the tablassert_dir param / {TABLASERT_DIR_ENV} env var"
            )
            raise RuntimeError(msg)

        qc_requested = bool(ctx.params.get("qc"))
        qc = qc_requested and qc_runtime_available()
        if qc_requested and not qc:
            logger.warning("tablassert --qc requested but the QC audit runtime (sentence-transformers) is not importable; running without --qc")
        release = bool(ctx.params.get("release"))

        command = self.build_command(graph_yaml, fullmap, tablassert_dir=tablassert_dir, qc=qc, release=release)
        cwd = Workdir(ctx.workdir).root

        logger.info("running Tablassert: {}", " ".join(command))
        completed = run_subprocess(command, cwd=cwd)
        status = "ok" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            logger.error("Tablassert exited {}: {}", completed.returncode, (completed.stderr or "").strip()[:2000])

        report = _base_report("real", assertion_refs, config_refs)
        report.update(
            {
                "status": status,
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "graph_config": str(graph_yaml),
                "fullmap": fullmap,
                "tablassert_dir": tablassert_dir,
                "qc": qc,
                "release": release,
            }
        )
        return [_write_report(report, assertion_refs, ctx)]


@dataclass(frozen=True)
class DeferredTablassertRunner:
    """Write a deferred-handoff report; never touch Tablassert (no fullmap trigger + tests)."""

    def run(self, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        report = _base_report("deferred", assertion_refs, config_refs)
        report.update(
            {
                "status": "deferred",
                "reason": "no fullmap provided (run_tablassert disabled); canonical resolution + KGX compilation delegated to the installed tablassert CLI",
            }
        )
        return [_write_report(report, assertion_refs, ctx)]


def run(assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Module entry point (stage harness / ``dags.dakp_build``): dispatch to a runner.

    Callers invoke this as a module attribute at call time (``tablassert.run(...)``) so
    ``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` replaces the callable they see.
    Real execution requires ``run_tablassert`` truthy in ``ctx.params`` (derived from a fullmap
    path); otherwise the deferred runner writes a deferred-handoff report. Returns a list with
    one ArtifactRef to the handoff report.
    """
    run_real = bool(ctx.params.get("run_tablassert"))
    runner: TablassertRunner | DeferredTablassertRunner = TablassertRunner() if run_real else DeferredTablassertRunner()
    return runner.run(assertion_refs, config_refs, ctx)


__all__ = [
    "AGENT_TYPE",
    "DEFAULT_TABLASERT_DIR",
    "FULLMAP_DEFAULT",
    "GRAPH_DESCRIPTION",
    "GRAPH_NAME",
    "INFORES_DAKP",
    "REPORT_NAME",
    "TABLASERT_DIR_ENV",
    "DeferredTablassertRunner",
    "TablassertRunner",
    "column_letter",
    "excel_column",
    "generate",
    "graph_config",
    "graph_yaml",
    "qc_runtime_available",
    "run",
    "run_subprocess",
    "tablassert_available",
    "table_config",
    "table_yaml",
]
