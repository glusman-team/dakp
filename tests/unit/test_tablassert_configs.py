"""Unit tests for Tablassert config generation + runner (Milestone 7).

Asserts the generated Graph + per-table configs are valid YAML matching the ACTUAL current
Tablassert 8.x schema (``template:``-wrapped ``Section``: ``source.kind=text``, column-encoded
subject/object/predicate, ``provenance.override`` ManualProvenance, column-encoded evidence
annotations), with column letters derived from the assertion-table contracts and provenance
matching the DINGO translator-ingest conventions. Also covers the runner: the mock runner's
handoff report and the real runner's monkeypatchable subprocess boundary.

These tests never import or require ``../Tablassert`` to be installed.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.tablassert import configs as tablassert_configs
from dakp_pipeline.tablassert.run import MockTablassertRunner, TablassertRunner
from dakp_pipeline.tablassert.run import run as run_tablassert

# The package ``__init__`` re-exports the ``run`` *function*, shadowing the ``run`` submodule
# attribute on ``dakp_pipeline.tablassert``. Patch the subprocess hook on the actual module
# object (import_module always returns the module), not the shadowed package attribute.
_RUN_MODULE = importlib.import_module("dakp_pipeline.tablassert.run")

TABLES = ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions")

INFORES_DAKP = "infores:multiomics-drugapprovals"
AGENT_TYPE = "manual_validation_of_automated_agent"

# assertion table -> (config basename, predicate, upstream chain, knowledge_level, agent_type):
# the DINGO translator-ingest provenance contract (../DINGO/tests/unit/ingests/dakp/test_dakp.py).
# Contraindications are text-mined from DailyMed (dailymed upstream, text_mining_agent).
EXPECTED_PROVENANCE = {
    "approved_treats_assertions": ("approved_treats", "treats", ["infores:dailymed", "infores:faers"], "knowledge_assertion", AGENT_TYPE),
    "faers_applied_to_treat_assertions": (
        "faers_applied_to_treat",
        "applied_to_treat",
        ["infores:faers", "infores:dailymed"],
        "observation",
        AGENT_TYPE,
    ),
    "contraindication_assertions": ("contraindications", "contraindicated_in", ["infores:dailymed"], "knowledge_assertion", "text_mining_agent"),
}

# assertion table -> {annotation name: assertion column it must encode}. case_count maps to the
# Translator ``number_of_cases`` slot; the rest keep their column name.
EXPECTED_ANNOTATIONS = {
    "approved_treats_assertions": {
        "approval_ids": "approval_ids",
        "supporting_spl_sets": "supporting_spl_sets",
        "clinical_approval_status": "clinical_approval_status",
    },
    "faers_applied_to_treat_assertions": {"number_of_cases": "case_count", "clinical_approval_status": "clinical_approval_status"},
    "contraindication_assertions": {
        "supporting_spl_sets": "supporting_spl_sets",
        "supporting_spl_documents": "supporting_spl_documents",
        "source_score": "source_score",
    },
}


# --- helpers ----------------------------------------------------------------------


def _letter_to_index(letters: str) -> int:
    """Inverse of configs.excel_column: Excel letters -> 0-based index (A->0, Z->25, AA->26)."""
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return index - 1


def _column_at(table: str, letters: str) -> str:
    """Assertion-table column name addressed by Excel ``letters``."""
    return schemas.columns_for(table)[_letter_to_index(letters)]


def _write_assertion_tsv(table: str, workdir: Workdir) -> Path:
    path = workdir.tabular / f"{table}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(schemas.columns_for(table)) + "\n", encoding="utf-8")
    return path


def _assertion_refs(workdir: Workdir) -> list[ArtifactRef]:
    store = ArtifactStore(workdir)
    return [store.register(_write_assertion_tsv(table, workdir), media_type=schemas.TSV_MEDIA_TYPE, rows=1) for table in TABLES]


def _ctx(workdir: Workdir, profile: str = "mock", **params: object) -> TaskContext:
    return TaskContext(profile=profile, workdir=workdir.root, fixture_root=None, threads=1, memory_budget_gb=1, params=params)


def _read_report(workdir: Workdir) -> dict:
    return json.loads((workdir.reports / "tablassert_handoff.json").read_text(encoding="utf-8"))


# --- Excel column-letter mapping --------------------------------------------------


def test_excel_column_letters() -> None:
    assert tablassert_configs.excel_column(0) == "A"
    assert tablassert_configs.excel_column(5) == "F"
    assert tablassert_configs.excel_column(25) == "Z"
    assert tablassert_configs.excel_column(26) == "AA"
    with pytest.raises(ValueError, match=">= 0"):
        tablassert_configs.excel_column(-1)


def test_column_letters_point_at_subject_and_object() -> None:
    # subject_text is column A and object_text is column F in every assertion contract.
    for table in TABLES:
        assert _column_at(table, tablassert_configs.column_letter(table, "subject_text")) == "subject_text"
        assert _column_at(table, tablassert_configs.column_letter(table, "object_text")) == "object_text"
        assert tablassert_configs.column_letter(table, "subject_text") == "A"
        assert tablassert_configs.column_letter(table, "object_text") == "F"


def test_column_letter_unknown_column_raises() -> None:
    with pytest.raises(KeyError, match="not_a_column"):
        tablassert_configs.column_letter("approved_treats_assertions", "not_a_column")


# --- table config structure -------------------------------------------------------


@pytest.mark.parametrize("table", TABLES)
def test_table_config_structure(table: str) -> None:
    _basename, predicate, upstream, knowledge_level, agent_type = EXPECTED_PROVENANCE[table]
    config = tablassert_configs.table_config(table)

    # text source over the uncompressed assertion TSV (tab delimiter; url required by the model).
    source = config["source"]
    assert source["kind"] == "text"
    assert source["local"] == f"data/tabular/{table}.tsv"
    assert source["delimiter"] == "\t"
    assert source["url"].startswith("https://")

    # column-encoded subject/object with drug / disease prioritization.
    statement = config["statement"]
    assert statement["predicate"] == predicate
    assert statement["subject"]["method"] == "column"
    assert statement["subject"]["encoding"] == tablassert_configs.column_letter(table, "subject_text")
    assert statement["subject"]["prioritize"] == ["Drug", "SmallMolecule", "ChemicalEntity"]
    assert statement["object"]["method"] == "column"
    assert statement["object"]["encoding"] == tablassert_configs.column_letter(table, "object_text")
    assert statement["object"]["prioritize"] == ["Disease", "PhenotypicFeature"]

    # ManualProvenance override matching the DINGO conventions (no publication alongside override).
    override = config["provenance"]["override"]
    assert override["infores"] == INFORES_DAKP
    assert override["upstream_resource_ids"] == upstream
    assert override["knowledge_level"] == knowledge_level
    assert override["agent_type"] == agent_type
    assert "publication" not in config["provenance"]


@pytest.mark.parametrize("table", TABLES)
def test_table_config_annotations_encode_expected_columns(table: str) -> None:
    annotations = tablassert_configs.table_config(table)["annotations"]
    by_name = {a["annotation"]: a for a in annotations}
    assert set(by_name) == set(EXPECTED_ANNOTATIONS[table])
    for annotation, column in EXPECTED_ANNOTATIONS[table].items():
        entry = by_name[annotation]
        assert entry["method"] == "column"
        # The encoding letter must address the expected assertion column.
        assert _column_at(table, entry["encoding"]) == column
        assert entry["encoding"] == tablassert_configs.column_letter(table, column)


# --- graph config structure -------------------------------------------------------


def test_graph_config_structure() -> None:
    graph = tablassert_configs.graph_config()
    assert graph["name"] == "dakp"
    assert isinstance(graph["version"], str)
    assert graph["version"]
    assert graph["description"]
    assert graph["infores"] == INFORES_DAKP
    assert graph["fullmap"] == ".fullmap"
    assert graph["tables"] == ["tables/approved_treats.yaml", "tables/faers_applied_to_treat.yaml", "tables/contraindications.yaml"]


# --- emitted YAML is valid + faithful (round-trips through yaml.safe_load) --------


@pytest.mark.parametrize("table", TABLES)
def test_table_yaml_is_valid_and_faithful(table: str) -> None:
    loaded = yaml.safe_load(tablassert_configs.table_yaml(table))
    # The table config is a template-wrapped Section (the shape Tablassert's loader requires).
    assert set(loaded) == {"template"}
    assert loaded == {"template": tablassert_configs.table_config(table)}
    # The tab delimiter survives serialization as a real tab character.
    assert loaded["template"]["source"]["delimiter"] == "\t"


def test_graph_yaml_is_valid_and_faithful() -> None:
    loaded = yaml.safe_load(tablassert_configs.graph_yaml())
    assert loaded == tablassert_configs.graph_config()
    assert loaded["name"] == "dakp"


# --- runtime generation into the workdir ------------------------------------------


def test_generate_writes_graph_and_table_configs(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    ctx = _ctx(workdir)

    refs = tablassert_configs.generate(assertion_refs, ctx)

    # Graph config first, then one config per table in canonical order.
    assert [ref.uri.name for ref in refs] == ["graph.yaml", "approved_treats.yaml", "faers_applied_to_treat.yaml", "contraindications.yaml"]
    for ref in refs:
        assert ref.uri.exists()
        assert ref.uri.parent == workdir.root / "tables"

    graph_text = refs[0].uri.read_text(encoding="utf-8")
    assert "name: dakp" in graph_text
    assert "tables/approved_treats.yaml" in graph_text
    assert "tables/faers_applied_to_treat.yaml" in graph_text
    assert "tables/contraindications.yaml" in graph_text

    # Each emitted table config parses and carries its predicate.
    predicates = {ref.uri.stem: yaml.safe_load(ref.uri.read_text(encoding="utf-8"))["template"]["statement"]["predicate"] for ref in refs[1:]}
    assert predicates == {"approved_treats": "treats", "faers_applied_to_treat": "applied_to_treat", "contraindications": "contraindicated_in"}


def test_generate_links_assertion_inputs_as_provenance(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    ctx = _ctx(workdir)

    refs = tablassert_configs.generate(assertion_refs, ctx)
    # The graph manifest lists the three table configs as inputs; each table config lists its TSV.
    assert refs[0].manifest is not None
    graph_manifest = json.loads(refs[0].manifest.read_text(encoding="utf-8"))
    assert len(graph_manifest["inputs"]) == 3


# --- runner: mock handoff report --------------------------------------------------


def test_mock_runner_writes_deferred_handoff_report(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    refs = MockTablassertRunner().run(assertion_refs, config_refs, _ctx(workdir))

    assert len(refs) == 1
    assert refs[0].uri == workdir.reports / "tablassert_handoff.json"
    report = _read_report(workdir)
    assert report["mode"] == "mock"
    assert report["status"] == "deferred"
    assert report["stage"] == "tablassert_handoff"
    assert {entry["table"] for entry in report["assertion_inputs"]} == set(TABLES)
    assert len(report["config_inputs"]) == 4  # graph + 3 tables


# --- runner: real subprocess boundary (monkeypatched; no real Tablassert) ---------


def test_real_runner_builds_exact_command(tmp_path: Path) -> None:
    command = TablassertRunner().build_command(Path("tables/graph.yaml"), ".fullmap", "../Tablassert")
    assert command == ["uv", "run", "--with-editable", "../Tablassert", "tablassert", "build-kg", "tables/graph.yaml", "--fullmap", ".fullmap"]


def test_real_runner_captures_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    calls: list[tuple[list[str], Path | None]] = []

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="built kg\n", stderr="")

    monkeypatch.setattr(_RUN_MODULE, "run_subprocess", fake_subprocess)

    refs = TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir))

    assert len(refs) == 1
    # The subprocess hook was invoked once, from the workdir root, with the build-kg command.
    assert len(calls) == 1
    command, cwd = calls[0]
    assert cwd == workdir.root
    assert command[:6] == ["uv", "run", "--with-editable", "../Tablassert", "tablassert", "build-kg"]
    assert command[-2:] == ["--fullmap", ".fullmap"]
    assert command[6] == str(workdir.root / "tables" / "graph.yaml")

    report = _read_report(workdir)
    assert report["mode"] == "real"
    assert report["status"] == "ok"
    assert report["exit_code"] == 0
    assert report["stdout"] == "built kg\n"
    assert report["command"] == command


def test_real_runner_records_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=command, returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(_RUN_MODULE, "run_subprocess", fake_subprocess)
    TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir))

    report = _read_report(workdir)
    assert report["mode"] == "real"
    assert report["status"] == "failed"
    assert report["exit_code"] == 2
    assert report["stderr"] == "boom"


def test_real_runner_honors_ctx_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    seen: list[list[str]] = []

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_RUN_MODULE, "run_subprocess", fake_subprocess)
    TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir, tablassert_dir="/opt/tablassert", fullmap="data/fullmap"))

    assert seen[0][3] == "/opt/tablassert"  # --with-editable <dir>
    assert seen[0][-1] == "data/fullmap"  # --fullmap <path>


# --- module-level dispatch --------------------------------------------------------


def test_run_dispatches_to_mock_in_mock_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    def no_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        msg = "subprocess must not run in the mock profile"
        raise AssertionError(msg)

    monkeypatch.setattr(_RUN_MODULE, "run_subprocess", no_subprocess)

    # run_tablassert=True is ignored under the mock profile -> deferred mock handoff, no subprocess.
    run_tablassert(assertion_refs, config_refs, _ctx(workdir, run_tablassert=True))
    assert _read_report(workdir)["mode"] == "mock"


def test_run_dispatches_to_real_outside_mock_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(_RUN_MODULE, "run_subprocess", fake_subprocess)

    run_tablassert(assertion_refs, config_refs, _ctx(workdir, profile="prod", run_tablassert=True))
    assert _read_report(workdir)["mode"] == "real"
