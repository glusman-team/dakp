"""Unit tests for Tablassert config generation + runner (Milestone 7).

Asserts the generated Graph + per-table configs are valid YAML matching the ACTUAL current
Tablassert 8.x schema (``template:``-wrapped ``Section``: ``source.kind=text``, column-encoded
subject/object/predicate, column-encoded ``statement.qualifiers`` where a column backs them,
``provenance.override`` ManualProvenance, column-encoded evidence annotations), with column
letters derived from the assertion-table contracts and provenance matching the DINGO
translator-ingest conventions. Also covers the runner: the mock runner's
handoff report and the real runner's monkeypatchable subprocess boundary.

These tests require the INSTALLED ``tablassert`` package (a core DAKP dependency): config
generation derives the hard category allow-lists from its Biolink ``Categories`` enum. The
sibling ``../Tablassert`` checkout is never imported or required.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from dakp_pipeline import tablassert as tablassert_configs
from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.tablassert import (
    OBJECT_COLUMN,
    OBJECT_PRIORITIZE,
    SUBJECT_PRIORITIZE,
    TABLASERT_DIR_ENV,
    DeferredTablassertRunner,
    TablassertError,
    TablassertRunner,
    _resolve_tablassert_dir,
    category_avoid_list,
    qc_runtime_available,
    tablassert_available,
)
from dakp_pipeline.tablassert import run as run_tablassert

# Config generation computes its category guards from the installed package's Biolink enum.
pytest.importorskip("tablassert", reason="tablassert is a core DAKP dependency; run `uv sync`")

# The flat ``dakp_pipeline.tablassert`` module IS the runner module: patch the subprocess hook
# and availability probes on the module object (import_module always returns the module) so the
# runner resolves the patched callables from its module globals at call time.
_RUN_MODULE = importlib.import_module("dakp_pipeline.tablassert")

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

# assertion table -> {qualifier slot: assertion column backing it}. Only tables whose columns carry
# a qualifier entity get entries (see ``_TABLE_QUALIFIERS`` for the per-table justification): FAERS
# adverse-event case reports carry their disease context; the other tables' columns back none.
EXPECTED_QUALIFIERS = {
    "approved_treats_assertions": {},
    "faers_applied_to_treat_assertions": {"disease_context_qualifier": OBJECT_COLUMN},
    "contraindication_assertions": {},
}

# assertion table -> {annotation name: (assertion column it encodes, multivalued delimiter)}.
# case_count maps to the Translator ``number_of_cases`` slot; the SPL-evidence columns map to names
# on Tablassert's edge-field allow-list (``has_evidence`` / ``supporting_documents``) so they stay
# first-class KGX fields, with ``delimiter: "|"`` so the pipe-joined cells emit as real JSON arrays;
# the rest keep their column name and fold into ``supporting_text`` (no Biolink slot on DAKP's
# association class).
EXPECTED_ANNOTATIONS = {
    "approved_treats_assertions": {
        "approval_ids": ("approval_ids", None),
        "has_evidence": ("supporting_spl_sets", "|"),
        "supporting_documents": ("supporting_spl_documents", "|"),
        "clinical_approval_status": ("clinical_approval_status", None),
    },
    "faers_applied_to_treat_assertions": {"number_of_cases": ("case_count", None), "clinical_approval_status": ("clinical_approval_status", None)},
    "contraindication_assertions": {
        "has_evidence": ("supporting_spl_sets", "|"),
        "supporting_documents": ("supporting_spl_documents", "|"),
        "source_score": ("source_score", None),
    },
}

# assertion table -> the REAL upstream dataset URL recorded as ``source.url`` (never a placeholder;
# Tablassert emits it as the edge ``sources[].source_record_urls``).
EXPECTED_SOURCE_URLS = {
    "approved_treats_assertions": "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm",
    "faers_applied_to_treat_assertions": "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html",
    "contraindication_assertions": "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm",
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


def _ctx(workdir: Workdir, **params: object) -> TaskContext:
    return TaskContext(workdir=workdir.root, fixture_root=None, params=params)


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
    # The real upstream dataset URL (a list since Tablassert 8.2.1) — never the example.invalid placeholder.
    assert source["url"] == [EXPECTED_SOURCE_URLS[table]]
    assert all("example.invalid" not in url for url in source["url"])

    # column-encoded subject/object with drug / disease prioritization + hard allow-list guards.
    statement = config["statement"]
    assert statement["predicate"] == predicate
    assert statement["subject"]["method"] == "column"
    assert statement["subject"]["encoding"] == tablassert_configs.column_letter(table, "subject_text")
    assert statement["subject"]["prioritize"] == ["Drug", "SmallMolecule", "ChemicalEntity"]
    assert statement["subject"]["avoid"] == category_avoid_list(SUBJECT_PRIORITIZE)
    assert statement["object"]["method"] == "column"
    assert statement["object"]["encoding"] == tablassert_configs.column_letter(table, "object_text")
    assert statement["object"]["prioritize"] == ["Disease", "PhenotypicFeature"]
    assert statement["object"]["avoid"] == category_avoid_list(OBJECT_PRIORITIZE)

    # ManualProvenance override matching the DINGO conventions (no publication alongside override).
    override = config["provenance"]["override"]
    assert "infores" not in override  # the DAKP infores is graph-level only (Tablassert >= 8.0.1 forbids it here)
    assert override["upstream_resource_ids"] == upstream
    assert override["knowledge_level"] == knowledge_level
    assert override["agent_type"] == agent_type
    assert "publication" not in config["provenance"]


@pytest.mark.parametrize("table", TABLES)
def test_table_config_qualifiers(table: str) -> None:
    statement = tablassert_configs.table_config(table)["statement"]
    expected = EXPECTED_QUALIFIERS[table]
    if not expected:
        # Nothing backs a qualifier on this table => no ``qualifiers`` key at all (never an empty list).
        assert "qualifiers" not in statement
        return
    qualifiers = statement["qualifiers"]
    assert [entry["qualifier"] for entry in qualifiers] == list(expected)
    for entry in qualifiers:
        backing = expected[entry["qualifier"]]
        assert entry["method"] == "column"
        # The encoding letter must address the backing assertion column.
        assert _column_at(table, entry["encoding"]) == backing
        assert entry["encoding"] == tablassert_configs.column_letter(table, backing)
        # The qualifier re-resolves its backing column with the object side's allow-list, so it
        # lands on exactly the same CURIE as the node it qualifies.
        assert entry["prioritize"] == list(OBJECT_PRIORITIZE)
        assert entry["avoid"] == category_avoid_list(OBJECT_PRIORITIZE)


def test_qualifier_slots_are_valid_biolink_qualifiers() -> None:
    # Every emitted qualifier slot must be a real member of the installed Tablassert's Biolink
    # ``Qualifiers`` enum — and never ``species_context_qualifier`` (Tablassert auto-derives that
    # one from the taxon constraint and rejects manual declarations at config load).
    from tablassert.biolink import Qualifiers

    valid = {qualifier.value for qualifier in Qualifiers}
    for table in TABLES:
        for qualifier, _column in tablassert_configs._TABLE_QUALIFIERS[table]:
            assert qualifier in valid
            assert qualifier != "species_context_qualifier"


@pytest.mark.parametrize("table", TABLES)
def test_table_config_annotations_encode_expected_columns(table: str) -> None:
    annotations = tablassert_configs.table_config(table)["annotations"]
    by_name = {a["annotation"]: a for a in annotations}
    assert set(by_name) == set(EXPECTED_ANNOTATIONS[table])
    for annotation, (column, delimiter) in EXPECTED_ANNOTATIONS[table].items():
        entry = by_name[annotation]
        assert entry["method"] == "column"
        # The encoding letter must address the expected assertion column.
        assert _column_at(table, entry["encoding"]) == column
        assert entry["encoding"] == tablassert_configs.column_letter(table, column)
        # Multivalued Biolink slots split the pipe-joined cell into a real JSON array.
        if delimiter is None:
            assert "delimiter" not in entry
        else:
            assert entry["delimiter"] == delimiter


# --- category guard (hard allow-lists via Tablassert ``avoid``) -------------------

# Canonical wacky categories present in the production fullmap that must never resolve into a
# drug↔disease graph. None is in either side's allow-list, so all must be on every avoid list.
WACKY_CATEGORIES = ("OrganismTaxon", "Publication", "Gene", "Protein", "CellLine", "GeographicLocation", "NamedThing")


def test_category_avoid_list_is_sorted_complement_of_allowed() -> None:
    from tablassert.biolink import Categories

    universe = {category.value for category in Categories}
    for allowed in (SUBJECT_PRIORITIZE, OBJECT_PRIORITIZE):
        avoid = category_avoid_list(allowed)
        assert avoid == sorted(avoid)  # deterministic YAML emission
        assert set(avoid).isdisjoint(allowed)
        # Exact partition of the installed Biolink universe: nothing off-list can ever resolve.
        assert set(avoid) | set(allowed) == universe


@pytest.mark.parametrize("table", TABLES)
def test_table_config_category_guards(table: str) -> None:
    statement = tablassert_configs.table_config(table)["statement"]
    for node, allowed in ((statement["subject"], SUBJECT_PRIORITIZE), (statement["object"], OBJECT_PRIORITIZE)):
        assert node["avoid"] == category_avoid_list(allowed)
        for wacky in WACKY_CATEGORIES:
            assert wacky in node["avoid"]
    # Cross-side exclusion: drug-side categories are avoided on the disease side and vice versa.
    for drug in SUBJECT_PRIORITIZE:
        assert drug in statement["object"]["avoid"]
    for disease in OBJECT_PRIORITIZE:
        assert disease in statement["subject"]["avoid"]


def test_category_avoid_list_unknown_allowed_category_raises() -> None:
    with pytest.raises(ValueError, match="not in the installed Tablassert Biolink model"):
        category_avoid_list(["Drug", "DefinitelyNotACategory"])


@pytest.mark.parametrize("table", TABLES)
def test_table_yaml_validates_against_tablassert_section_model(table: str) -> None:
    # The emitted config must load cleanly into Tablassert's own Section model — this proves every
    # avoid entry is a real Categories member and every qualifier a real Qualifiers member of the
    # INSTALLED tablassert (unknown names would raise a TablassertValidationError here).
    from tablassert.models import Section

    section = Section.model_validate(yaml.safe_load(tablassert_configs.table_yaml(table))["template"])
    # Qualifier entries survive model validation as Qualifier objects, in emission order.
    model_qualifiers = section.statement.qualifiers or []
    assert [str(qualifier.qualifier) for qualifier in model_qualifiers] == list(EXPECTED_QUALIFIERS[table])


def test_committed_table_configs_match_generator_output() -> None:
    # The committed ``tables/*.yaml`` must byte-equal the generator output (regenerate, never
    # hand-diverge).
    repo_tables = Path(__file__).resolve().parents[2] / "tables"
    for table in TABLES:
        basename = tablassert_configs._TABLE_SPECS[table][0]
        assert (repo_tables / f"{basename}.yaml").read_text(encoding="utf-8") == tablassert_configs.table_yaml(table)
    assert (repo_tables / "graph.yaml").read_text(encoding="utf-8") == tablassert_configs.graph_yaml()


# --- graph config structure -------------------------------------------------------


def test_graph_config_structure() -> None:
    graph = tablassert_configs.graph_config()
    assert graph["name"] == "dakp"
    assert isinstance(graph["version"], str)
    assert graph["version"]
    assert graph["description"]
    assert graph["infores"] == INFORES_DAKP
    assert graph["fullmap"] == ".fullmap"  # default placeholder; generate() writes the real ctx fullmap path
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


# --- runner: deferred handoff report ----------------------------------------------


def test_deferred_runner_writes_deferred_handoff_report(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    refs = DeferredTablassertRunner().run(assertion_refs, config_refs, _ctx(workdir))

    assert len(refs) == 1
    assert refs[0].uri == workdir.reports / "tablassert_handoff.json"
    report = _read_report(workdir)
    assert report["mode"] == "deferred"
    assert report["status"] == "deferred"
    assert report["stage"] == "tablassert_handoff"
    assert {entry["table"] for entry in report["assertion_inputs"]} == set(TABLES)
    assert len(report["config_inputs"]) == 4  # graph + 3 tables


# --- runner: availability probes (importlib.util.find_spec seams) -----------------


def _fake_find_spec(present: frozenset[str]):
    """An ``importlib.util.find_spec`` stand-in: a name is importable iff it is in ``present``."""

    def find_spec(name: str, *args: object, **kwargs: object) -> object:
        return object() if name in present else None

    return find_spec


def test_tablassert_available_reflects_importability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(frozenset({"tablassert"})))
    assert tablassert_available() is True
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(frozenset()))
    assert tablassert_available() is False


def test_qc_runtime_available_reflects_importability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(frozenset({"sentence_transformers"})))
    assert qc_runtime_available() is True
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(frozenset()))
    assert qc_runtime_available() is False


# --- runner: command construction (pure; no process spawned) ----------------------


def test_build_command_editable_override_prefix() -> None:
    command = TablassertRunner().build_command(Path("tables/graph.yaml"), tablassert_dir="../Tablassert")
    assert command == ["uv", "run", "--with-editable", "../Tablassert", "tablassert", "build-kg", "tables/graph.yaml"]


def test_build_command_prefers_installed_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/venv/bin/tablassert")
    command = TablassertRunner().build_command(Path("tables/graph.yaml"))
    assert command == ["/venv/bin/tablassert", "build-kg", "tables/graph.yaml"]


def test_build_command_falls_back_to_uv_run_tablassert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    command = TablassertRunner().build_command(Path("tables/graph.yaml"))
    assert command == ["uv", "run", "tablassert", "build-kg", "tables/graph.yaml"]


def test_build_command_appends_qc_and_release_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    command = TablassertRunner().build_command(Path("graph.yaml"), qc=True, release=True)
    assert command == ["uv", "run", "tablassert", "build-kg", "graph.yaml", "--qc", "--release"]


def test_resolve_tablassert_dir_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TABLASERT_DIR_ENV, raising=False)
    # ctx param wins over env and runner default.
    assert _resolve_tablassert_dir("/runner", "/param") == "/param"
    # env wins over the runner default when no param is given.
    monkeypatch.setenv(TABLASERT_DIR_ENV, "/env")
    assert _resolve_tablassert_dir("/runner", None) == "/env"
    # runner default is used when neither param nor env is set.
    monkeypatch.delenv(TABLASERT_DIR_ENV, raising=False)
    assert _resolve_tablassert_dir("/runner", None) == "/runner"
    # None everywhere -> the installed PyPI package.
    assert _resolve_tablassert_dir(None, None) is None


# --- runner: real subprocess boundary (monkeypatched; no real Tablassert) ---------


def _patch_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the runner's DEFAULT installed-package path deterministically (no editable dir)."""
    monkeypatch.delenv(TABLASERT_DIR_ENV, raising=False)  # ignore any real dev override in the env
    monkeypatch.setattr(_RUN_MODULE, "tablassert_available", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda name: None)  # -> the `uv run tablassert` prefix


def test_real_runner_captures_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    calls: list[tuple[list[str], Path | None]] = []

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="built kg\n", stderr="")

    _patch_installed(monkeypatch)
    monkeypatch.setattr(_RUN_MODULE, "stream_subprocess", fake_subprocess)

    refs = TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir, fullmap="data/fullmap"))

    assert len(refs) == 1
    # The subprocess hook was invoked once, from the workdir root, with the build-kg command.
    assert len(calls) == 1
    command, cwd = calls[0]
    assert cwd == workdir.root
    assert command[:4] == ["uv", "run", "tablassert", "build-kg"]
    assert command[4] == str(workdir.root / "tables" / "graph.yaml")
    # Tablassert >= 8.1 removed the build-kg --fullmap flag; graph.yaml carries the fullmap path.
    assert "--fullmap" not in command

    report = _read_report(workdir)
    assert report["mode"] == "real"
    assert report["status"] == "ok"
    assert report["exit_code"] == 0
    assert report["stdout"] == "built kg\n"
    assert report["command"] == command
    assert report["tablassert_dir"] is None
    assert report["qc"] is False
    assert report["release"] is False


def test_real_runner_records_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-zero exit writes the report (status=failed) AND raises TablassertError."""
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=command, returncode=2, stdout="", stderr="boom")

    _patch_installed(monkeypatch)
    monkeypatch.setattr(_RUN_MODULE, "stream_subprocess", fake_subprocess)
    with pytest.raises(TablassertError, match="exited 2"):
        TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir, fullmap="data/fullmap"))

    # The report must still be on disk (written before the exception was raised).
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

    # A tablassert_dir override selects the editable-checkout prefix AND bypasses the availability
    # check (uv resolves the local checkout transiently), so it runs even when tablassert is absent.
    monkeypatch.setattr(_RUN_MODULE, "tablassert_available", lambda: False)
    monkeypatch.setattr(_RUN_MODULE, "stream_subprocess", fake_subprocess)
    TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir, tablassert_dir="/opt/tablassert", fullmap="data/fullmap"))

    assert seen[0][:5] == ["uv", "run", "--with-editable", "/opt/tablassert", "tablassert"]
    assert seen[0][-1] == str(workdir.root / "tables" / "graph.yaml")  # no --fullmap flag (removed in Tablassert 8.1)


def test_real_runner_raises_when_tablassert_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    monkeypatch.delenv(TABLASERT_DIR_ENV, raising=False)
    monkeypatch.setattr(_RUN_MODULE, "tablassert_available", lambda: False)

    with pytest.raises(RuntimeError, match="uv sync"):
        TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir, fullmap="data/fullmap"))


def test_real_runner_raises_when_fullmap_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    _patch_installed(monkeypatch)
    with pytest.raises(RuntimeError, match="fullmap"):
        TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir))


def test_real_runner_appends_qc_when_runtime_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    seen: list[list[str]] = []

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    _patch_installed(monkeypatch)
    monkeypatch.setattr(_RUN_MODULE, "qc_runtime_available", lambda: True)
    monkeypatch.setattr(_RUN_MODULE, "stream_subprocess", fake_subprocess)
    TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir, qc=True, fullmap="data/fullmap"))

    assert "--qc" in seen[0]
    assert _read_report(workdir)["qc"] is True


def test_real_runner_skips_qc_when_runtime_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    seen: list[list[str]] = []

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    _patch_installed(monkeypatch)
    monkeypatch.setattr(_RUN_MODULE, "qc_runtime_available", lambda: False)
    monkeypatch.setattr(_RUN_MODULE, "stream_subprocess", fake_subprocess)
    TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir, qc=True, fullmap="data/fullmap"))

    assert "--qc" not in seen[0]
    assert _read_report(workdir)["qc"] is False


def test_real_runner_appends_release_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    seen: list[list[str]] = []

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    _patch_installed(monkeypatch)
    monkeypatch.setattr(_RUN_MODULE, "stream_subprocess", fake_subprocess)
    TablassertRunner().run(assertion_refs, config_refs, _ctx(workdir, release=True, fullmap="data/fullmap"))

    assert "--release" in seen[0]
    assert _read_report(workdir)["release"] is True


# --- module-level dispatch --------------------------------------------------------


def test_run_dispatches_to_deferred_without_run_tablassert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    def no_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        msg = "subprocess must not run when the handoff is deferred"
        raise AssertionError(msg)

    monkeypatch.setattr(_RUN_MODULE, "stream_subprocess", no_subprocess)

    # No run_tablassert (no fullmap) -> deferred handoff, no subprocess.
    run_tablassert(assertion_refs, config_refs, _ctx(workdir))
    assert _read_report(workdir)["mode"] == "deferred"


def test_run_dispatches_to_real_with_run_tablassert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    assertion_refs = _assertion_refs(workdir)
    config_refs = tablassert_configs.generate(assertion_refs, _ctx(workdir))

    def fake_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

    _patch_installed(monkeypatch)
    monkeypatch.setattr(_RUN_MODULE, "stream_subprocess", fake_subprocess)

    run_tablassert(assertion_refs, config_refs, _ctx(workdir, run_tablassert=True, fullmap="data/fullmap"))
    assert _read_report(workdir)["mode"] == "real"
