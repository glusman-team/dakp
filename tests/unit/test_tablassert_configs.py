"""Unit tests for Tablassert config generation + runner (Milestone 7).

Asserts the generated Graph + per-table configs are valid YAML matching the ACTUAL current
Tablassert >= 11 schema (``template:``-wrapped ``Section``: ``source.kind=text``, column-encoded
subject/object/predicate, column-encoded ``statement.qualifiers`` where a column backs them,
``provenance.override`` ManualProvenance, column-encoded evidence annotations; the graph config
carries the mandatory ``rig:`` section and no legacy top-level RIG keys), with column
letters derived from the assertion-table contracts and provenance matching the DINGO
translator-ingest conventions. Also covers the runner: the mock runner's
handoff report and the real runner's monkeypatchable subprocess boundary.

These tests require the INSTALLED ``tablassert`` package (a core DAKP dependency): config
generation derives the hard category allow-lists from its Biolink ``Categories`` enum. The
sibling ``../Tablassert`` checkout is never imported or required.
"""

from __future__ import annotations

import copy
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
    upstream_record_urls_supported,
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
# a qualifier entity get entries (see ``_TABLE_QUALIFIERS`` for the per-table justification).
# Contraindication context is sparse and nullable; it is not the same disease as the object.
EXPECTED_QUALIFIERS: dict[str, dict[str, str]] = {
    "approved_treats_assertions": {},
    "faers_applied_to_treat_assertions": {},
    "contraindication_assertions": {"disease_context_qualifier": "disease_context_text"},
}

# assertion table -> {annotation name: (assertion column it encodes, multivalued separator)}.
# Every name here must be a slot DAKP's association class
# (``ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation``) actually declares, or Tablassert
# relocates the value off the edge into the inlined supporting study's StudyResult description --
# a junk drawer no translator-ingests source models. So: the common ``edge_evidence`` column maps to
# ``has_evidence`` (ONE annotation, carrying the sorted ``dailymed:<spl_set_id>`` CURIEs, because
# duplicate annotation names silently overwrite
# each other), and ``case_count`` maps to ``evidence_count`` rather than the sibling-class-only
# ``number_of_cases``. ``split_by: "|"``
# makes the pipe-joined cells emit as real JSON arrays. ``approval_ids`` keeps its column name and,
# under Tablassert >= 12, reaches the edge as a curated TOP-LEVEL pass-through field (it is on
# Tablassert's edge allow-list as a known-pending Biolink slot, so it no longer folds); DAKP
# splits it with ``split_by`` so the edge carries the legacy ``approvals`` JSON-ARRAY shape;
# ``source_score`` still folds into ``supporting_text``.
EXPECTED_ANNOTATIONS = {
    "approved_treats_assertions": {
        "approval_ids": ("approval_ids", "|"),
        "has_evidence": ("edge_evidence", "|"),
        "clinical_approval_status": ("clinical_approval_status", None),
    },
    "faers_applied_to_treat_assertions": {
        "evidence_count": ("case_count", None),
        "approval_ids": ("approval_ids", "|"),
        "has_evidence": ("edge_evidence", "|"),
        "clinical_approval_status": ("clinical_approval_status", None),
    },
    "contraindication_assertions": {
        "approval_ids": ("approval_ids", "|"),
        "has_evidence": ("edge_evidence", "|"),
        "supporting_text": ("evidence_text", "|"),
        "source_score": ("source_score", None),
    },
}

# assertion table -> the REAL upstream dataset URL recorded as ``source.url`` (never a placeholder;
# with ``upstream_source_record_urls`` set, Tablassert keeps it for the RIG and places the edge
# ``sources[].source_record_urls`` per-upstream instead of on the primary DAKP entry).
EXPECTED_SOURCE_URLS = {
    "approved_treats_assertions": "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm",
    "faers_applied_to_treat_assertions": "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html",
    "contraindication_assertions": "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm",
}

# upstream infores -> the download URL that infores' supporting entry carries as its own
# ``source_record_urls`` on every edge where it appears (mirrors tablassert_configs._INFORES_RECORD_URLS).
EXPECTED_UPSTREAM_RECORD_URLS = {
    "infores:dailymed": ["https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"],
    "infores:faers": ["https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"],
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
    # Edge source_record_urls live on the per-upstream supporting entries, not the primary DAKP
    # entry — but only on Tablassert releases that model the slot (post-12.0.0); stock 12.0.0
    # forbids the key and keeps the URLs on the primary entry via source.url.
    if upstream_record_urls_supported():
        assert override["upstream_source_record_urls"] == {resource: EXPECTED_UPSTREAM_RECORD_URLS[resource] for resource in upstream}
    else:
        assert "upstream_source_record_urls" not in override
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
        if table == "contraindication_assertions":
            assert entry["nullable"] is True
            assert entry["prioritize"] == ["Disease"]
            assert entry["avoid"] == category_avoid_list(["Disease"])
        else:
            assert "nullable" not in entry
            assert entry["prioritize"] == list(OBJECT_PRIORITIZE)
            assert entry["avoid"] == category_avoid_list(OBJECT_PRIORITIZE)


def test_declared_qualifier_emits_a_column_encoding() -> None:
    # Exercise the emission path independently of the production backing-column choice.
    table = "contraindication_assertions"
    declared = dict(tablassert_configs._TABLE_QUALIFIERS, **{table: (("disease_context_qualifier", "object_text"),)})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tablassert_configs, "_TABLE_QUALIFIERS", declared)
        statement = tablassert_configs.table_config(table)["statement"]
    assert statement["qualifiers"] == [
        {
            "qualifier": "disease_context_qualifier",
            "method": "column",
            "encoding": tablassert_configs.column_letter(table, "object_text"),
            "nullable": True,
            "prioritize": ["Disease"],
            "avoid": category_avoid_list(["Disease"]),
        }
    ]


def test_qualifier_slots_are_valid_biolink_qualifiers() -> None:
    # No table declares a qualifier today, so this guards whatever the first one to come back is:
    # every emitted qualifier slot must be a real member of the installed Tablassert's Biolink
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
    for annotation, (column, split_by) in EXPECTED_ANNOTATIONS[table].items():
        entry = by_name[annotation]
        assert entry["method"] == "column"
        # The encoding letter must address the expected assertion column.
        assert _column_at(table, entry["encoding"]) == column
        assert entry["encoding"] == tablassert_configs.column_letter(table, column)
        # Multivalued Biolink slots split the pipe-joined cell into a real JSON array.
        # ``delimiter`` was the Tablassert <= 8.2.0 spelling and is REJECTED from 8.2.1 on
        # (extra="forbid"), so assert its absence too -- a regression would fail the build.
        assert "delimiter" not in entry
        if split_by is None:
            assert "split_by" not in entry
        else:
            assert entry["split_by"] == split_by


@pytest.mark.parametrize("table", TABLES)
def test_table_config_annotation_names_are_unique(table: str) -> None:
    """No two annotations in a section may share a name -- Tablassert would silently drop one.

    Annotations are applied in declaration order as ``with_columns(pl.col(src).alias(name))``
    with no duplicate check, so a second entry named ``has_evidence`` OVERWRITES the first and
    that column's values vanish without a warning. This is why the SPL set and section URLs are
    unioned into one ``supporting_spl_evidence`` column upstream instead of being declared as
    two ``has_evidence`` annotations.
    """
    names = [entry["annotation"] for entry in tablassert_configs.table_config(table)["annotations"]]
    assert len(names) == len(set(names)), f"duplicate annotation names in {table}: {names}"


def _dakp_association_classes(table: str) -> set[type]:
    """The association classes Tablassert resolves for every category pair this table allows.

    Derived, not hardcoded: the category comes from the (subject role, object role) pair and is
    then reconciled against the section predicate, exactly as ``Tcode._ops`` does at build time.
    """
    from tablassert.biolink import resolve_association_class
    from tablassert.lib import derived_edge_category

    predicate = f"biolink:{tablassert_configs._TABLE_SPECS[table][1]}"
    return {
        resolve_association_class(derived_edge_category(subject, obj), predicate)
        for subject in tablassert_configs.SUBJECT_PRIORITIZE
        for obj in tablassert_configs.OBJECT_PRIORITIZE
    }


@pytest.mark.parametrize("table", TABLES)
def test_annotation_slots_survive_dakp_association_class(table: str) -> None:
    """Every annotated Biolink slot is one DAKP's association class can actually hold.

    A name that is a slot of SOME association class but not of the one DAKP's edges resolve to
    passes Tablassert's edge-field allow-list, then gets nulled by ``prune_to_class`` and
    stringified into the inlined supporting study's StudyResult ``description`` -- exactly the
    junk drawer this contract exists to keep DAKP out of. ``number_of_cases`` and
    ``supporting_documents`` are the two that used to land there; this fails loudly if either
    (or a newly added annotation) comes back.
    """
    from tablassert.biolink import ALLOWED_EDGE_FIELDS, KNOWN_PENDING_EDGE_FIELDS, class_fields

    for cls in _dakp_association_classes(table):
        slots = class_fields(cls)
        for name in EXPECTED_ANNOTATIONS[table]:
            if name not in ALLOWED_EDGE_FIELDS:
                continue  # deliberately folded into ``supporting_text`` (e.g. ``source_score``)
            if name in KNOWN_PENDING_EDGE_FIELDS:
                continue  # curated Tablassert pass-through; no association class declares it
            assert name in slots, f"{name} is not a slot of {cls.__name__}; it would be relocated onto the supporting study"


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


def test_committed_table_configs_match_generator_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # The committed ``tables/*.yaml`` must byte-equal the generator output (regenerate, never
    # hand-diverge). They are committed in the WITH-``upstream_source_record_urls`` form (the
    # canonical new-Tablassert shape), so pin the probe ON: on a stock 12.0.0 install the generator
    # would otherwise omit the key and byte-diverge from the committed files.
    monkeypatch.setattr(tablassert_configs, "upstream_record_urls_supported", lambda: True)
    repo_tables = Path(__file__).resolve().parents[2] / "tables"
    for table in TABLES:
        basename = tablassert_configs._TABLE_SPECS[table][0]
        assert (repo_tables / f"{basename}.yaml").read_text(encoding="utf-8") == tablassert_configs.table_yaml(table)
    assert (repo_tables / "graph.yaml").read_text(encoding="utf-8") == tablassert_configs.graph_yaml()


# --- graph config structure -------------------------------------------------------


def test_graph_config_structure() -> None:
    graph = tablassert_configs.graph_config()
    assert graph["name"] == "DRUG_APPROVALS_KP"
    assert isinstance(graph["version"], str)
    assert graph["version"]
    assert graph["fullmap"] == ".fullmap"  # default placeholder; generate() writes the real ctx fullmap path
    assert graph["tables"] == ["tables/approved_treats.yaml", "tables/faers_applied_to_treat.yaml", "tables/contraindications.yaml"]
    # Tablassert >= 11 rejects the legacy top-level RIG keys; every RIG fact lives under `rig:`.
    for legacy_key in ("description", "infores", "contributions", "ui_explanation"):
        assert legacy_key not in graph
    rig = graph["rig"]
    assert rig["source_info"]["infores_id"] == INFORES_DAKP
    assert rig["source_info"]["name"] == "Drug Approvals Knowledge Provider (DAKP)"  # full title, not the bare acronym
    assert rig["source_info"]["description"]  # the former top-level graph description moved here
    assert any("https://pmc.ncbi.nlm.nih.gov/articles/PMC11601480/" in citation for citation in rig["source_info"]["citations"])
    assert rig["source_info"]["data_versioning_and_releases"]
    assert rig["source_info"]["terms_of_use_info"]
    assert all("https://" in location for location in rig["source_info"]["data_access_locations"])
    # Supporting data sources: exactly the two edge-backed upstreams (no infores:medi — this
    # rebuild text-mines contraindications from DailyMed SPL; no Drugs@FDA — it backs no edge).
    supporting = rig["supporting_data_source_info"]
    assert [entry["infores_id"] for entry in supporting] == ["infores:dailymed", "infores:faers"]
    # Each entry's file location is the URL constant the acquisition layer actually downloads.
    assert supporting[0]["relevant_files"][0]["location"] == tablassert_configs.dailymed_source.FULL_RELEASE_INDEX_URL
    assert supporting[1]["relevant_files"][0]["location"] == tablassert_configs.faers_source.FDA_FAERS_INDEX_URL
    assert all(entry["terms_of_use_info"]["terms_of_use_description"] for entry in supporting)
    assert rig["ingest_info"]["utility"]
    assert rig["ingest_info"]["scope"]
    assert rig["provenance_info"]["contributions"]
    assert rig["artifact_base_url"] == tablassert_configs.RIG_ARTIFACT_BASE_URL
    assert rig["artifact_base_path"] == tablassert_configs.RIG_ARTIFACT_BASE_PATH


def test_graph_config_validates_against_installed_tablassert() -> None:
    """The generated graph.yaml passes the INSTALLED Tablassert's ``Graph`` model (rig included)."""
    from tablassert.models import Graph

    graph = Graph.model_validate(yaml.safe_load(tablassert_configs.graph_yaml()))
    assert graph.rig.source_info.infores_id == INFORES_DAKP


def test_rig_section_validates_directly_against_tablassert_rig_config() -> None:
    """The ``rig:`` section passes the installed Tablassert's ``RIGConfig`` on its own.

    Guards against Tablassert schema drift: the Graph-level ``model_validate`` test validates the
    rig only as one nested field, so a RIG-schema change that renames or drops a source-info field
    (citations, versioning) could hide behind ``Graph`` defaults. Validating the section DIRECTLY
    pins the exact boundary Tablassert enforces when composing the RIG, and the content asserts
    below pin the enriched human-authored facts — a schema rename must fail loudly here, not
    silently shrink the generated RIG.
    """
    from tablassert.models import RIGConfig

    rig = RIGConfig.model_validate(tablassert_configs.graph_config()["rig"])
    source = rig.source_info
    # Supporting upstreams validate as real RIGSupportingDataSourceInfo entries (infores CURIE +
    # relevant-file URL checks included) and stay exactly the two edge-backed sources.
    assert rig.supporting_data_source_info is not None
    assert [entry.infores_id for entry in rig.supporting_data_source_info] == ["infores:dailymed", "infores:faers"]
    assert source.name == "Drug Approvals Knowledge Provider (DAKP)"
    assert source.citations is not None
    assert any("https://pmc.ncbi.nlm.nih.gov/articles/PMC11601480/" in citation for citation in source.citations)
    assert source.data_versioning_and_releases
    # Kept fields must stay untouched: terms of use, access locations, mechanisms, formats, status.
    assert source.terms_of_use_info.terms_of_use_url == "https://www.nlm.nih.gov/terms.html"
    assert source.data_provision_mechanisms == ["file_download"]
    assert source.data_formats == ["kgx"]
    assert source.source_status == "maintained_regular_updates"


def test_tablassert_rig_config_rejects_unknown_keys_and_bad_source_status() -> None:
    """The installed ``RIGConfig`` schema really BITES: unknown keys and bad enums are rejected.

    Guards the fail-loudly contract: every positive test above only proves the rig section is
    ACCEPTED. If a future Tablassert relaxed to permissive extras (or a free-form source
    status), those tests would keep passing while DINGO's RIG expectations silently stopped
    being enforced. Mutating DEEP COPIES of the generated rig section (never ``_rig_config``
    itself) must therefore raise: an unknown ``source_info`` key AND an out-of-enum
    ``source_status`` each fail validation with an error naming the offending field.
    """
    from pydantic import ValidationError
    from tablassert.models import RIGConfig

    rig = tablassert_configs.graph_config()["rig"]

    with_unknown_key = copy.deepcopy(rig)
    with_unknown_key["source_info"]["not_a_rig_field"] = "surprise"
    with pytest.raises(ValidationError) as extra_exc:
        RIGConfig.model_validate(with_unknown_key)
    assert "source_info.not_a_rig_field" in str(extra_exc.value)
    assert "Extra inputs are not permitted" in str(extra_exc.value)

    with_bad_status = copy.deepcopy(rig)
    with_bad_status["source_info"]["source_status"] = "maintained_never"
    with pytest.raises(ValidationError) as status_exc:
        RIGConfig.model_validate(with_bad_status)
    assert "source_info.source_status" in str(status_exc.value)

    # The generator itself is untouched: the pristine section still validates.
    assert RIGConfig.model_validate(rig).source_info.source_status == "maintained_regular_updates"


def test_rig_supporting_data_source_info_lists_only_edge_backed_upstreams() -> None:
    """``supporting_data_source_info`` carries exactly the two edge-backed upstreams — never MEDI.

    The legacy pipeline's translator-ingests RIG also listed ``infores:medi``, but this rebuild
    has no MEDI source module (contraindications are text-mined from DailyMed SPL), so a MEDI
    entry would be fabricated provenance while a dropped entry would under-attribute the edges.
    Pinning the exact id set through BOTH the raw config and the installed ``Graph`` model (i.e.
    the committed YAML shape) makes either drift fail loudly.
    """
    from tablassert.models import Graph

    ids = [entry["infores_id"] for entry in tablassert_configs.graph_config()["rig"]["supporting_data_source_info"]]
    assert ids == ["infores:dailymed", "infores:faers"]
    assert "infores:medi" not in ids  # legacy-pipeline source; no MEDI module backs it in this rebuild

    graph = Graph.model_validate(yaml.safe_load(tablassert_configs.graph_yaml()))
    supporting = graph.rig.supporting_data_source_info
    assert supporting is not None
    assert [entry.infores_id for entry in supporting] == ids


def test_rig_ingest_info_enrichment() -> None:
    """``ingest_info`` pins the explicit category, included/filtered content, and considerations.

    DAKP CREATES knowledge (it builds assertion tables) rather than passing a source through, so
    ``translator_knowledge_creator`` must be EXPLICIT, not merely the model default. The included
    content pins ALL FOUR mined section kinds; the filtered content pins the approved-treats-only
    approval gate and the section-kind filter; the future considerations pin the disease-only
    qualifier policy and the approval-status coercion. The upstream RIG's legacy KGX output files must stay rejected — no
    table section sources them, so Tablassert's RIG audit would fail the build. Asserting through
    BOTH the raw config and the installed ``RIGConfig`` model makes any drift fail loudly.
    """
    from tablassert.models import RIGConfig

    ingest = tablassert_configs.graph_config()["rig"]["ingest_info"]
    assert ingest["ingest_categories"] == ["translator_knowledge_creator"]
    # Kept grounded fields stay untouched, and the upstream RIG's legacy KGX-file relevant_files
    # never appear (pipeline OUTPUT artifacts, not inputs).
    assert ingest["utility"]
    assert ingest["scope"]
    assert all(not entry["file_name"].startswith("drug_approvals_kg") for entry in ingest["relevant_files"])
    # Included content keeps the DailyMed entry (all FOUR mined section kinds) plus the FAERS entry.
    included = ingest["included_content"]
    assert [entry["file_name"] for entry in included] == ["DailyMed SPL sections", "FAERS quarterly ASCII zips"]
    assert included[0]["included_records"] == (
        "indications_and_usage (LOINC 34067-9), contraindications (LOINC 34070-3), boxed warnings (LOINC 34066-1), and "
        "warnings/precautions (LOINC 43685-7, legacy 34071-1/42232-9) sections; FDA application numbers are carried "
        "as provenance where available"
    )
    assert included[0]["fields_used"] == (
        "indications_and_usage, contraindications, boxed-warning, and warnings/precautions section text, SPL set identifiers, FDA application numbers"
    )
    assert included[1]["included_records"] == "drug/indication case pairs; case counts"
    # Both scope filters pinned: the approval gate is approved-treats-only (observed-use and
    # text-mined contraindications are not approval-gated), and only the four mined section
    # kinds survive the section filter.
    filtered = ingest["filtered_content"]
    assert [entry["file_name"] for entry in filtered] == ["DailyMed SPL indication sections", "DailyMed SPL sections"]
    assert filtered[0]["filtered_records"] == "indication sections on SPL sets whose NDA lacks a DailyMed SPL approval"
    assert filtered[0]["rationale"] == (
        "approved-treats assertions require an FDA approval backing the indication; observed-use and "
        "text-mined contraindication assertions are deliberately not approval-gated"
    )
    assert filtered[1]["filtered_records"] == (
        "all SPL sections other than indications_and_usage (LOINC 34067-9), contraindications (LOINC 34070-3), "
        "boxed warnings (LOINC 34066-1), and warnings/precautions (LOINC 43685-7, legacy 34071-1/42232-9)"
    )
    assert filtered[1]["rationale"] == "the remaining sections carry no treatment or contraindication evidence"
    # Considerations carry ContentCategories members; the first defers the medication-context
    # interaction assertion (disease_context_qualifier is intentionally disease-only).
    considerations = ingest["future_considerations"]
    assert considerations[0]["category"] == "edge_content"
    assert "disease_context_qualifier" in considerations[0]["consideration"]

    # The same facts must survive the installed Tablassert model (enum members coerce to their
    # plain string values at validation time).
    info = RIGConfig.model_validate(tablassert_configs.graph_config()["rig"]).ingest_info
    assert info.ingest_categories == ["translator_knowledge_creator"]
    assert info.included_content is not None
    assert [entry.file_name for entry in info.included_content] == ["DailyMed SPL sections", "FAERS quarterly ASCII zips"]
    assert info.filtered_content is not None
    assert [entry.file_name for entry in info.filtered_content] == ["DailyMed SPL indication sections", "DailyMed SPL sections"]
    assert info.future_considerations is not None
    assert info.future_considerations[0].category == "edge_content"


def test_rig_provenance_info_credits_named_contributors_and_artifacts() -> None:
    """``provenance_info`` keeps the DINGO-reviewed contributors verbatim and adds Skye Lane Goetz.

    The three upstream contributor statements must survive VERBATIM (the upstream RIG passed
    DINGO review with them), and Skye Lane Goetz is credited because she authored the upstream
    Tablassert feature this pipeline's provenance override requires (SkyeAv/Tablassert#104)
    and co-authored the cited preprint (PMC11601480). People precede the pipeline/tooling
    statements. ``RIGProvenanceInfo`` has ONLY ``contributions`` + ``artifacts``, so
    validating through the installed model also proves no invented keys slip in.
    """
    from tablassert.models import RIGConfig

    provenance = tablassert_configs.graph_config()["rig"]["provenance_info"]
    contributions = provenance["contributions"]
    assert contributions == [
        "Gwenlyn Glusman - code author, domain expertise, data modeling",
        "Matthew Brush - data modeling",
        "Sierra Moxon - code, data modeling",
        "Skye Lane Goetz - code author, pipeline engineering, Tablassert integration",
        "DAKP pipeline (https://github.com/glusman-team/dakp): source acquisition, assertion modeling",
        "Tablassert: KGX and RIG generation",
    ]
    assert "Skye Lane Goetz - code author, pipeline engineering, Tablassert integration" in contributions
    assert provenance["artifacts"] == [
        "DAKP pipeline repository: https://github.com/glusman-team/dakp",
        (
            "Upstream DINGO-reviewed DAKP RIG: https://github.com/NCATSTranslator/translator-ingests/blob/main/src/"
            "translator_ingest/ingests/dakp/dakp_rig.yaml"
        ),
        "RIG review issue: https://github.com/NCATSTranslator/translator-ingests/issues/416",
    ]

    info = RIGConfig.model_validate(tablassert_configs.graph_config()["rig"]).provenance_info
    assert info.contributions == contributions
    assert info.artifacts == provenance["artifacts"]


def test_rig_target_info_pins_modeling_considerations_and_rejects_type_summaries() -> None:
    """``target_info`` carries ONLY the two ``RIGTargetInfoExtras`` fields, never type summaries.

    The upstream RIG's ``target_info.edge_type_info`` / ``node_type_info`` are rejected
    wholesale: Tablassert GENERATES node/edge type summaries from the observed build, so
    hand-authored summaries would both fail validation and drift from the graph. The
    consideration categories must be exact ``ModelingCategories`` members. Asserting through
    BOTH the raw config and the installed ``RIGConfig`` model makes any drift fail loudly.
    """
    from tablassert.models import RIGConfig

    target = tablassert_configs.graph_config()["rig"]["target_info"]
    assert set(target) == {"future_considerations", "additional_notes"}
    considerations = target["future_considerations"]
    assert [entry["category"] for entry in considerations] == ["qualifiers", "edge_properties"]
    assert "disease_context_qualifier" in considerations[0]["consideration"]
    assert "approval_ids" in considerations[1]["consideration"]
    assert any("generated by Tablassert" in note for note in target["additional_notes"])

    info = RIGConfig.model_validate(tablassert_configs.graph_config()["rig"]).target_info
    assert info is not None
    assert info.future_considerations is not None
    # ``use_enum_values``: validated ModelingCategories members compare as plain strings.
    assert [entry.category for entry in info.future_considerations] == ["qualifiers", "edge_properties"]
    assert info.additional_notes == target["additional_notes"]


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
    assert loaded["name"] == "DRUG_APPROVALS_KP"


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
    assert "name: DRUG_APPROVALS_KP" in graph_text
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


def test_upstream_record_urls_supported_reflects_installed_model() -> None:
    """The probe tracks the installed ManualProvenance, not the version string (a post-12.0.0
    checkout also reports 12.0.0, so the field set is the only truthful signal)."""
    from tablassert.models import ManualProvenance

    assert upstream_record_urls_supported() is ("upstream_source_record_urls" in ManualProvenance.model_fields)


def test_upstream_record_urls_supported_false_when_tablassert_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tablassert_configs, "tablassert_available", lambda: False)
    assert upstream_record_urls_supported() is False


def test_table_config_omits_upstream_record_urls_on_stock_tablassert(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a Tablassert without the #104 slot the override must omit the key entirely —
    pydantic rejects extra override inputs, which is what broke CI on stock 12.0.0."""
    monkeypatch.setattr(tablassert_configs, "upstream_record_urls_supported", lambda: False)
    for table in TABLES:
        override = tablassert_configs.table_config(table)["provenance"]["override"]
        assert "upstream_source_record_urls" not in override
        assert set(override) == {"upstream_resource_ids", "knowledge_level", "agent_type"}


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
