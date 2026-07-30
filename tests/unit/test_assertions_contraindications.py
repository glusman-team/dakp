"""Unit tests for contraindication assertion aggregation (text-mined from DailyMed via NER).

Contraindications are MINED directly from DailyMed SPL "Contraindications" sections (LOINC
``34070-3``) using a configurable NER backend (re-scoped to drop MEDI/Matrix). Covers:
mining a disease mention from a real contraindication section via the offline dictionary
backend; the DailyMed text-mining provenance columns (dailymed upstream, text_mining_agent,
NO MEDI); supporting SPL sets/documents; determinism; the empty-section edge case (a
contraindication section with no recognizable disease yields no row); the mock backend +
backend-resolution config; and the end-to-end shaper TSV output.
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.assertions.contraindications import ContraindicationsShaper, build_contraindication_rows, resolve_ner_backend
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.backends import DictionaryNERBackend, MockNERBackend

# --- helpers --------------------------------------------------------------------


def _dictionary_backend(fixture_root: Path) -> DictionaryNERBackend:
    """The offline dictionary baseline over the ontology fixture (mirrors the shaper default)."""
    return DictionaryNERBackend.from_tsv(fixture_root / "ontology" / "disease_map.tsv")


# --- mining from a real DailyMed contraindication section -----------------------


def test_mines_contraindication_from_dailymed_section(
    dailymed_refs: list[ArtifactRef], fixture_root: Path, disease_map: dict[str, dict[str, str]]
) -> None:
    backend = _dictionary_backend(fixture_root)
    rows = build_contraindication_rows(dailymed_refs, backend, disease_map)
    by_subject = {r["subject_text"]: r for r in rows}

    # The Ibuprofen label's contraindication section mentions "asthma" -> one mined assertion.
    assert set(by_subject) == {"Ibuprofen"}
    ibu = by_subject["Ibuprofen"]
    assert ibu["predicate"] == "biolink:contraindicated_in"
    # Subject = the SPL active ingredient (with its UNII).
    assert ibu["subject_curie"] == "UNII:WK2XYI10QM"
    assert ibu["subject_name"] == "Ibuprofen"
    assert ibu["subject_category"] == "ChemicalEntity"
    # Object = the mined disease mention, resolved to the dictionary baseline CURIE.
    assert ibu["object_text"] == "asthma"
    assert ibu["object_curie"] == "MONDO:0004979"
    assert ibu["object_name"] == "asthma"
    assert ibu["object_category"] == "Disease"
    # SPL provenance: the supporting set + the contraindication section document.
    assert ibu["supporting_spl_sets"] == "SETID-IBUPROFEN-002"
    assert ibu["supporting_spl_documents"] == "SETID-IBUPROFEN-002#34070-3"
    # The dictionary baseline is a high-confidence direct match (score 1.0).
    assert ibu["source_score"] == "1"


def test_provenance_columns_are_dailymed_text_mining(
    dailymed_refs: list[ArtifactRef], fixture_root: Path, disease_map: dict[str, dict[str, str]]
) -> None:
    rows = build_contraindication_rows(dailymed_refs, _dictionary_backend(fixture_root), disease_map)
    assert rows
    for row in rows:
        assert row["predicate"] == "biolink:contraindicated_in"
        assert row["knowledge_level"] == "knowledge_assertion"
        assert row["agent_type"] == "text_mining_agent"
        assert row["primary_knowledge_source"] == "infores:multiomics-drugapprovals"
        # Text-mined from DailyMed only — NO MEDI anywhere in the provenance.
        assert row["upstream_resource_ids"] == "infores:dailymed"
        assert "medi" not in row["upstream_resource_ids"]


# --- empty-section edge case ----------------------------------------------------


def test_section_without_recognizable_disease_yields_no_row(
    dailymed_refs: list[ArtifactRef], fixture_root: Path, disease_map: dict[str, dict[str, str]]
) -> None:
    # Examplestatin's contraindication section ("active liver disease ... transaminase
    # elevations") carries no dictionary-recognizable disease -> nothing mined for it.
    # Omeprazole has no contraindication section at all. Only Ibuprofen->asthma survives.
    rows = build_contraindication_rows(dailymed_refs, _dictionary_backend(fixture_root), disease_map)
    assert {r["subject_text"] for r in rows} == {"Ibuprofen"}


def test_backend_that_finds_nothing_yields_no_rows(dailymed_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    # An empty mock backend extracts no mentions -> no contraindication assertions.
    assert build_contraindication_rows(dailymed_refs, MockNERBackend(), disease_map) == []


def test_no_dailymed_inputs_yield_no_rows(disease_map: dict[str, dict[str, str]]) -> None:
    assert build_contraindication_rows([], MockNERBackend({"asthma": "disease"}), disease_map) == []


# --- mock backend + aggregation/determinism -------------------------------------


def test_mock_backend_mines_its_configured_vocabulary(dailymed_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    # A mock vocabulary covering both contraindication sections mines both ingredients.
    backend = MockNERBackend({"asthma": "disease", "liver disease": "disease"}, score=0.75)
    rows = build_contraindication_rows(dailymed_refs, backend, disease_map)
    by_subject = {r["subject_text"]: r for r in rows}
    assert set(by_subject) == {"Examplestatin", "Ibuprofen"}
    assert by_subject["Examplestatin"]["object_text"] == "liver disease"
    assert by_subject["Ibuprofen"]["object_text"] == "asthma"
    # source_score reflects the NER span score (deterministic string formatting).
    assert by_subject["Ibuprofen"]["source_score"] == "0.75"


def test_rows_are_deterministically_ordered(dailymed_refs: list[ArtifactRef], fixture_root: Path, disease_map: dict[str, dict[str, str]]) -> None:
    backend = _dictionary_backend(fixture_root)
    first = build_contraindication_rows(dailymed_refs, backend, disease_map)
    second = build_contraindication_rows(dailymed_refs, backend, disease_map)
    assert first == second
    keys = [(r["subject_text"], r["object_text"]) for r in first]
    assert keys == sorted(keys)


# --- backend resolution (configurable; offline default) -------------------------


def test_resolve_ner_backend_defaults_to_dictionary_over_fixture(fixture_root: Path) -> None:
    assert isinstance(resolve_ner_backend(fixture_root, {}), DictionaryNERBackend)


def test_resolve_ner_backend_honors_config_name_and_injected_instance() -> None:
    assert isinstance(resolve_ner_backend(None, {"ner_backend_name": "mock"}), MockNERBackend)
    sentinel = MockNERBackend()
    assert resolve_ner_backend(None, {"ner_backend": sentinel}) is sentinel
    # No fixture + no config -> empty mock backend (still import-free / offline).
    assert isinstance(resolve_ner_backend(None, {}), MockNERBackend)


# --- end-to-end shaper output ---------------------------------------------------


def test_shaper_writes_uncompressed_tsv_with_contract_columns(dailymed_refs: list[ArtifactRef], ctx: TaskContext) -> None:
    # The shaper resolves the offline dictionary backend from ctx.fixture_root on its own.
    refs = ContraindicationsShaper().transform([*dailymed_refs], ctx)
    assert len(refs) == 1
    out = refs[0]
    assert out.uri.name == "contraindication_assertions.tsv"

    frame = schemas.read_table(out.uri)
    assert frame.columns == schemas.CONTRAINDICATION_COLUMNS
    assert frame.height == 1  # Ibuprofen -> asthma
    assert out.uri.read_bytes().startswith(b"subject_text\t")
    row = frame.row(0, named=True)
    assert row["subject_text"] == "Ibuprofen"
    assert row["object_text"] == "asthma"
    assert row["agent_type"] == "text_mining_agent"
    assert row["upstream_resource_ids"] == "infores:dailymed"
