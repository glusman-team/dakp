"""Unit tests for contraindication assertion aggregation (text-mined from DailyMed via NER).

Contraindications are MINED directly from DailyMed SPL "Contraindications" sections (LOINC
``34070-3``) using the single composite NER backend. The object is the mined disease MENTION
TEXT; object CURIE/name/category are left EMPTY for Tablassert/fullmap to resolve (DAKP does no
ontology mapping). Covers: mining a mention from a real contraindication section via the offline
fixture gazetteer; the DailyMed text-mining provenance columns; supporting SPL sets/documents;
determinism; the empty-section edge case; an empty-gazetteer backend; a custom injected
gazetteer; ``default_ner`` resolution; and the end-to-end shaper TSV output.
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.assertions.contraindications import ContraindicationsShaper, build_contraindication_rows, default_ner
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.ner import DiseaseNER


def _fixture_ner(fixture_root: Path) -> DiseaseNER:
    """The offline backend over the ontology fixture gazetteer (mirrors the shaper default)."""
    return default_ner(fixture_root)


# --- mining from a real DailyMed contraindication section -----------------------


def test_mines_contraindication_from_dailymed_section(dailymed_refs: list[ArtifactRef], fixture_root: Path) -> None:
    rows = build_contraindication_rows(dailymed_refs, _fixture_ner(fixture_root))
    by_subject = {r["subject_text"]: r for r in rows}

    # The Ibuprofen label's contraindication section mentions "asthma" -> one mined assertion.
    assert set(by_subject) == {"Ibuprofen"}
    ibu = by_subject["Ibuprofen"]
    assert ibu["predicate"] == "biolink:contraindicated_in"
    # Subject = the SPL active ingredient (with its source-provided UNII).
    assert ibu["subject_curie"] == "UNII:WK2XYI10QM"
    assert ibu["subject_name"] == "Ibuprofen"
    assert ibu["subject_category"] == "ChemicalEntity"
    # Object = the mined mention TEXT; CURIE/name/category are EMPTY for Tablassert to resolve.
    assert ibu["object_text"] == "asthma"
    assert ibu["object_curie"] == ""
    assert ibu["object_name"] == ""
    assert ibu["object_category"] == ""
    # SPL provenance: the supporting set + the contraindication section document.
    assert ibu["supporting_spl_sets"] == "dailymed:SETID-IBUPROFEN-002"
    assert ibu["supporting_spl_documents"] == "SETID-IBUPROFEN-002#34070-3"
    # The offline gazetteer is a high-confidence direct match (score 1.0).
    assert ibu["source_score"] == "1"


def test_provenance_columns_are_dailymed_text_mining(dailymed_refs: list[ArtifactRef], fixture_root: Path) -> None:
    rows = build_contraindication_rows(dailymed_refs, _fixture_ner(fixture_root))
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


def test_section_without_recognizable_disease_yields_no_row(dailymed_refs: list[ArtifactRef], fixture_root: Path) -> None:
    # Examplestatin's contraindication section ("active liver disease ... transaminase
    # elevations") carries no fixture-gazetteer disease -> nothing mined for it. Omeprazole has
    # no contraindication section at all. Only Ibuprofen->asthma survives.
    rows = build_contraindication_rows(dailymed_refs, _fixture_ner(fixture_root))
    assert {r["subject_text"] for r in rows} == {"Ibuprofen"}


def test_backend_that_finds_nothing_yields_no_rows(dailymed_refs: list[ArtifactRef]) -> None:
    # An empty gazetteer extracts no mentions -> no contraindication assertions.
    assert build_contraindication_rows(dailymed_refs, DiseaseNER(gazetteer={})) == []


def test_no_dailymed_inputs_yield_no_rows() -> None:
    assert build_contraindication_rows([], DiseaseNER(gazetteer={"asthma": "disease"})) == []


# --- custom injected gazetteer + aggregation/determinism ------------------------


def test_custom_gazetteer_mines_its_configured_terms(dailymed_refs: list[ArtifactRef]) -> None:
    # A gazetteer covering both contraindication sections mines both ingredients.
    ner = DiseaseNER(gazetteer={"asthma": "disease", "liver disease": "disease"})
    rows = build_contraindication_rows(dailymed_refs, ner)
    by_subject = {r["subject_text"]: r for r in rows}
    assert set(by_subject) == {"Examplestatin", "Ibuprofen"}
    assert by_subject["Examplestatin"]["object_text"] == "liver disease"
    assert by_subject["Ibuprofen"]["object_text"] == "asthma"
    # Object CURIEs stay empty regardless of the mined term.
    assert all(r["object_curie"] == "" for r in rows)


def test_rows_are_deterministically_ordered(dailymed_refs: list[ArtifactRef], fixture_root: Path) -> None:
    ner = _fixture_ner(fixture_root)
    first = build_contraindication_rows(dailymed_refs, ner)
    second = build_contraindication_rows(dailymed_refs, ner)
    assert first == second
    keys = [(r["subject_text"], r["object_text"]) for r in first]
    assert keys == sorted(keys)


# --- default_ner resolution -----------------------------------------------------


def test_default_ner_loads_fixture_gazetteer(fixture_root: Path) -> None:
    ner = default_ner(fixture_root)
    assert isinstance(ner, DiseaseNER)
    # The fixture gazetteer knows "asthma" (disease) and "headache" (phenotype).
    assert sorted(m.text for m in ner.extract("asthma and headache")) == ["asthma", "headache"]


# --- end-to-end shaper output ---------------------------------------------------


def test_shaper_writes_uncompressed_tsv_with_contract_columns(dailymed_refs: list[ArtifactRef], ctx: TaskContext) -> None:
    # The shaper builds the offline fixture-gazetteer backend from ctx.fixture_root on its own.
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
    assert row["object_curie"] == ""
    assert row["agent_type"] == "text_mining_agent"
    assert row["upstream_resource_ids"] == "infores:dailymed"
