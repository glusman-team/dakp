"""Unit tests for contraindication assertion aggregation (MEDI + DailyMed, Milestone 5).

Covers first-scope DailyMed contraindication-section support, MEDI-provided subject/object CURIEs,
column-alias handling (canonical extractor vs legacy fixture/shim columns), multi-row aggregation
(max source_score, unioned supporting sets), medi_version resolution, provenance columns,
determinism, and the end-to-end shaper TSV output.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.assertions.contraindications import ContraindicationsShaper, build_contraindication_rows
from dakp_pipeline.assertions.evidence import build_dailymed_evidence
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext

_DEFAULT_VERSION = "MEDI-0.0-mock"


def _canonical_frame() -> pl.DataFrame:
    """MEDI rows in the canonical extractor column convention."""
    return pl.DataFrame(
        {
            "active_ingredient": ["Ibuprofen", "Warfarin"],
            "contraindication_text": ["hypersensitivity to ibuprofen", "contraindicated in pregnancy"],
            "disease_contraindicated": ["asthma", "pregnancy"],
            "normalized_drug_id": ["UNII:WK2XYI10QM", "UNII:5A9Q8V7QX1"],
            "normalized_drug_label": ["Ibuprofen", "Warfarin"],
            "normalized_disease_id": ["MONDO:0004979", ""],
            "normalized_disease_label": ["asthma", ""],
            "source_score": ["0.90", "0.00"],
            "medi_version": ["MEDI-1.0", "MEDI-1.0"],
        }
    )


# --- first-scope DailyMed support + MEDI-provided ids ---------------------------


def test_support_scoping_and_medi_curie_population(
    medi_refs: list[ArtifactRef], dailymed_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]
) -> None:
    medi_frame = schemas.read_table(next(r for r in medi_refs if r.uri.name == "contraindications.parquet").uri)
    ev = build_dailymed_evidence(dailymed_refs)

    rows = build_contraindication_rows(medi_frame, ev, disease_map, _DEFAULT_VERSION)
    by_subject = {r["subject_text"]: r for r in rows}

    assert set(by_subject) == {"Ibuprofen", "Warfarin", "Methotrexate"}

    # Ibuprofen matches a DailyMed contraindication section (first-scope by ingredient).
    ibu = by_subject["Ibuprofen"]
    assert ibu["supporting_spl_sets"] == "SETID-IBUPROFEN-002"
    assert ibu["subject_curie"] == "UNII:WK2XYI10QM"  # MEDI-provided drug id
    assert ibu["object_curie"] == "MONDO:0004979"  # MEDI-provided disease id
    assert ibu["predicate"] == "biolink:contraindicated_in"

    # Warfarin/Methotrexate have no DailyMed document -> no SPL support.
    assert by_subject["Warfarin"]["supporting_spl_sets"] == ""
    assert by_subject["Methotrexate"]["supporting_spl_sets"] == ""
    # Warfarin's condition has no MEDI disease id and is not in the dictionary -> empty object CURIE.
    assert by_subject["Warfarin"]["object_curie"] == ""
    assert by_subject["Warfarin"]["object_text"] == "pregnancy"
    # Methotrexate keeps the MEDI-provided disease id/label (renal failure).
    assert by_subject["Methotrexate"]["object_curie"] == "MONDO:0005154"


# --- column-alias handling (legacy fixture/shim columns) ------------------------


def test_reads_legacy_fixture_column_names(dailymed_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    # The pipeline's MEDI shim writes the fixture columns (final_normalized_*, drug_name, ...).
    legacy = pl.DataFrame(
        {
            "active_ingredient": ["Ibuprofen"],
            "contraindications": ["hypersensitivity to ibuprofen"],
            "disease_contraindicated": ["asthma"],
            "final_normalized_drug_id": ["UNII:WK2XYI10QM"],
            "final_normalized_drug_label": ["Ibuprofen"],
            "final_normalized_disease_id": ["MONDO:0004979"],
            "final_normalized_disease_label": ["asthma"],
            "drug_name": ["Ibuprofen"],
            "contraindicated_condition": ["asthma"],
            "source_score": ["0.9"],
        }
    )
    ev = build_dailymed_evidence(dailymed_refs)
    rows = build_contraindication_rows(legacy, ev, disease_map, _DEFAULT_VERSION)

    assert len(rows) == 1
    row = rows[0]
    assert row["subject_text"] == "Ibuprofen"
    assert row["subject_curie"] == "UNII:WK2XYI10QM"
    assert row["object_curie"] == "MONDO:0004979"
    assert row["source_score"] == "0.9"
    assert row["supporting_spl_sets"] == "SETID-IBUPROFEN-002"
    assert row["medi_version"] == _DEFAULT_VERSION  # no medi_version column -> default


# --- aggregation: max score + unioned support + determinism ---------------------


def test_duplicate_pairs_aggregate_max_score_and_union_sets(dailymed_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    ev = build_dailymed_evidence(dailymed_refs)
    dupes = pl.DataFrame(
        {
            "active_ingredient": ["Ibuprofen", "Ibuprofen"],
            "disease_contraindicated": ["asthma", "asthma"],
            "normalized_drug_id": ["UNII:WK2XYI10QM", "UNII:WK2XYI10QM"],
            "normalized_disease_id": ["MONDO:0004979", "MONDO:0004979"],
            "source_score": ["0.40", "0.90"],
            "medi_version": ["MEDI-1.0", "MEDI-1.0"],
        }
    )
    rows = build_contraindication_rows(dupes, ev, disease_map, _DEFAULT_VERSION)
    assert len(rows) == 1
    assert rows[0]["source_score"] == "0.90"  # max, original formatting preserved
    assert rows[0]["supporting_spl_sets"] == "SETID-IBUPROFEN-002"  # unioned, deduped


def test_rows_are_deterministically_ordered(
    medi_refs: list[ArtifactRef], dailymed_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]
) -> None:
    medi_frame = schemas.read_table(next(r for r in medi_refs if r.uri.name == "contraindications.parquet").uri)
    ev = build_dailymed_evidence(dailymed_refs)
    first = build_contraindication_rows(medi_frame, ev, disease_map, _DEFAULT_VERSION)
    second = build_contraindication_rows(medi_frame, ev, disease_map, _DEFAULT_VERSION)
    assert first == second
    keys = [(r["subject_text"], r["object_text"]) for r in first]
    assert keys == sorted(keys)


def test_provenance_columns_are_fixed(dailymed_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    ev = build_dailymed_evidence(dailymed_refs)
    rows = build_contraindication_rows(_canonical_frame(), ev, disease_map, _DEFAULT_VERSION)
    assert rows
    for row in rows:
        assert row["predicate"] == "biolink:contraindicated_in"
        assert row["knowledge_level"] == "knowledge_assertion"
        assert row["agent_type"] == "manual_validation_of_automated_agent"
        assert row["primary_knowledge_source"] == "infores:multiomics-drugapprovals"
        assert row["upstream_resource_ids"] == "infores:medi|infores:dailymed"
        assert row["medi_version"] == "MEDI-1.0"  # taken from the row


def test_no_medi_frame_yields_no_rows(dailymed_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]) -> None:
    assert build_contraindication_rows(None, build_dailymed_evidence(dailymed_refs), disease_map, _DEFAULT_VERSION) == []


# --- end-to-end shaper output ---------------------------------------------------


def test_shaper_writes_uncompressed_tsv_with_contract_columns(
    medi_refs: list[ArtifactRef], dailymed_refs: list[ArtifactRef], ctx: TaskContext
) -> None:
    refs = ContraindicationsShaper().transform([*medi_refs, *dailymed_refs], ctx)
    assert len(refs) == 1
    out = refs[0]
    assert out.uri.name == "contraindication_assertions.tsv"

    frame = schemas.read_table(out.uri)
    assert frame.columns == schemas.CONTRAINDICATION_COLUMNS
    assert frame.height == 3
    assert out.uri.read_bytes().startswith(b"subject_text\t")
