"""Edge-case tests for ``dakp_pipeline.assertions.evidence`` (drive to 100% branch coverage).

Targets the uncovered lines: ``find_table`` swallowing an unreadable matching input (103-104),
``find_faers_cases`` rejecting a case frame that lacks the required columns (134), the
``build_dailymed_evidence`` skip branches for approvals/ingredients/sections rows with missing
keys (191, 203, 214), and the ``build_drugsfda_ingredient_map`` skip on missing NDA/ingredient
(239->236). Plus FAERS case-table resolution preferences (global vs partition vs TSV).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.assertions.evidence import build_dailymed_evidence, build_drugsfda_ingredient_map, find_faers_cases, find_table
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _parquet(tmp_path: Path, name: str, data: dict[str, list[object]]) -> ArtifactRef:
    path = tmp_path / name
    pl.DataFrame(data).write_parquet(path)
    return _ref(path)


# --- find_table: unreadable matching input is swallowed (103-104) ---------------


def test_find_table_swallows_unreadable_matching_input(tmp_path: Path) -> None:
    corrupt = tmp_path / "spl_approvals.parquet"
    corrupt.write_bytes(b"this is not parquet")  # pl.read_parquet will raise
    assert find_table([_ref(corrupt)], "spl_approvals.parquet") is None


def test_find_table_recovers_after_an_unreadable_input(tmp_path: Path) -> None:
    corrupt = tmp_path / "bad" / "spl_approvals.parquet"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"garbage")
    good = _parquet(tmp_path, "spl_approvals.parquet", {"approval_id": ["012345"], "spl_set_id": ["SET-A"]})
    # The corrupt match is skipped (warning logged) and the readable match is returned.
    frame = find_table([_ref(corrupt), good], "spl_approvals.parquet")
    assert frame is not None
    assert frame.height == 1


def test_find_table_no_match_returns_none(tmp_path: Path) -> None:
    other = _parquet(tmp_path, "something_else.parquet", {"x": ["1"]})
    assert find_table([other], "spl_approvals.parquet") is None


# --- find_faers_cases: column validation + resolution preference ----------------


def test_find_faers_cases_rejects_frame_missing_required_columns(tmp_path: Path) -> None:
    # Named cases.parquet (global path) but lacks drugname/indication -> rejected -> None.
    cases = _parquet(tmp_path, "cases.parquet", {"unrelated": ["x"]})
    assert find_faers_cases([cases]) is None


def test_find_faers_cases_prefers_global_over_partition(tmp_path: Path) -> None:
    partition_dir = tmp_path / "quarter=2024Q3"
    partition_dir.mkdir()
    partition = partition_dir / "cases.parquet"
    pl.DataFrame({"drugname": ["PartDrug"], "indication": ["pain"]}).write_parquet(partition)
    global_cases = tmp_path / "cases.parquet"
    pl.DataFrame({"drugname": ["GlobalDrug"], "indication": ["pain"]}).write_parquet(global_cases)

    frame = find_faers_cases([_ref(partition), _ref(global_cases)])
    assert frame is not None
    assert frame["drugname"].to_list() == ["GlobalDrug"]  # global (no 'quarter=') wins


def test_find_faers_cases_uses_partition_when_no_global(tmp_path: Path) -> None:
    partition_dir = tmp_path / "quarter=2024Q3"
    partition_dir.mkdir()
    partition = partition_dir / "cases.parquet"
    pl.DataFrame({"drugname": ["PartDrug"], "indication": ["pain"]}).write_parquet(partition)
    frame = find_faers_cases([_ref(partition)])
    assert frame is not None
    assert frame["drugname"].to_list() == ["PartDrug"]


def test_find_faers_cases_falls_back_to_tsv(tmp_path: Path) -> None:
    tsv = tmp_path / "faers_cases.tsv"
    pl.DataFrame({"drugname": ["TsvDrug"], "indication": ["pain"]}).write_csv(tsv, separator="\t")
    frame = find_faers_cases([_ref(tsv)])
    assert frame is not None
    assert frame["drugname"].to_list() == ["TsvDrug"]


# --- build_dailymed_evidence: per-row skip branches (191, 203, 214) -------------


def test_build_dailymed_evidence_skips_approval_rows_missing_keys(tmp_path: Path) -> None:
    approvals = _parquet(
        tmp_path,
        "spl_approvals.parquet",
        {
            "approval_id": ["", "012345", "022222"],  # first -> empty norm -> skipped (191)
            "spl_set_id": ["SET-Z", "SET-A", ""],  # third -> empty set_id -> skipped (191)
        },
    )
    ev = build_dailymed_evidence([approvals])
    assert ev.approval_sets == {"12345": {"SET-A"}}  # only the complete row survives


def test_build_dailymed_evidence_skips_ingredient_rows_missing_or_duplicate(tmp_path: Path) -> None:
    ingredients = _parquet(
        tmp_path,
        "spl_ingredients.parquet",
        {
            "role": ["active", "active", "active", "active"],
            "spl_set_id": ["", "SET-A", "SET-A", "SET-B"],  # first -> empty set_id -> skipped (203)
            "ingredient_name": ["NoSet", "DrugA", "", "DrugB"],  # third -> empty name -> skipped (203)
            "ingredient_unii": ["U0", "UA", "U1", "UB"],
        },
    )
    # A second active row for SET-A would be skipped (set_id already present) -> add it.
    dup = _parquet(
        tmp_path,
        "spl_ingredients2.parquet",  # different filename; not read (only spl_ingredients.parquet)
        {"role": ["active"], "spl_set_id": ["SET-A"], "ingredient_name": ["Dup"], "ingredient_unii": ["UD"]},
    )
    ev = build_dailymed_evidence([ingredients, dup])
    assert ev.set_ingredient == {"SET-A": ("DrugA", "UA"), "SET-B": ("DrugB", "UB")}


def test_build_dailymed_evidence_ingredient_first_wins_per_set(tmp_path: Path) -> None:
    ingredients = _parquet(
        tmp_path,
        "spl_ingredients.parquet",
        {
            "role": ["active", "active"],
            "spl_set_id": ["SET-A", "SET-A"],  # duplicate set -> second skipped (203 'in set_ingredient')
            "ingredient_name": ["First", "Second"],
            "ingredient_unii": ["U1", "U2"],
        },
    )
    ev = build_dailymed_evidence([ingredients])
    assert ev.set_ingredient == {"SET-A": ("First", "U1")}


def test_build_dailymed_evidence_skips_section_rows_missing_set_id(tmp_path: Path) -> None:
    sections = _parquet(
        tmp_path,
        "spl_sections.parquet",
        {
            "spl_set_id": ["", "SET-A"],  # first -> empty set_id -> skipped (214)
            "spl_document_id": ["d0", "d1"],
            "clean_text": ["asthma", "asthma"],
            "loinc_code": ["34070-3", "34070-3"],
        },
    )
    ev = build_dailymed_evidence([sections])
    assert list(ev.contraindication_docs) == ["SET-A"]  # only the row with a set_id


def test_build_dailymed_evidence_section_text_falls_back_to_raw_text(tmp_path: Path) -> None:
    sections = _parquet(
        tmp_path,
        "spl_sections.parquet",
        {
            "spl_set_id": ["SET-A"],
            "spl_document_id": ["d1"],
            "raw_text": ["asthma from raw"],  # no clean_text -> falls back to raw_text
            "loinc_code": ["34067-9"],
        },
    )
    ev = build_dailymed_evidence([sections])
    assert ev.indication_docs["SET-A"] == [("d1", "asthma from raw")]


# --- build_drugsfda_ingredient_map: skip on missing NDA/ingredient (239->236) ---


def test_drugsfda_ingredient_map_skips_rows_missing_nda_or_ingredient(tmp_path: Path) -> None:
    products = _parquet(
        tmp_path,
        "products.parquet",
        {
            "appl_no_stripped": ["", "12345", "22222"],  # first -> empty norm -> skipped (239->236)
            "active_ingredient": ["ING", "EXAMPLESTATIN", ""],  # third -> empty ingredient -> skipped
        },
    )
    mapping = build_drugsfda_ingredient_map([products])
    assert mapping == {"12345": {"EXAMPLESTATIN"}}
