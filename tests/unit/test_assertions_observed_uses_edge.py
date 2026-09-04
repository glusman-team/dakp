"""Edge-case tests for ``dakp_pipeline.assertions.observed_uses`` (drive to 100% branch coverage).

Targets the uncovered skip branch (a FAERS row with a missing drugname or indication, line 54)
plus adversarial case-counting: empty-string primaryid falling back to a per-row id, an empty
frame, and object resolution falling back to the raw indication text.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.assertions.observed_uses import build_observed_use_rows


def test_rows_with_missing_drug_or_indication_are_skipped(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame(
        {
            "primaryid": ["1", "2", "3", "4"],
            "drugname": ["DrugX", "", "   ", "DrugX"],  # rows 1,2 -> blank drug -> skipped (54)
            "indication": ["hypercholesterolemia", "pain", "pain", ""],  # row 3 -> blank indication -> skipped
        }
    )
    rows = build_observed_use_rows(cases, disease_map)
    # Only the complete row (DrugX, hypercholesterolemia) survives.
    assert [(r["subject_text"], r["object_text"]) for r in rows] == [("DrugX", "hypercholesterolemia")]


def test_empty_string_primaryid_falls_back_to_row_count(disease_map: dict[str, dict[str, str]]) -> None:
    # primaryid column present but blank -> treated as no case id -> each row is its own count.
    cases = pl.DataFrame({"primaryid": ["", ""], "drugname": ["DrugX", "DrugX"], "indication": ["pain", "pain"]})
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["number_of_cases"] == "2"


def test_empty_frame_yields_no_rows(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame({"primaryid": [], "drugname": [], "indication": []})
    assert build_observed_use_rows(cases, disease_map) == []


def test_unknown_indication_falls_back_to_raw_text(disease_map: dict[str, dict[str, str]]) -> None:
    cases = pl.DataFrame({"primaryid": ["1"], "drugname": ["DrugX"], "indication": ["zzz_unknown"]})
    rows = build_observed_use_rows(cases, disease_map)
    assert len(rows) == 1
    assert rows[0]["object_text"] == "zzz_unknown"
    assert rows[0]["object_curie"] == ""  # no baseline match -> text-first, no CURIE
    assert rows[0]["object_category"] == "Disease"
