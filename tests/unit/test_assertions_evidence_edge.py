"""Edge-case tests for ``dakp_pipeline.assertions.evidence`` (drive to 100% branch coverage).

Targets the uncovered lines: ``find_table`` swallowing an unreadable matching input (103-104),
``find_faers_cases`` rejecting a case frame that lacks the required columns (134), the
``build_dailymed_evidence`` skip branches for approvals/ingredients/sections rows with missing
keys (191, 203, 214), and the ``build_drugsfda_ingredient_map`` skip on missing NDA/ingredient
(239->236). Plus FAERS case-table resolution preferences (global vs partition vs TSV).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from dakp_pipeline.assertions.evidence import (
    application_type_prefix,
    build_dailymed_evidence,
    build_drugsfda_ingredient_map,
    build_fda_approval_index,
    faers_quarter_url,
    faers_quarter_urls,
    find_faers_cases,
    find_table,
    pipe_safe_text,
    sorted_pipe,
    source_manifest_url,
    source_urls,
    split_application_number,
)
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.io.manifests import ArtifactManifest, SourceBlock


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _parquet(tmp_path: Path, name: str, data: dict[str, list[object]]) -> ArtifactRef:
    path = tmp_path / name
    pl.DataFrame(data).write_parquet(path)
    return _ref(path)


# --- FAERS quarter URLs and pipe encoding ---------------------------------------


def test_faers_quarter_url_uses_exact_fda_zip_fallback() -> None:
    expected = "https://fis.fda.gov/content/Exports/faers_ascii_2024q3.zip"
    assert faers_quarter_url("24Q3") == expected
    assert faers_quarter_url("2024q3") == expected


def test_faers_quarter_url_rejects_malformed_quarter() -> None:
    import pytest

    with pytest.raises(ValueError, match="invalid FAERS quarter"):
        faers_quarter_url("2024Q5")


def test_faers_quarter_urls_use_manifest_then_fallback(tmp_path: Path) -> None:
    loose = tmp_path / "faers_ascii_24Q3.zip"
    loose.write_bytes(b"fixture")
    fallback = _ref(loose)
    assert faers_quarter_urls([fallback]) == {"24Q3": "https://fis.fda.gov/content/Exports/faers_ascii_2024q3.zip"}

    manifest_path = tmp_path / "manifest.json"
    ArtifactManifest(
        artifact_id=fallback.blake3,
        path=str(loose),
        media_type="application/zip",
        source=SourceBlock(url="https://example.test/custom-faers-2024q3.zip"),
    ).write(manifest_path)
    exact = ArtifactRef(uri=loose, blake3=fallback.blake3, media_type="application/zip", manifest=manifest_path)
    assert faers_quarter_urls([exact]) == {"24Q3": "https://example.test/custom-faers-2024q3.zip"}
    assert source_manifest_url(exact) == "https://example.test/custom-faers-2024q3.zip"
    assert source_urls([exact]) == ["https://example.test/custom-faers-2024q3.zip"]
    corrupt_manifest = tmp_path / "corrupt-manifest.json"
    corrupt_manifest.write_text("not json", encoding="utf-8")
    corrupt = ArtifactRef(uri=loose, blake3=fallback.blake3, media_type="application/zip", manifest=corrupt_manifest)
    assert source_manifest_url(corrupt) == ""
    assert source_urls([]) == []


def test_sorted_pipe_rejects_delimiter_unsafe_values() -> None:
    import pytest

    with pytest.raises(ValueError, match="delimiter"):
        sorted_pipe(["faers:24Q3:1002|1"])


def test_pipe_safe_text_collapses_delimiter_runs() -> None:
    """Free-form label prose legitimately contains ``|`` bullets and line breaks — unlike
    identifier provenance, it is sanitized (not rejected) before pipe-encoded TSV output."""
    assert pipe_safe_text("Do not use|When using this product\nstop use") == "Do not use When using this product stop use"
    assert pipe_safe_text("  a\t\r\n||b  ") == "a b"
    assert pipe_safe_text(None) == ""
    assert pipe_safe_text(42) == "42"


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


# --- find_faers_cases: optional column projection ------------------------------


def test_find_faers_cases_projection_parquet(tmp_path: Path) -> None:
    cases = _parquet(tmp_path, "cases.parquet", {"drugname": ["DrugA"], "indication": ["pain"], "primaryid": ["1"], "extra": ["x"]})
    frame = find_faers_cases([cases], columns=("drugname", "indication", "primaryid"))
    assert frame is not None
    assert frame.columns == ["drugname", "indication", "primaryid"]  # extra column skipped
    assert frame["primaryid"].to_list() == ["1"]


def test_find_faers_cases_projection_tsv(tmp_path: Path) -> None:
    tsv = tmp_path / "faers_cases.tsv"
    pl.DataFrame({"drugname": ["TsvDrug"], "indication": ["pain"], "primaryid": ["9"]}).write_csv(tsv, separator="\t")
    frame = find_faers_cases([_ref(tsv)], columns=("drugname", "indication", "primaryid"))
    assert frame is not None
    assert frame.columns == ["drugname", "indication", "primaryid"]
    assert frame["drugname"].to_list() == ["TsvDrug"]


def test_find_faers_cases_projection_skips_absent_columns(tmp_path: Path) -> None:
    # No primaryid column in the table -> projection silently drops it (row-count fallback).
    cases = _parquet(tmp_path, "cases.parquet", {"drugname": ["DrugA"], "indication": ["pain"]})
    frame = find_faers_cases([cases], columns=("drugname", "indication", "primaryid"))
    assert frame is not None
    assert frame.columns == ["drugname", "indication"]


def test_find_faers_cases_projection_rejects_missing_required(tmp_path: Path) -> None:
    # None of the requested columns exist -> empty frame -> required-column check -> None.
    cases = _parquet(tmp_path, "cases.parquet", {"unrelated": ["x"]})
    assert find_faers_cases([cases], columns=("drugname", "indication", "primaryid")) is None


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


# --- DailyMed SPL-set -> approval reverse index -------------------------------


def test_dailymed_evidence_reverse_approval_index_deduplicates_and_skips_missing() -> None:
    approvals = pl.DataFrame(
        {
            "approval_id": ["012345", "012345", "", ""],
            "approval_code": ["", "", "099998", ""],
            "approval_type": ["NDA", "NDA", "BLA", "NDA"],
            "spl_set_id": ["SET-A", "SET-A", "SET-A", ""],
        }
    )
    # Keep the fixture isolated and avoid relying on a real acquisition artifact.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "spl_approvals.parquet"
        approvals.write_parquet(path)
        evidence = build_dailymed_evidence([_ref(path)])

    assert evidence.approval_ids_by_set == {"SET-A": {"NDA012345", "BLA099998"}}
    assert evidence.approval_ids_for_sets(["SET-A", "MISSING", "SET-A"]) == ["BLA099998", "NDA012345"]


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


# --- FDA application-number expansion (FDAApprovalIndex) ------------------------


def test_split_application_number_and_type_prefix() -> None:
    # The three shapes the sources actually write: SPL's prefixed extension, Drugs@FDA's spaced
    # lowercase prefix, and FAERS's bare (prefix- and zero-stripped) number.
    assert split_application_number("BLA125514") == ("BLA", "125514")
    assert split_application_number("bla 0042") == ("BLA", "0042")
    assert split_application_number("125514") == ("", "125514")
    assert split_application_number("no digits here") == ("", "")
    assert split_application_number(None) == ("", "")
    # Leading zeros are display, not join key: NDA017977 is the FDA form, "17977" the join key.
    assert split_application_number("NDA017977") == ("NDA", "017977")
    # The SPL <code> element carries an NCI Thesaurus code; a curated column carries the prefix.
    assert application_type_prefix("C73585") == "BLA"
    assert application_type_prefix("C73605") == "NDA"  # NDA authorized generic
    assert application_type_prefix("nda") == "NDA"
    assert application_type_prefix("") == ""
    assert application_type_prefix(None) == ""


def test_fda_approval_index_expands_faers_numbers_from_drugsfda(tmp_path: Path) -> None:
    # The truncated-approval bug: FAERS reports pembrolizumab's application as "125514", which is
    # only an identifier once the type prefix is restored -> "BLA125514". Drugs@FDA is the
    # register that knows the type; DailyMed carries no SPL approval for this one.
    applications = _parquet(
        tmp_path,
        "applications.parquet",
        {
            "appl_no": ["125514", "017977"],
            "appl_no_raw": ["BLA125514", "NDA017977"],
            "appl_no_stripped": ["125514", "17977"],
            "appl_type": ["BLA", "NDA"],
        },
    )
    index = build_fda_approval_index([applications])

    assert index.expand("125514") == ["BLA125514"]
    assert index.expand("0125514") == ["BLA125514"]  # any FAERS spelling normalizes to one entry
    assert index.expand("17977") == ["NDA017977"]  # padding restored from the register
    assert index.expand_all(["17977", "125514", "125514"]) == ["BLA125514", "NDA017977"]
    # Numbers in no register keep their own recorded form rather than being dropped.
    assert index.expand("099999") == ["099999"]
    assert index.expand("ANDA065432") == ["ANDA065432"]
    assert index.expand("") == []
    assert index.expand("not a number") == []


def test_fda_approval_index_prefers_drugsfda_over_dailymed_and_keeps_ambiguous_spl_prefixes(tmp_path: Path) -> None:
    # DailyMed embeds the prefix in the approval id itself and the <code> element holds an NCI
    # code, never a prefix: concatenating the two produced "C73584ANDA089160".
    approvals = _parquet(
        tmp_path,
        "spl_approvals.parquet",
        {
            "approval_id": ["ANDA089160", "NDA012345", "ANDA012345", "022329"],
            "approval_code": ["ANDA089160", "NDA012345", "ANDA012345", "022329"],
            "approval_type": ["C73584", "C73594", "C73584", "C73594"],
            "spl_set_id": ["SET-A", "SET-B", "SET-C", "SET-D"],
        },
    )
    products = _parquet(
        tmp_path, "products.parquet", {"appl_no": ["089160"], "appl_no_raw": ["ANDA089160"], "appl_no_stripped": ["89160"], "appl_type": ["ANDA"]}
    )
    index = build_fda_approval_index([approvals, products])

    assert index.expand("89160") == ["ANDA089160"]  # Drugs@FDA wins; never the NCI code
    # One number, two label prefixes and no Drugs@FDA entry: legacy emitted every prefix it saw
    # (Biolink's own example is ranitidine's ANADA200536/ANDA200536), and so does this.
    assert index.expand("12345") == ["ANDA012345", "NDA012345"]
    # An id with no prefix of its own falls back to the NCI code's application type.
    assert index.expand("22329") == ["NDA022329"]


def test_dailymed_display_never_concatenates_the_nci_application_code(tmp_path: Path) -> None:
    """Real SPL rows: the id already carries the prefix, ``approval_type`` is an NCI code.

    Concatenating the two produced ``C73584ANDA089160`` on every production build; the fixtures
    hid it by writing a friendly ``"NDA"`` into the type column.
    """
    approvals = _parquet(
        tmp_path,
        "spl_approvals.parquet",
        {
            "approval_id": ["ANDA089160", "BLA103795", "022329"],
            "approval_code": ["ANDA089160", "BLA103795", "022329"],
            "approval_type": ["C73584", "C73585", "C73594"],
            "spl_set_id": ["SET-A", "SET-B", "SET-C"],
        },
    )
    evidence = build_dailymed_evidence([approvals])

    assert evidence.approval_display["89160"] == "ANDA089160"
    assert evidence.approval_display["103795"] == "BLA103795"
    # No prefix on the id: the NCI code supplies it, and the padding is kept.
    assert evidence.approval_display["22329"] == "NDA022329"


def test_fda_approval_index_dedupes_repeated_labels_and_skips_unusable_rows(tmp_path: Path) -> None:
    # An application number appears on every label that bears it, so the same display form
    # arrives many times; and a row with no number, or with a number no source types, contributes
    # nothing rather than a prefix-less entry.
    approvals = _parquet(
        tmp_path,
        "spl_approvals.parquet",
        {
            "approval_id": ["NDA012345", "NDA012345", "022329", "", "no digits"],
            "approval_code": ["NDA012345", "NDA012345", "022329", "", "no digits"],
            "approval_type": ["C73594", "C73594", "", "C73594", "C73594"],
            "spl_set_id": ["SET-A", "SET-B", "SET-C", "SET-D", "SET-E"],
        },
    )
    index = build_fda_approval_index([approvals])

    assert index.displays["12345"] == ("NDA012345",)  # the repeat adds nothing
    assert "22329" not in index.displays  # no prefix anywhere: not an expansion, left alone
    assert index.expand("022329") == ["022329"]
    assert len(index.displays) == 1
