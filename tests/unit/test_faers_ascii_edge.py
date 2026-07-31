"""Edge-case tests for the FAERS ASCII extractor (100% branch coverage drive).

Targets the robustness branches the fixture-driven suite never reaches:

* ``_iter_faers_sources`` — zip directory entries, zip members without a family/quarter,
  loose ``.txt`` that is not a FAERS family, and non-``.txt``/``.zip`` artifacts (all skipped).
* ``_parse_source`` — empty content, a polars parse error (invalid UTF-8), header-only
  (no data rows), and input without a trailing ``$`` (no empty column to drop).
* ``extract`` — the loop's skip branch when a source parses to ``None``/empty; quarters with
  no DRUG/INDI (no cases); DRUG+INDI with disjoint join keys (empty join); DELETE listing a
  primaryid absent from DRUG (no rows dropped -> no warning); all-empty quarters reducing to
  an empty global table.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from dakp_pipeline.extract import faers_ascii
from dakp_pipeline.extract.faers_ascii import _FaersSource, _parse_source, _Warnings
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloads import infer_media_type

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_FAERS_DIR = _FIXTURE_ROOT / "faers"


def _ctx(wd: Path) -> TaskContext:
    return TaskContext(workdir=wd, fixture_root=_FIXTURE_ROOT, params={})


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type=infer_media_type(path))


def _write(tmp_path: Path, name: str, data: bytes) -> ArtifactRef:
    p = tmp_path / name
    p.write_bytes(data)
    return _ref(p)


# --- _parse_source robustness --------------------------------------------------


def test_parse_empty_content_records_empty_file_warning() -> None:
    warnings = _Warnings()
    assert _parse_source(_FaersSource("24Q3", "DEMO", b"   \r\n  ", "DEMO24Q3.txt", "b3:ab"), warnings) is None
    assert any(w.code == "empty_file" and w.message == "no bytes" for w in warnings.items)


def test_parse_invalid_utf8_records_parse_error() -> None:
    warnings = _Warnings()
    frame = _parse_source(_FaersSource("24Q3", "DEMO", b"PRIMARYID$\r\n\xff\xfe\xfa$\r\n", "DEMO24Q3.txt", "b3:ab"), warnings)
    assert frame is None
    assert any(w.code == "parse_error" for w in warnings.items)


def test_parse_header_only_records_no_data_rows() -> None:
    warnings = _Warnings()
    frame = _parse_source(_FaersSource("24Q3", "DEMO", b"PRIMARYID$CASEID$\r\n", "DEMO24Q3.txt", "b3:ab"), warnings)
    assert frame is None
    assert any(w.code == "empty_file" and w.message == "no data rows" for w in warnings.items)


def test_parse_without_trailing_dollar_keeps_all_columns() -> None:
    # No trailing "$" -> no empty column to drop (the empty_cols branch is skipped).
    frame = _parse_source(_FaersSource("24Q3", "DEMO", b"PRIMARYID$CASEID\n1001$5001\n", "DEMO24Q3.txt", "b3:ab"), _Warnings())
    assert frame is not None
    assert frame["primaryid"].to_list() == ["1001"]
    assert frame["caseid"].to_list() == ["5001"]


def test_warnings_frame_is_empty_schema_when_no_warnings() -> None:
    frame = _Warnings().frame()
    assert frame.is_empty()
    assert frame.columns == faers_ascii._WARNINGS_COLUMNS


# --- _iter_faers_sources skip branches -----------------------------------------


def test_iter_skips_zip_dirs_and_non_family_members(tmp_path: Path) -> None:
    zip_path = tmp_path / "faers_ascii_24Q3.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ascii/", b"")  # directory entry -> skipped
        zf.writestr("ascii/README.txt", b"not a family")  # no family/quarter -> skipped
        zf.writestr("ascii/DEMO24Q3.txt", b"PRIMARYID$\r\n1$\r\n")  # valid
    sources = list(faers_ascii._iter_faers_sources([_ref(zip_path)]))
    assert [s.family for s in sources] == ["DEMO"]


def test_extract_skips_loose_non_family_txt(tmp_path: Path) -> None:
    # A loose .txt that is not a FAERS family is skipped (and parses to nothing).
    out = faers_ascii.extract([_write(tmp_path, "notes.txt", b"hello\r\n")], _ctx(tmp_path / "work"))
    assert out == []


def test_extract_skips_non_txt_non_zip_artifacts(tmp_path: Path) -> None:
    out = faers_ascii.extract([_write(tmp_path, "data.csv", b"a,b\r\n1,2\r\n")], _ctx(tmp_path / "work"))
    assert out == []


def test_extract_skips_source_that_parses_to_none(tmp_path: Path) -> None:
    # A valid FAERS-named file with empty content parses to None -> skipped in the loop.
    out = faers_ascii.extract([_write(tmp_path, "DEMO24Q3.txt", b"")], _ctx(tmp_path / "work"))
    assert out == []


# --- case-join empty / no-match branches ---------------------------------------


def test_quarter_without_drug_or_indi_yields_no_cases(tmp_path: Path) -> None:
    # Only DEMO for the quarter -> _build_quarter_cases returns an empty frame; all quarters
    # empty -> _reduce_cases returns the empty global table + empty audit.
    refs = [_ref(_FAERS_DIR / "DEMO24Q3.txt")]
    out = faers_ascii.extract(refs, _ctx(tmp_path / "work"))
    cases = pl.read_parquet(next(r for r in out if r.uri.name == "cases.parquet").uri)
    assert cases.is_empty()
    audit = pl.read_parquet(tmp_path / "work" / "data" / "interim" / "faers" / "dedup_audit.parquet")
    assert audit.is_empty()


def test_drug_indi_disjoint_keys_yield_empty_join(tmp_path: Path) -> None:
    # DRUG and INDI present but on different primaryids -> inner join empty -> empty cases.
    drug = _write(tmp_path, "DRUG24Q3.txt", b"PRIMARYID$DRUG_SEQ$DRUGNAME$\r\n1001$1$DrugX$\r\n")
    indi = _write(tmp_path, "INDI24Q3.txt", b"PRIMARYID$INDI_DRUG_SEQ$INDI_PT$\r\n9999$1$pain$\r\n")
    out = faers_ascii.extract([drug, indi], _ctx(tmp_path / "work"))
    cases = pl.read_parquet(next(r for r in out if r.uri.name == "cases.parquet").uri)
    assert cases.is_empty()


def test_delete_listing_absent_primaryid_drops_no_rows(tmp_path: Path) -> None:
    # DELETE lists 9999 but DRUG only has 1001 -> deleted_pids non-empty yet zero drug rows
    # dropped, so no 'deleted_rows_dropped' warning is recorded.
    demo = _write(tmp_path, "DEMO24Q3.txt", b"PRIMARYID$CASEID$\r\n1001$5001$\r\n")
    drug = _write(tmp_path, "DRUG24Q3.txt", b"PRIMARYID$DRUG_SEQ$DRUGNAME$\r\n1001$1$DrugX$\r\n")
    indi = _write(tmp_path, "INDI24Q3.txt", b"PRIMARYID$INDI_DRUG_SEQ$INDI_PT$\r\n1001$1$pain$\r\n")
    delete = _write(tmp_path, "DELETE24Q3.txt", b"PRIMARYID$\r\n9999$\r\n")
    out = faers_ascii.extract([demo, drug, indi, delete], _ctx(tmp_path / "work"))
    cases = pl.read_parquet(next(r for r in out if r.uri.name == "cases.parquet").uri)
    assert cases["primaryid"].to_list() == ["1001"]  # 1001 survives (not deleted)
    warnings = pl.read_parquet(tmp_path / "work" / "data" / "interim" / "faers" / "warnings.parquet")
    assert "deleted_rows_dropped" not in warnings["code"].to_list()


def test_delete_without_primaryid_column_is_ignored(tmp_path: Path) -> None:
    # A DELETE frame lacking a primaryid column yields no deleted ids (defensive guard).
    demo = _ref(_FAERS_DIR / "DEMO24Q3.txt")
    drug = _ref(_FAERS_DIR / "DRUG24Q3.txt")
    indi = _ref(_FAERS_DIR / "INDI24Q3.txt")
    delete = _write(tmp_path, "DELETE24Q3.txt", b"CASEID$\r\n5001$\r\n")  # no PRIMARYID column
    out = faers_ascii.extract([demo, drug, indi, delete], _ctx(tmp_path / "work"))
    cases = pl.read_parquet(next(r for r in out if r.uri.name == "cases.parquet").uri)
    assert not cases.is_empty()  # nothing deleted


def test_intraquarter_dedup_keeps_first_and_sorts(tmp_path: Path) -> None:
    # Two distinct indications for the same drug row -> two case rows, deterministically sorted.
    drug = _write(tmp_path, "DRUG24Q3.txt", b"PRIMARYID$DRUG_SEQ$DRUGNAME$\r\n1001$1$DrugX$\r\n")
    indi = _write(tmp_path, "INDI24Q3.txt", b"PRIMARYID$INDI_DRUG_SEQ$INDI_PT$\r\n1001$1$zebra$\r\n1001$1$alpha$\r\n")
    out = faers_ascii.extract([drug, indi], _ctx(tmp_path / "work"))
    cases = pl.read_parquet(next(r for r in out if r.uri.name == "cases.parquet").uri)
    assert cases["indication"].to_list() == ["alpha", "zebra"]  # sorted by indication
