"""Unit tests for the FAERS ASCII extractor (Milestone 3).

Covers: per-family parsing ($-delimiter, trailing $, CRLF, header lowercasing, primaryid/isr
legacy), the per-quarter case join, DELETE filtering, cross-quarter caseid dedup
(most-recent-wins), NDA normalization, role/ingredient/effects/source preservation, the
uncompressed faers_cases.tsv contract, BLAKE3 determinism, manifest provenance, and zip input.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import polars as pl

from dakp_pipeline.extract import faers_ascii
from dakp_pipeline.extract.faers_ascii import _FaersSource, _parse_source, _Warnings
from dakp_pipeline.io import schemas
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloads import infer_media_type
from dakp_pipeline.io.manifests import read_manifest

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_FAERS_DIR = _FIXTURE_ROOT / "faers"


def _refs() -> list[ArtifactRef]:
    return [ArtifactRef(uri=p, blake3=hash_file(p), media_type=infer_media_type(p)) for p in sorted(_FAERS_DIR.glob("*.txt"))]


def _ctx(wd: Path, **params: Any) -> TaskContext:
    merged: dict[str, Any] = {"quarter_limit": None}
    merged.update(params)
    return TaskContext(workdir=wd, fixture_root=_FIXTURE_ROOT, params=merged)


def _extract(wd: Path, refs: list[ArtifactRef] | None = None) -> list[ArtifactRef]:
    return faers_ascii.extract(refs if refs is not None else _refs(), _ctx(wd))


def _cases(wd: Path) -> pl.DataFrame:
    _extract(wd)
    return pl.read_parquet(wd / "interim" / "faers" / "cases.parquet")


# --- parsing primitives ---------------------------------------------------------


def test_parse_lowercases_headers_and_drops_trailing_empty_column() -> None:
    src = _FaersSource("24Q3", "DEMO", b"PRIMARYID$CASEID$\r\n1001$5001$\r\n", "DEMO24Q3.txt", "b3:abcdef012345")
    frame = _parse_source(src, _Warnings())
    assert frame is not None
    assert frame.columns[:3] == ["quarter", "source_file", "source_record_id"]
    assert "primaryid" in frame.columns
    assert "" not in frame.columns  # no trailing empty col
    assert frame["primaryid"].to_list() == ["1001"]


def test_parse_resolves_legacy_isr_column_as_primaryid() -> None:
    # Pre-2014 FAERS used ISR instead of PRIMARYID.
    src = _FaersSource("12Q1", "DEMO", b"ISR$CASEID$\r\n9001$6001$\r\n", "DEMO12Q1.txt", "b3:abcdef012345")
    frame = _parse_source(src, _Warnings())
    assert frame is not None
    assert "primaryid" in frame.columns
    assert "isr" not in frame.columns
    assert frame["primaryid"].to_list() == ["9001"]


def test_parse_handles_crlf_and_trailing_dollar() -> None:
    # Real FAERS lines end with `$\r\n`; the fixture files are written that way.
    frame = _parse_source(_FaersSource("24Q3", "DRUG", (_FAERS_DIR / "DRUG24Q3.txt").read_bytes(), "DRUG24Q3.txt", "b3:abcdef012345"), _Warnings())
    assert frame is not None
    assert frame.height == 3
    assert frame["nda_num"].to_list() == ["012345", "017977", "099999"]  # leading zeroes preserved (Utf8)


def test_parse_missing_primaryid_records_warning() -> None:
    warnings = _Warnings()
    frame = _parse_source(_FaersSource("24Q3", "DEMO", b"CASEID$\r\n5001$\r\n", "DEMO24Q3.txt", "b3:abcdef012345"), warnings)
    assert frame is None
    assert any(w.code == "missing_primaryid" for w in warnings.items)


# --- normalized per-quarter tables ---------------------------------------------


def test_extract_writes_partitioned_normalized_parquets(tmp_path: Path) -> None:
    _extract(tmp_path)
    base = tmp_path / "interim" / "faers"
    for quarter, families in [
        ("24Q3", {"demo", "drug", "indi", "reac", "rpsr", "delete", "cases"}),
        ("24Q2", {"demo", "drug", "indi", "reac", "rpsr", "cases"}),
    ]:
        for fam in families:
            assert (base / f"quarter={quarter}" / f"{fam}.parquet").exists(), fam
    # Provenance columns present on a normalized table.
    drug = pl.read_parquet(base / "quarter=24Q3" / "drug.parquet")
    assert drug.columns[:3] == ["quarter", "source_file", "source_record_id"]
    assert drug["source_record_id"].n_unique() == drug.height  # one stable id per row


# --- case join semantics --------------------------------------------------------


def test_case_join_emits_expected_rows_and_columns(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    assert cases.columns == faers_ascii._CASE_COLUMNS
    # 1003 deleted; 2001 deduped -> 3 surviving cases: 1001, 1002, 2002.
    assert cases.height == 3
    assert set(cases["primaryid"].to_list()) == {"1001", "1002", "2002"}


def test_case_join_preserves_quarter_and_caseid(tmp_path: Path) -> None:
    cases = _cases(tmp_path).filter(pl.col("primaryid") == "1001")
    row = cases.row(0, named=True)
    assert row["quarter"] == "24Q3"
    assert row["caseid"] == "5001"


def test_nda_normalization_strips_leading_zeroes_and_keeps_raw(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    by_pid = {r["primaryid"]: r for r in cases.iter_rows(named=True)}
    assert by_pid["1001"]["nda"] == "12345"
    assert by_pid["1001"]["nda_raw"] == "012345"
    assert by_pid["2002"]["nda"] == "20000"
    assert by_pid["2002"]["nda_raw"] == "020000"


def test_preserves_role_cod_drugname_and_prod_ai_ingredient(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    by_pid = {r["primaryid"]: r for r in cases.iter_rows(named=True)}
    assert by_pid["1001"]["role_cod"] == "PS"
    # drugname (proprietary) differs from ingredient (prod_ai) for the Advil case.
    assert by_pid["1002"]["drugname"] == "Advil"
    assert by_pid["1002"]["ingredient"] == "Ibuprofen"


def test_effects_are_sorted_unique_dollar_joined(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    by_pid = {r["primaryid"]: r for r in cases.iter_rows(named=True)}
    # 1001 had myalgia + rhabdomyolysis -> sorted, "$"-joined.
    assert by_pid["1001"]["effects"] == "myalgia$rhabdomyolysis"
    assert by_pid["1002"]["effects"] == "nausea"
    assert by_pid["2002"]["effects"] == "dyspepsia"


def test_source_strips_trailing_whitespace_from_rpsr(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    by_pid = {r["primaryid"]: r for r in cases.iter_rows(named=True)}
    assert by_pid["1001"]["source"] == "PERIODIC"  # fixture value had a trailing space


# --- DELETE filtering ----------------------------------------------------------


def test_delete_filtering_drops_deleted_primaryid(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    assert "1003" not in cases["primaryid"].to_list()  # deleted in DELETE24Q3


def test_delete_audit_records_deleted_primaryid(tmp_path: Path) -> None:
    _extract(tmp_path)
    audit = pl.read_parquet(tmp_path / "interim" / "faers" / "delete_audit.parquet")
    assert audit.columns == ["quarter", "primaryid", "caseid", "source_file", "source_record_id"]
    assert "1003" in audit["primaryid"].to_list()


# --- cross-quarter caseid dedup -----------------------------------------------


def test_caseid_dedup_most_recent_wins(tmp_path: Path) -> None:
    # caseid 5001 appears in 24Q3 (primaryid 1001) and 24Q2 (primaryid 2001); 24Q3 wins.
    cases = _cases(tmp_path)
    assert "2001" not in cases["primaryid"].to_list()  # superseded
    assert "1001" in cases["primaryid"].to_list()  # survivor


def test_dedup_audit_records_superseded_case(tmp_path: Path) -> None:
    _extract(tmp_path)
    audit = pl.read_parquet(tmp_path / "interim" / "faers" / "dedup_audit.parquet")
    assert audit.columns == ["quarter", "primaryid", "caseid", "dedup_key", "winning_quarter", "source_file"]
    row = audit.filter(pl.col("primaryid") == "2001").row(0, named=True)
    assert row["dedup_key"] == "5001"
    assert row["winning_quarter"] == "24Q3"
    assert row["quarter"] == "24Q2"


def test_single_quarter_has_no_dedup(tmp_path: Path) -> None:
    refs = [r for r in _refs() if "24Q3" in r.uri.name]
    _extract(tmp_path, refs)
    audit = pl.read_parquet(tmp_path / "interim" / "faers" / "dedup_audit.parquet")
    assert audit.is_empty()


# --- public TSV contract -------------------------------------------------------


def test_faers_cases_tsv_is_uncompressed_with_contract_columns(tmp_path: Path) -> None:
    _extract(tmp_path)
    tsv = tmp_path / "interim" / "faers" / "faers_cases.tsv"
    assert tsv.exists()
    header = tsv.read_text(encoding="utf-8").splitlines()[0]
    assert header.split("\t") == schemas.FAERS_CASES_COLUMNS
    # Uncompressed: no gzip magic bytes.
    assert tsv.read_bytes()[:2] != b"\x1f\x8b"


# --- return contract / downstream compatibility -------------------------------


def test_cases_parquet_returned_first_and_readable(tmp_path: Path) -> None:
    refs = _extract(tmp_path)
    first = refs[0]
    assert first.uri.suffix == ".parquet"
    assert "faers" in str(first.uri)
    cases = schemas.read_table(first.uri)
    # Downstream observed-uses shaper reads these two columns.
    assert {"drugname", "indication"} <= set(cases.columns)


# --- BLAKE3 determinism --------------------------------------------------------


def test_extract_is_blake3_deterministic(tmp_path: Path) -> None:
    wd_a = tmp_path / "a"
    wd_b = tmp_path / "b"
    _extract(wd_a)
    _extract(wd_b)
    a = hash_file(wd_a / "interim" / "faers" / "cases.parquet")
    b = hash_file(wd_b / "interim" / "faers" / "cases.parquet")
    assert a == b
    assert a.startswith("b3:")


# --- manifest provenance -------------------------------------------------------


def test_manifest_records_b3_rows_and_schema_fingerprint(tmp_path: Path) -> None:
    refs = _extract(tmp_path)
    cases_ref = refs[0]
    assert cases_ref.manifest is not None
    manifest = read_manifest(cases_ref.manifest)
    assert manifest.artifact_id.startswith("b3:")
    assert manifest.artifact_id == cases_ref.blake3
    assert manifest.table.rows == 3
    assert manifest.table.schema_fingerprint == schemas.schema_fingerprint(faers_ascii._CASE_COLUMNS)
    assert manifest.operation is not None
    assert manifest.operation.name == "extract_faers_cases"
    # Inputs (source artifact ids) recorded for provenance traceability.
    assert manifest.inputs
    assert all(i.startswith("b3:") for i in manifest.inputs)


# --- zip input -----------------------------------------------------------------


def test_extract_reads_zip_members(tmp_path: Path) -> None:
    zip_path = tmp_path / "faers_ascii_24Q3.zip"
    members = {
        "ascii/DEMO24Q3.txt": (_FAERS_DIR / "DEMO24Q3.txt").read_bytes(),
        "ascii/DRUG24Q3.txt": (_FAERS_DIR / "DRUG24Q3.txt").read_bytes(),
        "ascii/INDI24Q3.txt": (_FAERS_DIR / "INDI24Q3.txt").read_bytes(),
        "ascii/REAC24Q3.txt": (_FAERS_DIR / "REAC24Q3.txt").read_bytes(),
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    ref = ArtifactRef(uri=zip_path, blake3=hash_file(zip_path), media_type="application/zip")
    out = faers_ascii.extract([ref], _ctx(tmp_path / "work"))
    cases_ref = out[0]
    assert cases_ref.uri.suffix == ".parquet"
    cases = schemas.read_table(cases_ref.uri)
    assert cases.height >= 2  # DEMO/DRUG/INDI/REAC for 24Q3 (no DELETE -> no deletion)


# --- empty / robustness --------------------------------------------------------


def test_empty_input_returns_empty_cases(tmp_path: Path) -> None:
    out = faers_ascii.extract([], _ctx(tmp_path))
    assert out == []
    # No crash; cases.parquet not written when nothing parsed.
    assert not (tmp_path / "interim" / "faers" / "cases.parquet").exists()


def test_intraquarter_duplicate_indi_rows_deduped(tmp_path: Path) -> None:
    # Duplicate INDI row for the same (primaryid, drug_seq, pt) must collapse to one case row.
    indi = tmp_path / "INDI24Q3.txt"
    indi.write_bytes(b"PRIMARYID$INDI_DRUG_SEQ$INDI_PT$\r\n1001$1$pain$\r\n1001$1$pain$\r\n")
    refs = [
        _ref(_FAERS_DIR / "DEMO24Q3.txt"),
        _ref(_FAERS_DIR / "DRUG24Q3.txt"),
        ArtifactRef(uri=indi, blake3=hash_file(indi), media_type=infer_media_type(indi)),
    ]
    out = faers_ascii.extract(refs, _ctx(tmp_path / "work"))
    cases = schemas.read_table(out[0].uri)
    dup = cases.filter((pl.col("primaryid") == "1001") & (pl.col("indication") == "pain"))
    # Only DEMO/DRUG/INDI here -> 1003 not deleted (no DELETE); dedup collapses the dup INDI.
    assert dup.height == 1


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type=infer_media_type(path))
