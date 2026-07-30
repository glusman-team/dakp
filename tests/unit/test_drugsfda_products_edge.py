"""Edge-case tests for the Drugs@FDA products extractor (100% branch coverage drive).

Covers:

* application-number normalization corners (``_parse_combined`` / ``_normalize_appl_fields``):
  combined ``NDA012345`` vs split ``ApplType``+``ApplNo``, all-zero, empty, prefix-with-no-digits
  falling back to ``ApplNo``, and a prefix present while ``ApplType`` is already set.
* filename classification (``_table_key``) and input collection (zip dir entries, non-table
  members, loose files with no table key).
* lookup derivation edges (empty appl_no_stripped, missing drug_name/active_ingredient,
  empty ingredient split part, duplicate keys) and ``_fill_appl_type_map`` skipping empties.
* record-id fallbacks (ndc / row).
* per-table missing-application-number warnings and submission appl_type inheritance
  (already-set vs map-miss).
* the opt-in Go delegation path's missing-frame branches, driven offline via ``MockGoRunner``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.extract import drugsfda_products as dp
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloads import infer_media_type
from dakp_pipeline.paths import Workdir
from dakp_pipeline.workers import go_runner

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(wd: Path, **params: object) -> TaskContext:
    Workdir(wd).create()
    return TaskContext(profile="mock", workdir=wd, fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params=params)


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type=infer_media_type(path))


def _tsv(tmp_path: Path, name: str, header: str, *rows: str) -> ArtifactRef:
    p = tmp_path / name
    p.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return _ref(p)


# --- _parse_combined / _normalize_appl_fields ----------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ("", "")),  # empty -> ("", "")
        ("NDA012345", ("NDA", "012345")),
        ("bla 0042", ("BLA", "0042")),  # lowercase prefix + space
        ("ANDA9", ("ANDA", "9")),
        ("987", ("", "987")),  # no prefix -> digits only
        ("nonsense", ("", "")),  # no prefix, no digits
    ],
)
def test_parse_combined(value: str, expected: tuple[str, str]) -> None:
    assert dp._parse_combined(value) == expected


@pytest.mark.parametrize(
    ("raw", "atype", "ano", "expected"),
    [
        ("NDA012345", "", "", ("NDA012345", "NDA", "012345", "12345")),  # combined prefix
        ("", "NDA", "012345", ("NDA012345", "NDA", "012345", "12345")),  # split
        ("000000", "", "", ("000000", "", "000000", "000000")),  # all-zero kept as-is
        ("NDA000000", "", "", ("NDA000000", "NDA", "000000", "000000")),  # prefix + all-zero
        ("", "", "", ("", "", "", "")),  # nothing
        ("NDA123", "BLA", "", ("BLA123", "BLA", "123", "123")),  # prefix present but type already set
        ("abc", "", "789", ("789", "", "789", "789")),  # raw has no digits -> fall back to ApplNo
        ("XYZ987", "", "", ("987", "", "987", "987")),  # non-prefix -> digits only
    ],
)
def test_normalize_appl_fields(raw: str, atype: str, ano: str, expected: tuple[str, str, str, str]) -> None:
    assert dp._normalize_appl_fields(raw, atype, ano) == expected


# --- _table_key + record-id fallbacks ------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Products.txt", "products"),
        ("product", "products"),
        ("Applications.txt", "applications"),
        ("Application", "applications"),
        ("Submissions.txt", "submissions"),
        ("submission", "submissions"),
        ("SubmissionPropertyType.txt", None),  # sub-table rejected
        ("foo.tsv", None),
    ],
)
def test_table_key(name: str, expected: str | None) -> None:
    assert dp._table_key(name) == expected


def test_product_record_id_fallbacks() -> None:
    assert dp._product_record_id("NDA", "12345", "001", "0001", 2) == "drugsfda:product:NDA12345:001"
    assert dp._product_record_id("NDA", "12345", "", "0001", 2) == "drugsfda:product:NDA12345:NA"  # empty product_no
    assert dp._product_record_id("", "", "", "0001-0002", 2) == "drugsfda:product:ndc:0001-0002"  # ndc fallback
    assert dp._product_record_id("", "", "", "", 7) == "drugsfda:product:row:7"  # row fallback


def test_record_id_fallbacks() -> None:
    assert dp._record_id("application", "NDA", "12345", 2) == "drugsfda:application:NDA12345"
    assert dp._record_id("submission", "NDA", "12345", 2, "3") == "drugsfda:submission:NDA12345:3"
    assert dp._record_id("application", "", "", 9) == "drugsfda:application:row:9"  # row fallback


# --- _build_lookups edges ------------------------------------------------------


def _lookup_frame(rows: list[dict[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=dict.fromkeys(dp.LOOKUPS_SOURCE_COLUMNS, pl.Utf8))


def test_build_lookups_handles_edges_and_dedup() -> None:
    frame = _lookup_frame(
        [
            # Full row: proprietary + nonproprietary + two ingredients (empty split part skipped) + ndc + marketing.
            {
                "drug_name": "DrugX",
                "active_ingredient": "ING-A; ;ING-B",
                "product_ndc": "0001",
                "marketing_status_name": "Rx",
                "appl_no": "012345",
                "appl_no_stripped": "12345",
                "appl_type": "NDA",
            },
            # appl_no_stripped present but no drug_name / ingredient / ndc / marketing -> no candidates.
            {
                "drug_name": "",
                "active_ingredient": "",
                "product_ndc": "",
                "marketing_status_name": "",
                "appl_no": "099999",
                "appl_no_stripped": "99999",
                "appl_type": "NDA",
            },
            # Duplicate proprietary_name key (same term + same stripped) -> skipped.
            {
                "drug_name": "DrugX",
                "active_ingredient": "",
                "product_ndc": "",
                "marketing_status_name": "",
                "appl_no": "012345",
                "appl_no_stripped": "12345",
                "appl_type": "NDA",
            },
            # Empty appl_no_stripped -> row skipped entirely.
            {
                "drug_name": "NoAppl",
                "active_ingredient": "X",
                "product_ndc": "9",
                "marketing_status_name": "Rx",
                "appl_no": "",
                "appl_no_stripped": "",
                "appl_type": "",
            },
        ]
    )
    lookups = dp._build_lookups(frame)
    keys = {(r["lookup_type"], r["term"]) for r in lookups.iter_rows(named=True)}
    assert ("proprietary_name", "DrugX") in keys
    assert ("nonproprietary_name", "ING-A; ;ING-B") in keys
    assert ("ingredient", "ING-A") in keys
    assert ("ingredient", "ING-B") in keys
    assert ("ingredient", "") not in keys  # empty split part dropped
    assert ("product_ndc", "0001") in keys
    assert ("marketing_status", "Rx") in keys
    # DrugX proprietary appears exactly once (dedup), and the empty-stripped row contributed nothing.
    assert [r for r in lookups.iter_rows(named=True) if r["lookup_type"] == "proprietary_name" and r["term"] == "DrugX"] == [
        {"lookup_type": "proprietary_name", "term": "DrugX", "appl_no": "012345", "appl_no_stripped": "12345", "appl_type": "NDA"}
    ]
    assert all(r["appl_no_stripped"] for r in lookups.iter_rows(named=True))  # no empty-stripped rows


def test_fill_appl_type_map_skips_empty_fields() -> None:
    frame = pl.DataFrame(
        [
            {"appl_no_stripped": "12345", "appl_type": "NDA"},  # added
            {"appl_no_stripped": "", "appl_type": "BLA"},  # skipped (empty stripped)
            {"appl_no_stripped": "999", "appl_type": ""},  # skipped (empty type)
        ],
        schema={"appl_no_stripped": pl.Utf8, "appl_type": pl.Utf8},
    )
    mapping: dict[str, str] = {}
    dp._fill_appl_type_map(mapping, frame)
    assert mapping == {"12345": "NDA"}


# --- extract: input collection edges -------------------------------------------


def test_collect_tables_skips_loose_non_table_file(tmp_path: Path) -> None:
    products = _tsv(tmp_path, "drugsfda_products.tsv", "ApplNo\tApplType\tDrugName", "012345\tNDA\tDrugX")
    stray = _tsv(tmp_path, "foo.tsv", "a\tb", "1\t2")  # no table key -> skipped
    refs = dp.extract([products, stray], _ctx(tmp_path / "work"))
    names = {r.uri.name for r in refs}
    assert "products.parquet" in names


def test_collect_tables_reads_zip_and_skips_dirs_and_non_tables(tmp_path: Path) -> None:
    zip_path = tmp_path / "drugsfda.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("media/", b"")  # directory entry -> skipped
        zf.writestr("media/Readme.txt", b"not a table")  # no table key -> skipped
        zf.writestr("media/Products.txt", "ApplNo\tApplType\tDrugName\n012345\tNDA\tDrugX\n")
    refs = dp.extract([_ref(zip_path)], _ctx(tmp_path / "work"))
    products = pl.read_parquet(next(r for r in refs if r.uri.name == "products.parquet").uri)
    assert products["drug_name"].to_list() == ["DrugX"]


def test_extract_with_no_recognized_tables_warns(tmp_path: Path) -> None:
    stray = _tsv(tmp_path, "foo.tsv", "a\tb", "1\t2")
    refs = dp.extract([stray], _ctx(tmp_path / "work"))
    warnings_path = next(r for r in refs if r.uri.name == "extract_warnings.jsonl").uri
    lines = [json.loads(line) for line in warnings_path.read_text(encoding="utf-8").splitlines()]
    assert any(w["table"] == "*" for w in lines)
    assert any(w["table"] == "products" for w in lines)  # products-missing warning


# --- extract: per-table missing-appl warnings + submission inheritance ---------


def test_missing_application_number_warnings_per_table(tmp_path: Path) -> None:
    products = _tsv(tmp_path, "drugsfda_products.tsv", "ApplNo\tApplType\tDrugName", "012345\tNDA\tDrugX", "\t\tNoAppl")  # row 3 missing appl
    applications = _tsv(tmp_path, "drugsfda_applications.tsv", "ApplNo\tApplType\tSponsorName", "012345\tNDA\tSponsor", "\t\tNoApplSponsor")
    submissions = _tsv(tmp_path, "drugsfda_submissions.tsv", "ApplNo\tSubmissionNo", "012345\t1", "\t9")  # row 3 missing appl
    refs = dp.extract([products, applications, submissions], _ctx(tmp_path / "work"))
    warnings_path = next(r for r in refs if r.uri.name == "extract_warnings.jsonl").uri
    lines = [json.loads(line) for line in warnings_path.read_text(encoding="utf-8").splitlines()]
    tables_with_missing = {w["table"] for w in lines if "missing application number" in w.get("message", "")}
    assert tables_with_missing == {"products", "applications", "submissions"}


def test_submission_appl_type_inheritance_set_vs_map_miss(tmp_path: Path) -> None:
    # Products seeds the appl_type map: 12345 -> NDA.
    products = _tsv(tmp_path, "drugsfda_products.tsv", "ApplNo\tApplType\tDrugName", "012345\tNDA\tDrugX")
    submissions = _tsv(
        tmp_path,
        "drugsfda_submissions.tsv",
        "ApplNo\tApplType\tSubmissionNo",
        "012345\t\t1",  # no ApplType, 12345 in map -> inherits NDA
        "00020000\tBLA\t2",  # ApplType already set -> inheritance skipped
        "077777\t\t3",  # no ApplType, 77777 NOT in map -> stays empty
    )
    refs = dp.extract([products, submissions], _ctx(tmp_path / "work"))
    subs = pl.read_parquet(next(r for r in refs if r.uri.name == "submissions.parquet").uri)
    by_no = {r["appl_no_stripped"]: r for r in subs.iter_rows(named=True)}
    assert by_no["12345"]["appl_type"] == "NDA"  # inherited
    assert by_no["12345"]["appl_no_raw"] == "NDA012345"  # raw rebuilt from inherited type
    assert by_no["20000"]["appl_type"] == "BLA"  # already set, untouched
    assert by_no["77777"]["appl_type"] == ""  # map miss -> empty


# --- Go delegation path: missing-frame branches (offline via MockGoRunner) -----


def _mock_go(monkeypatch: pytest.MonkeyPatch, write_frames: dict[str, list[str]]) -> None:
    """Route the extractor's Go runner to a MockGoRunner that writes selected empty TSVs."""
    monkeypatch.setattr(go_runner, "go_available", lambda: True)
    runner = go_runner.MockGoRunner()

    def handler(args: tuple[str, ...]) -> tuple[str, str]:
        out_dir = Path(args[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, columns in write_frames.items():
            pl.DataFrame(schema=dict.fromkeys(columns, pl.Utf8)).write_csv(out_dir / f"{name}.tsv", separator="\t")
        return ("", "")

    runner.set_handler("drugsfda", handler)
    monkeypatch.setattr(go_runner, "get_runner", lambda: runner)


def test_go_path_with_all_frames_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_go(
        monkeypatch,
        {
            "drugsfda_products": dp.PRODUCTS_COLUMNS,
            "drugsfda_applications": dp.APPLICATIONS_COLUMNS,
            "drugsfda_submissions": dp.SUBMISSIONS_COLUMNS,
            "drugsfda_lookups": dp.LOOKUPS_COLUMNS,
        },
    )
    products = _tsv(tmp_path, "drugsfda_products.tsv", "ApplNo\tApplType\tDrugName", "012345\tNDA\tDrugX")
    refs = dp.extract([products], _ctx(tmp_path / "work", use_go_workers=True))
    names = [r.uri.name for r in refs]
    # products parquet + products tsv + applications + submissions + lookups + warnings jsonl.
    assert names.count("products.parquet") == 1
    assert "drugsfda_products.tsv" in names
    assert "applications.parquet" in names
    assert "submissions.parquet" in names
    assert "lookups.parquet" in names
    assert "extract_warnings.jsonl" in names


def test_go_path_with_no_frames_emits_only_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_go(monkeypatch, {})  # worker writes nothing -> every frame is None
    products = _tsv(tmp_path, "drugsfda_products.tsv", "ApplNo\tApplType\tDrugName", "012345\tNDA\tDrugX")
    refs = dp.extract([products], _ctx(tmp_path / "work", use_go_workers=True))
    assert [r.uri.name for r in refs] == ["extract_warnings.jsonl"]
