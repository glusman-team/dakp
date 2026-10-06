"""Tests for the Canada Vigilance extractor (``dakp_pipeline.extract.canada_vigilance``).

The published extract is a series of ``$``-delimited, quote-enclosed ASCII files with NO
header row; the column ORDER is the contract. These tests pin: the three-member join
(indications x suspect roles x product ingredients), the Suspect-only rule (a concomitant
drug's indication is not an observed use), the brand-name subject fallback for products the
ingredients member does not cover, the missing-member warning path, and the extractor's
artifact contract (typed tables, fingerprints, operation provenance, byte determinism).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.extract import canada_vigilance as cv
from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"

# Rows match the published column order (REPORT_DRUG_ID, REPORT_ID, DRUG_PRODUCT_ID, DRUGNAME,
# INDICATION_NAME_ENG, INDICATION_NAME_FR); the French column is intentionally unused.
_INDICATIONS = [
    ("1", "100001", "10", "KEMPLAR", "Rheumatoid arthritis", "Polyarthrite rhumatoide"),
    ("2", "100001", "11", "CONCOMITANTIN", "Rheumatoid arthritis", "Polyarthrite rhumatoide"),
    ("3", "100002", "99", "GHOSTDRUG", "Migraine", "Migraine"),
]
# (REPORT_DRUG_ID, REPORT_ID, DRUG_PRODUCT_ID, DRUGNAME, DRUGINVOLV_ENG, ...)
_ROLES = [
    ("1", "100001", "10", "KEMPLAR", "Suspect"),
    ("2", "100001", "11", "CONCOMITANTIN", "Concomitant"),
    ("3", "100002", "99", "GHOSTDRUG", "Suspect"),
]
# (DRUG_PRODUCT_INGREDIENT_ID, DRUG_PRODUCT_ID, DRUGNAME, ACTIVE_INGREDIENT_ID, ACTIVE_INGREDIENT_NAME)
_INGREDIENTS = [("100", "10", "KEMPLAR", "500", "methotrexate"), ("101", "10", "KEMPLAR", "501", "folic acid")]


def _zip_with(tmp_path: Path, *, with_ingredients: bool = True, with_roles: bool = True) -> Path:
    """Build a minimal (headerless, ``$``-delimited, quoted) extract ZIP like the live one."""
    lines = {
        "report_drug_indication.txt": ["$".join(f'"{field}"' for field in row) for row in _INDICATIONS],
        "report_drug.txt": ["$".join(f'"{field}"' for field in row) for row in _ROLES],
        "drug_product_ingredients.txt": ["$".join(f'"{field}"' for field in row) for row in _INGREDIENTS]
        if with_ingredients
        # A valid member that matches NO report product: every subject falls back to the brand.
        else ["$".join(f'"{field}"' for field in ("900", "77", "UNRELATED", "700", "theophylline"))],
    }
    if not with_roles:
        del lines["report_drug.txt"]  # the member is ABSENT, not empty: that is the warning path
    path = tmp_path / "extract_extrait.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in lines.items():
            archive.writestr(f"cvponline_extract_20260531/{name}", "\n".join(content) + "\n")
    return path


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/zip")


def _ctx(workdir: Path) -> TaskContext:
    return TaskContext(workdir=workdir, fixture_root=_FIXTURE_ROOT, params={})


def test_parse_extract_joins_suspect_drugs_and_ingredients(tmp_path: Path) -> None:
    rows, warnings = cv.parse_extract(_zip_with(tmp_path))
    # GHOSTDRUG (product 99) has no ingredients row: warned, kept with the brand subject.
    assert warnings == [
        {
            "file": "drug_product_ingredients.txt",
            "code": "unknown_drug_product",
            "message": "1 drug_product_id(s) absent from the ingredients member (subject falls back to the brand name)",
            "count": "1",
        }
    ]
    by_key = {(row["report_drug_id"], row["ingredient"] or row["drugname"]): row for row in rows}
    # Only the Suspect drug's ingredients fan out; the concomitant drug never appears.
    assert set(by_key) == {("1", "methotrexate"), ("1", "folic acid"), ("3", "GHOSTDRUG")}
    row = by_key[("1", "methotrexate")]
    assert row["report_id"] == "100001"
    assert row["drugname"] == "KEMPLAR"
    assert row["indication"] == "Rheumatoid arthritis"
    assert row["source_record_id"] == "cv:100001:1:methotrexate"
    assert row["source_file"] == "report_drug_indication.txt"
    assert by_key[("3", "GHOSTDRUG")]["source_record_id"] == "cv:100002:3:GHOSTDRUG"


def test_parse_extract_falls_back_to_brand_subject(tmp_path: Path) -> None:
    """An ingredients member matching NO product leaves every subject on the brand name."""
    rows, warnings = cv.parse_extract(_zip_with(tmp_path, with_ingredients=False))
    assert warnings == [
        {
            "file": "drug_product_ingredients.txt",
            "code": "unknown_drug_product",
            "message": "2 drug_product_id(s) absent from the ingredients member (subject falls back to the brand name)",
            "count": "1",
        }
    ]
    by_subject = {(row["drugname"], row["indication"]) for row in rows}
    assert by_subject == {("KEMPLAR", "Rheumatoid arthritis"), ("GHOSTDRUG", "Migraine")}
    assert all(row["ingredient"] == "" for row in rows)
    assert any(row["source_record_id"] == "cv:100002:3:GHOSTDRUG" for row in rows)


def test_parse_extract_drops_concomitant_roles(tmp_path: Path) -> None:
    rows, _warnings = cv.parse_extract(_zip_with(tmp_path))
    assert all(row["drugname"] != "CONCOMITANTIN" for row in rows)  # role = Concomitant: dropped


def test_parse_extract_missing_member_is_a_warning(tmp_path: Path) -> None:
    rows, warnings = cv.parse_extract(_zip_with(tmp_path, with_roles=False))
    assert rows == []
    assert warnings == [{"file": "", "code": "missing_member", "message": "data-extract members absent: report_drug.txt", "count": "1"}]


def test_extract_writes_the_indications_and_warnings_tables(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    refs = cv.extract([_ref(_zip_with(tmp_path))], _ctx(workdir))
    names = sorted(ref.uri.name for ref in refs)
    assert names == ["cv_indications.parquet", "cv_warnings.parquet"]
    indications = pl.read_parquet(next(ref.uri for ref in refs if ref.uri.name == "cv_indications.parquet"))
    assert indications.columns == cv.CV_INDICATIONS_COLUMNS
    assert indications.height == 3
    indications_ref = next(ref for ref in refs if ref.uri.name == "cv_indications.parquet")
    assert indications_ref.schema_fingerprint == schemas.schema_fingerprint(cv.CV_INDICATIONS_COLUMNS)
    warnings = pl.read_parquet(next(ref.uri for ref in refs if ref.uri.name == "cv_warnings.parquet"))
    assert warnings.columns == cv.CV_WARNINGS_COLUMNS
    assert warnings.height == 1  # the GHOSTDRUG fallback warning
    store = ArtifactStore(Workdir(workdir))
    manifest = store.read_manifest(indications_ref.blake3)
    assert manifest is not None
    assert manifest.operation is not None
    assert manifest.operation.name == "extract_canada_vigilance"


def test_extract_is_byte_deterministic(tmp_path: Path) -> None:
    """Same inputs => same parquet bytes => same BLAKE3, so the shaping cache keys stay stable."""
    extract_zip = _zip_with(tmp_path)
    first = cv.extract([_ref(extract_zip)], _ctx(tmp_path / "a"))
    second = cv.extract([_ref(extract_zip)], _ctx(tmp_path / "b"))
    assert [ref.blake3 for ref in first] == [ref.blake3 for ref in second]


def test_extract_requires_the_zip(tmp_path: Path) -> None:
    stray = tmp_path / "not-a-zip.parquet"
    stray.write_text("x")
    with pytest.raises(ValueError, match="no Canada Vigilance data-extract ZIP"):
        cv.extract([_ref(stray)], _ctx(tmp_path / "work"))
