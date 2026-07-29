"""Tests for the Drugs@FDA product/application/submission extractor (Milestone 3).

Covers tab-delimited parsing, NDA/BLA/ANDA application-number normalization (raw + both
normalized forms), name/ingredient/ndc/marketing-status lookup tables, the uncompressed
Tablassert-facing source-section TSV, BLAKE3 determinism, ZIP-member inputs, and
submission appl_type inheritance.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import polars as pl

from dakp_pipeline.extract import drugsfda_products as ext
from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import read_manifest
from dakp_pipeline.paths import Workdir

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_DRUGSFDA_FIXTURE_DIR = _FIXTURE_ROOT / "drugsfda"
_ALL_FIXTURES = ("drugsfda_products.tsv", "drugsfda_applications.tsv", "drugsfda_submissions.tsv")


def _ctx(workdir: Path) -> TaskContext:
    return TaskContext(profile="mock", workdir=workdir, fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params={})


def _ingest(workdir: Path, names: tuple[str, ...]) -> list[ArtifactRef]:
    store = ArtifactStore(Workdir(workdir))
    refs: list[ArtifactRef] = []
    for name in names:
        ref, _ = store.ingest(_DRUGSFDA_FIXTURE_DIR / name, alias=f"drugsfda/{name}")
        refs.append(ref)
    return refs


def _extract(tmp_path: Path, names: tuple[str, ...] = _ALL_FIXTURES) -> tuple[list[ArtifactRef], Workdir]:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    ctx = _ctx(workdir.root)
    refs = ext.extract(_ingest(workdir.root, names), ctx)
    return refs, workdir


def _by_name(refs: list[ArtifactRef], filename: str) -> ArtifactRef:
    for ref in refs:
        if ref.uri.name == filename:
            return ref
    raise AssertionError(f"no output named {filename!r}; got {[r.uri.name for r in refs]}")


# --- application-number normalization (raw + both forms) ------------------------


def test_normalizes_nda_bla_anda_with_and_without_leading_zeroes(tmp_path: Path) -> None:
    _, workdir = _extract(tmp_path)
    products = pl.read_parquet(workdir.interim / "drugsfda" / "products.parquet")
    by_raw = {row["appl_no_raw"]: row for row in products.iter_rows(named=True)}

    # NDA with leading zero: NDA012345 -> 012345 (zeros kept) and 12345 (stripped).
    assert by_raw["NDA012345"]["appl_type"] == "NDA"
    assert by_raw["NDA012345"]["appl_no"] == "012345"
    assert by_raw["NDA012345"]["appl_no_stripped"] == "12345"

    # NDA without leading zero: 207500 stays 207500 in both forms.
    assert by_raw["NDA207500"]["appl_no"] == "207500"
    assert by_raw["NDA207500"]["appl_no_stripped"] == "207500"

    # BLA without leading zero.
    assert by_raw["BLA125557"]["appl_type"] == "BLA"
    assert by_raw["BLA125557"]["appl_no"] == "125557"
    assert by_raw["BLA125557"]["appl_no_stripped"] == "125557"

    # ANDA with leading zero: ANDA075123 -> 075123 / 75123.
    assert by_raw["ANDA075123"]["appl_type"] == "ANDA"
    assert by_raw["ANDA075123"]["appl_no"] == "075123"
    assert by_raw["ANDA075123"]["appl_no_stripped"] == "75123"


def test_raw_application_number_combined_form_is_preserved(tmp_path: Path) -> None:
    """Legacy NDC-style 'NDA012345' raw APPLICATIONNUMBER is preserved verbatim (readNDAproducts)."""
    _, workdir = _extract(tmp_path)
    products = pl.read_parquet(workdir.interim / "drugsfda" / "products.parquet")
    for row in products.iter_rows(named=True):
        assert row["appl_no_raw"] == f"{row['appl_type']}{row['appl_no']}"


def test_parses_combined_applicationnumber_column(tmp_path: Path) -> None:
    """A source that uses the NDC 'ApplicationNumber' combined column is normalized too."""
    tsv = (
        "ProductNDC\tApplicationNumber\tProprietaryName\tNonProprietary Name\n"
        "00000-001\tNDA012345\tExamplestatin\tExamplestatin\n"
        "00000-002\tBLA125557\tKeytruda\tPembrolizumab\n"
    )
    src = tmp_path / "ndc_products.tsv"
    src.write_text(tsv, encoding="utf-8")
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    store = ArtifactStore(workdir)
    ref, _ = store.ingest(src, alias="drugsfda/ndc_products.tsv")
    ext.extract([ref], _ctx(workdir.root))

    products = pl.read_parquet(workdir.interim / "drugsfda" / "products.parquet")
    rows = {r["appl_no_raw"]: r for r in products.iter_rows(named=True)}
    assert rows["NDA012345"]["appl_type"] == "NDA"
    assert rows["NDA012345"]["appl_no_stripped"] == "12345"
    assert rows["NDA012345"]["product_ndc"] == "00000-001"
    assert rows["BLA125557"]["drug_name"] == "Keytruda"


# --- lookup tables --------------------------------------------------------------


def test_builds_name_ingredient_ndc_marketing_lookups(tmp_path: Path) -> None:
    _, workdir = _extract(tmp_path)
    lookups = pl.read_parquet(workdir.interim / "drugsfda" / "lookups.parquet")

    def terms(lookup_type: str, appl_no_stripped: str) -> set[str]:
        return {
            row["term"]
            for row in lookups.filter((pl.col("lookup_type") == lookup_type) & (pl.col("appl_no_stripped") == appl_no_stripped)).iter_rows(named=True)
        }

    # Proprietary name -> BLA 125557 (Keytruda).
    assert terms("proprietary_name", "125557") == {"Keytruda"}
    # Nonproprietary (whole active-ingredient string) -> NDA 12345.
    assert terms("nonproprietary_name", "12345") == {"EXAMPLESTATIN"}
    # Multi-ingredient product: both ingredients map to ANDA 75123.
    assert terms("ingredient", "75123") == {"EZETIMIBE", "SIMVASTATIN"}
    # Product NDC lookup type is absent when the fixture has no NDC column.
    assert "product_ndc" not in set(lookups.get_column("lookup_type"))
    # Marketing status -> each application that carries it.
    assert "125557" in {
        row["appl_no_stripped"]
        for row in lookups.filter((pl.col("lookup_type") == "marketing_status") & (pl.col("term") == "Prescription")).iter_rows(named=True)
    }
    assert "90123" in {
        row["appl_no_stripped"]
        for row in lookups.filter((pl.col("lookup_type") == "marketing_status") & (pl.col("term") == "Over-the-counter")).iter_rows(named=True)
    }


# --- source-section TSV (Tablassert handoff) ------------------------------------


def test_emits_uncompressed_source_section_tsv(tmp_path: Path) -> None:
    refs, workdir = _extract(tmp_path)
    tsv_ref = _by_name(refs, "drugsfda_products.tsv")
    assert tsv_ref.media_type == "text/tab-separated-values"
    assert tsv_ref.uri == workdir.tabular / "drugsfda_products.tsv"

    # Uncompressed (no gzip magic bytes) and tab-delimited with a header row.
    raw = tsv_ref.uri.read_bytes()
    assert not raw.startswith(b"\x1f\x8b")
    first_line = raw.split(b"\n", 1)[0]
    assert b"source_record_id" in first_line
    assert first_line.count(b"\t") == len(ext.PRODUCTS_COLUMNS) - 1
    # Readable round-trip equals the parquet products row count.
    frame = pl.read_csv(tsv_ref.uri, separator="\t")
    assert frame.height == 6
    assert set(frame.columns) == set(ext.PRODUCTS_COLUMNS)


# --- determinism + provenance ---------------------------------------------------


def test_outputs_are_blake3_deterministic_across_runs(tmp_path: Path) -> None:
    refs_a, _ = _extract(tmp_path / "a")
    refs_b, _ = _extract(tmp_path / "b")
    a = {ref.uri.name: ref.blake3 for ref in refs_a}
    b = {ref.uri.name: ref.blake3 for ref in refs_b}
    assert a == b, "same inputs + code must produce identical artifact ids"
    # The products parquet content hash is the canonical b3:<hex> form.
    _, workdir_c = _extract(tmp_path / "c")
    assert hash_file(workdir_c.interim / "drugsfda" / "products.parquet").startswith("b3:")


def test_manifests_record_rows_schema_fingerprint_and_inputs(tmp_path: Path) -> None:
    refs, _ = _extract(tmp_path)
    products_ref = _by_name(refs, "products.parquet")
    assert products_ref.rows == 6
    assert products_ref.manifest is not None
    manifest = read_manifest(products_ref.manifest)
    assert manifest.table.rows == 6
    assert manifest.table.schema_fingerprint == schemas.schema_fingerprint(ext.PRODUCTS_COLUMNS)
    assert manifest.table.warnings == 0
    assert manifest.operation is not None
    assert manifest.operation.name == "extract_drugsfda_products"
    assert manifest.inputs  # provenance chain to the source fixture


def test_source_record_ids_are_stable_and_unique(tmp_path: Path) -> None:
    _, workdir = _extract(tmp_path)
    products = pl.read_parquet(workdir.interim / "drugsfda" / "products.parquet")
    ids = products.get_column("source_record_id").to_list()
    assert len(ids) == len(set(ids)), "source_record_id must be unique per product"
    assert all(uid.startswith("drugsfda:product:") for uid in ids)


# --- applications / submissions + appl_type inheritance -------------------------


def test_applications_and_submissions_parsed(tmp_path: Path) -> None:
    refs, workdir = _extract(tmp_path)
    _by_name(refs, "applications.parquet")
    _by_name(refs, "submissions.parquet")
    applications = pl.read_parquet(workdir.interim / "drugsfda" / "applications.parquet")
    assert applications.height == 4
    assert set(applications.get_column("appl_type").to_list()) == {"NDA", "BLA", "ANDA"}
    submissions = pl.read_parquet(workdir.interim / "drugsfda" / "submissions.parquet")
    assert submissions.height == 5


def test_submissions_inherit_appl_type_from_products(tmp_path: Path) -> None:
    """Submissions.txt carries no ApplType; the extractor inherits it from products."""
    _, workdir = _extract(tmp_path)
    submissions = pl.read_parquet(workdir.interim / "drugsfda" / "submissions.parquet").select(["appl_type", "appl_no_stripped"])
    by_stripped = {row["appl_no_stripped"]: row["appl_type"] for row in submissions.iter_rows(named=True)}
    assert by_stripped["125557"] == "BLA"
    assert by_stripped["75123"] == "ANDA"
    assert by_stripped["12345"] == "NDA"


# --- partial input + ZIP input --------------------------------------------------


def test_products_only_when_only_products_fixture_present(tmp_path: Path) -> None:
    refs, workdir = _extract(tmp_path, names=("drugsfda_products.tsv",))
    names = sorted(ref.uri.name for ref in refs)
    assert "products.parquet" in names
    assert "drugsfda_products.tsv" in names
    assert "lookups.parquet" in names
    # No applications/submissions tables were provided, so they are not emitted.
    assert "applications.parquet" not in names
    assert "submissions.parquet" not in names
    # The warnings record notes the missing tables.
    warnings_lines = (workdir.interim / "drugsfda" / "extract_warnings.jsonl").read_text(encoding="utf-8").splitlines()
    messages = [json.loads(line)["table"] for line in warnings_lines if line.strip()]
    assert "applications" in messages
    assert "submissions" in messages


def test_parses_zip_member_inputs(tmp_path: Path) -> None:
    """Real Drugs@FDA acquisition delivers a ZIP; members are parsed by filename."""
    zip_path = tmp_path / "drugsfda_data_files.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in _ALL_FIXTURES:
            arcname = name.replace("drugsfda_", "").capitalize()  # Products.tsv etc.
            archive.write(_DRUGSFDA_FIXTURE_DIR / name, arcname=arcname)
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    store = ArtifactStore(workdir)
    ref, _ = store.ingest(zip_path, alias="drugsfda/drugsfda_data_files.zip")
    refs = ext.extract([ref], _ctx(workdir.root))

    _by_name(refs, "products.parquet")
    _by_name(refs, "applications.parquet")
    _by_name(refs, "submissions.parquet")
    products = pl.read_parquet(workdir.interim / "drugsfda" / "products.parquet")
    assert products.height == 6


# --- empty input is a no-op (writes only the warnings record) -------------------


def test_empty_input_emits_only_warnings(tmp_path: Path) -> None:
    workdir = Workdir(tmp_path / "work")
    workdir.create()
    refs = ext.extract([], _ctx(workdir.root))
    assert len(refs) == 1
    assert refs[0].uri.name == "extract_warnings.jsonl"
    assert refs[0].rows is not None
    assert refs[0].rows >= 1
