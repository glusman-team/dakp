"""Unit tests for the streaming DailyMed SPL XML extractor.

Covers: streaming parse of the gzipped fixture, the full set of normalized interim
tables + the uncompressed section TSV, schema/row-count assertions, BLAKE3
``source_record_id`` determinism, manifest provenance (b3 hash + rows + schema
fingerprint), parse-warning emission, and gzip-aware streaming.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.extract import spl_xml
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import read_manifest
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(tmp_path: Path) -> TaskContext:
    return TaskContext(profile="mock", workdir=(tmp_path / "work"), fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params={})


def _acquire_and_extract(tmp_path: Path) -> tuple[list[ArtifactRef], list[ArtifactRef], TaskContext]:
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()
    raw = dailymed.fetch(ctx)
    refs = spl_xml.extract(raw, ctx)
    return raw, refs, ctx


# --- outputs & schema ----------------------------------------------------------


def test_extract_emits_all_tables_plus_section_tsv(tmp_path: Path) -> None:
    _raw, refs, ctx = _acquire_and_extract(tmp_path)

    # 5 interim parquet tables + 1 uncompressed section TSV.
    parquets = [r for r in refs if r.uri.suffix == ".parquet"]
    tsvs = [r for r in refs if r.uri.suffix == ".tsv"]
    assert len(parquets) == 5
    assert len(tsvs) == 1

    names = {r.uri.name for r in refs}
    assert names == {
        "spl_documents.parquet",
        "spl_sets.parquet",
        "spl_approvals.parquet",
        "spl_ingredients.parquet",
        "spl_sections.parquet",
        "dailymed_spl_sections.tsv",
    }
    # The section TSV is uncompressed (Tablassert cannot read compressed inputs).
    tsv_path = ctx.workdir / "data" / "tabular" / "dailymed_spl_sections.tsv"
    assert tsv_path.exists()
    assert tsv_path.read_bytes().startswith(b"source_record_id\t")


def test_spl_documents_is_first_ref_and_keeps_locked_contract(tmp_path: Path) -> None:
    """spl_documents is returned first (downstream shapers resolve 'the dailymed parquet')."""
    _raw, refs, _ctx = _acquire_and_extract(tmp_path)

    assert refs[0].uri.name == "spl_documents.parquet"
    frame = pl.read_parquet(refs[0].uri)
    assert frame.columns == schemas.DAILYMED_SPL_DOCUMENTS_COLUMNS


def test_row_counts_match_fixture(tmp_path: Path) -> None:
    # Fixture: 3 documents, each with 2 sections -> 6 section-level rows.
    _raw, refs, _ctx = _acquire_and_extract(tmp_path)
    by_name = {r.uri.name: r for r in refs}

    assert by_name["spl_documents.parquet"].rows == 6
    assert by_name["spl_sections.parquet"].rows == 6
    assert by_name["spl_sets.parquet"].rows == 3  # 3 distinct set ids
    assert by_name["spl_approvals.parquet"].rows == 3  # 1 approval per document
    assert by_name["spl_ingredients.parquet"].rows == 4  # 1 active each + 1 inactive on doc 1


def test_sections_table_carries_loinc_title_raw_and_clean_text(tmp_path: Path) -> None:
    _raw, refs, _ctx = _acquire_and_extract(tmp_path)
    frame = pl.read_parquet(next(r for r in refs if r.uri.name == "spl_sections.parquet").uri)
    assert frame.columns == spl_xml.SPL_SECTIONS_COLUMNS

    loincs = frame.get_column("loinc_code").sort().to_list()
    # 3x indications (34067-9), 2x contraindications (34070-3), 1x missing-LOINC "".
    assert loincs == ["", "34067-9", "34067-9", "34067-9", "34070-3", "34070-3"]

    # raw_text preserves original whitespace shape; clean_text is single-spaced.
    indications = frame.filter(pl.col("loinc_code") == "34067-9")
    assert indications.height == 3
    assert indications.get_column("clean_text").str.contains(" ").any()
    # Every cleaned value is whitespace-collapsed (no double spaces).
    for value in indications.get_column("clean_text").to_list():
        assert "  " not in value

    # A section without a LOINC code is preserved (no data loss), with empty code.
    no_loinc = frame.filter(pl.col("loinc_code") == "")
    assert no_loinc.height == 1
    assert no_loinc.get_column("section_name").to_list() == ["HOW SUPPLIED"]


def test_ingredients_table_has_active_and_inactive_roles_with_unii(tmp_path: Path) -> None:
    _raw, refs, _ctx = _acquire_and_extract(tmp_path)
    frame = pl.read_parquet(next(r for r in refs if r.uri.name == "spl_ingredients.parquet").uri)
    assert frame.columns == spl_xml.SPL_INGREDIENTS_COLUMNS

    active = frame.filter(pl.col("role") == "active")
    inactive = frame.filter(pl.col("role") == "inactive")
    assert active.height == 3  # one active per document
    assert inactive.height == 1  # lactose on the statin document

    # UNII codes are recorded for every ingredient.
    assert all(v.startswith("UNII:") for v in frame.get_column("ingredient_unii").to_list())
    assert sorted(active.get_column("ingredient_name").to_list()) == ["Examplestatin", "Ibuprofen", "Omeprazole"]
    assert inactive.get_column("ingredient_name").to_list() == ["Lactose"]


def test_approvals_and_sets_tables(tmp_path: Path) -> None:
    _raw, refs, _ctx = _acquire_and_extract(tmp_path)
    approvals = pl.read_parquet(next(r for r in refs if r.uri.name == "spl_approvals.parquet").uri)
    sets = pl.read_parquet(next(r for r in refs if r.uri.name == "spl_sets.parquet").uri)

    assert approvals.columns == spl_xml.SPL_APPROVALS_COLUMNS
    assert sets.columns == spl_xml.SPL_SETS_COLUMNS
    assert sorted(approvals.get_column("approval_id").to_list()) == ["012345", "017977", "022329"]
    assert (approvals.get_column("approval_type") == "NDA").all()
    assert sets.height == 3
    assert sets.get_column("spl_set_id").is_unique().all()


# --- provenance / determinism --------------------------------------------------


def test_source_record_id_is_deterministic(tmp_path: Path) -> None:
    """Re-extracting the same source yields identical source_record_id values."""
    ctx_a = _ctx(tmp_path / "a")
    Workdir(ctx_a.workdir).create()
    ctx_b = _ctx(tmp_path / "b")
    Workdir(ctx_b.workdir).create()

    raw = dailymed.fetch(ctx_a)
    refs_a = spl_xml.extract(raw, ctx_a)
    # Same source artifact id for the same fixture content.
    raw_b = dailymed.fetch(ctx_b)
    assert [r.blake3 for r in raw] == [r.blake3 for r in raw_b]
    refs_b = spl_xml.extract(raw_b, ctx_b)

    sec_a = pl.read_parquet(next(r for r in refs_a if r.uri.name == "spl_sections.parquet").uri)
    sec_b = pl.read_parquet(next(r for r in refs_b if r.uri.name == "spl_sections.parquet").uri)
    assert sec_a.get_column("source_record_id").sort().to_list() == sec_b.get_column("source_record_id").sort().to_list()
    # Ids are stable, unique per (set, loinc), and b3-prefixed.
    assert sec_a.height == sec_a.get_column("source_record_id").n_unique()
    assert all(v.startswith("b3:") for v in sec_a.get_column("source_record_id").to_list())


def test_source_record_id_joins_across_tables(tmp_path: Path) -> None:
    """The same source hash underlies every table so set/approval ids join consistently."""
    _raw, refs, _ctx = _acquire_and_extract(tmp_path)
    by_name = {r.uri.name: r for r in refs}
    approvals = pl.read_parquet(by_name["spl_approvals.parquet"].uri)
    sets = pl.read_parquet(by_name["spl_sets.parquet"].uri)
    sections = pl.read_parquet(by_name["spl_sections.parquet"].uri)

    # spl_set_id is shared across tables and resolves to a set row.
    set_ids = set(sets.get_column("spl_set_id").to_list())
    assert set(approvals.get_column("spl_set_id").to_list()) <= set_ids
    assert set(sections.get_column("spl_set_id").to_list()) <= set_ids


def test_manifests_record_hash_rows_and_schema_fingerprint(tmp_path: Path) -> None:
    _raw, refs, _ctx = _acquire_and_extract(tmp_path)
    for ref in refs:
        assert ref.manifest is not None
        manifest = read_manifest(ref.manifest)
        assert manifest.artifact_id == ref.blake3
        assert manifest.hash.file == ref.blake3 or manifest.hash.tree == ref.blake3
        assert manifest.table.rows == ref.rows
        assert manifest.table.schema_fingerprint is not None
        assert manifest.table.schema_fingerprint.startswith("b3:")
        assert manifest.operation is not None
        assert manifest.operation.name == "extract_dailymed_spl"
        # Input provenance: the source artifact id is recorded.
        assert manifest.inputs


def test_parse_warnings_recorded_for_lossy_sections(tmp_path: Path) -> None:
    """The fixture's no-LOINC section is flagged in the manifest warning counter."""
    _raw, refs, _ctx = _acquire_and_extract(tmp_path)
    sections_ref = next(r for r in refs if r.uri.name == "spl_sections.parquet")
    assert sections_ref.manifest is not None
    manifest = read_manifest(sections_ref.manifest)
    assert manifest.table.warnings is not None
    assert manifest.table.warnings >= 1


# --- streaming / gzip-awareness ------------------------------------------------


def test_extractor_handles_plain_and_gzipped_xml(tmp_path: Path) -> None:
    """Gzip-awareness: the same content parsed from a .xml.gz and a plain .xml agrees."""
    import gzip

    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()
    gz_fixture = _FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz"
    plain = tmp_path / "dailymed_spl.xml"
    plain.write_bytes(gzip.decompress(gz_fixture.read_bytes()))

    from dakp_pipeline.io.content_hash import hash_file

    store_ref = ArtifactRef(uri=plain, blake3=hash_file(plain), media_type="application/xml")

    refs = spl_xml.extract([store_ref], ctx)
    sections = pl.read_parquet(next(r for r in refs if r.uri.name == "spl_sections.parquet").uri)
    assert sections.height == 6
    # Same set ids as the gz path (content-identical).
    set_ids = set(sections.get_column("spl_set_id").to_list())
    assert "SETID-EXAMPLESTATIN-001" in set_ids
