"""Unit tests for the MEDI contraindication extractor and DailyMed-support scoring."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import polars as pl
import pytest

from dakp_pipeline.extract import medi as medi_extract
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.io.manifests import read_manifest
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import medi

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(tmp_path: Path, *, profile: str = "mock", params: dict[str, object] | None = None) -> TaskContext:
    return TaskContext(profile=profile, workdir=tmp_path / "work", fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params=params or {})


def _fixture_refs(tmp_path: Path, *, params: dict[str, object] | None = None) -> tuple[TaskContext, list]:
    ctx = _ctx(tmp_path, params=params)
    Workdir(ctx.workdir).create()
    return ctx, medi.fetch(ctx)


# === DailyMed-support scoring (pure) ===========================================


def test_words_in_text_strips_tags_lowercases_and_tokenizes() -> None:
    assert medi_extract.words_in_text("Patients <b>with</b> Asthma!") == ["patients", "with", "asthma"]
    # Stray '<' becomes 'lt' (legacy behavior) before non-alphanumeric stripping.
    assert medi_extract.words_in_text("a<b") == ["altb"]
    assert medi_extract.words_in_text("") == []
    assert medi_extract.words_in_text("   ") == []


def test_support_score_full_overlap_is_one() -> None:
    ci = "contraindicated in asthma"
    section = "This drug is contraindicated in asthma patients."
    assert medi_extract.support_score(ci, section) == pytest.approx(1.0)


def test_support_score_partial_overlap_fraction() -> None:
    # ci words: ["asthma", "pregnancy"] -> 1 of 2 present -> 0.5
    assert medi_extract.support_score("asthma pregnancy", "use avoids pregnancy") == pytest.approx(0.5)


def test_support_score_no_overlap_is_zero() -> None:
    assert medi_extract.support_score("asthma pregnancy", "headache and nausea") == pytest.approx(0.0)


def test_support_score_empty_inputs_are_zero() -> None:
    assert medi_extract.support_score("", "anything") == 0.0
    assert medi_extract.support_score("asthma", "") == 0.0


def test_best_support_score_picks_max_with_first_index_on_tie() -> None:
    sections = ["nope", "asthma mention", "also asthma here"]
    # Both asthma sections fully contain the single ci word -> tie at 1.0; first wins.
    score, idx = medi_extract.best_support_score("asthma", sections)
    assert score == pytest.approx(1.0)
    assert idx == 1


def test_best_support_score_empty_sections_returns_zero_and_minus_one() -> None:
    assert medi_extract.best_support_score("asthma", []) == (0.0, -1)


def test_rank_sections_sorted_desc_then_index() -> None:
    sections = ["nothing", "asthma", "asthma too"]
    ranked = medi_extract.rank_sections("asthma", sections)
    assert ranked[0] == (pytest.approx(1.0), 1)  # highest score, lowest index among ties
    assert ranked[-1][1] == 0  # the no-overlap section last


# === stdlib xlsx reader ========================================================


def _write_xlsx(path: Path, sheet_name: str, headers: list[str], data_rows: list[list[str]]) -> Path:
    """Write a minimal but valid .xlsx using a shared-strings table (no openpyxl)."""
    seen: dict[str, int] = {}
    shared: list[str] = []

    def sidx(value: str) -> int:
        if value not in seen:
            seen[value] = len(shared)
            shared.append(value)
        return seen[value]

    def col(n: int) -> str:
        out = ""
        n += 1
        while n > 0:
            n, rem = divmod(n - 1, 26)
            out = chr(65 + rem) + out
        return out

    def cell(j: int, i: int, value: str) -> str:
        return f'<c r="{col(j)}{i}" t="s"><v>{sidx(value)}</v></c>'

    all_rows = [headers, *data_rows]
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        + "".join(f'<row r="{i}">' + "".join(cell(j, i, v) for j, v in enumerate(row)) + "</row>" for i, row in enumerate(all_rows, start=1))
        + "</sheetData></worksheet>"
    )
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + "".join(f"<si><t>{xml_escape(s)}</t></si>" for s in shared)
        + "</sst>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{xml_escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/sharedStrings.xml", shared_xml)
    return path


def test_read_xlsx_resolves_shared_strings_and_sheet_name(tmp_path: Path) -> None:
    path = _write_xlsx(tmp_path / "cl.xlsx", "Contraindications", ["active ingredient", "contraindications"], [["Ibuprofen", "Do not use in asthma"]])
    sheets = medi_extract.read_xlsx_tables(path)
    assert len(sheets) == 1
    assert sheets[0].name == "Contraindications"
    assert sheets[0].rows[0] == ["active ingredient", "contraindications"]
    assert sheets[0].rows[1] == ["Ibuprofen", "Do not use in asthma"]


# === source_record_id =========================================================


def test_source_record_id_is_stable_and_distinct() -> None:
    a1 = medi_extract.source_record_id("b3:aaa", "tsv", 2)
    assert a1 == medi_extract.source_record_id("b3:aaa", "tsv", 2)  # deterministic
    assert a1.startswith("medi:")
    # Different row or different source -> different id.
    assert medi_extract.source_record_id("b3:aaa", "tsv", 3) != a1
    assert medi_extract.source_record_id("b3:bbb", "tsv", 2) != a1


# === extractor (TSV fixture) ===================================================


def test_extract_parses_fixture_rows_and_preserves_provenance(tmp_path: Path) -> None:
    ctx, refs = _fixture_refs(tmp_path, params={"medi_version": "1.4.1"})
    out = medi_extract.extract(refs, ctx)

    assert len(out) == 2  # parquet + tsv
    parquet_ref, tsv_ref = out
    assert parquet_ref.uri.suffix == ".parquet"
    assert tsv_ref.uri.suffix == ".tsv"

    frame = schemas.read_table(parquet_ref.uri)
    assert frame.columns == medi_extract.MEDI_CONTRAINDICATIONS_COLUMNS
    assert frame.height == 3  # three fixture rows
    assert set(frame["medi_version"]) == {"1.4.1"}
    assert set(frame["source_sheet"]) == {"tsv"}
    # Row numbers count the header as line 1 -> data rows start at 2.
    assert sorted(frame["source_row"]) == ["2", "3", "4"]
    assert (frame["source_file"] == "medi_contraindications.tsv").all()
    # Every row carries a stable, distinct source_record_id.
    assert frame["source_record_id"].n_unique() == 3
    assert (frame["source_record_id"].str.starts_with("medi:")).all()

    ibu = frame.filter(pl.col("active_ingredient") == "Ibuprofen").row(0, named=True)
    assert ibu["contraindication_text"].startswith("Patients with aspirin-exacerbated")
    assert ibu["normalized_drug_id"] == "UNII:WK2XYI10QM"
    assert ibu["normalized_drug_label"] == "Ibuprofen"


def test_extract_preserves_missing_disease_id_as_empty(tmp_path: Path) -> None:
    ctx, refs = _fixture_refs(tmp_path)
    frame = schemas.read_table(medi_extract.extract(refs, ctx)[0].uri)
    by_drug = {row["active_ingredient"]: row for row in frame.iter_rows(named=True)}

    # Warfarin row has no provided normalized disease id/label (kept as "", not dropped).
    assert by_drug["Warfarin"]["normalized_disease_id"] == ""
    assert by_drug["Warfarin"]["normalized_disease_label"] == ""
    # The other rows do carry provided disease ids.
    assert by_drug["Ibuprofen"]["normalized_disease_id"] == "MONDO:0004979"
    assert by_drug["Methotrexate"]["normalized_disease_id"] == "MONDO:0005154"


def test_extract_emits_uncompressed_tablassert_section_tsv(tmp_path: Path) -> None:
    ctx, refs = _fixture_refs(tmp_path)
    tsv_ref = medi_extract.extract(refs, ctx)[1]
    assert tsv_ref.media_type == schemas.TSV_MEDIA_TYPE
    # Uncompressed (Tablassert cannot read compressed inputs).
    raw = tsv_ref.uri.read_bytes()
    assert raw[:2] != b"\x1f\x8b"  # not gzip
    lines = tsv_ref.uri.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == medi_extract.MEDI_CONTRAINDICATION_SECTIONS_COLUMNS
    assert len(lines) == 4  # header + 3 rows


def test_extract_source_score_is_empty_without_sections(tmp_path: Path) -> None:
    ctx, refs = _fixture_refs(tmp_path)
    frame = schemas.read_table(medi_extract.extract(refs, ctx)[0].uri)
    assert set(frame["source_score"]) == {""}


def test_extract_source_score_computed_from_supplied_sections(tmp_path: Path) -> None:
    sections = ["Contraindicated in patients with asthma or hypersensitivity to ibuprofen."]
    ctx, refs = _fixture_refs(tmp_path, params={"dailymed_contraindication_sections": sections})
    frame = schemas.read_table(medi_extract.extract(refs, ctx)[0].uri)

    ibu_text = frame.filter(pl.col("active_ingredient") == "Ibuprofen").row(0, named=True)["contraindication_text"]
    expected_score, _ = medi_extract.best_support_score(ibu_text, sections)
    ibu_score = frame.filter(pl.col("active_ingredient") == "Ibuprofen").row(0, named=True)["source_score"]
    assert ibu_score == f"{expected_score:.4f}"
    # A contraindication with word overlap scores higher than zero when sections share words.
    assert float(ibu_score) > 0.0


def test_extract_manifest_records_rows_schema_and_warnings(tmp_path: Path) -> None:
    ctx, refs = _fixture_refs(tmp_path)
    parquet_ref = medi_extract.extract(refs, ctx)[0]
    assert parquet_ref.manifest is not None
    manifest = read_manifest(parquet_ref.manifest)
    assert manifest.table.rows == 3
    assert manifest.table.warnings == 0  # fixture rows are complete
    assert (manifest.table.schema_fingerprint or "").startswith("b3:")
    assert manifest.hash.file == parquet_ref.blake3
    assert parquet_ref.rows == 3
    assert parquet_ref.schema_fingerprint == manifest.table.schema_fingerprint


def test_extract_source_record_id_deterministic_across_runs(tmp_path: Path) -> None:
    ctx_a, refs_a = _fixture_refs(tmp_path / "a")
    ctx_b, refs_b = _fixture_refs(tmp_path / "b")
    ids_a = schemas.read_table(medi_extract.extract(refs_a, ctx_a)[0].uri)["source_record_id"].to_list()
    ids_b = schemas.read_table(medi_extract.extract(refs_b, ctx_b)[0].uri)["source_record_id"].to_list()
    assert ids_a == ids_b  # same fixture bytes -> identical row identities


def test_extract_counts_lossy_rows_as_warnings(tmp_path: Path) -> None:
    # Hand-crafted source with one row missing the required contraindication text.
    src = tmp_path / "medi_partial.tsv"
    src.write_text(
        "active_ingredient\tcontraindications\tdisease_contraindicated\nIbuprofen\tContraindicated in asthma\tasthma\nPlacebo\t\tnone\n",
        encoding="utf-8",
    )
    from dakp_pipeline.io.artifact_store import ArtifactStore

    wd = Workdir(tmp_path / "work")
    wd.create()
    ref, _ = ArtifactStore(wd).ingest(src, alias="medi/partial")
    ctx = _ctx(tmp_path)
    parquet_ref = medi_extract.extract([ref], ctx)[0]
    assert parquet_ref.manifest is not None
    manifest = read_manifest(parquet_ref.manifest)
    assert manifest.table.rows == 2  # kept (no silent data loss)
    assert manifest.table.warnings == 1  # the Placebo row is flagged


# === extractor (real xlsx path) ===============================================


def test_extract_parses_real_xlsx_asset(tmp_path: Path) -> None:
    from dakp_pipeline.io.artifact_store import ArtifactStore

    xlsx_path = tmp_path / "contraindicationList-1.4.1.xlsx"
    _write_xlsx(
        xlsx_path,
        "Contraindications",
        [
            "active ingredient",
            "contraindications",
            "disease contraindicated",
            "final normalized drug id",
            "final normalized drug label",
            "final normalized disease id",
            "final normalized disease label",
        ],
        [["Ibuprofen", "Contraindicated in asthma", "asthma", "UNII:WK2XYI10QM", "Ibuprofen", "MONDO:0004979", "asthma"]],
    )
    wd = Workdir(tmp_path / "work")
    wd.create()
    ref, _ = ArtifactStore(wd).ingest(xlsx_path, alias="medi/contraindicationList-1.4.1")
    ctx = _ctx(tmp_path, params={"medi_version": "1.4.1"})

    parquet_ref = medi_extract.extract([ref], ctx)[0]
    frame = schemas.read_table(parquet_ref.uri)
    assert frame.height == 1
    row = frame.row(0, named=True)
    # Space-bearing xlsx headers map to the same canonical fields as the snake_case TSV.
    assert row["active_ingredient"] == "Ibuprofen"
    assert row["contraindication_text"] == "Contraindicated in asthma"
    assert row["normalized_drug_id"] == "UNII:WK2XYI10QM"
    assert row["normalized_disease_id"] == "MONDO:0004979"
    assert row["source_sheet"] == "Contraindications"  # real sheet name preserved
    assert row["source_file"] == "contraindicationList-1.4.1.xlsx"
    assert row["medi_version"] == "1.4.1"


def test_extract_ignores_non_medi_inputs(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    # An ArtifactRef whose name does not look like MEDI is skipped -> no outputs.
    other = tmp_path / "faers_cases.tsv"
    other.write_text("x\n1\n", encoding="utf-8")
    from dakp_pipeline.io.contracts import ArtifactRef

    ref = ArtifactRef(uri=other, blake3="b3:other", media_type=schemas.TSV_MEDIA_TYPE)
    assert medi_extract.extract([ref], ctx) == []
