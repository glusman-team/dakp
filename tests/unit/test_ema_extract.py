"""Tests for the EMA medicines-registry extractor.

The committed fixture xlsx mirrors the real export's layout — a banner row ("Content type:" /
"Output automatically generated ...") above the real header row — so the header-location logic is
exercised against genuine structure. Covers the Authorised/Human filter, the normalized interim
parquet contract (including the Phase-2 ``therapeutic_indication`` column), and the defensive
raises (no xlsx input, no header row, missing required columns).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.extract.ema_registry import EMA_REGISTRY_COLUMNS, EMARegistryExtractor, parse_ema_registry
from dakp_pipeline.io import schemas
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_EMA_FIXTURE = _FIXTURE_ROOT / "ema" / "medicines-output-medicines-report_en.xlsx"


def _ctx(workdir: Path) -> TaskContext:
    Workdir(workdir).create()
    return TaskContext(workdir=workdir, fixture_root=_FIXTURE_ROOT, params={})


def _fixture_ref() -> ArtifactRef:
    return ArtifactRef(uri=_EMA_FIXTURE, blake3=hash_file(_EMA_FIXTURE), media_type="application/octet-stream")


# --- the real fixture: banner rows + header location + the Authorised/Human filter ------------


def test_parse_locates_header_below_banner_and_filters_to_authorised_human() -> None:
    frame = parse_ema_registry(_EMA_FIXTURE)
    assert frame.columns == EMA_REGISTRY_COLUMNS
    # Qoyvolma (Withdrawn), KemSu (Opinion under re-examination), Deqtynet (Refused), and
    # Poulvac (Veterinary) are all dropped; only the three Authorised Human medicines survive.
    assert frame["medicine_name"].to_list() == ["Twinrix Adult", "Pyrukynd", "Tecvayli"]  # sorted by product number
    assert set(frame["category"]) == {"Human"}
    assert set(frame["medicine_status"]) == {"Authorised"}


def test_parse_keeps_semicolon_cells_and_phase2_indication_text() -> None:
    frame = parse_ema_registry(_EMA_FIXTURE)
    by_name = {rec["medicine_name"]: rec for rec in frame.iter_rows(named=True)}

    pyrukynd = by_name["Pyrukynd"]
    assert pyrukynd["ema_product_number"] == "EMEA/H/C/005540"
    assert pyrukynd["active_substance"] == "mitapivat sulfate"
    assert pyrukynd["therapeutic_area_mesh"] == "Genetic Diseases, Inborn;Anemia, Hemolytic"
    assert "pyruvate kinase deficiency" in pyrukynd["therapeutic_indication"]
    assert pyrukynd["medicine_url"] == "https://www.ema.europa.eu/en/medicines/human/EPAR/pyrukynd"

    # Twinrix Adult carries a combo substance cell (semicolon-joined, split by the shaper).
    assert ";" in by_name["Twinrix Adult"]["active_substance"]
    # Tecvayli has no Active substance cell: the INN is the shaper's subject fallback.
    assert by_name["Tecvayli"]["active_substance"] == ""
    assert by_name["Tecvayli"]["inn"] == "teclistamab"


def test_extract_writes_and_registers_interim_parquet(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path / "work")
    refs = EMARegistryExtractor().extract([_fixture_ref()], ctx)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.uri == Workdir(ctx.workdir).interim / "ema" / "ema_registry.parquet"
    assert ref.media_type == schemas.PARQUET_MEDIA_TYPE
    assert ref.manifest is not None
    assert ref.manifest.exists()

    frame = pl.read_parquet(ref.uri)
    assert frame.columns == EMA_REGISTRY_COLUMNS
    assert frame.height == 3


# --- defensive raises ------------------------------------------------------------


def test_extract_raises_without_xlsx_input(tmp_path: Path) -> None:
    other = tmp_path / "not-a-workbook.txt"
    other.write_text("nope", encoding="utf-8")
    ref = ArtifactRef(uri=other, blake3=hash_file(other), media_type="text/plain")
    with pytest.raises(ValueError, match="no EMA medicines xlsx"):
        EMARegistryExtractor().extract([ref], _ctx(tmp_path / "work"))


def test_parse_raises_when_header_row_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    banner_only = pl.DataFrame({"column_1": ["Content type:", "Medicine"], "column_2": ["Medicine", "report"]})
    monkeypatch.setattr(pl, "read_excel", lambda *args, **kwargs: banner_only)
    with pytest.raises(ValueError, match="no EMA header row"):
        parse_ema_registry(Path("whatever.xlsx"))


def test_parse_raises_when_required_column_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Header markers present (so the row IS located) but most required columns are absent.
    frame = pl.DataFrame(
        {
            "column_1": ["Content type:", "Name of medicine", "Pyrukynd"],
            "column_2": ["Medicine", "Medicine status", "Authorised"],
            "column_3": ["generated", "Category", "Human"],
        }
    )
    monkeypatch.setattr(pl, "read_excel", lambda *args, **kwargs: frame)
    with pytest.raises(ValueError, match="missing required columns"):
        parse_ema_registry(Path("whatever.xlsx"))
