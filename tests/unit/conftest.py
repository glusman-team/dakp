"""Shared fixtures for the Milestone-5 assertion-aggregation tests.

Runs the *real* extractors on the tiny pipeline fixtures so every assertion test aggregates
genuine interim tables (not hand-rolled mocks). ``ctx`` carries the lexical disease baseline
loaded from the ontology fixture, exactly as ``run_pipeline`` does.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.extract import drugsfda_products, faers_ascii, medi, spl_xml
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


@pytest.fixture
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.fixture
def disease_map() -> dict[str, dict[str, str]]:
    """Lexical disease baseline loaded from the ontology fixture (mirrors run_pipeline)."""
    frame = pl.read_csv(FIXTURE_ROOT / "ontology" / "disease_map.tsv", separator="\t")
    mapping: dict[str, dict[str, str]] = {}
    for rec in frame.iter_rows(named=True):
        text = str(rec.get("text", "") or "").strip()
        if text:
            mapping[text] = {
                "curie": str(rec.get("curie", "") or ""),
                "name": str(rec.get("name", text) or text),
                "category": str(rec.get("category", "Disease") or "Disease"),
            }
    return mapping


@pytest.fixture
def ctx(tmp_path: Path, disease_map: dict[str, dict[str, str]]) -> TaskContext:
    context = TaskContext(
        profile="mock", workdir=tmp_path / "work", fixture_root=FIXTURE_ROOT, threads=1, memory_budget_gb=1, params={"disease_map": disease_map}
    )
    Workdir(context.workdir).create()
    return context


@pytest.fixture
def dailymed_refs(ctx: TaskContext) -> list[ArtifactRef]:
    return spl_xml.extract([_ref(FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz")], ctx)


@pytest.fixture
def drugsfda_refs(ctx: TaskContext) -> list[ArtifactRef]:
    return drugsfda_products.extract(
        [_ref(FIXTURE_ROOT / "drugsfda" / "drugsfda_products.tsv"), _ref(FIXTURE_ROOT / "drugsfda" / "drugsfda_applications.tsv")], ctx
    )


@pytest.fixture
def faers_refs(ctx: TaskContext) -> list[ArtifactRef]:
    """FAERS 24Q3 cases *without* the DELETE file, so all three cases (incl. Placebo) survive."""
    names = ("DEMO24Q3.txt", "DRUG24Q3.txt", "INDI24Q3.txt", "REAC24Q3.txt")
    return faers_ascii.extract([_ref(FIXTURE_ROOT / "faers" / name) for name in names], ctx)


@pytest.fixture
def medi_refs(ctx: TaskContext) -> list[ArtifactRef]:
    return medi.extract([_ref(FIXTURE_ROOT / "medi" / "medi_contraindications.tsv")], ctx)
