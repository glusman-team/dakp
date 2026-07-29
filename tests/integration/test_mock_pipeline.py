"""End-to-end mocked pipeline test (PLAN.md "Monkeypatch-first full-pipeline test sketch").

Runs the full pipeline with NO network and NO real Tablassert/Airflow installed
(``run_airflow=False``). Fetchers are monkeypatched to load fixtures via ``ctx.fixture()``,
and ``dakp_pipeline.tablassert.run`` is replaced with a fake — proving every external
boundary is substitutable. The default (unpatched) mock path is also exercised to mirror
the CLI acceptance command.
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.pipeline import run_pipeline
from dakp_pipeline.sources import dailymed, drugsfda, faers, medi

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _fake_tablassert_run(assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Stand-in for ../Tablassert: writes a placeholder KGX marker and returns a ref."""
    store = ArtifactStore(Workdir(ctx.workdir))
    out = Workdir(ctx.workdir).kgx / "fake_nodes.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"id":"MONDO:0005154","name":"hypercholesterolemia","category":["biolink:Disease"]}\n')
    return [store.register(out, media_type="application/x-ndjson", inputs=[r.blake3 for r in assertion_refs])]


def test_full_pipeline_uses_mocked_sources(monkeypatch, tmp_path: Path) -> None:
    # Fetchers -> ctx.fixture() (PLAN.md sketch boundary; no network).
    monkeypatch.setattr(dailymed, "fetch", lambda ctx: [ctx.fixture("dailymed/dailymed_spl.xml.gz")])
    monkeypatch.setattr(
        faers, "fetch", lambda ctx: [ctx.fixture("faers/DEMO24Q3.txt"), ctx.fixture("faers/DRUG24Q3.txt"), ctx.fixture("faers/INDI24Q3.txt")]
    )
    monkeypatch.setattr(drugsfda, "fetch", lambda ctx: [ctx.fixture("drugsfda/drugsfda_products.tsv")])
    monkeypatch.setattr(medi, "fetch", lambda ctx: [ctx.fixture("medi/medi_contraindications.tsv")])
    # No real Tablassert on the dev laptop.
    monkeypatch.setattr("dakp_pipeline.tablassert.run", _fake_tablassert_run)

    result = run_pipeline(profile="mock", fixture_root=_FIXTURE_ROOT, workdir=tmp_path / "work", run_airflow=False)

    assert result.table("approved_treats_assertions").rows > 0
    assert result.table("faers_applied_to_treat_assertions").rows > 0
    assert result.table("contraindication_assertions").rows > 0
    assert result.build_summary is not None
    assert result.build_summary.exists()

    # The fake Tablassert wrote its placeholder KGX marker.
    assert (tmp_path / "work" / "data" / "kgx" / "fake_nodes.jsonl").exists()


def test_default_mock_path_matches_cli_acceptance(tmp_path: Path) -> None:
    """The unpatched mock path (default fetchers + mock tablassert handoff) runs clean,
    mirroring `uv run dakp run --profile mock ...`."""
    result = run_pipeline(profile="mock", fixture_root=_FIXTURE_ROOT, workdir=tmp_path / "work", run_airflow=False)

    for table in ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions"):
        assert result.table(table).rows > 0
        # Tablassert-facing contracts are uncompressed TSV.
        assert result.table(table).path.suffix == ".tsv"
        assert result.table(table).path.exists()

    # Mock handoff manifest (no local KGX compiler) + build summary.
    assert (tmp_path / "work" / "data" / "reports" / "tablassert_handoff.json").exists()
    assert result.build_summary is not None
    assert result.build_summary.exists()
