"""End-to-end offline pipeline test (PLAN.md "Monkeypatch-first full-pipeline test sketch").

Runs the full pipeline via the shared ``run_stages`` stage harness (tests/integration/harness.py —
the same Python wiring the Airflow DAG drives, Airflow-free) with NO network and NO real Tablassert.
Fetchers always run their real branches, so they are monkeypatched to load fixtures via
``ctx.fixture()`` (the shared ``install_fixture_fetchers`` helper), and ``dakp_pipeline.tablassert.run``
is either replaced with a fake or left to dispatch to the deferred runner — proving every external
boundary is substitutable. (The native Go extract path is validated by ``uv run dakp up`` + the Go
parity tests; the Airflow DAG wiring by ``test_dag.py``.)

WHY this test matters: it is the monkeypatch-first full-pipeline guardrail — it proves the harness's
stage wiring is substitutable at every external boundary (each fetcher + the Tablassert handoff), so
a regression in the wiring (e.g. resolving ``tablassert.run`` through a from-import instead of the
module attribute) is caught immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness import install_fixture_fetchers, run_stages

from dakp_pipeline import __version__
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, drugsfda, faers
from dakp_pipeline.tablassert import GRAPH_NAME, REPORT_NAME

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _fake_tablassert_run(assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Stand-in for ../Tablassert following the REAL handoff contract.

    Writes a real-mode handoff report plus a ``<name>_<version>.{nodes,edges}.ndjson`` pair under
    ``data/`` (exactly what a successful ``build-kg`` leaves behind), so the downstream legacy
    TSV export stage exercises its real branch against the fake output.
    """
    del config_refs
    store = ArtifactStore(Workdir(ctx.workdir))
    data = Workdir(ctx.workdir).root / "data"
    data.mkdir(parents=True, exist_ok=True)
    nodes = data / f"{GRAPH_NAME}_{__version__}.nodes.ndjson"
    edges = data / f"{GRAPH_NAME}_{__version__}.edges.ndjson"
    nodes.write_text('{"id":"MONDO:0005154","name":"hypercholesterolemia","category":["biolink:Disease"]}\n', encoding="utf-8")
    edges.write_text(
        '{"id":"fake-edge","subject":"CHEBI:1000001","predicate":"biolink:treats","object":"MONDO:0005154",'
        '"original_subject":"Examplestatin","original_object":"hypercholesterolemia",'
        '"knowledge_level":"knowledge_assertion","agent_type":"manual_validation_of_automated_agent",'
        '"approval_ids":["NDA1"],"has_evidence":["dailymed:set-1"]}\n',
        encoding="utf-8",
    )
    report = Workdir(ctx.workdir).reports / REPORT_NAME
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"mode": "real", "status": "ok"}), encoding="utf-8")
    return [
        store.register(report, media_type="application/json", inputs=[r.blake3 for r in assertion_refs]),
        store.register(nodes, media_type="application/x-ndjson", inputs=[r.blake3 for r in assertion_refs]),
        store.register(edges, media_type="application/x-ndjson", inputs=[r.blake3 for r in assertion_refs]),
    ]


def test_full_pipeline_uses_mocked_sources(monkeypatch, tmp_path: Path) -> None:
    # Fetchers -> ctx.fixture() (PLAN.md sketch boundary; no network).
    monkeypatch.setattr(dailymed, "fetch", lambda ctx: [ctx.fixture("dailymed/dailymed_spl.xml.gz")])
    monkeypatch.setattr(
        faers, "fetch", lambda ctx: [ctx.fixture("faers/DEMO24Q3.txt"), ctx.fixture("faers/DRUG24Q3.txt"), ctx.fixture("faers/INDI24Q3.txt")]
    )
    monkeypatch.setattr(drugsfda, "fetch", lambda ctx: [ctx.fixture("drugsfda/drugsfda_products.tsv")])
    # No real Tablassert on the dev laptop.
    monkeypatch.setattr("dakp_pipeline.tablassert.run", _fake_tablassert_run)

    result = run_stages(fixture_root=_FIXTURE_ROOT, workdir=tmp_path / "work")

    assert result.table("approved_treats_assertions").rows > 0
    assert result.table("faers_applied_to_treat_assertions").rows > 0
    assert result.table("contraindication_assertions").rows > 0
    assert result.build_summary is not None
    assert result.build_summary.exists()
    summary = json.loads(result.build_summary.read_text(encoding="utf-8"))
    assert summary["translator_regression"]["ok"] is True
    assert set(summary["translator_regression"]["families_seen"]) == {"biolink:treats", "biolink:applied_to_treat", "biolink:contraindicated_in"}

    # The fake Tablassert wrote its KGX pair, and the legacy TSV stage retrofitted it.
    data = tmp_path / "work" / "data"
    assert (data / f"{GRAPH_NAME}_{__version__}.nodes.ndjson").exists()
    legacy_edges = (data / f"{GRAPH_NAME}_{__version__}.edges.tsv").read_text(encoding="utf-8").splitlines()
    assert legacy_edges[0].split("\t")[9:12] == ["approval", "N_cases", "supporting_spls"]
    # Subject CHEBI:1000001 is absent from the fake node set -> original_subject mention fallback;
    # object is resolved -> canonical node name; object_modifier is always NA.
    assert legacy_edges[1].split("\t")[4:7] == ["Examplestatin", "hypercholesterolemia", "NA"]
    assert legacy_edges[1].split("\t")[9:] == ["NDA1", "NA", "dailymed:set-1"]
    assert summary["legacy_tsv"]["exported"] is True
    assert {file["name"] for file in summary["legacy_tsv"]["files"]} == {f"{GRAPH_NAME}_{__version__}.nodes", f"{GRAPH_NAME}_{__version__}.edges"}


def test_default_deferred_handoff_runs_clean(monkeypatch, tmp_path: Path) -> None:
    """The default path (fixture fetchers + deferred Tablassert handoff, no fullmap) runs clean,
    mirroring a default `uv run dakp up` (no --fullmap => deferred handoff, never an error)."""
    install_fixture_fetchers(monkeypatch)
    result = run_stages(fixture_root=_FIXTURE_ROOT, workdir=tmp_path / "work")

    for table in ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions"):
        assert result.table(table).rows > 0
        # Tablassert-facing contracts are uncompressed TSV.
        assert result.table(table).path.suffix == ".tsv"
        assert result.table(table).path.exists()

    # Deferred handoff manifest (no fullmap => no real Tablassert; no local KGX compiler) + summary.
    handoff = json.loads((tmp_path / "work" / "data" / "reports" / "tablassert_handoff.json").read_text(encoding="utf-8"))
    assert handoff["mode"] == "deferred"
    assert result.build_summary is not None
    assert result.build_summary.exists()
    summary = json.loads(result.build_summary.read_text(encoding="utf-8"))
    assert summary["translator_regression"]["ok"] is True
    assert summary["translator_regression"]["violations"] == []
    # Deferred handoff => no KGX to retrofit: empty legacy_tsv section, no TSV pair.
    assert summary["legacy_tsv"] == {"exported": False, "files": []}
    assert list((tmp_path / "work" / "data").glob("*.nodes.tsv")) == []
