"""Unit tests for the legacy-named release publish stage (:mod:`dakp_pipeline.release`).

Covers the naming schema (``drug_approvals_kg_{nodes,edges}_v<version>.{ndjson,tsv}`` +
``drug_approvals_kg_v<version>.RIG.yaml`` — the Tablassert-emitted Resource Ingest Guide, not the
``tables/graph.yaml`` build config), the deferred-handoff empty return, and the loud
``RuntimeError`` guards on a missing/ambiguous handoff report, legacy TSV pair, or RIG.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dakp_pipeline import __version__
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.release import LEGACY_STEM, publish
from dakp_pipeline.tablassert import GRAPH_NAME, REPORT_NAME

_RIG_TEXT = "source_info:\n  infores_id: infores:drug-approvals-kp\n"


def _ref(path: Path, media_type: str = "application/octet-stream") -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type=media_type)


def _ctx(workdir: Path) -> TaskContext:
    return TaskContext(workdir=workdir, fixture_root=None, params={})


def _scaffold(workdir: Path, *, mode: str = "real") -> tuple[list[ArtifactRef], list[ArtifactRef]]:
    """Write the handoff report + KGX pair + RIG + legacy TSV pair; return the ref lists."""
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    stem = f"{GRAPH_NAME}_{__version__}"
    report = workdir / "reports" / REPORT_NAME
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"mode": mode, "status": "ok"}), encoding="utf-8")
    kgx_refs = [_ref(report, "application/json")]
    if mode != "deferred":
        for kind in ("nodes", "edges"):
            path = data / f"{stem}.{kind}.ndjson"
            path.write_text(f'{{"kind":"{kind}"}}\n', encoding="utf-8")
            kgx_refs.append(_ref(path, "application/x-ndjson"))
        (data / f"{stem}.RIG.yaml").write_text(_RIG_TEXT, encoding="utf-8")
    legacy_refs = []
    for kind in ("nodes", "edges"):
        path = data / f"{stem}.{kind}.tsv"
        path.write_text(f"{kind}\ttsved\n", encoding="utf-8")
        legacy_refs.append(_ref(path, "text/tab-separated-values"))
    return kgx_refs, legacy_refs


def test_legacy_name_schema() -> None:
    assert LEGACY_STEM == "drug_approvals_kg"
    data_names = {
        f"drug_approvals_kg_nodes_v{__version__}.ndjson",
        f"drug_approvals_kg_edges_v{__version__}.ndjson",
        f"drug_approvals_kg_nodes_v{__version__}.tsv",
        f"drug_approvals_kg_edges_v{__version__}.tsv",
        f"drug_approvals_kg_v{__version__}.RIG.yaml",
    }
    workdir = Path("/nonexistent")
    assert data_names  # documented here; the publish test below asserts them on disk
    assert not workdir.exists()


def test_publish_copies_everything_under_the_legacy_names(tmp_path: Path) -> None:
    kgx_refs, legacy_refs = _scaffold(tmp_path)
    refs = publish(kgx_refs, legacy_refs, _ctx(tmp_path))

    data = tmp_path / "data"
    expected = [
        f"drug_approvals_kg_nodes_v{__version__}.ndjson",
        f"drug_approvals_kg_edges_v{__version__}.ndjson",
        f"drug_approvals_kg_nodes_v{__version__}.tsv",
        f"drug_approvals_kg_edges_v{__version__}.tsv",
        f"drug_approvals_kg_v{__version__}.RIG.yaml",
    ]
    assert [ref.uri.name for ref in refs] == expected
    # Copies, not renames: the Tablassert-stemmed originals stay.
    stem = f"{GRAPH_NAME}_{__version__}"
    for kind in ("nodes", "edges"):
        assert (data / f"{stem}.{kind}.ndjson").exists()
        assert (data / f"{stem}.{kind}.tsv").exists()
        assert (data / f"drug_approvals_kg_{kind}_v{__version__}.ndjson").read_text(encoding="utf-8") == (data / f"{stem}.{kind}.ndjson").read_text(
            encoding="utf-8"
        )
        assert (data / f"drug_approvals_kg_{kind}_v{__version__}.tsv").read_text(encoding="utf-8") == (data / f"{stem}.{kind}.tsv").read_text(
            encoding="utf-8"
        )
    # The published yaml is the Tablassert RIG next to the KGX pair, NOT the graph build config.
    assert (data / expected[-1]).read_text(encoding="utf-8") == _RIG_TEXT
    assert (data / f"{stem}.RIG.yaml").exists()
    assert all(ref.blake3.startswith("b3:") for ref in refs)


def test_publish_deferred_handoff_returns_empty(tmp_path: Path) -> None:
    kgx_refs, legacy_refs = _scaffold(tmp_path, mode="deferred")
    assert publish(kgx_refs, legacy_refs, _ctx(tmp_path)) == []


def test_publish_requires_exactly_one_report(tmp_path: Path) -> None:
    _kgx_refs, legacy_refs = _scaffold(tmp_path)
    with pytest.raises(RuntimeError, match="exactly one"):
        publish([], legacy_refs, _ctx(tmp_path))
    kgx_refs, _ = _scaffold(tmp_path)
    with pytest.raises(RuntimeError, match="exactly one"):
        publish([*kgx_refs, kgx_refs[0]], legacy_refs, _ctx(tmp_path))


def test_publish_requires_the_legacy_tsv_pair(tmp_path: Path) -> None:
    kgx_refs, legacy_refs = _scaffold(tmp_path)
    with pytest.raises(RuntimeError, match=r"legacy \.nodes\.tsv/\.edges\.tsv pair"):
        publish(kgx_refs, legacy_refs[:1], _ctx(tmp_path))


def test_publish_requires_the_rig_yaml(tmp_path: Path) -> None:
    kgx_refs, legacy_refs = _scaffold(tmp_path)
    (tmp_path / "data" / f"{GRAPH_NAME}_{__version__}.RIG.yaml").unlink()
    with pytest.raises(RuntimeError, match=r"exactly one '.*\.RIG\.yaml'"):
        publish(kgx_refs, legacy_refs, _ctx(tmp_path))


def test_publish_requires_the_kgx_pair_on_disk(tmp_path: Path) -> None:
    kgx_refs, legacy_refs = _scaffold(tmp_path)
    (tmp_path / "data" / f"{GRAPH_NAME}_{__version__}.edges.ndjson").unlink()
    with pytest.raises(RuntimeError, match="exactly one"):
        publish(kgx_refs, legacy_refs, _ctx(tmp_path))
