"""Acquisition-task tests — offline (monkeypatched HTTP), import-safe WITHOUT airflow.

The Airflow acquisition tasks in :mod:`dakp_pipeline.dags.dakp_build` delegate to the shared
:mod:`dakp_pipeline.acquire` layer, so these tests exercise the task callables directly via
that layer (no network, no airflow needed). The Airflow task-graph assertions run only when
apache-airflow is importable and are skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline import acquire, runtime
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, drugsfda, faers

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(workdir: Path, *, params: dict[str, object] | None = None) -> TaskContext:
    Workdir(workdir).create()
    return TaskContext(workdir=workdir, fixture_root=_FIXTURE_ROOT, params=dict(params or {}))


# --- source acquisition helpers delegate to the real fetchers -------------------


@pytest.mark.parametrize(
    ("helper", "module"),
    [(acquire.acquire_dailymed, dailymed), (acquire.acquire_faers, faers), (acquire.acquire_drugsfda, drugsfda)],
    ids=["dailymed", "faers", "drugsfda"],
)
def test_acquire_source_helpers_delegate_to_fetcher(helper, module, monkeypatch, tmp_path: Path) -> None:
    calls: list[TaskContext] = []

    def fake_fetch(ctx: TaskContext) -> list[ArtifactRef]:
        calls.append(ctx)
        return [ArtifactRef(uri=tmp_path / "raw.bin", blake3="b3:deadbeef", media_type="application/octet-stream")]

    monkeypatch.setattr(module, "fetch", fake_fetch)
    ctx = _ctx(tmp_path, params={"quarter_limit": 2})

    refs = helper(ctx)

    # The fetcher received the exact context (params included) and its manifest is returned.
    assert calls == [ctx]
    assert calls[0].params["quarter_limit"] == 2
    assert [ref.blake3 for ref in refs] == ["b3:deadbeef"]


# --- NER model acquisition ------------------------------------------------------


def test_acquire_ner_models_downloads_and_is_idempotent(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_downloader(model_id: str, dest: Path) -> None:
        calls.append(model_id)
        (dest / "model.bin").write_bytes(b"weights")

    ctx = _ctx(tmp_path)
    first = acquire.acquire_ner_models(ctx, models=["fake/model"], downloader=fake_downloader)

    assert len(first) == 1
    ref = first[0]
    assert ref.blake3.startswith("b3:")
    assert ref.media_type == "application/x-directory"
    assert ref.uri.is_dir()
    assert ref.manifest is not None
    assert ref.manifest.exists()
    assert calls == ["fake/model"]

    # Cache hit: identical content tree hash -> no re-download.
    second = acquire.acquire_ner_models(ctx, models=["fake/model"], downloader=fake_downloader)
    assert second[0].blake3 == ref.blake3
    assert calls == ["fake/model"]


def test_acquire_ner_models_force_redownloads(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_downloader(model_id: str, dest: Path) -> None:
        calls.append(model_id)
        (dest / "model.bin").write_bytes(b"weights")

    ctx = _ctx(tmp_path, params={"force": True})
    acquire.acquire_ner_models(ctx, models=["fake/model"], downloader=fake_downloader)
    acquire.acquire_ner_models(ctx, models=["fake/model"], downloader=fake_downloader)
    assert calls == ["fake/model", "fake/model"]


def test_default_ner_models_is_the_gliner_checkpoint(tmp_path: Path) -> None:
    # With no explicit models= override, acquisition defaults to the GLiNER checkpoint the
    # production NER backend loads (a real run downloads its weights).
    from dakp_pipeline.ner.ner import DEFAULT_MODEL

    assert acquire.default_ner_models(_ctx(tmp_path)) == [DEFAULT_MODEL]


# --- aggregate acquisition ------------------------------------------------------


def test_acquire_all_runs_every_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = ArtifactRef(uri=tmp_path / "raw.bin", blake3="b3:deadbeef", media_type="application/octet-stream")
    monkeypatch.setattr(dailymed, "fetch", lambda ctx: [sentinel])
    monkeypatch.setattr(faers, "fetch", lambda ctx: [sentinel])
    monkeypatch.setattr(drugsfda, "fetch", lambda ctx: [sentinel])

    def fake_downloader(model_id: str, dest: Path) -> None:
        (dest / "model.bin").write_bytes(b"weights")

    results = acquire.acquire_all(_ctx(tmp_path), downloader=fake_downloader)
    assert set(results) == {"dailymed", "drugsfda", "faers", "ner_models"}
    assert results["dailymed"] == [sentinel]
    assert results["drugsfda"] == [sentinel]
    assert results["faers"] == [sentinel]
    assert len(results["ner_models"]) == 1  # DEFAULT_MODEL cached via the fake downloader
    assert all(ref.blake3.startswith("b3:") for refs in results.values() for ref in refs)


# --- DAG wiring (import-safe; graph assertions gated on airflow) ----------------


def test_dag_module_importable_and_acquisition_wired(dakp_build) -> None:
    assert dakp_build.DAG_ID == "dakp_build"
    assert dakp_build.DOWNLOAD_POOL == "dakp_download"
    # The acquisition tasks are present in the constructed DAG (they delegate to acquire.*, whose
    # fetcher delegation is covered by test_acquire_source_helpers_delegate_to_fetcher).
    task_ids = {t.task_id for t in dakp_build.dag_obj.tasks}
    assert {"acquire_dailymed", "acquire_faers", "acquire_drugsfda"} <= task_ids


def test_build_context_forwards_drugsfda_url_override(tmp_path: Path) -> None:
    ctx = runtime.build_context_from_config(
        {"workdir": str(tmp_path), "fixture_root": str(_FIXTURE_ROOT), "drugsfda_url": "https://example.test/drugsfda.zip"}
    )
    assert ctx.params["drugsfda_url"] == "https://example.test/drugsfda.zip"


def test_build_context_forwards_fullmap(tmp_path: Path) -> None:
    ctx = runtime.build_context_from_config({"workdir": str(tmp_path), "fixture_root": str(_FIXTURE_ROOT), "fullmap": "/maps/fullmap.redb"})
    assert ctx.params["fullmap"] == "/maps/fullmap.redb"
    # run_tablassert is derived from fullmap presence (a fullmap triggers the real handoff).
    assert ctx.params["run_tablassert"] is True


def test_dag_includes_acquisition_tasks_and_pools(dakp_build) -> None:
    dag = dakp_build.dag_obj
    acquire_ids = {"acquire_dailymed", "acquire_faers", "acquire_drugsfda", "acquire_ner_models"}
    assert acquire_ids <= {t.task_id for t in dag.tasks}

    def upstream(task_id: str) -> set[str]:
        return set(dag.get_task(task_id).upstream_task_ids)

    # The NER acquisition feeds the contraindication miner (download -> mine).
    assert "acquire_ner_models" in upstream("shape_contraindication_tables")

    # Every acquisition task is bounded by the download pool.
    for task_id in acquire_ids:
        assert dag.get_task(task_id).pool == dakp_build.DOWNLOAD_POOL
