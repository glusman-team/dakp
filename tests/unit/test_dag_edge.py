"""Edge-case tests for the DAG layer WITHOUT airflow installed.

Covers the pure ``dakp_build._ctx_from_params`` helper (quarter_limit/force overrides and the
``drugsfda_url`` download-config forwarding) and the ``dakp_pipeline.dags`` package helpers'
success paths (``get_dag`` returning the DAG object and ``run_dag`` forwarding ``run_conf``),
reached by monkeypatching the airflow-availability flag rather than installing airflow. The
Airflow-only task graph itself is pragma-excluded in the source (see ``dakp_build.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline import dags as dags_pkg
from dakp_pipeline.config import DownloadConfig, Profile
from dakp_pipeline.dags import dakp_build

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _params(workdir: Path, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {"workdir": str(workdir), "fixture_root": str(FIXTURE_ROOT), "profile": "mock", "log_level": "INFO"}
    base.update(extra)
    return base


def test_ctx_from_params_defaults(tmp_path: Path) -> None:
    ctx = dakp_build._ctx_from_params(_params(tmp_path))
    assert ctx.profile == "mock"
    assert ctx.workdir == tmp_path
    assert ctx.fixture_root == FIXTURE_ROOT
    # No drugsfda_url on the default mock profile -> not forwarded (the False branch).
    assert "drugsfda_url" not in ctx.params


def test_ctx_from_params_none_params_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # default workdir "data" is created relative to cwd
    ctx = dakp_build._ctx_from_params(None)
    assert ctx.profile == "mock"
    assert ctx.workdir == Path("data")  # the default is a relative, unresolved path
    assert (tmp_path / "data").is_dir()


def test_ctx_from_params_applies_quarter_limit_and_force_overrides(tmp_path: Path) -> None:
    ctx = dakp_build._ctx_from_params(_params(tmp_path, quarter_limit="5", force="1"))
    assert ctx.params["quarter_limit"] == 5
    assert ctx.params["force"] is True


def test_ctx_from_params_forwards_drugsfda_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = Profile(
        name="mock", threads=1, memory_budget_gb=1, mock_sources=True, download=DownloadConfig(drugsfda_url="https://example.invalid/x.zip")
    )
    monkeypatch.setattr(dakp_build, "load_profile", lambda _name, **_overrides: custom)
    ctx = dakp_build._ctx_from_params(_params(tmp_path))
    assert ctx.params["drugsfda_url"] == "https://example.invalid/x.zip"


def test_get_dag_returns_object_when_airflow_present(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(dakp_build, "_AIRFLOW_AVAILABLE", True)
    monkeypatch.setattr(dakp_build, "dag_obj", sentinel)
    assert dags_pkg.get_dag() is sentinel


def test_run_dag_forwards_run_conf(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object] | None] = []

    class _FakeDag:
        def test(self, run_conf: dict[str, object] | None = None) -> str:
            calls.append(run_conf)
            return "ran"

    monkeypatch.setattr(dags_pkg, "get_dag", lambda: _FakeDag())
    assert dags_pkg.run_dag() == "ran"
    assert dags_pkg.run_dag(profile="sample") == "ran"
    # No params -> dag.test() with no run_conf; params -> dag.test(run_conf=params).
    assert calls == [None, {"profile": "sample"}]
