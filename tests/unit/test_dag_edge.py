"""Edge-case tests for the config-driven run context + DAG package helpers (Airflow always present).

Covers ``runtime.build_context_from_config`` (the ``dakp_config`` Variable -> TaskContext path the
DAG tasks use): quarter_limit/force overrides, ``drugsfda_url`` download-config forwarding, and
workdir creation. Also the ``dakp_pipeline.dags`` package helpers (``get_dag`` / ``airflow_available``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline import dags as dags_pkg
from dakp_pipeline import runtime
from dakp_pipeline.config import DownloadConfig, Profile
from dakp_pipeline.dags import dakp_build
from dakp_pipeline.runtime import build_context_from_config

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _cfg(workdir: Path, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {"workdir": str(workdir), "fixture_root": str(FIXTURE_ROOT), "profile": "mock", "log_level": "INFO"}
    base.update(extra)
    return base


def test_build_context_from_config_defaults(tmp_path: Path) -> None:
    ctx = build_context_from_config(_cfg(tmp_path))
    assert ctx.profile == "mock"
    assert ctx.workdir == tmp_path
    assert ctx.fixture_root == FIXTURE_ROOT
    # No drugsfda_url on the default mock profile -> not forwarded (the False branch).
    assert "drugsfda_url" not in ctx.params


def test_build_context_from_config_applies_overrides(tmp_path: Path) -> None:
    ctx = build_context_from_config(_cfg(tmp_path, quarter_limit="5", force="1", release_limit="2"))
    assert ctx.params["quarter_limit"] == 5
    assert ctx.params["force"] is True
    assert ctx.params["release_limit"] == 2


def test_build_context_from_config_forwards_drugsfda_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = Profile(
        name="mock", threads=1, memory_budget_gb=1, mock_sources=True, download=DownloadConfig(drugsfda_url="https://example.invalid/x.zip")
    )
    monkeypatch.setattr(runtime, "load_profile", lambda _name, **_overrides: custom)
    ctx = build_context_from_config(_cfg(tmp_path))
    assert ctx.params["drugsfda_url"] == "https://example.invalid/x.zip"


def test_get_dag_returns_dag_object() -> None:
    assert dags_pkg.get_dag() is dakp_build.dag_obj


def test_airflow_available_true() -> None:
    assert dags_pkg.airflow_available() is True
