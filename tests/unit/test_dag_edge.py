"""Edge-case tests for the config-driven run context + DAG package helpers (Airflow always present).

Covers ``runtime.build_context_from_config`` (the ``dakp_config`` Variable -> TaskContext path the
DAG tasks use): quarter_limit/force overrides, ``drugsfda_url`` forwarding, the fullmap-derived
``run_tablassert`` trigger, and workdir creation. Also the ``dakp_pipeline.dags`` package helpers
(``get_dag`` / ``airflow_available``).
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline import dags as dags_pkg
from dakp_pipeline.runtime import build_context_from_config

# ``dakp_build`` (and ``import airflow``) arrives via the lazy ``dakp_build`` fixture (conftest.py).

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _cfg(workdir: Path, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {"workdir": str(workdir), "fixture_root": str(FIXTURE_ROOT), "log_level": "INFO"}
    base.update(extra)
    return base


def test_build_context_from_config_defaults(tmp_path: Path) -> None:
    ctx = build_context_from_config(_cfg(tmp_path))
    assert ctx.workdir == tmp_path
    assert ctx.fixture_root == FIXTURE_ROOT
    # No fullmap -> the real Tablassert handoff is not triggered; no drugsfda_url forwarded.
    assert ctx.params["run_tablassert"] is False
    assert ctx.params["release"] is False  # Tablassert --release defaults off unless the config sets it
    assert "drugsfda_url" not in ctx.params
    assert "fullmap" not in ctx.params


def test_build_context_from_config_forwards_release(tmp_path: Path) -> None:
    ctx = build_context_from_config(_cfg(tmp_path, release=True))
    assert ctx.params["release"] is True


def test_build_context_from_config_applies_overrides(tmp_path: Path) -> None:
    ctx = build_context_from_config(_cfg(tmp_path, quarter_limit="5", force="1", release_limit="2"))
    assert ctx.params["quarter_limit"] == 5
    assert ctx.params["force"] is True
    assert ctx.params["release_limit"] == 2


def test_build_context_from_config_forwards_drugsfda_url(tmp_path: Path) -> None:
    ctx = build_context_from_config(_cfg(tmp_path, drugsfda_url="https://example.invalid/x.zip"))
    assert ctx.params["drugsfda_url"] == "https://example.invalid/x.zip"


def test_build_context_from_config_resolves_relative_fullmap(tmp_path: Path) -> None:
    """A relative fullmap path is resolved to absolute (anchored to the caller's CWD)."""
    ctx = build_context_from_config(_cfg(tmp_path, fullmap="some/relative/fullmap"))
    resolved = ctx.params["fullmap"]
    assert Path(resolved).is_absolute()
    assert resolved.endswith("some/relative/fullmap")


def test_build_context_from_config_keeps_absolute_fullmap(tmp_path: Path) -> None:
    """An absolute fullmap path is passed through unchanged."""
    abs_path = str((tmp_path / "deep" / "fullmap").resolve())
    ctx = build_context_from_config(_cfg(tmp_path, fullmap=abs_path))
    assert ctx.params["fullmap"] == abs_path


def test_get_dag_returns_dag_object(dakp_build) -> None:
    assert dags_pkg.get_dag() is dakp_build.dag_obj


def test_airflow_available_true() -> None:
    assert dags_pkg.airflow_available() is True
