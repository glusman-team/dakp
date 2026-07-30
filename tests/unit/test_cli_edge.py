"""Edge-case tests for the CLI ``run`` command body (``dakp_pipeline.cli.main``).

The existing ``test_cli.py`` covers parser construction, ``--help``, and the mock-profile
fixture-root guard. These tests exercise the *success* body of ``main`` (the param-override
forwarding and the build-summary printing) by monkeypatching ``cli.run_pipeline`` — the
documented seam — so every override branch is covered without a full pipeline run, plus one
faithful end-to-end invocation against the tiny mock fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline import cli
from dakp_pipeline.config import load_profile
from dakp_pipeline.paths import Workdir
from dakp_pipeline.pipeline import PipelineResult

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _fake_result(workdir: Path, build_summary: Path | None) -> PipelineResult:
    return PipelineResult(workdir=Workdir(workdir), profile=load_profile("mock"), tables={}, build_summary=build_summary)


def test_cli_run_forwards_all_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs: object) -> PipelineResult:
        captured.update(kwargs)
        return _fake_result(tmp_path, tmp_path / "build_summary.json")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    exit_code = cli.main(
        [
            "run",
            "--profile",
            "mock",
            "--fixture-root",
            str(FIXTURE_ROOT),
            "--workdir",
            str(tmp_path),
            "--quarter-limit",
            "2",
            "--release-limit",
            "3",
            "--force",
            "--run-airflow",
        ]
    )

    assert exit_code == 0
    assert captured["profile"] == "mock"
    assert captured["fixture_root"] == Path(FIXTURE_ROOT)
    assert captured["workdir"] == Path(tmp_path)
    assert captured["run_airflow"] is True
    # All three scope/force overrides are merged into the forwarded params dict.
    assert captured["params"] == {"quarter_limit": 2, "release_limit": 3, "force": True}
    out = capsys.readouterr().out
    assert "Pipeline complete" in out
    assert "Build summary" in out


def test_cli_run_no_overrides_forwards_none_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs: object) -> PipelineResult:
        captured.update(kwargs)
        return _fake_result(tmp_path, tmp_path / "build_summary.json")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    assert cli.main(["run", "--profile", "sample", "--workdir", str(tmp_path)]) == 0

    assert captured["profile"] == "sample"
    assert captured["fixture_root"] is None
    assert captured["run_airflow"] is False
    # No overrides -> empty params dict -> forwarded as None (params or None).
    assert captured["params"] is None


def test_cli_run_without_build_summary_prints_only_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "run_pipeline", lambda **_kwargs: _fake_result(tmp_path, None))
    assert cli.main(["run", "--profile", "sample", "--workdir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "Pipeline complete" in out
    assert "Build summary" not in out


def test_cli_run_unknown_profile_is_an_argparse_error() -> None:
    # `choices` validation rejects an unknown profile before main() dispatches.
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--profile", "bogus"])
    assert exc_info.value.code == 2


def test_cli_run_mock_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A faithful run through the CLI entrypoint on the tiny mock fixtures (no monkeypatching)."""
    exit_code = cli.main(["run", "--profile", "mock", "--fixture-root", str(FIXTURE_ROOT), "--workdir", str(tmp_path)])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "Pipeline complete" in out
    assert "Build summary" in out
    assert (tmp_path / "data" / "reports" / "build_summary.json").exists()
