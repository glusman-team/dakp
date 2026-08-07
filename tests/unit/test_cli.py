"""Tests for the ``dakp`` CLI orchestrator (:mod:`dakp_pipeline.cli`).

Every side effect in ``cli`` goes through a monkeypatchable module function (``run_subprocess``,
``api_up``, ``dag_registered``, ``run_state``, ``start_standalone``, ``sleep``, ``pid_alive``,
``terminate``, ``_tail``), so these tests drive every orchestration branch with no real Airflow / Go
/ network — the same convention as ``test_tablassert_configs.py``. The real side-effect *bodies* are
exercised separately in ``test_cli_edge.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from dakp_pipeline import cli

# --- helpers ----------------------------------------------------------------------


def _bools(*values: bool):
    """A no-arg callable yielding ``values`` in order, then sticking on the last one."""
    seq = list(values)

    def next_value() -> bool:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return next_value


class FakeSubprocess:
    """Records every ``run_subprocess`` call; fails (rc=1) when a command contains a fail marker."""

    def __init__(self, fail_markers: tuple[str, ...] = (), stderr: str = "boom") -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self.fail_markers = fail_markers
        self.stderr = stderr

    def __call__(self, command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        self.envs.append(env)
        joined = " ".join(command)
        failed = any(marker in joined for marker in self.fail_markers)
        return subprocess.CompletedProcess(args=command, returncode=1 if failed else 0, stdout="", stderr=self.stderr if failed else "")

    def commands_containing(self, marker: str) -> list[list[str]]:
        return [call for call in self.calls if any(marker in part for part in call)]


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the hardcoded run locations at ``tmp_path`` and make poll loops instant."""
    monkeypatch.setattr(cli, "_DEFAULT_WORKDIR", tmp_path / "work")
    monkeypatch.setattr(cli, "_DEFAULT_AIRFLOW_HOME", tmp_path / "home")
    monkeypatch.setattr(cli, "_DEFAULT_FIXTURE_ROOT", tmp_path / "fixtures")
    monkeypatch.setattr(cli, "sleep", lambda seconds: None)
    return tmp_path


def _patch_happy(monkeypatch: pytest.MonkeyPatch, *, api_up: bool | Callable[[], bool] = True, run_state: str = "success") -> FakeSubprocess:
    """Monkeypatch the healthy-path boundaries; return the recording subprocess fake.

    ``api_up`` may be a bool (constant result) or a no-arg callable yielding a sequence; either way
    it is wrapped to accept the ``base_url`` argument the orchestrator passes.
    """
    monkeypatch.setattr(cli, "airflow_importable", lambda: True)
    if isinstance(api_up, bool):
        monkeypatch.setattr(cli, "api_up", lambda url: api_up)
    else:
        sequence = api_up
        monkeypatch.setattr(cli, "api_up", lambda url: sequence())
    monkeypatch.setattr(cli, "dag_registered", lambda db, dag_id: True)
    monkeypatch.setattr(cli, "run_state", lambda db, dag_id: run_state)
    fake = FakeSubprocess()
    monkeypatch.setattr(cli, "run_subprocess", fake)
    return fake


def _write_summary(tmp_path: Path) -> Path:
    summary = tmp_path / "work" / "data" / "reports" / "build_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text('{"ok": true}', encoding="utf-8")
    return summary


# --- up: happy paths --------------------------------------------------------------


def test_up_mock_happy_path_reuses_running_airflow(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = _patch_happy(monkeypatch)  # api_up True -> reuse
    _write_summary(sandbox)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 0
    out = capsys.readouterr().out
    assert "reusing" in out
    assert "SUCCESS" in out
    assert '{"ok": true}' in out  # build summary was printed
    # Orchestration ran the go pack + unpause + both pools + variables set + trigger.
    assert fake.commands_containing("airflow-go-pack")
    assert fake.commands_containing("unpause")
    assert len(fake.commands_containing("pools")) == 2
    assert fake.commands_containing("variables")
    assert fake.commands_containing("trigger")


def test_up_starts_airflow_when_not_running(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    started: list[int] = []
    monkeypatch.setattr(cli, "start_standalone", lambda log_path, env: started.append(1) or 4242)
    _patch_happy(monkeypatch, api_up=_bools(False, True))  # down once, then up
    _write_summary(sandbox)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 0
    assert started == [1]
    # The pidfile records the pid start_standalone returned.
    assert (sandbox / "home" / "standalone.pid").read_text(encoding="utf-8") == "4242"
    assert "starting Airflow standalone" in capsys.readouterr().out


def test_up_config_variable_carries_null_limits_and_fullmap(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    fake = _patch_happy(monkeypatch)

    cli.run_up(fullmap="/maps/fullmap.redb", port=8090, log_level="DEBUG", detach=False)

    (variables_call,) = fake.commands_containing("variables")
    config = json.loads(variables_call[-1])
    assert config["quarter_limit"] is None  # null => unbounded full build
    assert config["release_limit"] is None
    assert config["dailymed_max_age_days"] == 7  # a fresh stored release is reused for a week
    assert config["force"] is False
    assert config["threads"] == os.cpu_count()  # Go all-cores contract
    assert config["log_level"] == "DEBUG"
    assert config["fullmap"] == "/maps/fullmap.redb"
    assert config["workdir"] == str(sandbox / "work")
    # The machine-sizing / profile keys are gone from the Variable.
    assert "profile" not in config
    assert "memory_budget_gb" not in config


def test_up_small_sets_scope_bounds(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    fake = _patch_happy(monkeypatch)

    cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False, small=True)

    (variables_call,) = fake.commands_containing("variables")
    config = json.loads(variables_call[-1])
    assert config["quarter_limit"] == 1  # --small => ~1 FAERS quarter
    assert config["release_limit"] == 1  # --small => ~1 DailyMed release
    assert config["dailymed_max_age_days"] == 7  # freshness window is scope-independent
    assert config["threads"] == os.cpu_count()  # scope bound does NOT touch threads (Go contract)
    assert config["fullmap"] is None


def test_up_success_without_summary_file_still_succeeds(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_happy(monkeypatch)  # no summary file written

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 0
    assert "SUCCESS" in capsys.readouterr().out


def test_up_detach_returns_after_trigger(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    states: list[str] = []
    monkeypatch.setattr(cli, "run_state", lambda db, dag_id: states.append("polled") or "success")
    fake = _patch_happy(monkeypatch)
    monkeypatch.setattr(cli, "run_state", lambda db, dag_id: states.append("polled") or "success")

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=True)

    assert code == 0
    assert "detached" in capsys.readouterr().out
    assert fake.commands_containing("trigger")
    assert states == []  # never polled the run state


# --- up: preflight self-heal ------------------------------------------------------


def test_up_preflight_reinstall_heals(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, True))
    fake = _patch_happy(monkeypatch)
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, True))

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 0
    assert fake.commands_containing("--reinstall-package")
    assert not fake.commands_containing("cache")  # healed on the first reinstall, no cache clean


def test_up_preflight_cache_clean_heals(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, False, True))
    fake = _patch_happy(monkeypatch)
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, False, True))

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 0
    assert fake.commands_containing("cache")  # a corrupt uv cache was cleaned


def test_up_preflight_still_broken(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "airflow_importable", _bools(False))
    fake = _patch_happy(monkeypatch)
    fake.fail_markers = ("import airflow",)  # the bail diagnostic probe fails with stderr
    monkeypatch.setattr(cli, "airflow_importable", _bools(False))

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    out = capsys.readouterr().out
    assert code == 1
    assert "still fails to import" in out
    assert "boom" in out  # the actual import error is surfaced, not swallowed
    assert not fake.commands_containing("airflow-go-pack")  # bailed before the bundle step


def test_up_preflight_still_broken_without_import_stderr(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "airflow_importable", _bools(False))
    fake = FakeSubprocess(fail_markers=("import airflow",), stderr="")
    _patch_happy(monkeypatch)
    monkeypatch.setattr(cli, "run_subprocess", fake)
    monkeypatch.setattr(cli, "airflow_importable", _bools(False))

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    out = capsys.readouterr().out
    assert code == 1
    assert "still fails to import" in out


def test_up_preflight_heal_failure_prints_stderr_and_bounds_lock_wait(
    monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, True))
    fake = FakeSubprocess(fail_markers=("sync",), stderr="error: Timeout (60s) when waiting for lock on /home/u/.cache/uv")
    _patch_happy(monkeypatch)
    monkeypatch.setattr(cli, "run_subprocess", fake)
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, True))

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    out = capsys.readouterr().out
    assert code == 0  # the second probe passed; the failed heal step did not abort the run
    assert "heal step failed" in out
    assert "waiting for lock" in out  # a held cache lock is visible, never swallowed
    heal_env = fake.envs[0]
    assert heal_env is not None
    assert heal_env["UV_LOCK_TIMEOUT"] == str(cli._UV_HEAL_LOCK_TIMEOUT_SECONDS)


def test_up_preflight_heal_failure_without_stderr(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, True))
    fake = FakeSubprocess(fail_markers=("sync",), stderr="")
    _patch_happy(monkeypatch)
    monkeypatch.setattr(cli, "run_subprocess", fake)
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, True))

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    out = capsys.readouterr().out
    assert code == 0
    assert "heal step failed" in out
    assert "waiting for lock" not in out  # no stderr to echo


# --- up: failure paths ------------------------------------------------------------


def test_up_go_pack_failure_with_stderr(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeSubprocess(fail_markers=("airflow-go-pack",), stderr="go: build failed")
    monkeypatch.setattr(cli, "airflow_importable", lambda: True)
    monkeypatch.setattr(cli, "run_subprocess", fake)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 1
    out = capsys.readouterr().out
    assert "bundle pack failed" in out
    assert "go: build failed" in out


def test_up_go_pack_failure_without_stderr(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeSubprocess(fail_markers=("airflow-go-pack",), stderr="")
    monkeypatch.setattr(cli, "airflow_importable", lambda: True)
    monkeypatch.setattr(cli, "run_subprocess", fake)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 1
    assert "bundle pack failed" in capsys.readouterr().out


def test_up_api_never_comes_up(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tailed: list[Path] = []
    monkeypatch.setattr(cli, "airflow_importable", lambda: True)
    monkeypatch.setattr(cli, "api_up", lambda url: False)  # never up
    monkeypatch.setattr(cli, "start_standalone", lambda log_path, env: 1)
    monkeypatch.setattr(cli, "run_subprocess", FakeSubprocess())
    monkeypatch.setattr(cli, "_tail", lambda path, lines=40: tailed.append(path))

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 1
    assert "did not come up" in capsys.readouterr().out
    assert len(tailed) == 1  # the startup log was tailed for diagnostics


def test_up_dag_never_registers(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_happy(monkeypatch)
    monkeypatch.setattr(cli, "dag_registered", lambda db, dag_id: False)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 1
    assert "never registered" in capsys.readouterr().out


def test_up_trigger_failure_with_stderr(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeSubprocess(fail_markers=("dags trigger",), stderr="trigger exploded")
    monkeypatch.setattr(cli, "airflow_importable", lambda: True)
    monkeypatch.setattr(cli, "api_up", lambda url: True)
    monkeypatch.setattr(cli, "dag_registered", lambda db, dag_id: True)
    monkeypatch.setattr(cli, "run_subprocess", fake)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 1
    out = capsys.readouterr().out
    assert "trigger failed" in out
    assert "trigger exploded" in out


def test_up_trigger_failure_without_stderr(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeSubprocess(fail_markers=("dags trigger",), stderr="")
    monkeypatch.setattr(cli, "airflow_importable", lambda: True)
    monkeypatch.setattr(cli, "api_up", lambda url: True)
    monkeypatch.setattr(cli, "dag_registered", lambda db, dag_id: True)
    monkeypatch.setattr(cli, "run_subprocess", fake)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 1
    assert "trigger failed" in capsys.readouterr().out


def test_up_run_failed(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_happy(monkeypatch, run_state="failed")

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 1
    assert "final=failed" in capsys.readouterr().out


def test_up_run_times_out(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_happy(monkeypatch, run_state="running")  # never success/failed -> exhausts the poll budget

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 1
    assert "final=timeout" in capsys.readouterr().out


# --- down -------------------------------------------------------------------------


def test_down_without_pidfile(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = _patch_happy(monkeypatch)

    code = cli.run_down()

    assert code == 0
    assert "Airflow stopped" in capsys.readouterr().out
    assert fake.commands_containing("pkill")  # catch-all reap still runs


def test_down_stops_alive_pid(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    pidfile = sandbox / "home" / "standalone.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text("4242", encoding="utf-8")
    terminated: list[int] = []
    monkeypatch.setattr(cli, "pid_alive", lambda pid: True)
    monkeypatch.setattr(cli, "terminate", lambda pid: terminated.append(pid))
    monkeypatch.setattr(cli, "run_subprocess", FakeSubprocess())

    code = cli.run_down()

    assert code == 0
    assert terminated == [4242]
    assert not pidfile.exists()  # pidfile removed


def test_down_skips_dead_pid(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    pidfile = sandbox / "home" / "standalone.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text("4242", encoding="utf-8")
    terminated: list[int] = []
    monkeypatch.setattr(cli, "pid_alive", lambda pid: False)
    monkeypatch.setattr(cli, "terminate", lambda pid: terminated.append(pid))
    monkeypatch.setattr(cli, "run_subprocess", FakeSubprocess())

    code = cli.run_down()

    assert code == 0
    assert terminated == []  # dead pid is not signalled
    assert not pidfile.exists()


def test_down_ignores_non_numeric_pidfile(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    pidfile = sandbox / "home" / "standalone.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text("not-a-pid", encoding="utf-8")
    terminated: list[int] = []
    monkeypatch.setattr(cli, "pid_alive", lambda pid: terminated.append(pid) or True)
    monkeypatch.setattr(cli, "terminate", lambda pid: terminated.append(pid))
    monkeypatch.setattr(cli, "run_subprocess", FakeSubprocess())

    code = cli.run_down()

    assert code == 0
    assert terminated == []  # non-numeric pidfile is skipped
    assert not pidfile.exists()


# --- clean ------------------------------------------------------------------------


def test_clean_removes_expected_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path)
    # A mix of dirs, a file, and an absent target.
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / "tmp").mkdir()
    (tmp_path / ".coverage").write_text("x", encoding="utf-8")
    # __pycache__ dirs: one outside .venv (removed), one inside .venv (kept).
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "__pycache__").mkdir(parents=True)
    (tmp_path / "go").mkdir()
    (tmp_path / "go" / "dakp-worker").write_text("bin", encoding="utf-8")

    code = cli.run_clean()

    assert code == 0
    assert "cleaned" in capsys.readouterr().out
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "tmp").exists()
    assert not (tmp_path / ".coverage").exists()
    assert not (tmp_path / "src" / "__pycache__").exists()
    assert (tmp_path / ".venv" / "lib" / "__pycache__").exists()  # .venv is never touched
    assert not (tmp_path / "go" / "dakp-worker").exists()


# --- cyclopts command wrappers (exit codes) ---------------------------------------


def test_clean_command_raises_systemexit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        cli.clean()
    assert excinfo.value.code == 0


def test_down_command_raises_systemexit(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    monkeypatch.setattr(cli, "run_subprocess", FakeSubprocess())
    with pytest.raises(SystemExit) as excinfo:
        cli.down()
    assert excinfo.value.code == 0


def test_up_command_raises_systemexit(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    _patch_happy(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        cli.up()
    assert excinfo.value.code == 0


def test_up_command_surfaces_failure_code(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    monkeypatch.setattr(cli, "airflow_importable", lambda: True)
    monkeypatch.setattr(cli, "run_subprocess", FakeSubprocess(fail_markers=("airflow-go-pack",)))
    with pytest.raises(SystemExit) as excinfo:
        cli.up()  # the Go bundle pack fails -> run_up returns 1, surfaced as the exit code
    assert excinfo.value.code == 1
