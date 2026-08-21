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

import polars as pl
import pytest

from dakp_pipeline import cli
from dakp_pipeline.paths import Workdir

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"

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
    # Orchestration ran the go pack + nercache build + unpause + all three pools + variables set + trigger.
    assert fake.commands_containing("airflow-go-pack")
    assert fake.commands_containing("dakp-nercache")
    assert fake.commands_containing("unpause")
    assert len(fake.commands_containing("pools")) == 3
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
    assert config["release"] is True  # `tablassert build-kg --release` (slim significant-only graph)
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
    assert config["release"] is True  # release mode is scope-independent too
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
    assert not fake.commands_containing("clean")  # healed on the first reinstall, no uv cache clean


def test_up_preflight_cache_clean_heals(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, False, True))
    fake = _patch_happy(monkeypatch)
    monkeypatch.setattr(cli, "airflow_importable", _bools(False, False, True))

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 0
    assert fake.commands_containing("clean")  # a corrupt uv cache was cleaned (uv cache clean)


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


# --- env: UI customization (plans/airflow-ui-optimizations.md) -------------------


def test_airflow_env_carries_ui_customization(sandbox: Path) -> None:
    """The Airflow env identifies the UI and wraps task logs by default."""
    env = cli._airflow_env(sandbox / "home", sandbox / "home" / "executable-bundles", port=8090)

    assert env["AIRFLOW__API__INSTANCE_NAME"] == "DAKP"
    assert env["AIRFLOW__API__DEFAULT_WRAP"] == "True"
    # Keep the custom color palette from being reintroduced accidentally.
    assert "AIRFLOW__API__THEME" not in env


# --- up: dakp-nercache build (optional) --------------------------------------------


def test_up_nercache_build_failure_is_non_fatal(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A failed dakp-nercache build only disables mention caching; the run proceeds."""
    fake = _patch_happy(monkeypatch)
    fake.fail_markers = ("dakp-nercache",)
    _write_summary(sandbox)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 0
    out = capsys.readouterr().out
    assert "dakp-nercache build failed" in out
    assert "boom" in out  # the build's stderr tail was printed
    assert "SUCCESS" in out


def test_up_nercache_build_failure_without_stderr(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A failed nercache build with EMPTY stderr prints the failure line and nothing more."""
    fake = _patch_happy(monkeypatch)
    fake.fail_markers = ("dakp-nercache",)
    fake.stderr = ""
    _write_summary(sandbox)

    code = cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False)

    assert code == 0
    assert "dakp-nercache build failed" in capsys.readouterr().out


def test_up_builds_nercache_into_workdir_bin(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    """The mention-cache server lands at <workdir>/bin/dakp-nercache (the client's default lookup)."""
    fake = _patch_happy(monkeypatch)
    _write_summary(sandbox)

    assert cli.run_up(fullmap=None, port=8090, log_level="INFO", detach=False) == 0
    (build,) = [call for call in fake.commands_containing("dakp-nercache") if "build" in call]
    assert str(sandbox / "work" / "bin" / "dakp-nercache") in build


# --- cache clear --------------------------------------------------------------------


def test_cache_clear_removes_the_store(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = sandbox / "work" / "cache" / "ner"
    cache_dir.mkdir(parents=True)
    (cache_dir / "LOCK").write_bytes(b"")

    assert cli.run_cache_clear() == 0
    assert not cache_dir.exists()
    assert "cleared" in capsys.readouterr().out


def test_cache_clear_without_cache_is_a_noop(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.run_cache_clear() == 0
    assert "no NER mention cache" in capsys.readouterr().out


def test_cache_clear_stops_a_live_server_first(monkeypatch: pytest.MonkeyPatch, sandbox: Path) -> None:
    cache_dir = sandbox / "work" / "cache" / "ner"
    cache_dir.mkdir(parents=True)
    (cache_dir / "server.json").write_text(json.dumps({"pid": 4242, "port": 9999}), encoding="utf-8")
    stopped: list[int] = []
    alive = iter([True, False])  # alive, then gone after SIGTERM
    monkeypatch.setattr(cli, "pid_alive", lambda pid: next(alive))
    monkeypatch.setattr(cli, "terminate", stopped.append)

    assert cli.run_cache_clear() == 0
    assert stopped == [4242]
    assert not cache_dir.exists()


def test_cache_clear_tolerates_a_corrupt_server_json(monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An unreadable server.json means no live server to stop — the store is simply deleted."""
    cache_dir = sandbox / "work" / "cache" / "ner"
    cache_dir.mkdir(parents=True)
    (cache_dir / "server.json").write_text("not json", encoding="utf-8")
    terminated: list[int] = []
    monkeypatch.setattr(cli, "terminate", terminated.append)

    assert cli.run_cache_clear() == 0
    assert terminated == []  # pid probe failed -> nothing to stop
    assert not cache_dir.exists()
    assert "cleared" in capsys.readouterr().out


def test_cache_clear_refuses_when_the_server_survives_sigterm(
    monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = sandbox / "work" / "cache" / "ner"
    cache_dir.mkdir(parents=True)
    (cache_dir / "server.json").write_text(json.dumps({"pid": 4242, "port": 9999}), encoding="utf-8")
    monkeypatch.setattr(cli, "pid_alive", lambda pid: True)  # never dies
    monkeypatch.setattr(cli, "terminate", lambda pid: None)

    assert cli.run_cache_clear() == 1
    assert cache_dir.exists()  # untouched
    assert "refusing" in capsys.readouterr().out


def test_cache_clear_command_raises_systemexit(sandbox: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.clear()
    assert excinfo.value.code == 0


# --- export-medliner (MEDliNER training-data bundle) ------------------------------


def _write_interim_tables(workdir: Path) -> None:
    """Materialize the two interim tables the export consumes (minimal but real parquets)."""
    interim = Workdir(workdir).interim
    (interim / "dailymed").mkdir(parents=True, exist_ok=True)
    (interim / "faers").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "spl_document_id": ["set-1", "set-2"],
            "loinc_code": ["34070-3", "34067-9"],
            "section_text": ["Contraindicated in patients with active liver disease.", "For the treatment of headache."],
        }
    ).write_parquet(interim / "dailymed" / "spl_documents.parquet")
    pl.DataFrame(
        {
            "quarter": ["24Q3", "24Q3"],
            "primaryid": ["1001", "1002"],
            "drugname": ["EXAMPLESTATIN", "EXAMPLESTATIN"],
            "indication": ["headache", "hypercholesterolemia"],
        }
    ).write_parquet(interim / "faers" / "cases.parquet")


def test_export_medliner_happy_path_copies_to_out(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Default mode exports from a materialized workdir — no acquisition, no downloads."""
    workdir = sandbox / "materialized"
    _write_interim_tables(workdir)
    out = sandbox / "bundle"

    code = cli.run_export_medliner(out=str(out), workdir=str(workdir))

    assert code == 0
    assert "bundle ready" in capsys.readouterr().out
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "dakp.medliner.export.v1"
    # 2 DailyMed sections (one per export LOINC) + 2 FAERS indications.
    assert manifest["files"]["candidates.jsonl"]["rows"] == 4
    assert manifest["task_counts"] == {"contraindication": 1, "indication": 3}
    assert manifest["family_counts"] == {"dailymed": 2, "faers": 2}
    # The store copy (where the export stage writes) exists alongside the --out copy.
    assert (workdir / "data" / "store" / "medliner-export" / "manifest.json").exists()


def test_export_medliner_default_out_is_the_store_bundle(sandbox: Path) -> None:
    """Without --out/--workdir the bundle stays where the export stage wrote it (no copy)."""
    _write_interim_tables(sandbox / "work")  # the sandbox fixture points _DEFAULT_WORKDIR here

    code = cli.run_export_medliner()

    assert code == 0
    bundle = sandbox / "work" / "data" / "store" / "medliner-export"
    assert sorted(path.name for path in bundle.iterdir()) == ["candidates.jsonl", "manifest.json", "ner_gold.json"]


def test_export_medliner_missing_interim_tables_fail_loudly(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An unmaterialized workdir is a loud error naming BOTH tables — never a download."""
    code = cli.run_export_medliner()

    assert code == 1
    out = capsys.readouterr().out
    assert "!!!" in out
    assert "dailymed/spl_documents.parquet" in out
    assert "faers/cases.parquet" in out
    assert "never downloads" in out
    assert not (sandbox / "work" / "data" / "store" / "medliner-export").exists()  # no partial bundle


def test_export_medliner_names_only_the_missing_table(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When one interim table exists, the error names exactly the one that is missing."""
    interim = sandbox / "work" / "data" / "interim" / "dailymed"
    interim.mkdir(parents=True)
    (interim / "spl_documents.parquet").write_bytes(b"payload")

    code = cli.run_export_medliner()

    assert code == 1
    out = capsys.readouterr().out
    assert "faers/cases.parquet" in out
    assert out.count(".parquet") == 1  # only the missing table is named


def test_export_medliner_fixtures_runs_reference_extractors_offline(
    monkeypatch: pytest.MonkeyPatch, sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--fixtures runs the REAL reference extractors over the committed pipeline fixtures.

    WHY: the offline path must produce the identical bundle shape a materialized workdir does,
    with no network — it is also the documented way to regenerate MEDliNER's sample bundle.
    """
    monkeypatch.setattr(cli, "_DEFAULT_FIXTURE_ROOT", _FIXTURE_ROOT)  # restore the real fixtures

    code = cli.run_export_medliner(fixtures=True)

    assert code == 0
    assert "bundle ready" in capsys.readouterr().out
    bundle = sandbox / "work" / "data" / "store" / "medliner-export"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "dakp.medliner.export.v1"
    assert manifest["files"]["candidates.jsonl"]["rows"] > 0
    assert manifest["family_counts"]["dailymed"] > 0
    assert manifest["family_counts"]["faers"] > 0
    # The reference extractors materialized the interim layer on the way through.
    assert (sandbox / "work" / "data" / "interim" / "dailymed" / "spl_documents.parquet").exists()
    assert (sandbox / "work" / "data" / "interim" / "faers" / "cases.parquet").exists()


def test_export_medliner_fixtures_missing_fixture_raises_loudly(sandbox: Path) -> None:
    """A missing fixture file surfaces as a loud FileNotFoundError from ctx.fixture.

    The sandbox points _DEFAULT_FIXTURE_ROOT at an absent dir; the error must propagate rather
    than producing a silently empty bundle.
    """
    with pytest.raises(FileNotFoundError, match="fixture not found"):
        cli.run_export_medliner(fixtures=True)


def test_copy_export_bundle_overwrites_only_the_known_files(tmp_path: Path) -> None:
    """An already-populated --out dir: the three bundle files overwrite; nothing else is touched."""
    src = tmp_path / "src"
    src.mkdir()
    for name, body in (("manifest.json", "{}"), ("candidates.jsonl", '{"a": 1}\n'), ("ner_gold.json", '{"gold": true}')):
        (src / name).write_text(body, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "unrelated.txt").write_text("keep me", encoding="utf-8")
    (out / "manifest.json").write_text("stale", encoding="utf-8")

    paths = cli.copy_export_bundle(src, out)

    assert set(paths) == {"manifest.json", "candidates.jsonl", "ner_gold.json"}
    assert (out / "manifest.json").read_text(encoding="utf-8") == "{}"
    assert (out / "unrelated.txt").read_text(encoding="utf-8") == "keep me"


def test_export_medliner_command_raises_systemexit(sandbox: Path) -> None:
    _write_interim_tables(sandbox / "work")
    with pytest.raises(SystemExit) as excinfo:
        cli.export_medliner()
    assert excinfo.value.code == 0
