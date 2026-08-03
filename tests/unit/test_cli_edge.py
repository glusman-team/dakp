"""Edge tests for the REAL side-effect bodies in :mod:`dakp_pipeline.cli`.

``test_cli.py`` monkeypatches the boundary functions to drive the orchestration branches; here we
exercise the actual bodies (real subprocesses, real sqlite, real signals), faking only the OS
process boundary where spawning a real Airflow would be absurd (``start_standalone``). Mirrors
``test_tablassert_run_edge.py``.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from dakp_pipeline import cli

# --- run_subprocess ---------------------------------------------------------------


def test_run_subprocess_captures_stdout() -> None:
    completed = cli.run_subprocess([sys.executable, "-c", "print('hello')"])
    assert completed.returncode == 0
    assert completed.stdout.strip() == "hello"


def test_run_subprocess_captures_nonzero_without_raising() -> None:
    completed = cli.run_subprocess([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert completed.returncode == 3


# --- airflow_importable -----------------------------------------------------------


def test_airflow_importable_true_when_installed() -> None:
    # Real body: spawns ``[sys.executable, "-c", "import airflow"]`` (a real import, a few seconds).
    assert cli.airflow_importable() is True  # apache-airflow is a hard DAKP dependency


# --- api_up -----------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_api_up_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse(200))
    assert cli.api_up("http://127.0.0.1:8090") is True


def test_api_up_false_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse(500))
    assert cli.api_up("http://127.0.0.1:8090") is False


def test_api_up_false_when_unreachable() -> None:
    # A real connection to a closed port raises -> the except branch -> False (no monkeypatch).
    assert cli.api_up("http://127.0.0.1:1") is False


# --- dag_registered / run_state (real sqlite) -------------------------------------


def _make_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE dag (dag_id TEXT)")
    con.execute("CREATE TABLE dag_run (dag_id TEXT, state TEXT, run_after INTEGER)")
    return con


def test_dag_registered_true_when_present(tmp_path: Path) -> None:
    db = tmp_path / "airflow.db"
    con = _make_db(db)
    con.execute("INSERT INTO dag VALUES ('dakp_build')")
    con.commit()
    con.close()
    assert cli.dag_registered(db, "dakp_build") is True
    assert cli.dag_registered(db, "other") is False


def test_dag_registered_false_on_unreadable_db(tmp_path: Path) -> None:
    # Pointing sqlite at a directory raises -> except branch -> False.
    assert cli.dag_registered(tmp_path, "dakp_build") is False


def test_run_state_returns_latest_state(tmp_path: Path) -> None:
    db = tmp_path / "airflow.db"
    con = _make_db(db)
    con.execute("INSERT INTO dag_run VALUES ('dakp_build', 'failed', 1)")
    con.execute("INSERT INTO dag_run VALUES ('dakp_build', 'success', 2)")  # most recent
    con.commit()
    con.close()
    assert cli.run_state(db, "dakp_build") == "success"


def test_run_state_none_when_no_rows(tmp_path: Path) -> None:
    db = tmp_path / "airflow.db"
    _make_db(db).close()
    assert cli.run_state(db, "dakp_build") == "none"


def test_run_state_none_when_state_null(tmp_path: Path) -> None:
    db = tmp_path / "airflow.db"
    con = _make_db(db)
    con.execute("INSERT INTO dag_run VALUES ('dakp_build', NULL, 1)")
    con.commit()
    con.close()
    assert cli.run_state(db, "dakp_build") == "none"


def test_run_state_none_on_unreadable_db(tmp_path: Path) -> None:
    assert cli.run_state(tmp_path, "dakp_build") == "none"


# --- start_standalone (real body; OS process faked) -------------------------------


class _FakePopen:
    last: _FakePopen | None = None

    def __init__(self, argv: list[str], stdout: object = None, stderr: object = None, env: object = None) -> None:
        self.argv = argv
        self.pid = 31337
        _FakePopen.last = self


def test_start_standalone_launches_airflow_and_returns_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    log = tmp_path / "standalone.log"

    pid = cli.start_standalone(log, {"AIRFLOW_HOME": str(tmp_path)})

    assert pid == 31337
    assert log.exists()  # the log file was opened for the child's stdout
    assert _FakePopen.last is not None
    assert _FakePopen.last.argv == ["uv", "run", "airflow", "standalone"]


# --- sleep / pid_alive / terminate (real) -----------------------------------------


def test_sleep_returns_none() -> None:
    assert cli.sleep(0) is None


def test_pid_alive_true_for_self() -> None:
    assert cli.pid_alive(os.getpid()) is True


def test_pid_alive_false_for_reaped_process() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert cli.pid_alive(proc.pid) is False


def test_terminate_stops_a_live_process() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    cli.terminate(proc.pid)
    assert proc.wait(timeout=10) is not None  # reaped (signalled)


def test_terminate_on_dead_pid_does_not_raise() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    cli.terminate(proc.pid)  # already gone -> except -> swallowed


# --- _tail ------------------------------------------------------------------------


def test_tail_prints_only_the_last_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "standalone.log"
    log.write_text("\n".join(f"line{i}" for i in range(50)), encoding="utf-8")
    cli._tail(log, lines=5)
    out = capsys.readouterr().out
    assert "line49" in out
    assert "line45" in out
    assert "line44" not in out


def test_tail_on_missing_file_is_a_noop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli._tail(tmp_path / "does-not-exist.log")
    assert capsys.readouterr().out == ""
