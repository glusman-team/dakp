"""Edge-case tests for ``dakp_pipeline.workers.go_runner``.

Covers the branches the toolchain-free unit tests and the Go parity integration tests miss:
``GoWorkerError`` construction (stderr-tail truncation), ``GoResult.summary`` invalid-JSON,
the ``DAKP_GO_CACHE`` override, ``_relay_slog`` blank/non-dict lines, the cached-binary reuse
path, a failed ``go build``, the *real* subprocess ``_exec`` path (streaming a non-JSON stderr
line plus a non-zero exit) driven by a tiny ``sys.executable`` script (no Go toolchain needed),
the lazy ``get_runner`` construction, the ``stage_inputs`` hardlink-fallback, and the
``go_warnings`` skip/unconvertible branches.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.workers import go_runner
from dakp_pipeline.workers.go_runner import GoResult, GoRunner, GoUnavailableError, GoWorkerError, go_warnings, hash_go_sources, stage_inputs

_B3 = "b3:" + "a" * 64


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3="b3:" + "0" * 64, media_type="application/octet-stream")


def _executable(tmp_path: Path, name: str, body_lines: list[str]) -> Path:
    script = tmp_path / name
    script.write_text("\n".join([f"#!{sys.executable}", *body_lines, ""]), encoding="utf-8")
    script.chmod(0o755)
    return script


# --- GoWorkerError + GoResult parsing --------------------------------------------


def test_go_worker_error_carries_context_and_truncates_tail() -> None:
    lines = [f"line{i}" for i in range(30)]
    err = GoWorkerError("faers", 2, "\n".join(lines))
    assert err.subcommand == "faers"
    assert err.returncode == 2
    assert "dakp-worker faers exited 2" in str(err)
    assert "line29" in str(err)  # most recent lines kept
    assert "line0" not in str(err)  # older lines truncated to the last 20


def test_go_result_summary_invalid_json_returns_none() -> None:
    assert GoResult("drugsfda", 0, "{not valid json", "", logs=()).summary is None


# --- binary discovery / cache ----------------------------------------------------


def test_default_cache_dir_honors_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(go_runner.ENV_CACHE_DIR, str(tmp_path / "custom-cache"))
    assert go_runner._default_cache_dir() == tmp_path / "custom-cache"


def test_ensure_binary_reuses_cached_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(go_runner.ENV_BINARY, raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/fake/go")  # Go "present"
    go_dir = tmp_path / "go"
    go_dir.mkdir()
    (go_dir / "go.mod").write_text("module dakp-worker\n")
    (go_dir / "main.go").write_text("package main\n")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / f"dakp-worker-{hash_go_sources(go_dir)}"
    cached.write_bytes(b"#!/bin/sh\n")

    runner = GoRunner(go_dir=go_dir, cache_dir=cache_dir)
    assert runner.ensure_binary() == cached  # cached binary present -> no build


def test_ensure_binary_build_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(go_runner.ENV_BINARY, raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/fake/go")
    go_dir = tmp_path / "go"
    go_dir.mkdir()
    (go_dir / "go.mod").write_text("module x\n")
    runner = GoRunner(go_dir=go_dir, cache_dir=tmp_path / "cache")
    failed = subprocess.CompletedProcess(args=["go"], returncode=1, stdout="", stderr="compile error")
    monkeypatch.setattr(go_runner.subprocess, "run", lambda *a, **k: failed)
    with pytest.raises(GoUnavailableError, match="go build failed"):
        runner.ensure_binary()


def test_build_success_returns_normally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = GoRunner(go_dir=tmp_path, cache_dir=tmp_path / "cache")
    ok = subprocess.CompletedProcess(args=["go"], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(go_runner.subprocess, "run", lambda *a, **k: ok)
    runner._build(tmp_path / "worker")  # returncode 0 -> no raise (the success branch)


# --- real subprocess _exec (no Go toolchain required) ----------------------------


def test_exec_streams_stderr_and_parses_stdout(tmp_path: Path) -> None:
    script = _executable(
        tmp_path,
        "ok-worker",
        [
            "import sys",
            'print("plain non-json line", file=sys.stderr)',
            'print(\'{"level": "INFO", "msg": "hi", "warnings": 3}\', file=sys.stderr)',
            f'print("{_B3}")',
        ],
    )
    result = GoRunner(binary=script).run("hash", ["arg1"])
    assert result.ok
    assert result.artifact_id == _B3
    # Only the JSON stderr line is parsed into logs; the plain line was relayed, not captured.
    assert [record["msg"] for record in result.logs] == ["hi"]
    assert "plain non-json line" in result.stderr
    assert go_warnings(result) == 3


def test_exec_raises_go_worker_error_on_nonzero_exit(tmp_path: Path) -> None:
    script = _executable(tmp_path, "fail-worker", ["import sys", 'print("boom", file=sys.stderr)', "sys.exit(3)"])
    with pytest.raises(GoWorkerError) as exc_info:
        GoRunner(binary=script).run("hash")
    assert exc_info.value.returncode == 3
    assert exc_info.value.subcommand == "hash"
    assert "boom" in exc_info.value.stderr


# --- stderr relay ----------------------------------------------------------------


def test_relay_slog_blank_and_non_dict_return_none() -> None:
    log = go_runner.bind(task_id="edge")
    assert go_runner._relay_slog("", log) is None
    assert go_runner._relay_slog("   ", log) is None
    assert go_runner._relay_slog("[1, 2, 3]", log) is None  # JSON array, not an object
    assert go_runner._relay_slog('"bare string"', log) is None  # JSON scalar


# --- module accessor + helpers ---------------------------------------------------


def test_get_runner_builds_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(go_runner, "_RUNNER", None)
    assert isinstance(go_runner.get_runner(), GoRunner)


def test_stage_inputs_copies_when_hardlink_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "DEMO24Q3.txt"
    src.write_text("data")

    def no_link(*_a: object, **_k: object) -> None:
        msg = "cross-device link"
        raise OSError(msg)

    monkeypatch.setattr(go_runner.os, "link", no_link)
    stage = stage_inputs([_ref(src)], tmp_path / "stage")
    assert (stage / "DEMO24Q3.txt").read_text() == "data"


def test_go_warnings_skips_records_without_the_field() -> None:
    # reversed: {"other": 1} (no "warnings" -> loop continues) then {"warnings": 5}.
    result = GoResult("x", 0, "", "", logs=({"warnings": 5}, {"other": 1}))
    assert go_warnings(result) == 5


def test_go_warnings_continues_past_unconvertible_values() -> None:
    assert go_warnings(GoResult("x", 0, "", "", logs=({"warnings": "abc"},))) == 0  # ValueError
    assert go_warnings(GoResult("x", 0, "", "", logs=({"warnings": None},))) == 0  # TypeError
