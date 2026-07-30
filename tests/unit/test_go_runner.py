"""Unit tests for the Go worker runner (``dakp_pipeline.workers.go_runner``).

Covers the pieces that need no Go toolchain: ``go_available()`` detection (and that it is
monkeypatchable), ``GoResult`` stdout parsing (``b3:<hex>`` artifact id vs JSON summary),
``should_use_go()`` gating, the source-hash build-cache key, input staging, TSV read-back, and
the full :class:`MockGoRunner` contract (subcommand routing, stdout/stderr handling, and the
Go-missing error). The real-binary build/run path is exercised in ``tests/integration/test_go_parity.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.workers import go_runner
from dakp_pipeline.workers.go_runner import (
    GoResult,
    GoRunner,
    GoUnavailableError,
    MockGoRunner,
    go_available,
    go_warnings,
    hash_go_sources,
    read_go_tsv,
    should_use_go,
    stage_inputs,
)


def _ctx(use_go: bool | None = None) -> TaskContext:
    params: dict[str, object] = {}
    if use_go is not None:
        params["use_go_workers"] = use_go
    return TaskContext(profile="mock", workdir=Path("work"), fixture_root=None, threads=1, memory_budget_gb=1, params=params)


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3="b3:" + "0" * 64, media_type="application/octet-stream")


# --- go_available detection ------------------------------------------------------


def test_go_available_returns_bool() -> None:
    assert isinstance(go_available(), bool)


def test_go_available_true_with_prebuilt_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "dakp-worker"
    binary.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setenv(go_runner.ENV_BINARY, str(binary))
    # Even with no `go` on PATH, a configured prebuilt binary makes Go "available".
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert go_available() is True


def test_go_available_false_without_go_or_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(go_runner.ENV_BINARY, raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert go_available() is False


def test_go_available_is_monkeypatchable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module function is the documented test seam (patched directly)."""
    monkeypatch.setattr(go_runner, "go_available", lambda: False)
    assert go_runner.go_available() is False
    monkeypatch.setattr(go_runner, "go_available", lambda: True)
    assert go_runner.go_available() is True


# --- GoResult stdout parsing -----------------------------------------------------


def test_go_result_artifact_id_from_b3_line() -> None:
    result = GoResult(subcommand="dailymed", returncode=0, stdout="b3:" + "ab" * 32, stderr="", logs=())
    assert result.ok
    assert result.artifact_id == "b3:" + "ab" * 32
    assert result.summary is None


def test_go_result_summary_from_json_stdout() -> None:
    payload = {"tables": {"drugsfda_products.tsv": {"artifact_id": "b3:" + "cd" * 32, "rows": 3}}}
    result = GoResult(subcommand="drugsfda", returncode=0, stdout=json.dumps(payload), stderr="", logs=())
    assert result.summary == payload
    # A JSON summary carries no bare b3: line.
    assert result.artifact_id is None


def test_go_result_nonzero_is_not_ok() -> None:
    result = GoResult(subcommand="faers", returncode=1, stdout="", stderr="boom", logs=())
    assert result.ok is False


# --- should_use_go gating --------------------------------------------------------


def test_should_use_go_off_by_default_even_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(go_runner, "go_available", lambda: True)
    assert should_use_go(_ctx(use_go=None)) is False
    assert should_use_go(_ctx(use_go=False)) is False


def test_should_use_go_requires_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(go_runner, "go_available", lambda: False)
    assert should_use_go(_ctx(use_go=True)) is False


def test_should_use_go_on_when_flagged_and_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(go_runner, "go_available", lambda: True)
    assert should_use_go(_ctx(use_go=True)) is True


def test_should_use_go_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(go_runner, "go_available", lambda: True)
    assert should_use_go(_ctx(use_go=True), override=False) is False
    assert should_use_go(_ctx(use_go=False), override=True) is True


# --- build-cache key + staging + TSV read-back -----------------------------------


def test_hash_go_sources_is_deterministic() -> None:
    go_dir = GoRunner().go_dir
    assert (go_dir / "go.mod").is_file()
    first = hash_go_sources(go_dir)
    second = hash_go_sources(go_dir)
    assert first == second
    assert len(first) == 16
    assert all(ch in "0123456789abcdef" for ch in first)


def test_stage_inputs_preserves_basenames(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    a = src / "DEMO24Q3.txt"
    b = src / "DRUG24Q3.txt"
    a.write_text("demo")
    b.write_text("drug")
    stage = stage_inputs([_ref(a), _ref(b)], tmp_path / "stage")
    assert (stage / "DEMO24Q3.txt").read_text() == "demo"
    assert (stage / "DRUG24Q3.txt").read_text() == "drug"


def test_stage_inputs_resolves_name_collisions(tmp_path: Path) -> None:
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "same.txt").write_text("one")
    (d2 / "same.txt").write_text("two")
    stage = stage_inputs([_ref(d1 / "same.txt"), _ref(d2 / "same.txt")], tmp_path / "stage")
    names = sorted(p.name for p in stage.iterdir())
    assert len(names) == 2
    assert "same.txt" in names  # first keeps its basename; second is disambiguated


def test_read_go_tsv_is_all_utf8_and_keeps_formatting(tmp_path: Path) -> None:
    tsv = tmp_path / "t.tsv"
    # Leading zeroes + a quoted empty cell (polars/Go quoting) must survive as Utf8.
    tsv.write_text('appl_no\tnote\n000123\t""\n')
    frame = read_go_tsv(tsv)
    assert frame.schema["appl_no"] == pl.Utf8
    assert frame.get_column("appl_no").to_list() == ["000123"]
    # Quoted empty -> "" (not null, not the literal string "None").
    assert frame.get_column("note").to_list() == [""]


def test_go_warnings_reads_summary_field() -> None:
    result = GoResult("faers", 0, "b3:x", "", logs=({"warnings": 7},))
    assert go_warnings(result) == 7
    assert go_warnings(GoResult("faers", 0, "b3:x", "", logs=())) == 0


# --- GoRunner binary resolution --------------------------------------------------


def test_go_runner_raises_when_go_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(go_runner.ENV_BINARY, raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    runner = GoRunner(go_dir=tmp_path / "go", cache_dir=tmp_path / "cache")
    with pytest.raises(GoUnavailableError, match="Go toolchain not found"):
        runner.ensure_binary()


def test_go_runner_uses_prebuilt_binary_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "prebuilt-dakp-worker"
    binary.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setenv(go_runner.ENV_BINARY, str(binary))
    # No `go` needed when a prebuilt binary is configured.
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert GoRunner(cache_dir=tmp_path / "cache").ensure_binary() == binary


def test_go_runner_explicit_binary_must_exist(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(GoUnavailableError, match="does not exist"):
        GoRunner(binary=missing).ensure_binary()
    present = tmp_path / "here"
    present.write_bytes(b"#!/bin/sh\n")
    assert GoRunner(binary=present).ensure_binary() == present


def test_go_runner_env_binary_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(go_runner.ENV_BINARY, str(tmp_path / "absent"))
    with pytest.raises(GoUnavailableError, match="does not exist"):
        GoRunner(cache_dir=tmp_path / "cache").ensure_binary()


# --- MockGoRunner contract -------------------------------------------------------


def test_mock_runner_routes_subcommands_and_records_calls() -> None:
    mock = MockGoRunner()
    canned_id = "b3:" + "ab" * 32
    mock.set_handler("hash", lambda args: (canned_id, json.dumps({"level": "INFO", "msg": "hashed", "path": args[0]})))

    result = mock.run("hash", ["somefile.txt"])
    assert result.ok
    assert result.artifact_id == canned_id
    assert mock.calls == [("hash", ("somefile.txt",))]
    # The structured stderr line was parsed into logs.
    assert result.logs
    assert result.logs[0]["msg"] == "hashed"
    assert result.logs[0]["path"] == "somefile.txt"


def test_mock_runner_default_stdout_per_subcommand() -> None:
    mock = MockGoRunner()
    assert mock.run("dailymed").artifact_id == "b3:" + "1" * 64
    assert mock.run("faers").artifact_id == "b3:" + "2" * 64
    summary = mock.run("drugsfda").summary
    assert summary is not None
    assert "drugsfda_products.tsv" in summary["tables"]


def test_mock_runner_run_table_appends_in_out_dirs(tmp_path: Path) -> None:
    mock = MockGoRunner()
    mock.run_table("dailymed", tmp_path / "in", tmp_path / "out")
    assert mock.calls == [("dailymed", (str(tmp_path / "in"), str(tmp_path / "out")))]


def test_mock_runner_parses_stderr_and_relays_non_json() -> None:
    mock = MockGoRunner()
    stderr = "\n".join([json.dumps({"level": "WARN", "msg": "careful", "warnings": 2}), "plain text line"])
    mock.set_handler("faers", lambda _args: ("b3:" + "9" * 64, stderr))
    result = mock.run("faers")
    # Only the JSON line lands in logs; the plain line was relayed to the logger, not captured.
    assert [rec["msg"] for rec in result.logs] == ["careful"]
    assert go_warnings(result) == 2


def test_mock_runner_raises_when_go_missing() -> None:
    mock = MockGoRunner(go_present=False)
    with pytest.raises(GoUnavailableError, match="unavailable"):
        mock.ensure_binary()
    with pytest.raises(GoUnavailableError, match="unavailable"):
        mock.run("hash", ["x"])


def test_get_runner_set_runner_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockGoRunner()
    monkeypatch.setattr(go_runner, "_RUNNER", None)
    go_runner.set_runner(mock)
    try:
        assert go_runner.get_runner() is mock
    finally:
        go_runner.set_runner(None)
