"""Edge-case tests for ``dakp_pipeline.release``.

Covers the robustness branches the happy-path tests miss: ``detect_git_commit`` degrading to
``None`` on ``OSError`` / ``TimeoutExpired`` / non-zero exit / blank stdout, and ``write_release``
skipping missing assertion tables, an absent ``tables/`` config dir, and an unparseable
``build_summary.json``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dakp_pipeline import release
from dakp_pipeline.io import schemas
from dakp_pipeline.paths import Workdir
from dakp_pipeline.release import detect_git_commit, write_release


def _write_table(wd: Workdir, name: str, rows: int = 1) -> None:
    columns = schemas.columns_for(name)
    path = wd.tabular / f"{name}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(columns)]
    lines.extend("\t".join("v" for _ in columns) for _ in range(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- detect_git_commit degradation ------------------------------------------------


def test_detect_git_commit_oserror_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        msg = "git not found"
        raise OSError(msg)

    monkeypatch.setattr(release.subprocess, "run", boom)
    assert detect_git_commit() is None


def test_detect_git_commit_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=10)

    monkeypatch.setattr(release.subprocess, "run", boom)
    assert detect_git_commit() is None


def test_detect_git_commit_nonzero_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = subprocess.CompletedProcess(args=["git"], returncode=128, stdout="", stderr="not a git repo")
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: fake)
    assert detect_git_commit() is None


def test_detect_git_commit_blank_stdout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="  \n", stderr="")
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: fake)
    assert detect_git_commit() is None


# --- write_release robustness -----------------------------------------------------


def test_write_release_skips_missing_assertion_tables(tmp_path: Path) -> None:
    wd = Workdir(tmp_path / "work")
    wd.create()
    _write_table(wd, "approved_treats_assertions")  # only one of the three tables exists
    result = write_release(wd, git_commit="0" * 40)
    assert [path.name for path in result.tables] == ["approved_treats_assertions.tsv"]
    assert result.manifest["row_counts"] == {"approved_treats_assertions": 1}


def test_write_release_handles_absent_configs_dir(tmp_path: Path) -> None:
    wd = Workdir(tmp_path / "work")
    wd.create()  # creates data/* but NOT the top-level tables/ config dir
    _write_table(wd, "approved_treats_assertions")
    result = write_release(wd, git_commit="0" * 40)
    assert result.configs == []
    assert result.manifest["inputs"]["configs"] == []


def test_write_release_handles_unparseable_build_summary(tmp_path: Path) -> None:
    wd = Workdir(tmp_path / "work")
    wd.create()
    _write_table(wd, "approved_treats_assertions")
    (wd.reports / "build_summary.json").write_text("{not valid json", encoding="utf-8")
    result = write_release(wd, git_commit="0" * 40)
    assert result.manifest["profile"] is None  # JSON decode error -> profile unknown
    assert result.manifest["inputs"]["build_summary"] is not None  # the file is still copied
