"""Unit tests for the FAERS fetcher (Milestone 2).

Covers: FDA index parsing (pure), mock fixture discovery + ``quarter_limit`` truncation,
and the real network path driven entirely through monkeypatched boundaries (no network).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import faers as faers_source

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(wd: Path, *, profile: str = "mock", **params: Any) -> TaskContext:
    merged: dict[str, Any] = {"quarter_limit": None}
    merged.update(params)
    return TaskContext(profile=profile, workdir=wd, fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params=merged)


# --- discover_quarters (pure parser) --------------------------------------------


def test_discover_quarters_parses_and_orders_most_recent_first() -> None:
    html = (
        "<a href='faers_ascii_2024q3.zip'>q3</a>"
        "<a href='faers_ascii_2023q4.zip'>q4</a>"
        "<a href='faers_ascii_2024q1.zip'>q1</a>"
        "<a href='faers_ascii_derivations_2024q3.zip'>deriv</a>"  # not an ASCII zip -> ignored
    )
    quarters = faers_source.discover_quarters(html)
    labels = [q.quarter for q in quarters]
    assert labels == ["24Q3", "24Q1", "23Q4"]  # descending = most-recent first
    assert all(q.url.endswith(".zip") for q in quarters)
    assert "faers_ascii_2024q3.zip" in quarters[0].url


def test_discover_quarters_canonicalizes_full_year_to_two_digit() -> None:
    quarters = faers_source.discover_quarters("<a href='faers_ascii_2018q2.zip'>x</a>")
    assert quarters[0].quarter == "18Q2"


def test_discover_quarters_empty_html() -> None:
    assert faers_source.discover_quarters("<html>no links</html>") == []


# --- mock profile ---------------------------------------------------------------


def test_mock_fetch_returns_all_fixture_families(tmp_path: Path) -> None:
    refs = faers_source.fetch(_ctx(tmp_path))
    # 11 fixture .txt files across two quarters (24Q3: 6 families, 24Q2: 5 families).
    assert len(refs) == 11
    names = {ref.uri.name for ref in refs}
    assert {"DEMO24Q3.txt", "DELETE24Q3.txt", "REAC24Q2.txt"} <= names
    # Every ref resolves to a readable, content-addressed store path with a b3: id.
    assert all(ref.uri.exists() for ref in refs)
    assert all(ref.blake3.startswith("b3:") for ref in refs)


def test_mock_fetch_honors_quarter_limit(tmp_path: Path) -> None:
    refs = faers_source.fetch(_ctx(tmp_path, quarter_limit=1))
    assert len(refs) == 6  # most-recent quarter (24Q3) only
    assert all("24Q3" in ref.uri.name for ref in refs)


def test_mock_fetch_quarter_limit_two_returns_both_quarters(tmp_path: Path) -> None:
    assert len(faers_source.fetch(_ctx(tmp_path, quarter_limit=2))) == 11


@pytest.mark.parametrize("bad", [0, -3])
def test_mock_fetch_non_positive_limit_means_all(tmp_path: Path, bad: int) -> None:
    assert len(faers_source.fetch(_ctx(tmp_path, quarter_limit=bad))) == 11


def test_mock_fetch_requires_fixture_root(tmp_path: Path) -> None:
    ctx = TaskContext(profile="mock", workdir=tmp_path, fixture_root=None, threads=1, memory_budget_gb=1, params={})
    with pytest.raises(ValueError, match="fixture_root"):
        faers_source.fetch(ctx)


# --- real profile (monkeypatched network boundaries) ---------------------------


def _build_fake_zip(dest: Path) -> None:
    """A minimal FAERS-style zip: members named <FAMILY>24Q3.txt with trailing-$ lines."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    members = {
        "DEMO24Q3.txt": b"PRIMARYID$CASEID$\r\n1001$5001$\r\n",
        "DRUG24Q3.txt": b"PRIMARYID$DRUG_SEQ$DRUGNAME$\r\n1001$1$DrugX$\r\n",
        "INDI24Q3.txt": b"PRIMARYID$INDI_DRUG_SEQ$INDI_PT$\r\n1001$1$pain$\r\n",
    }
    with zipfile.ZipFile(dest, "w") as zf:
        for name, data in members.items():
            zf.writestr(f"ascii/{name}", data)


def _ingest_fake_zip(workdir: Path, source: faers_source.QuarterSource):
    wd = Workdir(workdir)
    dest = wd.raw / "downloads" / f"faers_ascii_{source.quarter}.zip"
    _build_fake_zip(dest)
    ref, _ = ArtifactStore(wd).ingest(dest, alias=f"faers/faers_ascii_{source.quarter}.zip")
    return ref


def test_real_fetch_uses_monkeypatched_index_and_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = faers_source.FAERSFetcher()
    fake_index = "<a href='faers_ascii_2024q3.zip'>a</a><a href='faers_ascii_2024q1.zip'>b</a>"
    monkeypatch.setattr(fetcher, "fetch_index", lambda ctx: fake_index)

    captured: list[str] = []

    def fake_download(ctx: TaskContext, source: faers_source.QuarterSource):
        captured.append(source.quarter)
        return _ingest_fake_zip(ctx.workdir, source)

    monkeypatch.setattr(fetcher, "download_quarter", fake_download)

    refs = fetcher.fetch(_ctx(tmp_path, profile="sample"))
    assert len(refs) == 2  # both discovered quarters
    assert captured == ["24Q3", "24Q1"]  # most-recent first
    assert all(ref.uri.suffix == ".zip" for ref in refs)


def test_real_fetch_honors_quarter_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = faers_source.FAERSFetcher()
    monkeypatch.setattr(fetcher, "fetch_index", lambda ctx: "<a href='faers_ascii_2024q3.zip'>a</a><a href='faers_ascii_2024q1.zip'>b</a>")
    monkeypatch.setattr(fetcher, "download_quarter", lambda ctx, source: _ingest_fake_zip(ctx.workdir, source))
    refs = fetcher.fetch(_ctx(tmp_path, profile="sample", quarter_limit=1))
    assert len(refs) == 1
    assert "24Q3" in refs[0].uri.name


def test_module_level_fetch_is_monkeypatchable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing ``faers_source.fetch`` takes effect for callers that resolve it at call time."""
    called: list[int] = []
    monkeypatch.setattr(faers_source, "fetch", lambda ctx: called.append(1) or [])
    assert faers_source.fetch(_ctx(tmp_path)) == []
    assert called == [1]
