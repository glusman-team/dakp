"""Unit tests for the FAERS fetcher.

Covers: FDA index parsing (pure) and the real network path driven entirely through monkeypatched
boundaries (no network). The real HTTP bodies (``fetch_index`` / ``_http_get_text`` /
``_http_download``) are covered end-to-end by the integration ``test_prod_smoke.py``.
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


def _ctx(wd: Path, **params: Any) -> TaskContext:
    merged: dict[str, Any] = {"quarter_limit": None}
    merged.update(params)
    return TaskContext(workdir=wd, fixture_root=_FIXTURE_ROOT, params=merged)


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


# --- real path (monkeypatched network boundaries) -------------------------------


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

    refs = fetcher.fetch(_ctx(tmp_path))
    assert len(refs) == 2  # both discovered quarters
    assert captured == ["24Q3", "24Q1"]  # most-recent first
    assert all(ref.uri.suffix == ".zip" for ref in refs)


def test_real_fetch_honors_quarter_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = faers_source.FAERSFetcher()
    monkeypatch.setattr(fetcher, "fetch_index", lambda ctx: "<a href='faers_ascii_2024q3.zip'>a</a><a href='faers_ascii_2024q1.zip'>b</a>")
    monkeypatch.setattr(fetcher, "download_quarter", lambda ctx, source: _ingest_fake_zip(ctx.workdir, source))
    refs = fetcher.fetch(_ctx(tmp_path, quarter_limit=1))
    assert len(refs) == 1
    assert "24Q3" in refs[0].uri.name


@pytest.mark.parametrize("bad", [0, -3])
def test_real_fetch_non_positive_limit_means_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: int) -> None:
    fetcher = faers_source.FAERSFetcher()
    monkeypatch.setattr(fetcher, "fetch_index", lambda ctx: "<a href='faers_ascii_2024q3.zip'>a</a><a href='faers_ascii_2024q1.zip'>b</a>")
    monkeypatch.setattr(fetcher, "download_quarter", lambda ctx, source: _ingest_fake_zip(ctx.workdir, source))
    assert len(fetcher.fetch(_ctx(tmp_path, quarter_limit=bad))) == 2


def test_module_level_fetch_is_monkeypatchable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing ``faers_source.fetch`` takes effect for callers that resolve it at call time."""
    called: list[int] = []
    monkeypatch.setattr(faers_source, "fetch", lambda ctx: called.append(1) or [])
    assert faers_source.fetch(_ctx(tmp_path)) == []
    assert called == [1]


# --- download_quarter: cache-first acquisition (no re-download of cached quarters) ---


def _source() -> faers_source.QuarterSource:
    return faers_source.QuarterSource(quarter="24Q3", url="https://example.invalid/faers_ascii_2024q3.zip")


def test_download_quarter_skips_network_when_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    primed = _ingest_fake_zip(tmp_path, source)  # quarter already content-addressed under its alias

    def no_network(url: str, dest: Path, *, timeout: float) -> Path:
        pytest.fail("cached quarter must not hit the network")

    monkeypatch.setattr(faers_source, "_http_download", no_network)
    ref = faers_source.FAERSFetcher().download_quarter(_ctx(tmp_path), source)
    assert ref.blake3 == primed.blake3
    assert ref.uri == primed.uri


def test_download_quarter_force_redownloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    primed = _ingest_fake_zip(tmp_path, source)
    calls: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float) -> Path:
        calls.append(url)
        _build_fake_zip(dest)
        return dest

    monkeypatch.setattr(faers_source, "_http_download", fake_download)
    ref = faers_source.FAERSFetcher().download_quarter(_ctx(tmp_path, force=True), source)
    assert calls == [source.url]
    assert ref.blake3 == primed.blake3  # identical bytes -> same content id


def test_download_quarter_cache_miss_downloads_and_cleans_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()

    def fake_download(url: str, dest: Path, *, timeout: float) -> Path:
        _build_fake_zip(dest)
        return dest

    monkeypatch.setattr(faers_source, "_http_download", fake_download)
    ref = faers_source.FAERSFetcher().download_quarter(_ctx(tmp_path), source)
    assert ref.uri.exists()
    assert ref.uri.name == "faers_ascii_24Q3.zip"
    # The staged download copy is removed once the zip is content-addressed.
    assert not (Workdir(tmp_path).raw / "downloads" / "faers_ascii_24Q3.zip").exists()
    # A second call is now a cache hit (no network).
    monkeypatch.setattr(faers_source, "_http_download", lambda *a, **k: pytest.fail("network called"))
    again = faers_source.FAERSFetcher().download_quarter(_ctx(tmp_path), source)
    assert again.blake3 == ref.blake3


def test_download_quarter_redownloads_when_alias_points_at_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    primed = _ingest_fake_zip(tmp_path, source)
    primed.uri.unlink()  # stored file vanished; the alias record remains

    def fake_download(url: str, dest: Path, *, timeout: float) -> Path:
        _build_fake_zip(dest)
        return dest

    monkeypatch.setattr(faers_source, "_http_download", fake_download)
    ref = faers_source.FAERSFetcher().download_quarter(_ctx(tmp_path), source)
    assert ref.uri.exists()
    assert ref.blake3 == primed.blake3
