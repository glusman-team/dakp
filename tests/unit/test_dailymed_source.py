"""Unit tests for the DailyMed fetcher (:mod:`dakp_pipeline.sources.dailymed`).

Covers monkeypatchability of the module-level ``fetch``, the release-index ZIP parser, and the
wiring that always routes ``fetch`` to the real downloader (stubbed, no network). The full offline
release pipeline (index -> release ZIP -> per-member SPL ingest, conditional GET, 304s) is
exercised in ``test_sources_edge.py``; the real HTTP bodies are covered end-to-end by the
integration ``test_prod_smoke.py`` (monkeypatched ``urllib.request.urlopen``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.sources import dailymed

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(tmp_path: Path) -> TaskContext:
    return TaskContext(workdir=(tmp_path / "work"), fixture_root=_FIXTURE_ROOT, params={})


def test_fetch_is_monkeypatchable(tmp_path: Path) -> None:
    """The module-level `fetch` is replaceable (the pipeline calls dailymed.fetch(ctx))."""
    ctx = _ctx(tmp_path)
    sentinel = ArtifactRef(uri=Path("/tmp/sentinel.xml.gz"), blake3="b3:deadbeef", media_type="application/gzip")

    dailymed.fetch = lambda _ctx: [sentinel]  # type: ignore[method-assign]
    try:
        assert dailymed.fetch(ctx) == [sentinel]
    finally:
        # Restore the real bound method so other tests in the session are unaffected.
        dailymed.fetch = dailymed.DailyMedFetcher().fetch  # type: ignore[method-assign]


def test_fetch_always_routes_to_the_real_downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch always routes to _download_full_release; tests stub it (no network)."""
    ctx = _ctx(tmp_path)
    stub_ref = ArtifactRef(uri=tmp_path / "staged.zip", blake3="b3:cafebabe", media_type="application/zip")
    calls: list[TaskContext] = []

    def fake_download(c: TaskContext) -> list[ArtifactRef]:
        calls.append(c)
        return [stub_ref]

    monkeypatch.setattr(dailymed, "_download_full_release", fake_download)

    refs = dailymed.fetch(ctx)

    assert refs == [stub_ref]
    assert calls == [ctx]


def test_real_download_parses_release_zip_urls_from_index() -> None:
    """The index parser only keeps ZIP hrefs under the 'Full Releases' heading."""
    html = (
        "<html><h2>Partial Releases</h2>"
        '<a href="https://x.example/partial.zip">partial.zip</a>'
        "<h2>Full Releases</h2>"
        '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip">part1</a>'
        '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_otc_part1.zip">otc1</a>'
        '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip">dup</a>'
        "</html>"
    )
    urls = dailymed._parse_release_zips(html)
    # Only Full Releases kept; duplicate de-duplicated; order preserved.
    assert urls == [
        "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip",
        "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_otc_part1.zip",
    ]
    assert "partial.zip" not in urls[0]
