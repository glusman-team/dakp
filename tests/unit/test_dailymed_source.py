"""Unit tests for the DailyMed fetcher (:mod:`dakp_pipeline.sources.dailymed`).

Covers monkeypatchability of the module-level ``fetch``, the release-index ZIP parser, the
declared-file-count parser + 404 coverage floor, and the wiring that always routes ``fetch``
to the real downloader (stubbed, no network). The full offline release pipeline (index ->
release ZIP -> per-member SPL ingest, conditional GET, 304s) is exercised in
``test_sources_edge.py``; the real HTTP bodies are covered end-to-end by the integration
``test_prod_smoke.py`` (monkeypatched ``urllib.request.urlopen``).
"""

from __future__ import annotations

import io
import urllib.error
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.io.xcom import REFS_FILE_MEDIA_TYPE, refs_from_xcom, refs_to_xcom
from dakp_pipeline.paths import Workdir
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


def test_write_refs_manifest_round_trip(tmp_path: Path) -> None:
    """The single-file XCom handoff: refs -> one store JSON -> ONE sentinel ref -> same refs back.

    This is the shrink that keeps tens of thousands of per-SPL-member refs out of the XCom
    payload: the DAG pushes only the returned sentinel ref, and consumers (refs_from_xcom; the
    Go DecodeArtifactRefs mirror) resolve it back to the identical refs.
    """
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()
    refs = [
        ArtifactRef(uri=tmp_path / "a.xml.gz", blake3="b3:aaa", media_type="application/gzip"),
        ArtifactRef(uri=tmp_path / "b.xml", blake3="b3:bbb", media_type="application/xml"),
    ]

    refs_ref = dailymed.write_refs_manifest(ctx, refs)

    assert refs_ref.media_type == REFS_FILE_MEDIA_TYPE
    assert refs_ref.blake3.startswith("b3:")
    assert refs_ref.uri.exists()
    # The XCom payload is ONE small dict; refs_from_xcom resolves it to the identical refs.
    payload = refs_to_xcom([refs_ref])
    assert len(payload) == 1
    assert refs_from_xcom(payload) == refs
    # Idempotent: identical refs re-ingest as a store cache hit (same content hash).
    assert dailymed.write_refs_manifest(ctx, refs).blake3 == refs_ref.blake3


# --- declared file counts + 404 coverage floor -----------------------------------------


def _http_error(code: int) -> urllib.error.HTTPError:
    import email.message

    return urllib.error.HTTPError("https://dailymed-data.nlm.nih.gov/x.zip", code, "boom", email.message.Message(), None)


def _seed_release(store: ArtifactStore, tmp_path: Path, name: str, *, retrieved_at: str) -> None:
    """Ingest a one-SPL-document release ZIP under its ``dailymed/<name>`` alias."""
    doc = io.BytesIO()
    with zipfile.ZipFile(doc, "w") as inner:
        inner.writestr("00000000-0000-0000-0000-000000000000.xml", "<document/>")
    staged = tmp_path / name
    with zipfile.ZipFile(staged, "w") as outer:
        outer.writestr("prescription/20260910_00000000.zip", doc.getvalue())
    url = f"https://dailymed-data.nlm.nih.gov/public-release-files/{name}"
    store.ingest(staged, alias=f"dailymed/{name}", source=SourceBlock(url=url, retrieved_at=retrieved_at))
    staged.unlink()


def test_parse_declared_file_counts_reads_index_manifest() -> None:
    """The index's per-release 'Number of files' bullets become the coverage denominator."""
    html = (
        "<html><h2>Full Releases</h2>"
        '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/a.zip">a</a>'
        "<ul><li><strong>Number of files:</strong> 9,521</li>"
        "<li><strong>File size:</strong> 3.00GB</li>"
        "<li><strong>MD5 checksum:</strong> 919c9efe1984047debf6d9abeb173884</li></ul>"
        '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/b.zip">b</a>'
        "<ul><li><strong>Number of files:</strong> 35</li></ul>"
        '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/c.zip">c</a>'
        "</html>"
    )
    counts = dailymed._parse_declared_file_counts(html)
    assert counts == {
        "https://dailymed-data.nlm.nih.gov/public-release-files/a.zip": 9521,
        "https://dailymed-data.nlm.nih.gov/public-release-files/b.zip": 35,
    }
    # A count before the heading is never attributed (same section anchor as the URL parser).
    pre = '<html><strong>Number of files:</strong> 77</strong><h2>Full Releases</h2><a href="https://x/a.zip">a</a></html>'
    assert dailymed._parse_declared_file_counts(pre) == {}


def test_download_one_404_without_cache_is_a_missing_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A listed-but-absent release with no cached copy yields ([], missing=True), no raise."""
    Workdir(tmp_path / "work").create()
    store = ArtifactStore(Workdir(tmp_path / "work"))

    def boom(url: str, dest: Path, source: SourceBlock | None) -> tuple[str | None, str | None]:
        raise _http_error(404)

    monkeypatch.setattr(dailymed, "_conditional_download", boom)
    refs, missing = dailymed._download_one(
        "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_otc_part12.zip", tmp_path / "staging", store, force=True
    )
    assert refs == []
    assert missing is True


def test_download_one_404_reuses_stale_cached_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 release with a stale cached copy is served from cache, not skipped."""
    Workdir(tmp_path / "work").create()
    store = ArtifactStore(Workdir(tmp_path / "work"))
    stale = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    name = "dm_spl_release_human_otc_part12.zip"
    _seed_release(store, tmp_path, name, retrieved_at=stale)

    def boom(url: str, dest: Path, source: SourceBlock | None) -> tuple[str | None, str | None]:
        raise _http_error(404)

    monkeypatch.setattr(dailymed, "_conditional_download", boom)
    refs, missing = dailymed._download_one(
        f"https://dailymed-data.nlm.nih.gov/public-release-files/{name}", tmp_path / "staging", store, max_age_days=7.0, force=True
    )
    assert missing is False
    assert len(refs) == 1  # the seeded nested SPL document was re-expanded from cache
    # Second run: the expansion is now the release's cached refs, so the 404 fallback serves
    # those directly (no re-expansion, and no freshness window to warn against).
    refs_again, missing_again = dailymed._download_one(
        f"https://dailymed-data.nlm.nih.gov/public-release-files/{name}", tmp_path / "staging", store, force=True
    )
    assert missing_again is False
    assert refs_again == refs


def test_download_one_non_404_http_error_still_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only 404 is tolerated; a 500 still fails the release loudly."""
    Workdir(tmp_path / "work").create()
    store = ArtifactStore(Workdir(tmp_path / "work"))

    def boom(url: str, dest: Path, source: SourceBlock | None) -> tuple[str | None, str | None]:
        raise _http_error(500)

    monkeypatch.setattr(dailymed, "_conditional_download", boom)
    with pytest.raises(urllib.error.HTTPError):
        dailymed._download_one("https://dailymed-data.nlm.nih.gov/public-release-files/a.zip", tmp_path / "staging", store, force=True)


def test_enforce_release_coverage_measures_declared_share() -> None:
    """The floor is the missing share of the index-declared corpus, not the release count."""
    declared = {"https://x/a.zip": 9000, "https://x/b.zip": 35}
    # A 35-file tail shard out of 9035 declared files (0.4%) degrades gracefully.
    dailymed._enforce_release_coverage(["https://x/b.zip"], declared)
    # A 9000-file part (99.6%) refuses to build a hollow corpus, naming the release.
    with pytest.raises(RuntimeError, match=r"a\.zip.*9000/9035"):
        dailymed._enforce_release_coverage(["https://x/a.zip"], declared)
    # The boundary itself passes: a loss of exactly the tolerance (1 of 100 files = 1%) does not
    # exceed it — the floor is `>`, so exactly-at-tolerance degrades gracefully.
    dailymed._enforce_release_coverage(["https://x/b.zip"], {"https://x/a.zip": 99, "https://x/b.zip": 1})


def test_parse_declared_file_counts_matches_the_real_index_markup() -> None:
    """A verbatim capture of the live listing pins the parser to the real page structure.

    Note the page writes the release href TWICE (label link + HTTPS mirror link) before the
    metadata bullets; the count must attribute to that URL exactly once.
    """
    base = "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_otc_part12.zip"
    real = (
        '<li data-ddfilter="human otc labels">\n            \n\n            '
        f'<a href="{base}">dm_spl_release_human_otc_part12.zip</a>\n            [ '
        f'<a href="{base}">HTTPS</a> / '
        '<a href="ftp://public.nlm.nih.gov/nlmdata/.dailymed/dm_spl_release_human_otc_part12.zip">FTP</a> ]\n\n            '
        "<ul>\n                \n                <li><strong>Number of files:</strong> 35</li>\n                \n\n                "
        "<li><strong>File size:</strong> 10.91MB</li>\n                "
        "<li><strong>MD5 checksum:</strong> 09b8587a93f4559edf19b3f97d13f4dd</li>\n                "
        "<li><strong>Last Modified:</strong> Sep 15, 2026</li>\n            </ul>\n        </li>"
    )
    html = f"<html><h2>Full Releases</h2>{real}</html>"
    assert dailymed._parse_declared_file_counts(html) == {base: 35}
    assert dailymed._parse_release_zips(html) == [base]


def test_enforce_release_coverage_falls_back_to_release_count() -> None:
    """Without declared counts (layout drift) the floor is an absolute release count."""
    urls = [f"https://x/{i}.zip" for i in range(3)]
    dailymed._enforce_release_coverage(urls[: dailymed._MAX_MISSING_RELEASES], {})
    with pytest.raises(RuntimeError, match="coverage cannot be measured"):
        dailymed._enforce_release_coverage(urls, {})
    # A missing release with no declared count routes to the count floor, never silently
    # through the measured floor (its size is unknown, so it cannot be judged by share).
    unknown = [f"https://x/unknown{i}.zip" for i in range(dailymed._MAX_MISSING_RELEASES + 1)]
    with pytest.raises(RuntimeError, match="coverage cannot be measured"):
        dailymed._enforce_release_coverage(unknown, {"https://x/a.zip": 9000})


def test_download_full_release_tolerates_a_broken_tail_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: one small declared release 404s, the rest of the corpus still acquires."""
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()
    html = (
        "<html><h2>Full Releases</h2>"
        '<a href="https://x/a.zip">a</a><ul><li><strong>Number of files:</strong> 9,000</li></ul>'
        '<a href="https://x/b.zip">b</a><ul><li><strong>Number of files:</strong> 35</li></ul>'
        "</html>"
    )
    monkeypatch.setattr(dailymed, "_fetch_index", lambda ctx, staging, store: html)
    good = [ArtifactRef(uri=tmp_path / "a.xml.gz", blake3="b3:a", media_type="application/gzip")]

    def fake_one(
        url: str, staging: Path, store: ArtifactStore, *, max_age_days: float | None = None, force: bool = False
    ) -> tuple[list[ArtifactRef], bool]:
        return ([], True) if url.endswith("b.zip") else (good, False)

    monkeypatch.setattr(dailymed, "_download_one", fake_one)

    refs = dailymed._download_full_release(ctx)

    assert refs == good


def test_download_full_release_fails_when_a_big_part_goes_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: a 404 release carrying most of the declared corpus fails the acquisition."""
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()
    html = (
        "<html><h2>Full Releases</h2>"
        '<a href="https://x/a.zip">a</a><ul><li><strong>Number of files:</strong> 9,000</li></ul>'
        '<a href="https://x/b.zip">b</a><ul><li><strong>Number of files:</strong> 35</li></ul>'
        "</html>"
    )
    monkeypatch.setattr(dailymed, "_fetch_index", lambda ctx, staging, store: html)
    good = [ArtifactRef(uri=tmp_path / "b.xml.gz", blake3="b3:b", media_type="application/gzip")]

    def fake_one(
        url: str, staging: Path, store: ArtifactStore, *, max_age_days: float | None = None, force: bool = False
    ) -> tuple[list[ArtifactRef], bool]:
        return ([], True) if url.endswith("a.zip") else (good, False)

    monkeypatch.setattr(dailymed, "_download_one", fake_one)

    with pytest.raises(RuntimeError, match=r"a\.zip"):
        dailymed._download_full_release(ctx)
