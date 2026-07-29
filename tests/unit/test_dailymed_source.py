"""Unit tests for the DailyMed fetcher (:mod:`dakp_pipeline.sources.dailymed`).

Covers the mock-profile fixture ingest (with source provenance in the manifest),
idempotent re-fetch, monkeypatchability of the module-level ``fetch``, and wiring of the
real (non-mock) download path without any network.
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import read_manifest
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(tmp_path: Path, *, profile: str = "mock") -> TaskContext:
    return TaskContext(profile=profile, workdir=(tmp_path / "work"), fixture_root=_FIXTURE_ROOT, threads=1, memory_budget_gb=1, params={})


def test_mock_fetch_ingests_fixture_with_source_provenance(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()

    refs = dailymed.fetch(ctx)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.blake3.startswith("b3:")
    # Content-addressed store copy exists and is readable as gzip SPL.
    assert ref.uri.exists()
    assert ref.uri.name.endswith(".xml.gz")
    assert ref.manifest is not None

    manifest = read_manifest(ref.manifest)
    # Source provenance recorded (URL / retrieved_at) into the artifact manifest.
    assert manifest.source.url is not None
    assert manifest.source.url.startswith("fixture://")
    assert manifest.source.retrieved_at is not None
    assert manifest.hash.file == ref.blake3


def test_mock_fetch_is_idempotent(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()

    first = dailymed.fetch(ctx)
    second = dailymed.fetch(ctx)

    assert len(first) == len(second) == 1
    # Identical content -> same artifact id, same store path (cache hit, no duplicate).
    assert first[0].blake3 == second[0].blake3
    assert first[0].uri == second[0].uri


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

    # Real fetcher still works after restore.
    Workdir(ctx.workdir).create()
    assert dailymed.fetch(ctx)[0].uri.exists()


def test_real_profile_invokes_downloader_without_network(tmp_path: Path, monkeypatch) -> None:
    """A non-mock profile routes to _download_full_release; tests stub it (no network)."""
    ctx = _ctx(tmp_path, profile="sample")
    Workdir(ctx.workdir).create()

    stub_ref = ArtifactRef(uri=tmp_path / "staged.zip", blake3="b3:cafebabe", media_type="application/zip")
    calls: list[TaskContext] = []

    def fake_download(c: TaskContext) -> list[ArtifactRef]:
        calls.append(c)
        return [stub_ref]

    monkeypatch.setattr(dailymed, "_download_full_release", fake_download)

    refs = dailymed.fetch(ctx)

    assert refs == [stub_ref]
    assert len(calls) == 1
    assert calls[0].profile == "sample"


def test_real_download_parses_release_zip_urls_from_index(monkeypatch) -> None:
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


def test_store_alias_records_artifact_for_conditional_lookup(tmp_path: Path) -> None:
    """The fetcher's alias write enables _prior_source lookups (for conditional GET)."""
    ctx = _ctx(tmp_path)
    Workdir(ctx.workdir).create()
    store = ArtifactStore(Workdir(ctx.workdir))

    [ref] = dailymed.fetch(ctx)

    prior = dailymed._prior_source(store, alias="dailymed/dailymed/dailymed_spl.xml.gz")
    assert prior is not None
    assert prior.url is not None
    # Unknown alias -> None (no crash).
    assert dailymed._prior_source(store, alias="dailymed/never.zip") is None
    assert ref.uri.exists()
