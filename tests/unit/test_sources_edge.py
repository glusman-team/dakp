"""Edge-case tests for ``dakp_pipeline.sources`` (100% branch coverage drive).

Drives the REAL (non-mock) acquisition paths fully offline by monkeypatching the stdlib
``urllib.request.urlopen`` boundary (DailyMed) and the module download helpers (Drugs@FDA /
FAERS), plus the small defensive branches the happy-path suite never reaches:

* ``sources/__init__`` — ``require_mock`` non-mock raise; ``ingest_fixtures`` None-root raise.
* ``sources/drugsfda`` — real fetch success + idempotent cache hit + URL override; the
  ``finally`` cleanup branch when the staged file is absent.
* ``sources/faers`` — mock discovery skipping quarter-less fixtures, empty-fixture-dir
  ``[]``, remote no-quarters ``[]``, ``iter_quarter_sources`` passthrough, real
  ``download_quarter`` via a monkeypatched HTTP download.
* ``sources/dailymed`` — mock fixture-missing/None-root errors; the full real release
  pipeline (index fetch -> release ZIP -> per-member SPL ingest) with canned HTTP responses;
  release_limit; conditional-GET headers; HTTP 304 handling for both the index and a release
  (cached-present and cached-absent); release ZIPs with directory entries / non-SPL members /
  no SPL members; ``_cached_ref`` pointer resolution.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, drugsfda, faers, ingest_fixtures, require_mock

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(tmp_path: Path, *, profile: str = "mock", fixture_root: Path | None = _FIXTURE_ROOT, **params: Any) -> TaskContext:
    wd = tmp_path / "work"
    Workdir(wd).create()
    return TaskContext(profile=profile, workdir=wd, fixture_root=fixture_root, threads=1, memory_budget_gb=1, params=params)


# --- sources/__init__: require_mock + ingest_fixtures --------------------------


def test_require_mock_raises_for_non_mock_profile(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="Milestone 2"):
        require_mock(_ctx(tmp_path, profile="prod"), "faers")


def test_require_mock_passes_for_mock_profile(tmp_path: Path) -> None:
    require_mock(_ctx(tmp_path, profile="mock"), "faers")  # no raise


def test_ingest_fixtures_requires_fixture_root(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, fixture_root=None)
    with pytest.raises(ValueError, match="fixture_root is None"):
        ingest_fixtures(ctx, ("faers/DEMO24Q3.txt",), namespace="faers")


def test_ingest_fixtures_content_addresses_each_name(tmp_path: Path) -> None:
    refs = ingest_fixtures(_ctx(tmp_path), ("faers/DEMO24Q3.txt", "faers/DRUG24Q3.txt"), namespace="faers")
    assert len(refs) == 2
    assert all(r.blake3.startswith("b3:") and r.uri.exists() for r in refs)


# --- sources/drugsfda: real fetch ----------------------------------------------


def _write_fake_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Products.txt", "ApplicationNumber\tDrugName\nNDA012345\tDrugX\n")


def test_drugsfda_real_fetch_success_and_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    fetcher = drugsfda.DrugsFDAFetcher()

    refs = fetcher.fetch(_ctx(tmp_path, profile="prod"))
    assert len(refs) == 1
    assert refs[0].media_type == "application/zip"
    first_id = refs[0].blake3

    # Identical bytes -> cache hit on re-fetch (same artifact id).
    refs2 = fetcher.fetch(_ctx(tmp_path, profile="prod"))
    assert refs2[0].blake3 == first_id


def test_drugsfda_real_fetch_honors_url_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        seen.append(url)
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    drugsfda.DrugsFDAFetcher().fetch(_ctx(tmp_path, profile="prod", drugsfda_url="https://example.invalid/custom.zip"))
    assert seen == ["https://example.invalid/custom.zip"]


def test_drugsfda_real_fetch_cleans_up_absent_staged_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``finally`` cleanup tolerates an absent staged file (65->68); ingest then fails loudly."""

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        dest.unlink(missing_ok=True)  # downloader left nothing behind
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    with pytest.raises(FileNotFoundError):
        drugsfda.DrugsFDAFetcher().fetch(_ctx(tmp_path, profile="prod"))


# --- sources/faers: discovery + mock/remote edges ------------------------------


def test_faers_mock_fetch_skips_quarterless_fixtures(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    (fixture_root / "faers").mkdir(parents=True)
    (fixture_root / "faers" / "DEMO24Q3.txt").write_bytes(b"PRIMARYID$\r\n1$\r\n")
    (fixture_root / "faers" / "README.txt").write_bytes(b"not a quarter file\r\n")  # no quarter -> skipped
    refs = faers.fetch(_ctx(tmp_path, fixture_root=fixture_root))
    assert [r.uri.name for r in refs] == ["DEMO24Q3.txt"]


def test_faers_mock_fetch_empty_fixture_dir_returns_empty(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    (fixture_root / "faers").mkdir(parents=True)
    assert faers.fetch(_ctx(tmp_path, fixture_root=fixture_root)) == []


def test_faers_remote_no_quarters_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = faers.FAERSFetcher()
    monkeypatch.setattr(fetcher, "fetch_index", lambda ctx: "<html>nothing here</html>")
    assert fetcher.fetch(_ctx(tmp_path, profile="sample")) == []


def test_faers_iter_quarter_sources_is_identity_passthrough() -> None:
    quarters = [faers.QuarterSource("24Q3", "u1"), faers.QuarterSource("24Q1", "u2")]
    assert list(faers.iter_quarter_sources(quarters)) == quarters


def test_faers_download_quarter_ingests_via_monkeypatched_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_http_download(url: str, dest: Path, *, timeout: float) -> Path:
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(faers, "_http_download", fake_http_download)
    ctx = _ctx(tmp_path, profile="sample")
    ref = faers.FAERSFetcher().download_quarter(ctx, faers.QuarterSource("24Q3", "https://x/faers_ascii_2024q3.zip"))
    assert ref.uri.exists()
    assert ref.blake3.startswith("b3:")
    # Idempotent: re-download identical bytes is a cache hit (same id).
    ref2 = faers.FAERSFetcher().download_quarter(ctx, faers.QuarterSource("24Q3", "https://x/faers_ascii_2024q3.zip"))
    assert ref2.blake3 == ref.blake3


def test_faers_discover_quarters_keeps_absolute_http_url() -> None:
    # A match whose surrounding text is a full URL still resolves to base_url/filename because
    # the regex match itself is the bare filename; relative names are joined onto base_url.
    quarters = faers.discover_quarters("see faers_ascii_2024q3.zip now", base_url="https://fis.fda.gov/content/Exports/")
    assert quarters[0].url == "https://fis.fda.gov/content/Exports/faers_ascii_2024q3.zip"


# --- sources/dailymed: mock-profile error paths --------------------------------


def test_dailymed_mock_requires_fixture_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixture_root is None"):
        dailymed.fetch(_ctx(tmp_path, profile="mock", fixture_root=None))


def test_dailymed_mock_missing_fixture_raises(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty-fixtures"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="DailyMed fixture not found"):
        dailymed.fetch(_ctx(tmp_path, profile="mock", fixture_root=empty_root))


# --- sources/dailymed: real release pipeline (monkeypatched urllib) ------------


class _FakeResponse:
    """A minimal ``urlopen`` return value: context manager with headers + streaming read."""

    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        self._buf = io.BytesIO(data)
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def close(self) -> None:  # HTTPError.close() parity for the 304 path
        pass


_RELEASE_URL = "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip"
_INDEX_HTML = f'<html><h2>Full Releases</h2><a href="{_RELEASE_URL}">part1</a></html>'


def _release_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch, responses: dict[str, _FakeResponse], requests: list[urllib.request.Request] | None = None
) -> None:
    """Route ``urllib.request.urlopen`` to canned responses keyed by URL substring."""

    def fake_urlopen(request: Any, *, timeout: float = 60.0) -> _FakeResponse:
        if requests is not None:
            requests.append(request)
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        for key, resp in responses.items():
            if key in url:
                resp._buf.seek(0)  # responses are reused across calls; rewind the stream
                return resp
        msg = f"unexpected URL: {url}"
        raise AssertionError(msg)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_dailymed_real_fetch_expands_release_spl_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spl = b"<splBatch><document><setId>S1</setId></document></splBatch>"
    zip_bytes = _release_zip({"xmls/drug1.xml": spl, "xmls/drug2.xml.gz": spl, "readme.txt": b"skip me"})
    _patch_urlopen(
        monkeypatch,
        {
            "spl-resources-all-drug-labels.cfm": _FakeResponse(_INDEX_HTML.encode(), {"ETag": "e1", "Last-Modified": "lm1"}),
            "dm_spl_release_human_rx_part1.zip": _FakeResponse(zip_bytes, {"ETag": "ze", "Last-Modified": "zlm"}),
        },
    )
    refs = dailymed.fetch(_ctx(tmp_path, profile="prod"))
    # Two SPL members ingested (.xml + .xml.gz); the .txt member is skipped.
    assert len(refs) == 2
    assert all(r.uri.suffix in {".xml", ".gz"} for r in refs)
    assert all(r.uri.exists() for r in refs)


def test_dailymed_real_fetch_release_limit_slices_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    two_release_index = (
        "<html><h2>Full Releases</h2>"
        f'<a href="{_RELEASE_URL}">part1</a>'
        '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part2.zip">part2</a>'
        "</html>"
    )
    zip_bytes = _release_zip({"drug1.xml": b"<splBatch/>"})
    _patch_urlopen(
        monkeypatch,
        {
            "spl-resources-all-drug-labels.cfm": _FakeResponse(two_release_index.encode()),
            "part1.zip": _FakeResponse(zip_bytes),
            "part2.zip": _FakeResponse(zip_bytes),
        },
    )
    refs = dailymed.fetch(_ctx(tmp_path, profile="prod", release_limit=1))
    assert len(refs) == 1  # only the first release processed


def test_dailymed_real_fetch_no_releases_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, {"spl-resources-all-drug-labels.cfm": _FakeResponse(b"<html>no zips</html>")})
    with pytest.raises(RuntimeError, match="no DailyMed full-release ZIPs"):
        dailymed.fetch(_ctx(tmp_path, profile="prod"))


def test_dailymed_release_zip_with_only_dirs_and_non_spl_warns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A release whose members are all directories / non-SPL yields no refs (logged, not fatal).
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xmls/", b"")  # directory entry
        zf.writestr("notes.txt", b"not spl")  # non-SPL member
    _patch_urlopen(
        monkeypatch,
        {
            "spl-resources-all-drug-labels.cfm": _FakeResponse(_INDEX_HTML.encode()),
            "dm_spl_release_human_rx_part1.zip": _FakeResponse(buf.getvalue()),
        },
    )
    assert dailymed.fetch(_ctx(tmp_path, profile="prod")) == []


def test_dailymed_conditional_get_sends_etag_and_last_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spl = b"<splBatch/>"
    zip_bytes = _release_zip({"drug1.xml": spl})
    requests: list[urllib.request.Request] = []
    _patch_urlopen(
        monkeypatch,
        {
            "spl-resources-all-drug-labels.cfm": _FakeResponse(_INDEX_HTML.encode(), {"ETag": "idx-e", "Last-Modified": "idx-lm"}),
            "dm_spl_release_human_rx_part1.zip": _FakeResponse(zip_bytes, {"ETag": "zip-e", "Last-Modified": "zip-lm"}),
        },
        requests,
    )
    ctx = _ctx(tmp_path, profile="prod")
    dailymed.fetch(ctx)  # first run records source etag/last-modified under the release alias
    requests.clear()
    dailymed.fetch(ctx)  # second run: _prior_source finds them -> conditional headers sent
    release_reqs = [r for r in requests if "part1.zip" in r.full_url]
    assert release_reqs
    assert release_reqs[0].get_header("If-none-match") == "zip-e"
    assert release_reqs[0].get_header("If-modified-since") == "zip-lm"


def test_dailymed_non_304_http_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-304 HTTP error from the index fetch is re-raised (not swallowed as cache-fresh)."""

    def fake_urlopen(request: Any, *, timeout: float = 60.0) -> _FakeResponse:
        raise urllib.error.HTTPError(request.full_url, 500, "Server Error", {}, io.BytesIO(b""))  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        dailymed.fetch(_ctx(tmp_path, profile="prod"))


def _raise_304(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 304, "Not Modified", {}, io.BytesIO(b""))  # type: ignore[arg-type]


def test_dailymed_index_304_without_cache_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, *, timeout: float = 60.0) -> _FakeResponse:
        raise _raise_304(request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="no cached copy exists"):
        dailymed.fetch(_ctx(tmp_path, profile="prod"))


def test_dailymed_index_304_with_cache_returns_cached_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spl = b"<splBatch/>"
    zip_bytes = _release_zip({"drug1.xml": spl})
    state = {"not_modified": False}

    def fake_urlopen(request: Any, *, timeout: float = 60.0) -> _FakeResponse:
        url = request.full_url
        if state["not_modified"] and "spl-resources-all-drug-labels.cfm" in url:
            raise _raise_304(url)
        if "spl-resources-all-drug-labels.cfm" in url:
            return _FakeResponse(_INDEX_HTML.encode(), {"ETag": "idx-e"})
        return _FakeResponse(zip_bytes, {"ETag": "zip-e"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ctx = _ctx(tmp_path, profile="prod")
    first = dailymed.fetch(ctx)
    assert len(first) == 1
    state["not_modified"] = True
    second = dailymed.fetch(ctx)  # index 304 -> cached index text reused -> same release ref
    assert [r.blake3 for r in second] == [r.blake3 for r in first]


def test_dailymed_release_304_returns_cached_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spl = b"<splBatch/>"
    zip_bytes = _release_zip({"drug1.xml": spl})
    state = {"not_modified": False}

    def fake_urlopen(request: Any, *, timeout: float = 60.0) -> _FakeResponse:
        url = request.full_url
        if "spl-resources-all-drug-labels.cfm" in url:
            return _FakeResponse(_INDEX_HTML.encode())
        if state["not_modified"]:
            raise _raise_304(url)
        return _FakeResponse(zip_bytes, {"ETag": "zip-e"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ctx = _ctx(tmp_path, profile="prod")
    first = dailymed.fetch(ctx)
    assert len(first) == 1
    state["not_modified"] = True
    # Release 304: the cached release ZIP ref is returned (the ZIP itself, not its members).
    second = dailymed.fetch(ctx)
    assert len(second) == 1
    assert second[0].uri.name.endswith(".zip")


def test_dailymed_release_304_without_cache_returns_empty(tmp_path: Path) -> None:
    store = _store_for(tmp_path)
    # No alias recorded for this release -> _cached_ref None -> [] on a 304.
    staging = Workdir(store.workdir.root).root / ".staging" / "dailymed"
    staging.mkdir(parents=True, exist_ok=True)

    def fake_conditional(url: str, dest: Path, source: Any) -> tuple[None, None]:
        dest.unlink(missing_ok=True)  # simulate 304: nothing downloaded
        return None, None

    import dakp_pipeline.sources.dailymed as dm

    orig = dm._conditional_download
    dm._conditional_download = fake_conditional  # type: ignore[method-assign]
    try:
        assert dm._download_one("https://x/never.zip", staging, store) == []
    finally:
        dm._conditional_download = orig  # type: ignore[method-assign]


def _store_for(tmp_path: Path) -> ArtifactStore:
    wd = Workdir(tmp_path / "work")
    wd.create()
    return ArtifactStore(wd)


# --- dailymed: _cached_ref / _prior_source pointer resolution ------------------


def test_dailymed_cached_ref_missing_alias_and_path(tmp_path: Path) -> None:
    store = _store_for(tmp_path)
    assert dailymed._cached_ref(store, alias="dailymed/absent.zip") is None  # no alias
    # Alias present but sibling .path pointer missing -> None.
    wd = Workdir(store.workdir.root)
    alias_file = wd.aliases / "dailymed" / "half.zip"
    alias_file.parent.mkdir(parents=True, exist_ok=True)
    alias_file.write_text("b3:deadbeef", encoding="utf-8")
    assert dailymed._cached_ref(store, alias="dailymed/half.zip") is None


def test_dailymed_prior_source_none_when_manifest_absent(tmp_path: Path) -> None:
    store = _store_for(tmp_path)
    wd = Workdir(store.workdir.root)
    alias_file = wd.aliases / "dailymed" / "ghost.zip"
    alias_file.parent.mkdir(parents=True, exist_ok=True)
    alias_file.write_text("b3:" + "f" * 64, encoding="utf-8")  # alias points at an unknown artifact
    assert dailymed._prior_source(store, alias="dailymed/ghost.zip") is None


def test_dailymed_apply_release_limit_non_positive_means_all(tmp_path: Path) -> None:
    urls = ["u1", "u2", "u3"]
    assert dailymed._apply_release_limit(urls, _ctx(tmp_path, release_limit=0)) == urls
    assert dailymed._apply_release_limit(urls, _ctx(tmp_path, release_limit=-1)) == urls
    assert dailymed._apply_release_limit(urls, _ctx(tmp_path, release_limit="x")) == urls  # non-int -> all
    assert dailymed._apply_release_limit(urls, _ctx(tmp_path, release_limit=2)) == ["u1", "u2"]
