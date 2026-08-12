"""Edge-case tests for ``dakp_pipeline.sources`` (100% branch coverage drive).

Drives the REAL acquisition paths fully offline by monkeypatching the stdlib
``urllib.request.urlopen`` boundary (DailyMed) and the module download helpers (Drugs@FDA /
FAERS), plus the small defensive branches the happy-path suite never reaches:

* ``sources/drugsfda`` — real fetch success + idempotent cache hit + URL override; the
  ``finally`` cleanup branch when the staged file is absent.
* ``sources/faers`` — remote no-quarters ``[]``, real ``download_quarter`` via a monkeypatched
  HTTP download, quarter-label URL resolution.
* ``sources/dailymed`` — the full real release pipeline (index fetch -> release ZIP ->
  per-member SPL ingest) with canned HTTP responses; release_limit; the freshness gate
  (stored releases younger than ``dailymed_max_age_days`` skip the network; ``force`` /
  stale / missing cache fall through); conditional-GET headers;
  HTTP 304 handling for both the index and a release (cached-present and cached-absent);
  release ZIPs with directory entries / non-SPL members / no SPL members; ``ArtifactStore.cached_ref``
  pointer resolution.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed, drugsfda, faers

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ctx(tmp_path: Path, *, fixture_root: Path | None = _FIXTURE_ROOT, **params: Any) -> TaskContext:
    wd = tmp_path / "work"
    Workdir(wd).create()
    return TaskContext(workdir=wd, fixture_root=fixture_root, params=params)


# --- sources/drugsfda: real fetch ----------------------------------------------


def _write_fake_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Products.txt", "ApplicationNumber\tDrugName\nNDA012345\tDrugX\n")


def test_drugsfda_real_fetch_success_and_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        calls.append(url)
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    fetcher = drugsfda.DrugsFDAFetcher()

    refs = fetcher.fetch(_ctx(tmp_path))
    assert len(refs) == 1
    assert refs[0].media_type == "application/zip"
    first_id = refs[0].blake3

    # The alias is checked before opening a network connection on re-fetch.
    refs2 = fetcher.fetch(_ctx(tmp_path))
    assert refs2[0].blake3 == first_id
    assert calls == [drugsfda.DRUGSFDA_DATA_FILES_URL]


def test_drugsfda_force_bypasses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        calls.append(url)
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    fetcher = drugsfda.DrugsFDAFetcher()
    fetcher.fetch(_ctx(tmp_path))
    fetcher.fetch(_ctx(tmp_path, force=True))
    assert calls == [drugsfda.DRUGSFDA_DATA_FILES_URL, drugsfda.DRUGSFDA_DATA_FILES_URL]


def test_drugsfda_real_fetch_honors_url_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        seen.append(url)
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    fetcher = drugsfda.DrugsFDAFetcher()
    fetcher.fetch(_ctx(tmp_path))
    fetcher.fetch(_ctx(tmp_path, drugsfda_url="https://example.invalid/custom.zip"))
    assert seen == [drugsfda.DRUGSFDA_DATA_FILES_URL, "https://example.invalid/custom.zip"]


def test_drugsfda_stale_cache_rechecks_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        calls.append(url)
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    fetcher = drugsfda.DrugsFDAFetcher()
    fetcher.fetch(_ctx(tmp_path))
    _age_release(_store_for(tmp_path), "drugsfda/drugsfda_data_files.zip", days=8)
    fetcher.fetch(_ctx(tmp_path))
    assert calls == [drugsfda.DRUGSFDA_DATA_FILES_URL, drugsfda.DRUGSFDA_DATA_FILES_URL]


def test_drugsfda_non_positive_max_age_disables_cache_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        calls.append(url)
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    fetcher = drugsfda.DrugsFDAFetcher()
    fetcher.fetch(_ctx(tmp_path))
    fetcher.fetch(_ctx(tmp_path, drugsfda_max_age_days=0))
    assert calls == [drugsfda.DRUGSFDA_DATA_FILES_URL, drugsfda.DRUGSFDA_DATA_FILES_URL]


def test_drugsfda_non_numeric_max_age_falls_back_to_the_default_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A param that is not a real number falls back to the 7-day default — never to 'no gate'.

    Airflow params arrive untyped, so a string ``"14"`` (or a ``True`` that ``isinstance(_, int)``
    would otherwise accept) must not silently disable or widen the freshness window.
    """
    calls: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        calls.append(url)
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    fetcher = drugsfda.DrugsFDAFetcher()
    fetcher.fetch(_ctx(tmp_path))
    # Both bogus values resolve to the default 7 days, so the fresh release is still reused.
    fetcher.fetch(_ctx(tmp_path, drugsfda_max_age_days="14"))
    fetcher.fetch(_ctx(tmp_path, drugsfda_max_age_days=True))
    assert calls == [drugsfda.DRUGSFDA_DATA_FILES_URL]

    # And it really is the DEFAULT window, not an unbounded one: past 7 days the gate reopens.
    _age_release(_store_for(tmp_path), "drugsfda/drugsfda_data_files.zip", days=8)
    fetcher.fetch(_ctx(tmp_path, drugsfda_max_age_days="14"))
    assert calls == [drugsfda.DRUGSFDA_DATA_FILES_URL, drugsfda.DRUGSFDA_DATA_FILES_URL]


def test_drugsfda_unparsable_retrieved_at_refetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest whose ``retrieved_at`` will not parse has no measurable age, so the gate cannot apply."""
    calls: list[str] = []

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        calls.append(url)
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    fetcher = drugsfda.DrugsFDAFetcher()
    fetcher.fetch(_ctx(tmp_path))

    store = _store_for(tmp_path)
    cached = store.cached_ref("drugsfda/drugsfda_data_files.zip")
    assert cached is not None
    manifest_path = store.manifest_path(cached.blake3)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["source"]["retrieved_at"] = "whenever"  # provenance present but not a timestamp
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    fetcher.fetch(_ctx(tmp_path))
    assert calls == [drugsfda.DRUGSFDA_DATA_FILES_URL, drugsfda.DRUGSFDA_DATA_FILES_URL]


def test_drugsfda_real_fetch_cleans_up_absent_staged_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``finally`` cleanup tolerates an absent staged file (65->68); ingest then fails loudly."""

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        dest.unlink(missing_ok=True)  # downloader left nothing behind
        return dest

    monkeypatch.setattr(drugsfda, "download_drugsfda_zip", fake_download)
    with pytest.raises(FileNotFoundError):
        drugsfda.DrugsFDAFetcher().fetch(_ctx(tmp_path))


# --- sources/faers: discovery + remote edges -----------------------------------


def test_faers_remote_no_quarters_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = faers.FAERSFetcher()
    monkeypatch.setattr(fetcher, "fetch_index", lambda ctx: "<html>nothing here</html>")
    assert fetcher.fetch(_ctx(tmp_path)) == []


def test_faers_download_quarter_ingests_via_monkeypatched_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_http_download(url: str, dest: Path, *, timeout: float) -> Path:
        _write_fake_zip(dest)
        return dest

    monkeypatch.setattr(faers, "_http_download", fake_http_download)
    ctx = _ctx(tmp_path)
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
    refs = dailymed.fetch(_ctx(tmp_path))
    # Two SPL members ingested (.xml + .xml.gz); the .txt member is skipped.
    assert len(refs) == 2
    assert all(r.uri.suffix in {".xml", ".gz"} for r in refs)
    assert all(r.uri.exists() for r in refs)


def test_dailymed_real_fetch_descends_into_nested_doc_zips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Real DailyMed releases nest one zip per SPL document; the SPL .xml lives inside each inner zip.
    spl = b"<splBatch><document><setId>N1</setId></document></splBatch>"
    doc_zip = _release_zip({"subdir/": b"", "ABCD-1234.xml": spl, "ABCD-1234-image01.jpg": b"jpeg"})
    release_bytes = _release_zip(
        {"prescription/20060131_ABCD-1234.zip": doc_zip, "prescription/20060131_EFGH-5678.zip": _release_zip({"EFGH-5678.xml": spl})}
    )
    _patch_urlopen(
        monkeypatch,
        {"spl-resources-all-drug-labels.cfm": _FakeResponse(_INDEX_HTML.encode()), "dm_spl_release_human_rx_part1.zip": _FakeResponse(release_bytes)},
    )
    refs = dailymed.fetch(_ctx(tmp_path))
    # Two nested doc zips -> two SPL .xml ingested; the dir entry + .jpg media are skipped.
    assert len(refs) == 2
    assert all(r.uri.suffix == ".xml" for r in refs)
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
    refs = dailymed.fetch(_ctx(tmp_path, release_limit=1))
    assert len(refs) == 1  # only the first release processed


def test_dailymed_real_fetch_no_releases_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, {"spl-resources-all-drug-labels.cfm": _FakeResponse(b"<html>no zips</html>")})
    with pytest.raises(RuntimeError, match="no DailyMed full-release ZIPs"):
        dailymed.fetch(_ctx(tmp_path))


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
    assert dailymed.fetch(_ctx(tmp_path)) == []


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
    # max_age_days=0 disables the freshness gate so the second run exercises conditional GET.
    ctx = _ctx(tmp_path, dailymed_max_age_days=0)
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
        dailymed.fetch(_ctx(tmp_path))


def _raise_304(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 304, "Not Modified", {}, io.BytesIO(b""))  # type: ignore[arg-type]


def test_dailymed_index_304_without_cache_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, *, timeout: float = 60.0) -> _FakeResponse:
        raise _raise_304(request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="no cached copy exists"):
        dailymed.fetch(_ctx(tmp_path))


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
    ctx = _ctx(tmp_path)
    first = dailymed.fetch(ctx)
    assert len(first) == 1
    state["not_modified"] = True
    second = dailymed.fetch(ctx)  # index 304 -> cached index text reused -> same release ref
    assert [r.blake3 for r in second] == [r.blake3 for r in first]


def test_dailymed_release_304_reexpands_cached_spl_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    # max_age_days=0 disables the freshness gate so the second run exercises the release-304 path.
    ctx = _ctx(tmp_path, dailymed_max_age_days=0)
    first = dailymed.fetch(ctx)
    assert len(first) == 1
    state["not_modified"] = True
    # Release 304: the cached release ZIP is re-expanded into its SPL members (no re-download) —
    # the same SPL refs the first run produced (what the SPL extractor consumes), not the ZIP itself.
    second = dailymed.fetch(ctx)
    assert [r.blake3 for r in second] == [r.blake3 for r in first]
    assert second[0].uri.name.endswith(".xml")


def test_dailymed_release_304_reexpands_when_the_member_record_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 304 with no completion record falls back to re-expanding the cached ZIP, not to ``[]``.

    The record is written only after an expansion finishes, so an interrupted run leaves the
    release ZIP cached with no member set — the recovery path must rebuild the SPL refs the
    extractor consumes, without re-downloading.
    """
    zip_bytes = _release_zip({"drug1.xml": b"<splBatch/>"})
    state = {"not_modified": False}

    def fake_urlopen(request: Any, *, timeout: float = 60.0) -> _FakeResponse:
        url = request.full_url
        if "spl-resources-all-drug-labels.cfm" in url:
            return _FakeResponse(_INDEX_HTML.encode())
        if state["not_modified"]:
            raise _raise_304(url)
        return _FakeResponse(zip_bytes, {"ETag": "zip-e"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ctx = _ctx(tmp_path, dailymed_max_age_days=0)  # freshness gate off, so the 304 path runs
    first = dailymed.fetch(ctx)
    assert len(first) == 1

    # Simulate the interrupted expansion: the ZIP is still cached, its member record is not.
    _store_for(tmp_path).invalidate_cached_refs("dailymed/dm_spl_release_human_rx_part1.zip")
    state["not_modified"] = True
    second = dailymed.fetch(ctx)

    assert [ref.blake3 for ref in second] == [ref.blake3 for ref in first]
    assert second[0].uri.name.endswith(".xml")


def test_dailymed_release_304_without_cache_returns_empty(tmp_path: Path) -> None:
    store = _store_for(tmp_path)
    # No alias recorded for this release -> cached_ref None -> [] on a 304.
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


def _age_release(store: ArtifactStore, alias: str, *, days: float) -> None:
    """Rewrite the stored release's manifest so its ``retrieved_at`` is ``days`` in the past."""
    cached = store.cached_ref(alias)
    assert cached is not None
    manifest_path = store.manifest_path(cached.blake3)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["source"]["retrieved_at"] = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


# --- sources/dailymed: freshness gate (no re-download of a fresh release) ------


def test_dailymed_fresh_release_skips_zip_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spl = b"<splBatch/>"
    zip_bytes = _release_zip({"drug1.xml": spl})
    requests: list[urllib.request.Request] = []
    _patch_urlopen(
        monkeypatch,
        {"spl-resources-all-drug-labels.cfm": _FakeResponse(_INDEX_HTML.encode()), "dm_spl_release_human_rx_part1.zip": _FakeResponse(zip_bytes)},
        requests,
    )
    ctx = _ctx(tmp_path)
    first = dailymed.fetch(ctx)
    assert len(first) == 1
    requests.clear()
    # The stored release was fetched moments ago (< 7 days): the gate skips the ZIP download
    # entirely — only the small index conditional GET goes out, and the cached ZIP is re-expanded
    # into the same SPL refs.
    second = dailymed.fetch(ctx)
    assert [r.full_url for r in requests] == [dailymed.FULL_RELEASE_INDEX_URL]
    assert [r.blake3 for r in second] == [r.blake3 for r in first]


def test_dailymed_missing_completion_record_reexpands_without_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spl = b"<splBatch/>"
    zip_bytes = _release_zip({"drug1.xml": spl})
    requests: list[urllib.request.Request] = []
    _patch_urlopen(
        monkeypatch,
        {"spl-resources-all-drug-labels.cfm": _FakeResponse(_INDEX_HTML.encode()), "dm_spl_release_human_rx_part1.zip": _FakeResponse(zip_bytes)},
        requests,
    )
    ctx = _ctx(tmp_path)
    first = dailymed.fetch(ctx)
    marker = Workdir(ctx.workdir).aliases / "dailymed" / "dm_spl_release_human_rx_part1.zip.members.json"
    marker.unlink()
    requests.clear()

    second = dailymed.fetch(ctx)

    assert [r.blake3 for r in second] == [r.blake3 for r in first]
    assert [r.full_url for r in requests] == [dailymed.FULL_RELEASE_INDEX_URL]
    assert marker.exists()


def test_dailymed_stale_release_redownloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spl = b"<splBatch/>"
    zip_bytes = _release_zip({"drug1.xml": spl})
    requests: list[urllib.request.Request] = []
    _patch_urlopen(
        monkeypatch,
        {"spl-resources-all-drug-labels.cfm": _FakeResponse(_INDEX_HTML.encode()), "dm_spl_release_human_rx_part1.zip": _FakeResponse(zip_bytes)},
        requests,
    )
    ctx = _ctx(tmp_path)
    dailymed.fetch(ctx)
    _age_release(_store_for(tmp_path), "dailymed/dm_spl_release_human_rx_part1.zip", days=10)
    requests.clear()
    dailymed.fetch(ctx)  # stored copy older than the 7-day window -> conditional GET goes out
    assert [r for r in requests if "part1.zip" in r.full_url]


def test_dailymed_mixed_freshness_downloads_only_stale_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    two_release_index = (
        "<html><h2>Full Releases</h2>"
        f'<a href="{_RELEASE_URL}">part1</a>'
        '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part2.zip">part2</a>'
        "</html>"
    )
    requests: list[urllib.request.Request] = []
    _patch_urlopen(
        monkeypatch,
        {
            "spl-resources-all-drug-labels.cfm": _FakeResponse(two_release_index.encode()),
            "part1.zip": _FakeResponse(_release_zip({"a.xml": b"<splBatch>A</splBatch>"})),
            "part2.zip": _FakeResponse(_release_zip({"b.xml": b"<splBatch>B</splBatch>"})),
        },
        requests,
    )
    ctx = _ctx(tmp_path)
    dailymed.fetch(ctx)
    _age_release(_store_for(tmp_path), "dailymed/dm_spl_release_human_rx_part2.zip", days=8)
    requests.clear()
    dailymed.fetch(ctx)
    # Only the stale part2 is re-checked; fresh part1 never hits the network.
    assert [r.full_url for r in requests if r.full_url.endswith(".zip")] == [
        "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part2.zip"
    ]


def test_dailymed_force_bypasses_freshness_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zip_bytes = _release_zip({"drug1.xml": b"<splBatch/>"})
    requests: list[urllib.request.Request] = []
    _patch_urlopen(
        monkeypatch,
        {"spl-resources-all-drug-labels.cfm": _FakeResponse(_INDEX_HTML.encode()), "dm_spl_release_human_rx_part1.zip": _FakeResponse(zip_bytes)},
        requests,
    )
    dailymed.fetch(_ctx(tmp_path))
    requests.clear()
    dailymed.fetch(_ctx(tmp_path, force=True))  # force re-checks even a fresh release
    assert [r for r in requests if "part1.zip" in r.full_url]


def test_dailymed_fresh_gate_falls_through_when_cache_vanishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # If the cached ZIP disappears between the age check and the reuse, the gate falls through to
    # the download flow (here simulated as a 304 with no cache -> []).
    store = _store_for(tmp_path)
    staging = Workdir(store.workdir.root).root / ".staging" / "dailymed"
    staging.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dailymed, "_release_age_days", lambda *args, **kwargs: 0.0)

    def fake_conditional(url: str, dest: Path, source: Any) -> tuple[None, None]:
        dest.unlink(missing_ok=True)  # simulate 304: nothing downloaded
        return None, None

    monkeypatch.setattr(dailymed, "_conditional_download", fake_conditional)
    assert dailymed._download_one("https://x/never.zip", staging, store, max_age_days=7.0) == []


def test_dailymed_release_age_days_edges(tmp_path: Path) -> None:
    store = _store_for(tmp_path)
    assert dailymed._release_age_days(store, "dailymed/absent.zip") is None  # no alias -> stale

    src = tmp_path / "rel.zip"
    src.write_bytes(b"zip-bytes")
    # Missing / unparseable / naive retrieved_at all mean "not reusable" (stale).
    for retrieved_at in (None, "not-a-date", "2026-08-01T00:00:00"):
        store.ingest(src, alias="dailymed/rel.zip", source=SourceBlock(url="https://x/rel.zip", retrieved_at=retrieved_at))
        assert dailymed._release_age_days(store, "dailymed/rel.zip") is None, retrieved_at
    # Valid aware timestamp -> age in days against an explicit now.
    store.ingest(src, alias="dailymed/rel.zip", source=SourceBlock(url="https://x/rel.zip", retrieved_at="2026-08-01T00:00:00+00:00"))
    age = dailymed._release_age_days(store, "dailymed/rel.zip", now=datetime(2026, 8, 8, 12, 0, 0, tzinfo=UTC))
    assert age == pytest.approx(7.5)
    # Alias present but the cached file vanished from the store -> stale.
    ref, _ = store.ingest(src, alias="dailymed/gone.zip", source=SourceBlock(url="https://x/gone.zip", retrieved_at="2026-08-01T00:00:00+00:00"))
    ref.uri.unlink()
    assert dailymed._release_age_days(store, "dailymed/gone.zip") is None


def test_dailymed_max_age_days_param_resolution(tmp_path: Path) -> None:
    assert dailymed._max_age_days(_ctx(tmp_path)) == 7.0  # absent -> default window
    assert dailymed._max_age_days(_ctx(tmp_path, dailymed_max_age_days=None)) == 7.0  # null config -> default
    assert dailymed._max_age_days(_ctx(tmp_path, dailymed_max_age_days=14)) == 14.0
    assert dailymed._max_age_days(_ctx(tmp_path, dailymed_max_age_days=0.5)) == 0.5
    for bad in (0, -1, "week", True):  # non-positive / non-numeric -> gate disabled (always re-check)
        assert dailymed._max_age_days(_ctx(tmp_path, dailymed_max_age_days=bad)) is None


# --- store.cached_ref: alias / .path pointer resolution (shared by fetchers) ---


def test_cached_ref_missing_alias_and_path(tmp_path: Path) -> None:
    store = _store_for(tmp_path)
    assert store.cached_ref("dailymed/absent.zip") is None  # no alias
    # Alias present but sibling .path pointer missing -> None.
    wd = Workdir(store.workdir.root)
    alias_file = wd.aliases / "dailymed" / "half.zip"
    alias_file.parent.mkdir(parents=True, exist_ok=True)
    alias_file.write_text("b3:deadbeef", encoding="utf-8")
    assert store.cached_ref("dailymed/half.zip") is None


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
