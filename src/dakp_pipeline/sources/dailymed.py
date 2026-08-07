"""DailyMed SPL acquisition — real fetcher behind the :class:`Fetcher` protocol.

Idempotent and content-addressed (BLAKE3 manifest + checksums), recording source provenance
(URL / ETag / Last-Modified) into the artifact manifest. Downloads the DailyMed full-release
index + release ZIPs using **stdlib ``urllib`` only** (no ``requests``), streaming each release
into the content-addressed store. Re-ingesting an identical release is a cache hit; conditional
GET (``If-None-Match`` / ``If-Modified-Since``) skips bytes the server confirms unchanged. Each
release ZIP's SPL XML members are extracted and ingested individually (the SPL extractor reads
``.xml``/``.xml.gz``, not ZIPs), mirroring the legacy ``getFullRelease.pl`` "extract XMLs into
``xmls/<bin>/...xml.gz``" step. ``release_limit`` (run param) bounds how many releases a bounded
run processes. Offline tests monkeypatch the module-level ``fetch`` / the stdlib ``urlopen`` seam.

The legacy ``ref/legacy/DailyMed/bin/getFullRelease.pl`` is intentionally **not** replicated: it
destructively stashed whole download directories (``rm -r $ddir.prev``; ``mv $ddir
$ddir.prev``) before re-fetching. This fetcher never moves or deletes shared state — it
only ever *adds* immutable, content-addressed artifacts and overwrites human-readable
aliases in place.

``fetch = DailyMedFetcher().fetch`` is exposed at module scope so tests can
``monkeypatch.setattr(dailymed, "fetch", ...)``. The internal ``_download_full_release``
is also module-level so wiring tests can stub it (no real network in CI).
"""

from __future__ import annotations

import http
import io
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import logger, progress, stats, step
from dakp_pipeline.paths import Workdir

# DailyMed "all drug labels" full-release listing (legacy getFullRelease.pl target).
FULL_RELEASE_INDEX_URL = "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"
# Anchors the parser to the "Full Releases" section of the index page.
_FULL_RELEASES_HEADING = "Full Releases"
# Matches release ZIP hrefs as written by the DailyMed index page, e.g.
#   <a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip">
_RELEASE_ZIP_HREF = re.compile(r'href="(?P<url>https?://[^"]+\.zip)"', re.IGNORECASE)

_DOWNLOAD_TIMEOUT = 60.0
_CHUNK = 1 << 20  # 1 MiB streaming window.
#: Narration prefix for every log line this fetcher emits (one stat per line).
_EVENT = "acquire_dailymed"
#: One INFO progress line per this many ingested SPL documents (per-document detail is DEBUG).
_SPL_PROGRESS_EVERY = 2500


class DailyMedFetcher:
    """Acquire DailyMed SPL full-release artifacts over the network (stdlib only)."""

    def fetch(self, ctx: TaskContext) -> list[ArtifactRef]:
        return _download_full_release(ctx)


fetch = DailyMedFetcher().fetch


def _download_full_release(ctx: TaskContext) -> list[ArtifactRef]:
    """Download the DailyMed full release into the content-addressed store.

    Idempotent and non-destructive: each release ZIP is streamed to a workdir-local
    staging file, then :meth:`ArtifactStore.ingest` copies it into ``by-hash/<hex>/``
    (cache hit if identical). The staging file is removed after ingest; shared state is
    never renamed or deleted. Conditional headers skip bytes the server reports unchanged.
    """
    with step(logger, _EVENT):
        store = ArtifactStore(Workdir(ctx.workdir))
        staging = Workdir(ctx.workdir).root / ".staging" / "dailymed"
        staging.mkdir(parents=True, exist_ok=True)

        index_html = _fetch_index(ctx, staging, store)
        release_urls = _apply_release_limit(_parse_release_zips(index_html), ctx)
        stats(logger, _EVENT, releases_discovered=len(release_urls))
        if not release_urls:
            msg = "no DailyMed full-release ZIPs found in index (the listing page layout may have changed)"
            raise RuntimeError(msg)

        refs: list[ArtifactRef] = []
        for url in release_urls:
            refs.extend(_download_one(url, staging, store))
        stats(logger, _EVENT, spl_artifacts_acquired=len(refs))
        return refs


def _fetch_index(ctx: TaskContext, staging: Path, store: ArtifactStore) -> str:
    """Fetch (and cache) the full-release index HTML, returning its text.

    On HTTP 304 the previously cached store copy is read instead of re-downloading.
    """
    alias = "dailymed/spl-resources-all-drug-labels.html"
    dest = staging / "spl-resources-all-drug-labels.html"
    source = _prior_source(store, alias=alias)
    index_event = f"{_EVENT} index"
    stats(logger, index_event, url=FULL_RELEASE_INDEX_URL)
    started = time.monotonic()
    etag, last_modified = _conditional_download(FULL_RELEASE_INDEX_URL, dest, source)
    if not dest.exists():
        cached = store.cached_ref(alias)
        if cached is None:
            msg = "server returned 304 for the DailyMed index but no cached copy exists"
            raise RuntimeError(msg)
        stats(logger, index_event, cache_fresh=True)
        return cached.uri.read_text(encoding="utf-8", errors="replace")
    stats(logger, index_event, bytes=dest.stat().st_size, elapsed_s=round(time.monotonic() - started, 3))
    store.ingest(dest, alias=alias, source=SourceBlock(url=FULL_RELEASE_INDEX_URL, etag=etag, last_modified=last_modified, retrieved_at=_now_iso()))
    text = dest.read_text(encoding="utf-8", errors="replace")
    dest.unlink(missing_ok=True)
    return text


def _parse_release_zips(index_html: str) -> list[str]:
    """Return the ordered, de-duplicated full-release ZIP URLs from the index page.

    Only links appearing after the ``Full Releases`` heading are kept, mirroring the
    legacy Perl parser's section scan.
    """
    head = index_html.find(_FULL_RELEASES_HEADING)
    section = index_html[head:] if head != -1 else index_html
    seen: set[str] = set()
    urls: list[str] = []
    for match in _RELEASE_ZIP_HREF.finditer(section):
        url = match.group("url")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _download_one(url: str, staging: Path, store: ArtifactStore) -> list[ArtifactRef]:
    """Download one release ZIP and ingest its SPL XML members.

    The release ZIP is recorded under its ``dailymed/<name>`` alias (provenance anchor +
    conditional-GET source for future runs), then its SPL XML members are extracted and
    ingested individually — those ``.xml``/``.xml.gz`` refs are what the SPL extractor
    consumes. On HTTP 304 the cached release ZIP is re-expanded into its SPL members (no
    re-download), so a re-run yields the same SPL refs the extractor needs; returns ``[]`` only
    if a 304 occurs with no cached copy resolvable.
    """
    name = url.rsplit("/", 1)[-1]
    alias = f"dailymed/{name}"
    dest = staging / name
    out_dir = staging / f"{name}-xml"
    release_event = f"{_EVENT} release"
    stats(logger, release_event, release=name)
    stats(logger, release_event, level="DEBUG", url=url)
    source = _prior_source(store, alias=alias)
    started = time.monotonic()
    etag, last_modified = _conditional_download(url, dest, source)
    if not dest.exists():
        # 304 Not Modified: the cached release ZIP is still current — re-expand it into its SPL XML
        # members (no re-download). The SPL extractor consumes the .xml/.xml.gz refs, not the ZIP.
        stats(logger, release_event, cache_fresh=True)
        cached = store.cached_ref(alias)
        if cached is None:
            return []
        refs = _expand_release_zip(cached.uri, out_dir, store, SourceBlock(url=url, retrieved_at=_now_iso()), release_name=name)
        stats(logger, release_event, spl_documents_ingested=len(refs), elapsed_s=round(time.monotonic() - started, 3))
        return refs
    stats(logger, release_event, bytes=dest.stat().st_size, elapsed_s=round(time.monotonic() - started, 3))
    src_block = SourceBlock(url=url, etag=etag, last_modified=last_modified, retrieved_at=_now_iso())
    ref, cache_hit = store.ingest(dest, alias=alias, source=src_block)
    stats(logger, release_event, blake3=ref.blake3, cache_hit=cache_hit)
    refs = _expand_release_zip(dest, out_dir, store, src_block, release_name=name)
    dest.unlink(missing_ok=True)  # staged zip no longer needed once its XMLs are content-addressed
    stats(logger, release_event, spl_documents_ingested=len(refs), elapsed_s=round(time.monotonic() - started, 3))
    return refs


def _apply_release_limit(urls: list[str], ctx: TaskContext) -> list[str]:
    """Slice to the first N full releases when ``release_limit`` is set (<=0 / None = all).

    Bounds a run to a tiny real DailyMed scope (e.g. ``release_limit=1`` via the integration test
    harness, as the offline smoke test does), mirroring FAERS ``quarter_limit``.
    """
    limit = ctx.params.get("release_limit")
    if not isinstance(limit, int) or limit <= 0:
        return urls
    return urls[:limit]


def _expand_release_zip(zip_path: Path, out_dir: Path, store: ArtifactStore, source: SourceBlock, *, release_name: str) -> list[ArtifactRef]:
    """Ingest the SPL XML documents of a DailyMed release ZIP as individual artifacts.

    DailyMed full-release ZIPs are nested: the outer archive holds one ``.zip`` per SPL document
    (e.g. ``prescription/<date>_<uuid>.zip``), and each inner archive holds that document's
    ``.xml`` (plus media we ignore). The SPL extractor reads ``.xml``/``.xml.gz`` (not ZIPs), so
    this descends into each per-document ZIP and content-addresses the SPL XML inside under
    ``dailymed/<release>::<member>`` with the release's source provenance. A top-level
    ``.xml``/``.xml.gz`` member (older release layouts) is ingested directly. Non-SPL members are
    skipped; a release yielding no SPL documents returns ``[]`` (logged) rather than failing the
    whole acquisition. Extracted members are staged under ``out_dir`` (caller-owned scratch).
    """
    expansion_event = f"{_EVENT} expansion"
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: list[ArtifactRef] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        total = sum(1 for info in members if _looks_like_spl_target(Path(info.filename).name))
        stats(logger, expansion_event, release=release_name, spl_targets=total)
        done = 0
        for info in members:
            name = Path(info.filename).name
            if name.lower().endswith(".zip"):
                # Nested per-document archive: ingest the SPL XML it contains.
                ingested = _expand_doc_zip(archive.read(info), out_dir, store, source, release_name=release_name)
                refs.extend(ingested)
                done += len(ingested)
            elif _looks_like_spl_name(name):
                refs.append(_ingest_spl_bytes(archive.read(info), name, out_dir, store, source, release_name=release_name))
                done += 1
            progress(logger, expansion_event, done, total, every=_SPL_PROGRESS_EVERY)
    if not refs:
        logger.warning("{}: release {} contained no SPL XML members", _EVENT, release_name)
    return refs


def _expand_doc_zip(doc_bytes: bytes, out_dir: Path, store: ArtifactStore, source: SourceBlock, *, release_name: str) -> list[ArtifactRef]:
    """Ingest the SPL XML member(s) of one nested per-document DailyMed ZIP (held in memory)."""
    refs: list[ArtifactRef] = []
    with zipfile.ZipFile(io.BytesIO(doc_bytes)) as doc_archive:
        for info in doc_archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename).name
            if _looks_like_spl_name(member):
                refs.append(_ingest_spl_bytes(doc_archive.read(info), member, out_dir, store, source, release_name=release_name))
    return refs


def _ingest_spl_bytes(data: bytes, member: str, out_dir: Path, store: ArtifactStore, source: SourceBlock, *, release_name: str) -> ArtifactRef:
    """Write one SPL member to staging and content-address it under the release alias.

    A flat ``::<member>`` alias (not nested under the release) never collides with the release
    ZIP's own ``dailymed/<release>`` alias file; per-document SPL names are UUIDs, so they are
    unique within a release.
    """
    target = out_dir / member
    target.write_bytes(data)
    ref, _ = store.ingest(target, alias=f"dailymed/{release_name}::{member}", source=source)
    target.unlink(missing_ok=True)
    return ref


def _looks_like_spl_name(name: str) -> bool:
    """Whether a release member is an SPL document the extractor can read."""
    return name.lower().endswith((".xml.gz", ".xml"))


def _looks_like_spl_target(name: str) -> bool:
    """Whether a release member can yield SPL documents (a nested doc zip or an SPL file)."""
    return name.lower().endswith((".zip", ".xml.gz", ".xml"))


def _conditional_download(url: str, dest: Path, source: SourceBlock | None) -> tuple[str | None, str | None]:
    """Stream ``url`` to ``dest`` with conditional headers; return (etag, last_modified).

    On HTTP 304 the destination is left absent and ``(None, None)`` is returned so the
    caller treats the artifact as cache-fresh. Any other HTTP error is raised. A stale
    staging file is removed before the request so it can never masquerade as fresh.
    """
    headers: dict[str, str] = {}
    if source is not None:
        if source.etag:
            headers["If-None-Match"] = source.etag
        if source.last_modified:
            headers["If-Modified-Since"] = source.last_modified
    dest.unlink(missing_ok=True)  # never let a stale staging file masquerade as fresh
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as resp:
            etag = resp.headers.get("ETag")
            last_modified = resp.headers.get("Last-Modified")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as handle:
                shutil.copyfileobj(resp, handle, length=_CHUNK)
            return etag, last_modified
    except urllib.error.HTTPError as exc:
        if exc.code == http.HTTPStatus.NOT_MODIFIED:
            exc.close()
            return None, None
        raise


def _prior_source(store: ArtifactStore, *, alias: str) -> SourceBlock | None:
    """Read the source block previously recorded for ``alias`` (for conditional GET)."""
    alias_path = Workdir(store.workdir.root).aliases / alias
    if not alias_path.exists():
        return None
    artifact_id = alias_path.read_text(encoding="utf-8").strip()
    manifest = store.read_manifest(artifact_id)
    return manifest.source if manifest is not None else None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["FULL_RELEASE_INDEX_URL", "DailyMedFetcher", "fetch"]
