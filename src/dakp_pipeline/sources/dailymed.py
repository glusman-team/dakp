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
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.downloads import infer_media_type
from dakp_pipeline.io.manifests import SourceBlock
from dakp_pipeline.logging_setup import bind
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
    store = ArtifactStore(Workdir(ctx.workdir))
    staging = Workdir(ctx.workdir).root / ".staging" / "dailymed"
    staging.mkdir(parents=True, exist_ok=True)
    log = bind(task_id="fetch_dailymed")

    index_html = _fetch_index(ctx, staging, store)
    release_urls = _apply_release_limit(_parse_release_zips(index_html), ctx)
    log.info("discovered dailyMed full releases", count=len(release_urls))
    if not release_urls:
        msg = "no DailyMed full-release ZIPs found in index (the listing page layout may have changed)"
        raise RuntimeError(msg)

    refs: list[ArtifactRef] = []
    for url in release_urls:
        refs.extend(_download_one(url, staging, store))
    return refs


def _fetch_index(ctx: TaskContext, staging: Path, store: ArtifactStore) -> str:
    """Fetch (and cache) the full-release index HTML, returning its text.

    On HTTP 304 the previously cached store copy is read instead of re-downloading.
    """
    alias = "dailymed/spl-resources-all-drug-labels.html"
    dest = staging / "spl-resources-all-drug-labels.html"
    source = _prior_source(store, alias=alias)
    etag, last_modified = _conditional_download(FULL_RELEASE_INDEX_URL, dest, source)
    if not dest.exists():
        cached = _cached_ref(store, alias=alias)
        if cached is None:
            msg = "server returned 304 for the DailyMed index but no cached copy exists"
            raise RuntimeError(msg)
        return cached.uri.read_text(encoding="utf-8", errors="replace")
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
    consumes. On HTTP 304 the previously cached release ref is returned unchanged
    (idempotent re-run); returns ``[]`` only if a 304 occurs with no cached copy resolvable.
    """
    name = url.rsplit("/", 1)[-1]
    alias = f"dailymed/{name}"
    dest = staging / name
    source = _prior_source(store, alias=alias)
    etag, last_modified = _conditional_download(url, dest, source)
    if not dest.exists():
        # 304 Not Modified: the cached artifact is still current — return it, do not drop it.
        cached = _cached_ref(store, alias=alias)
        return [cached] if cached is not None else []
    src_block = SourceBlock(url=url, etag=etag, last_modified=last_modified, retrieved_at=_now_iso())
    store.ingest(dest, alias=alias, source=src_block)
    refs = _expand_release_zip(dest, store, src_block, release_name=name)
    dest.unlink(missing_ok=True)  # staged zip no longer needed once its XMLs are content-addressed
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


def _expand_release_zip(zip_path: Path, store: ArtifactStore, source: SourceBlock, *, release_name: str) -> list[ArtifactRef]:
    """Ingest the SPL XML documents of a DailyMed release ZIP as individual artifacts.

    DailyMed full-release ZIPs are nested: the outer archive holds one ``.zip`` per SPL document
    (e.g. ``prescription/<date>_<uuid>.zip``), and each inner archive holds that document's
    ``.xml`` (plus media we ignore). The SPL extractor reads ``.xml``/``.xml.gz`` (not ZIPs), so
    this descends into each per-document ZIP and content-addresses the SPL XML inside under
    ``dailymed/<release>::<member>`` with the release's source provenance. A top-level
    ``.xml``/``.xml.gz`` member (older release layouts) is ingested directly. Non-SPL members are
    skipped; a release yielding no SPL documents returns ``[]`` (logged) rather than failing the
    whole acquisition.
    """
    log = bind(task_id="fetch_dailymed", release=release_name)
    out_dir = zip_path.parent / f"{zip_path.stem}-xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: list[ArtifactRef] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name.lower().endswith(".zip"):
                # Nested per-document archive: ingest the SPL XML it contains.
                refs.extend(_expand_doc_zip(archive.read(info), out_dir, store, source, release_name=release_name))
            elif _looks_like_spl_name(name):
                refs.append(_ingest_spl_bytes(archive.read(info), name, out_dir, store, source, release_name=release_name))
    if not refs:
        log.warning("release ZIP contained no SPL XML members", release=release_name)
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


def _cached_ref(store: ArtifactStore, *, alias: str) -> ArtifactRef | None:
    """Reconstruct an :class:`ArtifactRef` for an already-ingested artifact (used on 304).

    Reads the alias + sibling ``.path`` pointer written by :meth:`ArtifactStore.ingest`.
    Returns ``None`` if the alias or path pointer is missing.
    """
    wd = Workdir(store.workdir.root)
    id_path = wd.aliases / alias
    if not id_path.exists():
        return None
    artifact_id = id_path.read_text(encoding="utf-8").strip()
    path_file = wd.aliases / f"{alias}.path"
    if not path_file.exists():
        return None
    uri = Path(path_file.read_text(encoding="utf-8").strip())
    return ArtifactRef(uri=uri, blake3=artifact_id, media_type=infer_media_type(uri), manifest=store.manifest_path(artifact_id))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["FULL_RELEASE_INDEX_URL", "DailyMedFetcher", "fetch"]
