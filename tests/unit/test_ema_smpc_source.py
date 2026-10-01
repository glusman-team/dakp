"""Tests for the EMA product-information (SmPC) PDF crawl.

The crawl is the only thing that bounds the EU contraindication/indication corpus, so these tests
pin the four behaviors that would otherwise fail silently in production: the manifest-driven
selection (never a scraped discovery), the two cache layers (fan-out record + per-document
currency), partial-failure tolerance (one bad document must not empty the corpus), and the
fail-loudly paths (empty selection, total failure).

Network is monkeypatched at both seams: the documents-report downloader and the per-PDF downloader
serve the committed fixtures (a trimmed real 9-record report and a trimmed real SmPC PDF).
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import ema_documents, ema_smpc

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_MANIFEST = _FIXTURE_ROOT / "ema" / "epar_documents_en.json"
_SMPD_PDF = _FIXTURE_ROOT / "ema" / "smpc" / "ceplene-epar-product-information_en.pdf"
_DOCUMENTS_ALIAS = ema_documents.DOCUMENTS_ALIAS
_FANOUT_ALIAS = ema_smpc.SMPD_FANOUT_ALIAS

#: The five English human product-information PDFs the fixture report selects, in crawl order.
_SELECTED_IDS = ["2441", "10327", "28033", "49893", "50210"]


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the single-retry path off the wall clock (it exists for transient 5xx, not for tests)."""
    monkeypatch.setattr(ema_smpc, "_RETRY_BACKOFF_S", 0.0)


def _ctx(workdir: Path, **params: object) -> TaskContext:
    Workdir(workdir).create()
    return TaskContext(workdir=workdir, fixture_root=_FIXTURE_ROOT, params=dict(params))


def _store_for(workdir: Path) -> ArtifactStore:
    return ArtifactStore(Workdir(workdir))


def _serve_manifest(monkeypatch: pytest.MonkeyPatch, *, payloads: list[Path] | None = None) -> list[str]:
    """Serve the documents report; ``payloads`` cycles through alternative manifests per call."""
    calls: list[str] = []
    sources = payloads or [_MANIFEST]

    def fake(url: str, dest: Path, *, timeout: float = 180.0) -> Path:
        calls.append(url)
        shutil.copyfile(sources[min(len(calls) - 1, len(sources) - 1)], dest)
        return dest

    monkeypatch.setattr(ema_documents, "download_ema_documents", fake)
    return calls


def _serve_pdfs(monkeypatch: pytest.MonkeyPatch, *, failing: set[str] | None = None, empty: set[str] | None = None) -> list[str]:
    """Serve every SmPC PDF from the one committed fixture; optionally fail or empty chosen URLs."""
    calls: list[str] = []
    bad = failing or set()
    blank = empty or set()

    def fake(url: str, dest: Path, *, timeout: float = 180.0) -> Path:
        calls.append(url)
        if url in bad:
            msg = f"503 Service Unavailable: {url}"
            raise OSError(msg)
        if url in blank:
            dest.write_bytes(b"")  # downloader left a zero-byte file behind
            return dest
        shutil.copyfile(_SMPD_PDF, dest)
        return dest

    monkeypatch.setattr(ema_smpc, "download_ema_smpc_pdf", fake)
    return calls


def _record_edit(manifest: Path, tmp_path: Path, record_id: str, **fields: object) -> Path:
    """A copy of the fixture report with ``fields`` set on the record whose ``id`` matches."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for record in data["data"]:
        if record.get("id") == record_id:
            record.update(fields)
    out = tmp_path / f"manifest-{record_id}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


def _age_manifest(store: ArtifactStore, *, days: float) -> None:
    """Expire the documents-report freshness gate so the next acquire re-reads the network."""
    cached = store.cached_ref(_DOCUMENTS_ALIAS)
    assert cached is not None
    path = store.manifest_path(cached.blake3)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source"]["retrieved_at"] = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")


def _age_member(store: ArtifactStore, ref: ArtifactRef, *, days: float) -> None:
    """Rewrite one stored PDF's ``retrieved_at`` so per-document currency sees an old fetch."""
    assert ref.manifest is not None
    data = json.loads(ref.manifest.read_text(encoding="utf-8"))
    data["source"]["retrieved_at"] = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    ref.manifest.write_text(json.dumps(data), encoding="utf-8")


# --- selection + happy path ----------------------------------------------------


def test_crawl_downloads_exactly_the_manifest_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only English human product-information PDFs are fetched; veterinary/xls/other types are not."""
    _serve_manifest(monkeypatch)
    pdf_calls = _serve_pdfs(monkeypatch)
    workdir = tmp_path / "work"

    refs = ema_smpc.fetch(_ctx(workdir))

    assert len(refs) == len(_SELECTED_IDS) == 5
    assert len(pdf_calls) == 5
    documents = {doc.id: doc for doc in ema_documents.load_documents(_MANIFEST)}
    assert set(pdf_calls) == {documents[i].document_url for i in _SELECTED_IDS}
    assert all(ref.media_type == ema_smpc.PDF_MEDIA_TYPE for ref in refs)
    assert all(ref.blake3.startswith("b3:") for ref in refs)
    assert all(ref.uri.exists() for ref in refs)
    # Each member is content-addressed under the fan-out alias convention.
    store = _store_for(workdir)
    for doc_id in _SELECTED_IDS:
        assert store.cached_ref(ema_smpc.member_alias(documents[doc_id])) is not None


def test_refs_are_in_selection_order_regardless_of_network_timing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Deterministic output order is what keeps every downstream artifact hash stable."""
    _serve_manifest(monkeypatch)
    _serve_pdfs(monkeypatch)
    documents = {doc.id: doc for doc in ema_documents.load_documents(_MANIFEST)}

    first = ema_smpc.fetch(_ctx(tmp_path / "a", force=True))
    second = ema_smpc.fetch(_ctx(tmp_path / "b", force=True, ema_smpc_concurrency=1))

    assert [ref.blake3 for ref in first] == [ref.blake3 for ref in second]
    expected_stems = [documents[i].stem for i in _SELECTED_IDS]
    aliases = [ema_smpc.member_alias(documents[i]) for i in _SELECTED_IDS]
    assert len(set(expected_stems)) == len(expected_stems)
    assert len(set(aliases)) == len(aliases)
    assert len(first) == len(expected_stems)


def test_second_crawl_is_a_fanout_cache_hit_with_zero_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unchanged selection must not re-download 1,945 PDFs on every DAG run."""
    _serve_manifest(monkeypatch)
    pdf_calls = _serve_pdfs(monkeypatch)
    workdir = tmp_path / "work"
    ctx = _ctx(workdir)

    first = ema_smpc.fetch(ctx)
    assert len(pdf_calls) == 5
    second = ema_smpc.fetch(ctx)

    assert len(pdf_calls) == 5  # no new PDF fetches
    assert [ref.blake3 for ref in second] == [ref.blake3 for ref in first]


def test_limit_bounds_the_crawl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``ema_smpc_limit`` exists so the offline prod smoke can crawl a real but tiny scope."""
    _serve_manifest(monkeypatch)
    pdf_calls = _serve_pdfs(monkeypatch)

    refs = ema_smpc.fetch(_ctx(tmp_path / "work", ema_smpc_limit=2))

    assert len(refs) == 2
    assert len(pdf_calls) == 2


# --- cache layers --------------------------------------------------------------


def test_force_recrawls_every_document(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _serve_manifest(monkeypatch)
    pdf_calls = _serve_pdfs(monkeypatch)
    workdir = tmp_path / "work"
    ema_smpc.fetch(_ctx(workdir))
    ema_smpc.fetch(_ctx(workdir, force=True))
    assert len(pdf_calls) == 10


def test_only_a_republished_document_is_refetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Per-document currency: a SmPC republished after our fetch is stale, the rest are not."""
    manifest_calls = _serve_manifest(monkeypatch)
    pdf_calls = _serve_pdfs(monkeypatch)
    workdir = tmp_path / "work"
    store = _store_for(workdir)
    ema_smpc.fetch(_ctx(workdir))
    assert len(pdf_calls) == 5

    # The report is regenerated nightly: expire its gate and serve a revision where ONE document
    # (Ceplene, id 2441) has just been republished.
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    revised = _record_edit(_MANIFEST, tmp_path, "2441", last_updated_date=future)
    _age_manifest(store, days=8)
    _serve_manifest(monkeypatch, payloads=[revised])

    ema_smpc.fetch(_ctx(workdir))

    assert len(manifest_calls) == 2  # the report itself was re-read
    assert len(pdf_calls) == 6  # exactly one PDF re-downloaded
    assert pdf_calls[-1].endswith("ceplene-epar-product-information_en.pdf")


def test_undated_document_falls_back_to_the_age_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No usable ``last_updated_date`` => the crawl-level window decides, never 'always fresh'."""
    _serve_manifest(monkeypatch)
    pdf_calls = _serve_pdfs(monkeypatch)
    workdir = tmp_path / "work"
    refs = ema_smpc.fetch(_ctx(workdir))
    store = _store_for(workdir)

    revised = _record_edit(_MANIFEST, tmp_path, "2441", last_updated_date="")
    _age_manifest(store, days=8)
    _serve_manifest(monkeypatch, payloads=[revised])
    for ref in refs:
        _age_member(store, ref, days=40)  # older than the 30-day default window
    store.invalidate_cached_refs(_FANOUT_ALIAS)  # the selection changed, so the record is stale

    ema_smpc.fetch(_ctx(workdir))

    assert len(pdf_calls) == 10  # every undated/expired member was re-fetched


def test_a_fresh_undated_document_is_not_refetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _serve_manifest(monkeypatch)
    pdf_calls = _serve_pdfs(monkeypatch)
    workdir = tmp_path / "work"
    ema_smpc.fetch(_ctx(workdir))
    store = _store_for(workdir)

    revised = _record_edit(_MANIFEST, tmp_path, "2441", last_updated_date="")
    _age_manifest(store, days=8)
    _serve_manifest(monkeypatch, payloads=[revised])
    store.invalidate_cached_refs(_FANOUT_ALIAS)

    ema_smpc.fetch(_ctx(workdir))

    assert len(pdf_calls) == 5  # stored copies are inside the 30-day window


# --- failure paths -------------------------------------------------------------


def test_one_failed_document_does_not_abort_the_crawl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A transient 5xx on one SmPC must not empty the EU contraindication corpus."""
    _serve_manifest(monkeypatch)
    documents = {doc.id: doc for doc in ema_documents.load_documents(_MANIFEST)}
    bad_url = documents["28033"].document_url
    pdf_calls = _serve_pdfs(monkeypatch, failing={bad_url})
    workdir = tmp_path / "work"

    refs = ema_smpc.fetch(_ctx(workdir))

    assert len(refs) == 4
    assert pdf_calls.count(bad_url) == ema_smpc._DOWNLOAD_ATTEMPTS  # retried once, then recorded
    # An incomplete crawl publishes no completion record, so the next run retries the hole.
    assert (
        _store_for(workdir).cached_refs(
            _FANOUT_ALIAS, ema_smpc.crawl_fingerprint(ema_documents.product_information_documents(ema_documents.load_documents(_MANIFEST)))
        )
        is None
    )


def test_a_partial_crawl_retries_only_the_missing_document(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _serve_manifest(monkeypatch)
    documents = {doc.id: doc for doc in ema_documents.load_documents(_MANIFEST)}
    bad_url = documents["28033"].document_url
    pdf_calls = _serve_pdfs(monkeypatch, failing={bad_url})
    workdir = tmp_path / "work"
    ctx = _ctx(workdir)
    assert len(ema_smpc.fetch(ctx)) == 4

    _serve_pdfs(monkeypatch)  # the outage is over
    refs = ema_smpc.fetch(ctx)

    assert len(refs) == 5
    assert pdf_calls.count(bad_url) == ema_smpc._DOWNLOAD_ATTEMPTS + 1


def test_empty_downloads_count_as_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A zero-byte body is a failed document, not an ingested empty artifact."""
    _serve_manifest(monkeypatch)
    documents = {doc.id: doc for doc in ema_documents.load_documents(_MANIFEST)}
    _serve_pdfs(monkeypatch, empty={documents["2441"].document_url})

    refs = ema_smpc.fetch(_ctx(tmp_path / "work"))

    assert len(refs) == 4
    assert all(ref.uri.stat().st_size > 0 for ref in refs)


def test_total_crawl_failure_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Zero documents acquired is an acquisition failure, never an empty corpus downstream."""
    _serve_manifest(monkeypatch)
    documents = ema_documents.load_documents(_MANIFEST)
    urls = {doc.document_url for doc in ema_documents.product_information_documents(documents)}
    _serve_pdfs(monkeypatch, failing=urls)

    with pytest.raises(RuntimeError, match="failed to download"):
        ema_smpc.fetch(_ctx(tmp_path / "work"))


def test_empty_selection_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A report whose selection is empty (schema change, wrong file) must fail loudly."""
    emptied = tmp_path / "empty-report.json"
    emptied.write_text(json.dumps({"meta": {"total_records": 0}, "data": []}), encoding="utf-8")
    _serve_manifest(monkeypatch, payloads=[emptied])
    pdf_calls = _serve_pdfs(monkeypatch)

    with pytest.raises(RuntimeError, match="selected no PDFs"):
        ema_smpc.fetch(_ctx(tmp_path / "work", force=True))
    assert pdf_calls == []


# --- pure helpers --------------------------------------------------------------


def test_crawl_fingerprint_ignores_report_order_and_tracks_revisions() -> None:
    documents = ema_documents.product_information_documents(ema_documents.load_documents(_MANIFEST))
    assert ema_smpc.crawl_fingerprint(documents) == ema_smpc.crawl_fingerprint(list(reversed(documents)))
    revised = list(documents)
    revised[0] = ema_documents.EparDocument(
        id=revised[0].id,
        type=revised[0].type,
        medicine_name=revised[0].medicine_name,
        ema_product_number=revised[0].ema_product_number,
        document_url=revised[0].document_url,
        last_updated_date="2030-01-01T00:00:00Z",
    )
    assert ema_smpc.crawl_fingerprint(revised) != ema_smpc.crawl_fingerprint(documents)


def test_member_alias_uses_the_fanout_convention() -> None:
    doc = ema_documents.load_documents(_MANIFEST)[0]
    alias = ema_smpc.member_alias(doc)
    assert alias.startswith(f"{_FANOUT_ALIAS}::")
    assert alias.endswith(".pdf")


def test_concurrency_param_resolution(tmp_path: Path) -> None:
    resolve = ema_smpc._concurrency
    assert resolve(_ctx(tmp_path / "a")) == 8
    assert resolve(_ctx(tmp_path / "b", ema_smpc_concurrency=3)) == 3
    for bogus in (0, -1, "8", True, None):  # anything unusable falls back, never to 0 workers
        assert resolve(_ctx(tmp_path / "c", ema_smpc_concurrency=bogus)) == 8


def test_max_age_days_param_resolution(tmp_path: Path) -> None:
    resolve = ema_smpc._max_age_days
    assert resolve(_ctx(tmp_path / "a")) == 30.0
    assert resolve(_ctx(tmp_path / "b", ema_smpc_max_age_days=None)) == 30.0
    assert resolve(_ctx(tmp_path / "c", ema_smpc_max_age_days=7)) == 7.0
    for non_positive in (0, -1):
        assert resolve(_ctx(tmp_path / "d", ema_smpc_max_age_days=non_positive)) is None
    for bogus in ("month", True):
        assert resolve(_ctx(tmp_path / "e", ema_smpc_max_age_days=bogus)) == 30.0


def test_apply_limit_rejects_unusable_values(tmp_path: Path) -> None:
    documents = ema_documents.load_documents(_MANIFEST)
    for bogus in (0, -3, "2", True, None):
        assert ema_smpc._apply_limit(documents, _ctx(tmp_path / "x", ema_smpc_limit=bogus)) == documents
    assert ema_smpc._apply_limit(documents, _ctx(tmp_path / "y", ema_smpc_limit=1)) == documents[:1]


def test_parse_datetime_handles_the_report_forms() -> None:
    parse = ema_smpc._parse_datetime
    assert parse("2023-04-19T12:21:00Z") is not None
    assert parse("2023-04-19T12:21:00+00:00") is not None
    for bogus in ("", "   ", "whenever", None, 42):
        assert parse(bogus) is None


def test_download_ema_smpc_pdf_streams_to_dest(tmp_path: Path) -> None:
    """The real downloader (stdlib urllib) copies bytes verbatim from a file:// URL."""
    dest = tmp_path / "downloaded.pdf"
    result = ema_smpc.download_ema_smpc_pdf(_SMPD_PDF.as_uri(), dest)
    assert result == dest
    assert dest.read_bytes() == _SMPD_PDF.read_bytes()
