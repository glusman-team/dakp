"""Tests for the EMA EPAR documents-report fetcher (the SmPC crawl manifest).

Covers the real download path (network downloader monkeypatched to serve the committed trimmed
report), content-addressing idempotence, the seven-day freshness gate (cache hit / ``force``
bypass / stale / disabled / misconfigured windows), the ``ema_documents_url`` override, the
fail-loudly parse gate on a non-JSON body, the stdlib downloader itself (via a ``file://`` URL),
and the manifest parsing + product-information selection rule.

Every test names WHY it exists: this manifest is the only thing that bounds the SmPC PDF crawl,
so a silently wrong selection or a silently empty report would turn into a missing edge family
rather than an error.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import ema_documents, ema_smpc

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_FIXTURE = _FIXTURE_ROOT / "ema" / "epar_documents_en.json"
_ALIAS = ema_documents.DOCUMENTS_ALIAS


def _ctx(workdir: Path, **params: object) -> TaskContext:
    Workdir(workdir).create()
    return TaskContext(workdir=workdir, fixture_root=_FIXTURE_ROOT, params=dict(params))


def _fake_download(calls: list[str]):
    def fake(url: str, dest: Path, *, timeout: float = 180.0) -> Path:
        calls.append(url)
        shutil.copyfile(_FIXTURE, dest)
        return dest

    return fake


def _store_for(workdir: Path) -> ArtifactStore:
    return ArtifactStore(Workdir(workdir))


def _age_cached(store: ArtifactStore, *, days: float) -> None:
    """Rewrite the cached manifest so its ``retrieved_at`` is ``days`` in the past."""
    cached = store.cached_ref(_ALIAS)
    assert cached is not None
    manifest_path = store.manifest_path(cached.blake3)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["source"]["retrieved_at"] = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


# --- real download path (network monkeypatched) --------------------------------


def test_real_fetch_ingests_json_and_cache_hit_skips_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema_documents, "download_ema_documents", _fake_download(calls))
    ctx = _ctx(tmp_path / "work")

    first = ema_documents.fetch(ctx)
    assert len(first) == 1
    ref = first[0]
    assert ref.media_type == "application/json"
    assert ref.blake3.startswith("b3:")
    assert ref.manifest is not None
    assert ref.manifest.exists()
    assert calls == [ema_documents.EMA_DOCUMENTS_URL]

    # Fresh cache (< 7 days): the second fetch is a gate hit, so the crawl manifest never
    # re-downloads on a re-run of the same DAG.
    second = ema_documents.fetch(ctx)
    assert second[0].blake3 == ref.blake3
    assert calls == [ema_documents.EMA_DOCUMENTS_URL]


def test_real_fetch_url_overridable_via_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pinned snapshot URL must be injectable so a build can be reproduced against old data."""
    calls: list[str] = []
    monkeypatch.setattr(ema_documents, "download_ema_documents", _fake_download(calls))
    workdir = tmp_path / "work"
    ema_documents.fetch(_ctx(workdir))
    override = "https://example.test/epar-documents-snapshot.json"
    ema_documents.fetch(_ctx(workdir, ema_documents_url=override))
    assert calls == [ema_documents.EMA_DOCUMENTS_URL, override]


def test_force_bypasses_freshness_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema_documents, "download_ema_documents", _fake_download(calls))
    workdir = tmp_path / "work"
    ema_documents.fetch(_ctx(workdir))
    ema_documents.fetch(_ctx(workdir, force=True))
    assert calls == [ema_documents.EMA_DOCUMENTS_URL, ema_documents.EMA_DOCUMENTS_URL]


def test_stale_cache_rechecks_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema_documents, "download_ema_documents", _fake_download(calls))
    workdir = tmp_path / "work"
    ema_documents.fetch(_ctx(workdir))
    _age_cached(_store_for(workdir), days=8)
    ema_documents.fetch(_ctx(workdir))
    assert calls == [ema_documents.EMA_DOCUMENTS_URL, ema_documents.EMA_DOCUMENTS_URL]


def test_non_positive_max_age_disables_cache_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema_documents, "download_ema_documents", _fake_download(calls))
    workdir = tmp_path / "work"
    ema_documents.fetch(_ctx(workdir))
    ema_documents.fetch(_ctx(workdir, ema_documents_max_age_days=0))
    assert calls == [ema_documents.EMA_DOCUMENTS_URL, ema_documents.EMA_DOCUMENTS_URL]


def test_non_numeric_max_age_falls_back_to_the_default_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A param that is not a real number falls back to the 7-day default, never to 'no gate'."""
    calls: list[str] = []
    monkeypatch.setattr(ema_documents, "download_ema_documents", _fake_download(calls))
    workdir = tmp_path / "work"
    ema_documents.fetch(_ctx(workdir))
    ema_documents.fetch(_ctx(workdir, ema_documents_max_age_days="14"))
    ema_documents.fetch(_ctx(workdir, ema_documents_max_age_days=True))
    assert calls == [ema_documents.EMA_DOCUMENTS_URL]

    _age_cached(_store_for(workdir), days=8)
    ema_documents.fetch(_ctx(workdir, ema_documents_max_age_days="14"))
    assert calls == [ema_documents.EMA_DOCUMENTS_URL, ema_documents.EMA_DOCUMENTS_URL]


def test_unparsable_retrieved_at_refetches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A manifest whose ``retrieved_at`` will not parse has no measurable age, so the gate cannot apply."""
    calls: list[str] = []
    monkeypatch.setattr(ema_documents, "download_ema_documents", _fake_download(calls))
    workdir = tmp_path / "work"
    ema_documents.fetch(_ctx(workdir))

    store = _store_for(workdir)
    cached = store.cached_ref(_ALIAS)
    assert cached is not None
    manifest_path = store.manifest_path(cached.blake3)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["source"]["retrieved_at"] = "whenever"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    ema_documents.fetch(_ctx(workdir))
    assert calls == [ema_documents.EMA_DOCUMENTS_URL, ema_documents.EMA_DOCUMENTS_URL]


def test_max_age_days_param_resolution(tmp_path: Path) -> None:
    resolve = ema_documents._max_age_days
    assert resolve(_ctx(tmp_path / "a")) == 7.0  # absent -> default window
    assert resolve(_ctx(tmp_path / "b", ema_documents_max_age_days=None)) == 7.0
    assert resolve(_ctx(tmp_path / "c", ema_documents_max_age_days=14)) == 14.0
    assert resolve(_ctx(tmp_path / "d", ema_documents_max_age_days=0.5)) == 0.5
    for non_positive in (0, -1):  # non-positive -> gate disabled (always re-check)
        assert resolve(_ctx(tmp_path / "e", ema_documents_max_age_days=non_positive)) is None
    for bogus in ("week", True):  # non-numeric -> the default window, never 'no gate'
        assert resolve(_ctx(tmp_path / "f", ema_documents_max_age_days=bogus)) == 7.0


def test_download_ema_documents_streams_to_dest(tmp_path: Path) -> None:
    """The real downloader (stdlib urllib) copies bytes verbatim from a file:// URL."""
    dest = tmp_path / "downloaded.json"
    result = ema_documents.download_ema_documents(_FIXTURE.as_uri(), dest)
    assert result == dest
    assert dest.read_bytes() == _FIXTURE.read_bytes()


# --- fail-loudly acquisition gates ---------------------------------------------


def test_non_json_body_fails_before_ingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An HTML error page must fail the acquisition, not land in the store as a bogus manifest."""

    def fake(url: str, dest: Path, *, timeout: float = 180.0) -> Path:
        dest.write_text("<html><body>503 Service Unavailable</body></html>", encoding="utf-8")
        return dest

    monkeypatch.setattr(ema_documents, "download_ema_documents", fake)
    workdir = tmp_path / "work"
    with pytest.raises(ValueError, match="not valid JSON"):
        ema_documents.fetch(_ctx(workdir))
    assert _store_for(workdir).cached_ref(_ALIAS) is None  # nothing was registered


def test_missing_staged_file_fails_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The ``finally`` cleanup tolerates an absent staged file; the parse gate then fails loudly."""

    def fake_download(url: str, dest: Path, *, timeout: float = 180.0) -> Path:
        dest.unlink(missing_ok=True)
        return dest

    monkeypatch.setattr(ema_documents, "download_ema_documents", fake_download)
    with pytest.raises(ValueError, match="cannot read"):
        ema_documents.EmaDocumentsFetcher().fetch(_ctx(tmp_path / "work"))


def test_load_documents_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        ema_documents.load_documents(tmp_path / "absent.json")


# --- manifest parsing ----------------------------------------------------------


def test_parse_documents_maps_every_field() -> None:
    documents = ema_documents.load_documents(_FIXTURE)
    assert len(documents) == 9
    by_id = {doc.id: doc for doc in documents}
    ceplene = by_id["2441"]
    assert ceplene.type == "product-information"
    assert ceplene.medicine_name == "Ceplene"
    assert ceplene.ema_product_number == "EMEA/H/C/000796"
    assert ceplene.document_url.endswith("ceplene-epar-product-information_en.pdf")
    assert ceplene.first_published_date.startswith("2008-10-28")
    assert ceplene.last_updated_date.startswith("2023-04-19")
    assert ceplene.status == "unknown"
    # Optional keys absent from a record default to "" rather than raising.
    assert by_id["4056"].reference_number == ""


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "expected a JSON object"),
        ({"data": {}}, "'data' must be a list"),
        ({"data": ["nope"]}, "record 0 must be an object"),
        ({"data": [{"id": "1", "type": "product-information"}]}, "missing required key\\(s\\): document_url"),
        ({"data": [{"id": "1", "document_url": "https://x/y_en.pdf"}]}, "missing required key\\(s\\): type"),
        (
            {"data": [{"id": "1", "type": "product-information", "medicine_name": "X", "ema_product_number": "EMEA/H/C/1", "document_url": "  "}]},
            "document_url",
        ),
    ],
)
def test_parse_documents_rejects_malformed_reports(payload: object, match: str) -> None:
    """A malformed report is an acquisition failure, never a partial (silently small) crawl."""
    with pytest.raises(ValueError, match=match):
        ema_documents.parse_documents(payload)


def test_records_without_medicine_metadata_parse_and_are_never_crawled() -> None:
    """The live report has 2 of 20,234 records (an orphan-maintenance report, a tracked-changes
    document) with no medicine_name/ema_product_number. They must not fail the acquisition, and a
    product-information row lacking them must not be crawled: it could not be joined to a substance."""
    payload = {
        "data": [
            {
                "id": "75411",
                "type": "orphan-maintenance-report",
                "document_url": "https://www.ema.europa.eu/en/documents/orphan-maintenance-report/x_en.pdf",
            },
            {
                "id": "9",
                "type": "product-information",
                "ema_product_number": "EMEA/H/C/000009",
                "document_url": "https://www.ema.europa.eu/en/documents/product-information/nameless_en.pdf",
            },
            {
                "id": "10",
                "type": "product-information",
                "medicine_name": "Named",
                "ema_product_number": "EMEA/H/C/000010",
                "document_url": "https://www.ema.europa.eu/en/documents/product-information/named_en.pdf",
            },
        ]
    }
    documents = ema_documents.parse_documents(payload)
    assert [doc.id for doc in documents] == ["75411", "9", "10"]
    assert documents[0].medicine_name == ""
    assert [doc.id for doc in ema_documents.product_information_documents(documents)] == ["10"]


def test_the_live_report_shape_parses(tmp_path: Path) -> None:
    """Regression: the first draft required medicine_name and rejected the WHOLE live report."""
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    payload["data"].append(
        {
            "id": "75411",
            "name": "Nezglyal : Orphan maintenance assessment report",
            "type": "orphan-maintenance-report",
            "medicine_name": "",
            "ema_product_number": "",
            "document_url": "https://www.ema.europa.eu/en/documents/orphan-maintenance-report/nezglyal-orphan-maintenance-assessment-report_en.pdf",
        }
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(ema_documents.load_documents(path)) == 10


# --- product-information selection ---------------------------------------------


def test_product_information_selection_keeps_english_human_pdfs() -> None:
    """The crawl bound: veterinary rows, non-PDF members and other document types stay out."""
    documents = ema_documents.load_documents(_FIXTURE)
    selected = ema_documents.product_information_documents(documents)
    assert [doc.id for doc in selected] == ["2441", "10327", "28033", "49893", "50210"]

    dropped = {doc.id for doc in documents} - {doc.id for doc in selected}
    assert dropped == {
        "4056",  # EMEA/V/C veterinary product information
        "60975",  # product-information type but an .xls member-state URL list, not a SmPC
        "2410",  # variation report
        "2460",  # assessment report
    }


def test_product_information_selection_accepts_the_dotted_pdf_suffix_variant() -> None:
    """Drupal serves a few product-information files as ``..._en.pdf-0``; they are real SmPCs."""
    documents = ema_documents.load_documents(_FIXTURE)
    selected = ema_documents.product_information_documents(documents)
    odd = next(doc for doc in selected if doc.id == "28033")
    assert odd.document_url.endswith("_en.pdf-0")
    # The Drupal revision stays in the key: stripping it collided with the plain ``.pdf``
    # document of a DIFFERENT product in the 2026-10-06 live report (both named
    # ``dimethyl-fumarate-accord-epar-product-information_en``), so two PDFs staged onto one
    # path and the crawl died on a missing file mid-ingest.
    assert odd.stem == "clopidogrel-ratiopharm-epar-product-information_en-0"


def test_pdf_revision_suffix_keeps_live_collision_apart() -> None:
    """The 2026-10-06 production report pairs ``.pdf`` and ``.pdf-0`` across two products."""
    plain = ema_documents.EparDocument(
        id="64041",
        type="product-information",
        medicine_name="Dimethyl Fumarate Accord",
        ema_product_number="EMEA/H/C/006471",
        document_url="https://www.ema.europa.eu/en/documents/product-information/dimethyl-fumarate-accord-epar-product-information_en.pdf",
    )
    revision = ema_documents.EparDocument(
        id="61597",
        type="product-information",
        medicine_name="Dimethyl Fumarate Accord",
        ema_product_number="EMEA/H/C/005950",
        document_url="https://www.ema.europa.eu/en/documents/product-information/dimethyl-fumarate-accord-epar-product-information_en.pdf-0",
    )
    assert plain.stem == "dimethyl-fumarate-accord-epar-product-information_en"
    assert revision.stem == "dimethyl-fumarate-accord-epar-product-information_en-0"

    # The crawl keys (member alias, staging path, fingerprint) all derive from the stem, so
    # distinct stems keep the two PDFs apart end to end.
    assert ema_smpc.member_alias(revision) != ema_smpc.member_alias(plain)
    assert ema_smpc.crawl_fingerprint([plain, revision]) == ema_smpc.crawl_fingerprint([revision, plain])


def test_product_information_selection_is_deterministic() -> None:
    """Crawl order must not depend on report order: sorted by (product number, id)."""
    documents = ema_documents.load_documents(_FIXTURE)
    assert ema_documents.product_information_documents(list(reversed(documents))) == ema_documents.product_information_documents(documents)


def test_selection_keeps_unauthorised_documents_for_the_registry_join() -> None:
    """The report's ``status`` is unusable ('unknown' everywhere), so withdrawn products are
    still crawled and dropped later by the medicines-registry join, not by a guessed filter."""
    documents = ema_documents.load_documents(_FIXTURE)
    selected = ema_documents.product_information_documents(documents)
    vantobra = next(doc for doc in selected if doc.id == "10327")
    assert "no-longer-authorised" in vantobra.document_url
    assert vantobra.status == "unknown"


def test_document_stem_is_unique_across_a_duplicate_product_number() -> None:
    """One medicine can carry two product-information documents; the stem keeps their aliases apart."""
    documents = ema_documents.load_documents(_FIXTURE)
    dupes = [doc for doc in documents if doc.ema_product_number == "EMEA/H/C/005675"]
    assert len(dupes) == 2
    stems = {doc.stem for doc in dupes}
    assert len(stems) == 2
    assert all(stem for stem in stems)


def test_document_stem_falls_back_to_the_basename_without_a_pdf_suffix() -> None:
    doc = ema_documents.EparDocument(
        id="1",
        type="product-information",
        medicine_name="X",
        ema_product_number="EMEA/H/C/000001",
        document_url="https://www.ema.europa.eu/en/documents/product-information/x_en.xls",
    )
    assert doc.stem == "x_en.xls"
