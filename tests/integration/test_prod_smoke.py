"""Offline ``prod`` smoke run: REAL fetchers/extractors over mocked HTTP.

Runs :func:`dakp_pipeline.pipeline.run_pipeline` with a ``prod``-like profile
(``mock_sources=False``) so **every fetcher takes its real (download) branch**, but
monkeypatches the stdlib HTTP seam (``urllib.request.urlopen``) to serve the tiny pipeline
fixtures *as if downloaded*. Contraindications are text-mined from the downloaded DailyMed
SPL contraindication sections (offline dictionary NER backend over the ontology fixture).
This validates the real fetcher → extractor → aggregation → Tablassert-handoff path
end-to-end with no network and without the multi-TB full build: ``quarter_limit`` /
``release_limit`` bound FAERS and DailyMed to a single quarter / release.

The artifacts are derived from the "downloaded" bytes through the REAL code path (DailyMed
release-ZIP expansion, ``$``-delimited FAERS ZIP parsing, SPL XML extraction, Drugs@FDA ZIP
parsing, content-addressed ingest) — **not** the mock fixture shortcut (the
``profile == "mock"`` fixture-ingest branch, which is structurally unreachable here). The
Tablassert *handoff* runs its real runner; only the ``../Tablassert`` subprocess is faked
(the documented ``run_subprocess`` seam), since Tablassert is not a dependency. Stays fully
offline so it passes in CI.
"""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pytest

from dakp_pipeline.pipeline import run_pipeline

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"

# Two releases / two quarters are advertised so the bounded smoke run (release_limit=1 /
# quarter_limit=1) provably truncates to the most-recent single one.
_DAILYMED_INDEX_HTML = (
    "<html><body><h2>Full Releases</h2>"
    '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip">part1</a>'
    '<a href="https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part2.zip">part2</a>'
    "</body></html>"
)
_FAERS_INDEX_HTML = "<html><body><a href='faers_ascii_2024q3.zip'>2024q3</a><a href='faers_ascii_2024q2.zip'>2024q2</a></body></html>"


# --- fixture-bytes-as-downloads helpers -----------------------------------------


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP (the shape each real downloader would serve)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


def _read_fixture(relative: str) -> bytes:
    return (_FIXTURE_ROOT / relative).read_bytes()


def _faers_quarter_zip(quarter: str, families: tuple[str, ...]) -> bytes:
    """A FAERS quarterly ASCII ZIP: members ``ascii/<FAMILY><quarter>.txt``."""
    return _zip_bytes({f"ascii/{family}{quarter}.txt": _read_fixture(f"faers/{family}{quarter}.txt") for family in families})


def _downloaded_payloads() -> dict[str, bytes]:
    """The bytes the mocked HTTP layer serves for each real source URL."""
    return {
        # DailyMed: one release ZIP carrying the SPL fixture as an inner .xml.gz member.
        "dailymed_release": _zip_bytes({"dailymed_spl.xml.gz": _read_fixture("dailymed/dailymed_spl.xml.gz")}),
        # FAERS: quarterly ASCII ZIPs (24Q3 carries the DELETE family; 24Q2 has none).
        "faers_24q3": _faers_quarter_zip("24Q3", ("DEMO", "DRUG", "INDI", "REAC", "RPSR", "DELETE")),
        "faers_24q2": _faers_quarter_zip("24Q2", ("DEMO", "DRUG", "INDI", "REAC", "RPSR")),
        # Drugs@FDA: the data-files ZIP (Products/Applications/Submissions members).
        "drugsfda": _zip_bytes(
            {
                "Products.tsv": _read_fixture("drugsfda/drugsfda_products.tsv"),
                "Applications.tsv": _read_fixture("drugsfda/drugsfda_applications.tsv"),
                "Submissions.tsv": _read_fixture("drugsfda/drugsfda_submissions.tsv"),
            }
        ),
    }


class _FakeHTTPResponse:
    """Minimal ``urllib`` response: context manager + ``.read()`` + ``.headers.get()``."""

    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)
        self.headers: dict[str, str] = {}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _install_fake_http(monkeypatch: pytest.MonkeyPatch, requested: list[str]) -> None:
    """Route every real source URL to fixture bytes; fail loudly on any other access."""
    payloads = _downloaded_payloads()
    # Ordered most-specific-first: quarter/release ZIP URLs contain the index URL as a prefix.
    routes: list[tuple[str, bytes]] = [
        ("dm_spl_release_human_rx_part1.zip", payloads["dailymed_release"]),
        ("dm_spl_release_human_rx_part2.zip", payloads["dailymed_release"]),
        ("spl-resources-all-drug-labels.cfm", _DAILYMED_INDEX_HTML.encode("utf-8")),
        ("faers_ascii_2024q3.zip", payloads["faers_24q3"]),
        ("faers_ascii_2024q2.zip", payloads["faers_24q2"]),
        ("fis.fda.gov/content/Exports", _FAERS_INDEX_HTML.encode("utf-8")),
        ("fda.gov/media/89850/download", payloads["drugsfda"]),
    ]

    def fake_urlopen(url: object, timeout: float | None = None, **kwargs: object) -> _FakeHTTPResponse:
        full_url = url.full_url if hasattr(url, "full_url") else str(url)  # type: ignore[union-attr]
        requested.append(str(full_url))
        for needle, payload in routes:
            if needle in full_url:
                return _FakeHTTPResponse(payload)
        msg = f"unexpected network access in offline prod smoke test: {full_url}"
        raise AssertionError(msg)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _fake_tablassert_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """The real TablassertRunner runs; only the ``../Tablassert`` process is faked (offline)."""
    return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")


def _manifest_source_urls(workdir: Path) -> set[str]:
    """Collect every ``source.url`` recorded across the run's artifact manifests."""
    urls: set[str] = set()
    for manifest_path in (workdir / "data" / "manifests").glob("*.json"):
        source = json.loads(manifest_path.read_text(encoding="utf-8")).get("source") or {}
        if source.get("url"):
            urls.add(str(source["url"]))
    return urls


# --- the smoke run --------------------------------------------------------------


def test_prod_smoke_run_executes_real_path_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    requested: list[str] = []
    _install_fake_http(monkeypatch, requested)
    run_module = importlib.import_module("dakp_pipeline.tablassert.run")
    # The real runner guards on tablassert being importable (a core DAKP dependency); fake the
    # availability probe too so this smoke run stays offline and tablassert-independent, exactly
    # like the faked subprocess below.
    monkeypatch.setattr(run_module, "tablassert_available", lambda: True)
    monkeypatch.setattr(run_module, "run_subprocess", _fake_tablassert_subprocess)

    workdir = tmp_path / "work"
    result = run_pipeline(
        profile="prod",
        fixture_root=_FIXTURE_ROOT,  # only loads the disease map; fetchers DOWNLOAD (mock_sources=False)
        workdir=workdir,
        params={"quarter_limit": 1, "release_limit": 1},
    )

    # A real (non-mock) profile drove the run.
    assert result.profile.name == "prod"
    assert result.profile.mock_sources is False

    # All three assertion tables were produced by the real aggregation, with rows.
    for table in ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions"):
        assert result.table(table).rows > 0
        assert result.table(table).path.exists()

    # ... and the rows came from the "downloaded" fixture bytes through the real extractors:
    approved = result.table("approved_treats_assertions").path.read_text(encoding="utf-8")
    assert "Examplestatin" in approved  # DailyMed SPL + Drugs@FDA join
    assert "hypercholesterolemia" in approved
    uses = result.table("faers_applied_to_treat_assertions").path.read_text(encoding="utf-8")
    assert "Examplestatin" in uses  # FAERS quarter ZIP parsed
    assert "hypercholesterolemia" in uses
    contra = result.table("contraindication_assertions").path.read_text(encoding="utf-8")
    assert "Ibuprofen" in contra  # NER-mined from the DailyMed contraindication section
    assert "asthma" in contra

    # Build summary + REAL Tablassert handoff (mode "real", not the mock deferred report).
    assert result.build_summary is not None
    assert result.build_summary.exists()
    handoff = json.loads((workdir / "data" / "reports" / "tablassert_handoff.json").read_text(encoding="utf-8"))
    assert handoff["mode"] == "real"

    # The real download branches ran (index + per-source URLs were requested over HTTP)...
    assert any("spl-resources-all-drug-labels.cfm" in u for u in requested)
    assert any("fis.fda.gov/content/Exports" in u for u in requested)
    assert any("fda.gov/media/89850/download" in u for u in requested)
    # ...bounded: release_limit=1 fetched DailyMed part1 only, quarter_limit=1 fetched 24Q3 only.
    assert any("dm_spl_release_human_rx_part1.zip" in u for u in requested)
    assert not any("dm_spl_release_human_rx_part2.zip" in u for u in requested)
    assert any("faers_ascii_2024q3.zip" in u for u in requested)
    assert not any("faers_ascii_2024q2.zip" in u for u in requested)

    # Provenance proves the real path: a DailyMed release URL is recorded, and NO artifact
    # came from the mock fixture shortcut (fixture:// or fixture: source URLs).
    source_urls = _manifest_source_urls(workdir)
    assert any(u.startswith("https://dailymed-data.nlm.nih.gov") for u in source_urls)
    assert not any(u.startswith("fixture") for u in source_urls)
