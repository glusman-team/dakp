"""Edge-case tests for ``dakp_pipeline.acquire``.

Covers the acquisition branches the happy-path tests miss: ``acquire_ontologies`` with no
configured sources, the mock profile with no ``fixture_root`` and with a fixture root lacking
``ontology/*.tsv``, and the stdlib-urllib download path of ``_download_to`` (via a ``file://``
URL, so no network is needed).
"""

from __future__ import annotations

from pathlib import Path

from dakp_pipeline import acquire
from dakp_pipeline.io.contracts import TaskContext


def _ctx(profile: str, workdir: Path, fixture_root: Path | None = None) -> TaskContext:
    return TaskContext(profile=profile, workdir=workdir, fixture_root=fixture_root, threads=1, memory_budget_gb=1, params={})


def test_acquire_ontologies_no_sources_returns_empty(tmp_path: Path) -> None:
    # Non-mock profile with an explicitly empty source map -> nothing to acquire.
    assert acquire.acquire_ontologies(_ctx("sample", tmp_path), sources={}) == []


def test_acquire_ontologies_mock_without_fixture_root(tmp_path: Path) -> None:
    assert acquire.acquire_ontologies(_ctx("mock", tmp_path, fixture_root=None)) == []


def test_acquire_ontologies_mock_fixture_without_ontology_files(tmp_path: Path) -> None:
    empty_fixture = tmp_path / "fixtures"
    empty_fixture.mkdir()
    assert acquire.acquire_ontologies(_ctx("mock", tmp_path, fixture_root=empty_fixture)) == []


def test_download_to_uses_stdlib_urllib_for_file_url(tmp_path: Path) -> None:
    src = tmp_path / "source.txt"
    src.write_text("payload", encoding="utf-8")
    dest = tmp_path / "nested" / "dest.txt"
    acquire._download_to(src.as_uri(), dest, None)  # no injected downloader -> urllib path
    assert dest.read_text(encoding="utf-8") == "payload"
