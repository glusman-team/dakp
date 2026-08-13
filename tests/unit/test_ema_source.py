"""Tests for the EMA medicines source fetcher.

Covers the real download path (with the network downloader monkeypatched to serve the committed
fixture xlsx), content-addressing idempotence, the seven-day freshness gate (cache hit / ``force``
bypass / stale / disabled / misconfigured windows), the ``ema_url`` override, and the stdlib
downloader itself (via a ``file://`` URL) — mirroring the Drugs@FDA fetcher suite.
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
from dakp_pipeline.sources import ema

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_EMA_FIXTURE = _FIXTURE_ROOT / "ema" / "medicines-output-medicines-report_en.xlsx"
_ALIAS = "ema/medicines.xlsx"


def _ctx(workdir: Path, **params: object) -> TaskContext:
    Workdir(workdir).create()
    return TaskContext(workdir=workdir, fixture_root=_FIXTURE_ROOT, params=dict(params))


def _fake_download(calls: list[str]):
    def fake(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        calls.append(url)
        shutil.copyfile(_EMA_FIXTURE, dest)
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


def test_real_fetch_ingests_xlsx_and_cache_hit_skips_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema, "download_ema_table", _fake_download(calls))
    ctx = _ctx(tmp_path / "work")

    first = ema.fetch(ctx)
    assert len(first) == 1
    ref = first[0]
    assert ref.media_type == ema.XLSX_MEDIA_TYPE
    assert ref.blake3.startswith("b3:")
    assert ref.manifest is not None
    assert ref.manifest.exists()
    assert calls == [ema.EMA_MEDICINES_URL]

    # Fresh cache (< 7 days): the second fetch is a gate hit — no network at all.
    second = ema.fetch(ctx)
    assert second[0].blake3 == ref.blake3
    assert calls == [ema.EMA_MEDICINES_URL]


def test_real_fetch_url_overridable_via_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema, "download_ema_table", _fake_download(calls))
    workdir = tmp_path / "work"
    ema.fetch(_ctx(workdir))
    override = "https://example.test/ema-snapshot.xlsx"
    ema.fetch(_ctx(workdir, ema_url=override))
    assert calls == [ema.EMA_MEDICINES_URL, override]


def test_force_bypasses_freshness_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema, "download_ema_table", _fake_download(calls))
    workdir = tmp_path / "work"
    ema.fetch(_ctx(workdir))
    ema.fetch(_ctx(workdir, force=True))
    assert calls == [ema.EMA_MEDICINES_URL, ema.EMA_MEDICINES_URL]


def test_stale_cache_rechecks_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema, "download_ema_table", _fake_download(calls))
    workdir = tmp_path / "work"
    ema.fetch(_ctx(workdir))
    _age_cached(_store_for(workdir), days=8)
    ema.fetch(_ctx(workdir))
    assert calls == [ema.EMA_MEDICINES_URL, ema.EMA_MEDICINES_URL]


def test_non_positive_max_age_disables_cache_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ema, "download_ema_table", _fake_download(calls))
    workdir = tmp_path / "work"
    ema.fetch(_ctx(workdir))
    ema.fetch(_ctx(workdir, ema_max_age_days=0))
    assert calls == [ema.EMA_MEDICINES_URL, ema.EMA_MEDICINES_URL]


def test_non_numeric_max_age_falls_back_to_the_default_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A param that is not a real number falls back to the 7-day default — never to 'no gate'."""
    calls: list[str] = []
    monkeypatch.setattr(ema, "download_ema_table", _fake_download(calls))
    workdir = tmp_path / "work"
    ema.fetch(_ctx(workdir))
    # Both bogus values resolve to the default 7 days, so the fresh download is still reused.
    ema.fetch(_ctx(workdir, ema_max_age_days="14"))
    ema.fetch(_ctx(workdir, ema_max_age_days=True))
    assert calls == [ema.EMA_MEDICINES_URL]

    # And it really is the DEFAULT window, not an unbounded one: past 7 days the gate reopens.
    _age_cached(_store_for(workdir), days=8)
    ema.fetch(_ctx(workdir, ema_max_age_days="14"))
    assert calls == [ema.EMA_MEDICINES_URL, ema.EMA_MEDICINES_URL]


def test_unparsable_retrieved_at_refetches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A manifest whose ``retrieved_at`` will not parse has no measurable age, so the gate cannot apply."""
    calls: list[str] = []
    monkeypatch.setattr(ema, "download_ema_table", _fake_download(calls))
    workdir = tmp_path / "work"
    ema.fetch(_ctx(workdir))

    store = _store_for(workdir)
    cached = store.cached_ref(_ALIAS)
    assert cached is not None
    manifest_path = store.manifest_path(cached.blake3)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["source"]["retrieved_at"] = "whenever"  # provenance present but not a timestamp
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    ema.fetch(_ctx(workdir))
    assert calls == [ema.EMA_MEDICINES_URL, ema.EMA_MEDICINES_URL]


def test_real_fetch_cleans_up_absent_staged_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The ``finally`` cleanup tolerates an absent staged file; ingest then fails loudly."""

    def fake_download(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
        dest.unlink(missing_ok=True)  # downloader left nothing behind
        return dest

    monkeypatch.setattr(ema, "download_ema_table", fake_download)
    with pytest.raises(FileNotFoundError):
        ema.EMAFetcher().fetch(_ctx(tmp_path / "work"))


def test_max_age_days_param_resolution(tmp_path: Path) -> None:
    assert ema._max_age_days(_ctx(tmp_path / "a")) == 7.0  # absent -> default window
    assert ema._max_age_days(_ctx(tmp_path / "b", ema_max_age_days=None)) == 7.0  # null config -> default
    assert ema._max_age_days(_ctx(tmp_path / "c", ema_max_age_days=14)) == 14.0
    assert ema._max_age_days(_ctx(tmp_path / "d", ema_max_age_days=0.5)) == 0.5
    for non_positive in (0, -1):  # non-positive numbers -> gate disabled (always re-check)
        assert ema._max_age_days(_ctx(tmp_path / "e", ema_max_age_days=non_positive)) is None
    for bogus in ("week", True):  # non-numeric -> the default window, never 'no gate'
        assert ema._max_age_days(_ctx(tmp_path / "f", ema_max_age_days=bogus)) == 7.0


def test_download_ema_table_streams_to_dest(tmp_path: Path) -> None:
    """The real downloader (stdlib urllib) copies bytes verbatim from a file:// URL."""
    dest = tmp_path / "downloaded.xlsx"
    result = ema.download_ema_table(_EMA_FIXTURE.as_uri(), dest)
    assert result == dest
    assert dest.read_bytes() == _EMA_FIXTURE.read_bytes()
