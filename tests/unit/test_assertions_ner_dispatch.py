"""Unit tests for the shared NER dispatch plumbing (assertions/ner_dispatch.py).

Covers the generalized multi-pass GPU dispatch (mine_passes_multi_gpu / _group_devices)
beyond what the contraindication shaper's historical two-pass tests exercise, plus the
persistent mention-cache seam (mine_with_cache). All tests run offline with gazetteer-only
DiseaseNERs on "cpu" devices (the spawn pool is real); cache tests use a fake in-memory
cache and a production-mode backend whose ``extract`` is monkeypatched — GLiNER never loads.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from dakp_pipeline.assertions.ner_dispatch import _group_devices, default_ner, mine_passes_multi_gpu, mine_with_cache
from dakp_pipeline.ner import model_cache
from dakp_pipeline.ner.ner import DiseaseNER, Mention


def _ner(*terms: str) -> DiseaseNER:
    return DiseaseNER(gazetteer=dict.fromkeys(terms, "disease"))


# --- mine_passes_multi_gpu ------------------------------------------------------


def test_mine_passes_multi_gpu_all_passes_empty() -> None:
    """No work in any pass: empty result map, no pool dispatched."""
    assert mine_passes_multi_gpu([[], []], _ner("asthma"), ("cpu", "cpu")) == {}


def test_mine_passes_multi_gpu_single_nonempty_pass_uses_multi_gpu() -> None:
    """One nonempty pass: all devices go to it (the _mine_multi_gpu fallback)."""
    results = mine_passes_multi_gpu([[], [("SET-B", "DOC-B", "diabetes")]], _ner("diabetes"), ("cpu", "cpu"))
    assert [m.text for m in results[("SET-B", "DOC-B")]] == ["diabetes"]


def test_mine_passes_multi_gpu_splits_devices_across_three_passes() -> None:
    """Three passes share the device list contiguously; every pass's mentions are collected."""
    passes = [[("SET-A", "DOC-A", "asthma")], [("SET-B", "DOC-B", "diabetes")], [("SET-C", "DOC-C", "epilepsy")]]
    results = mine_passes_multi_gpu(passes, _ner("asthma", "diabetes", "epilepsy"), ("cpu", "cpu", "cpu"))
    assert set(results.keys()) == {("SET-A", "DOC-A"), ("SET-B", "DOC-B"), ("SET-C", "DOC-C")}
    assert [m.text for m in results[("SET-C", "DOC-C")]] == ["epilepsy"]


def test_mine_passes_multi_gpu_one_device_two_passes_shares_it() -> None:
    """Fewer devices than passes: the empty group falls back to the first device."""
    passes = [[("SET-A", "DOC-A", "asthma")], [("SET-B", "DOC-B", "diabetes")]]
    results = mine_passes_multi_gpu(passes, _ner("asthma", "diabetes"), ("cpu",))
    assert [m.text for m in results[("SET-A", "DOC-A")]] == ["asthma"]
    assert [m.text for m in results[("SET-B", "DOC-B")]] == ["diabetes"]


# --- _group_devices --------------------------------------------------------------


def test_group_devices_even_split() -> None:
    assert _group_devices(["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 2) == [["cuda:0", "cuda:1"], ["cuda:2", "cuda:3"]]


def test_group_devices_remainder_goes_to_earlier_groups() -> None:
    assert _group_devices(["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 3) == [["cuda:0", "cuda:1"], ["cuda:2"], ["cuda:3"]]


def test_group_devices_never_empty() -> None:
    """k > len(devices): groups that would be empty defensively take the first device."""
    assert _group_devices(["cuda:0"], 3) == [["cuda:0"], ["cuda:0"], ["cuda:0"]]


# --- default_ner -----------------------------------------------------------------


def test_default_ner_without_fixture_root_uses_embedded_gazetteer() -> None:
    ner = default_ner(None)
    assert ner._offline
    assert [m.text for m in ner.extract("patients with asthma")] == ["asthma"]


# --- mine_with_cache ------------------------------------------------------------


class _FakeCache:
    """In-memory stand-in for MentionCache, serializing values like the real server does.

    Stores ``[mention.to_dict(), ...]`` and deserializes on read, emulating the Go server's
    verbatim-bytes round-trip — a hit is only byte-identical if the Mention (de)serialization
    is lossless.
    """

    def __init__(self) -> None:
        self.store: dict[str, list[dict[str, Any]]] = {}
        self.get_calls = 0
        self.put_calls = 0

    def get_many(self, keys: list[str]) -> dict[str, list[Mention]]:
        self.get_calls += 1
        return {key: [Mention.from_dict(item) for item in self.store[key]] for key in keys if key in self.store}

    def put_many(self, items: dict[str, list[Mention]]) -> None:
        self.put_calls += 1
        self.store.update({key: [mention.to_dict() for mention in mentions] for key, mentions in items.items()})


def _production_ner(tmp_path: Path) -> DiseaseNER:
    """A production-mode backend whose model is cached on disk (manifest only, no GLiNER load)."""

    def _write_model(_id: str, dest: Path) -> None:
        (dest / "w.bin").write_bytes(b"w")

    model_cache.ensure_model("acme/test-ner", cache_dir=tmp_path, downloader=_write_model)
    return DiseaseNER(offline=False, model_id="acme/test-ner", cache_dir=tmp_path)


def _counting_extract(ner: DiseaseNER, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``ner.extract`` with a counting fake; returns the call log."""
    calls: list[str] = []

    def fake_extract(text: str) -> list[Mention]:
        calls.append(text)
        return [Mention(text=text, start=0, end=len(text), type="disease", score=0.9)]

    monkeypatch.setattr(ner, "extract", fake_extract)
    return calls


def _sequential_mine(ner: DiseaseNER):
    def mine(items: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        return {(item[0], item[1]): ner.extract(item[2]) for item in items}

    return mine


def test_mine_with_cache_none_cache_passes_through() -> None:
    ner = _ner("asthma")
    items = [("S1", "D1", "asthma")]
    assert mine_with_cache(items, ner, _sequential_mine(ner), None) == {("S1", "D1"): ner.extract("asthma")}


def test_mine_with_cache_offline_backend_never_touches_cache() -> None:
    """The offline gazetteer is deterministic and CPU-cheap: deliberately not cached."""
    ner = _ner("asthma")
    cache = _FakeCache()
    mine_with_cache([("S1", "D1", "asthma")], ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert (cache.get_calls, cache.put_calls) == (0, 0)


def test_mine_with_cache_misses_mine_once_per_distinct_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Duplicate texts across work items are mined exactly once; results merge per item."""
    ner = _production_ner(tmp_path)
    calls = _counting_extract(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma"), ("S2", "D2", "diabetes"), ("S3", "D3", "asthma")]
    results = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert sorted(calls) == ["asthma", "diabetes"]
    assert set(results.keys()) == {("S1", "D1"), ("S2", "D2"), ("S3", "D3")}
    assert results[("S1", "D1")] == results[("S3", "D3")]  # same text -> same mentions
    assert len(cache.store) == 2


def test_mine_with_cache_second_run_is_all_hits_and_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A warm cache serves everything: extract is never called, output is byte-identical."""
    ner = _production_ner(tmp_path)
    calls = _counting_extract(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma"), ("S2", "D2", "diabetes")]
    first = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert len(calls) == 2

    calls.clear()
    second = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert calls == []  # all hits: the backend never ran
    assert second == first
    assert cache.put_calls == 1  # nothing re-put on the warm run


def test_mine_with_cache_results_match_no_cache_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cache hits + misses merge into exactly what a no-cache run produces."""
    ner = _production_ner(tmp_path)
    _counting_extract(ner, monkeypatch)
    items = [("S1", "D1", "asthma"), ("S2", "D2", "diabetes")]
    no_cache = mine_with_cache(items, ner, _sequential_mine(ner), None)
    cache = _FakeCache()
    cached_cold = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    cached_warm = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert cached_cold == no_cache
    assert cached_warm == no_cache
