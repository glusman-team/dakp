"""Unit tests for the shared NER dispatch plumbing (assertions/ner_dispatch.py).

Covers the per-GPU dispatch primitives (``_mine_shard`` worker, ``_mine_multi_gpu`` LPT
orchestrator, device resolution) plus the persistent mention-cache seam (``mine_with_cache``). All tests run offline with gazetteer-only
DiseaseNERs on "cpu" devices (the spawn pool is real); cache tests use a fake in-memory
cache and a production-mode backend whose ``extract`` is monkeypatched — GLiNER never loads.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

import dakp_pipeline.assertions.ner_dispatch as dispatch
from dakp_pipeline.assertions.ner_dispatch import _mentions_fit, default_ner, mine_by_position, mine_with_cache
from dakp_pipeline.logging_setup import WORKER_LOG_SUBDIR
from dakp_pipeline.ner import model_cache
from dakp_pipeline.ner.ner import DiseaseNER, Mention


def _ner(*terms: str) -> DiseaseNER:
    return DiseaseNER(gazetteer=dict.fromkeys(terms, "disease"))


@contextmanager
def _captured_logs() -> Iterator[list[str]]:
    """Collect rendered loguru lines emitted inside the block (sink removed on exit)."""
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(sink_id)


# --- _mine_shard worker ---------------------------------------------------------


def test_shard_uses_one_batch_per_gpu_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    ner = _ner("asthma")
    calls: list[list[str]] = []

    def extract_batch(texts: Sequence[str]) -> list[list[Mention]]:
        calls.append(list(texts))
        return [ner.extract(text) for text in texts]

    class WorkerNER:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def extract_batch(self, texts: Sequence[str]) -> list[list[Mention]]:
            return extract_batch(texts)

    monkeypatch.setattr(dispatch, "DiseaseNER", WorkerNER)
    result = dispatch._mine_shard([("S1", "D1", "asthma"), ("S2", "D2", "asthma")], ner._config(), "cpu")
    assert calls == [["asthma", "asthma"]]
    assert {(set_id, doc_id) for set_id, doc_id, _mentions in result} == {("S1", "D1"), ("S2", "D2")}


def test_mine_shard_defaults_to_leaving_process_logging_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``in_worker`` the process-global logging reconfiguration never runs.

    This default is what protects the Airflow TASK process: ``configure_worker_logging`` calls
    ``logger.remove()`` and ``basicConfig(force=True)``, which would wipe the task's own sinks.
    An inferred guard could not do this safely -- ``multiprocessing.parent_process()`` is
    non-None in the Airflow task process itself under LocalExecutor.
    """
    reconfigured: list[tuple[Any, str]] = []
    monkeypatch.setattr(dispatch, "configure_worker_logging", lambda workdir, name: reconfigured.append((workdir, name)))
    dispatch._mine_shard([("S1", "D1", "asthma")], _ner("asthma")._config(), "cpu")
    assert reconfigured == []


def test_mine_shard_in_worker_configures_logging_before_loading_the_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``in_worker=True`` configures logging FIRST, before the backend (and its imports) load.

    Ordering is the point: ``DiseaseNER`` construction pulls in transformers/torch, which write
    to stderr as they initialize. Configuring afterwards would let exactly the noise this fixes
    escape, so both steps record into one list and the ORDER is asserted.
    """
    calls: list[Any] = []

    class WorkerNER:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("construct_ner", kwargs["device"]))

        def extract_batch(self, texts: Sequence[str]) -> list[list[Mention]]:
            return [[] for _ in texts]

    monkeypatch.setattr(dispatch, "configure_worker_logging", lambda workdir, name: calls.append(("configure_logging", workdir, name)))
    monkeypatch.setattr(dispatch, "_set_parent_death_signal", lambda: calls.append("pdeathsig"))
    monkeypatch.setattr(dispatch, "DiseaseNER", WorkerNER)
    ner = DiseaseNER(gazetteer={"asthma": "disease"}, workdir=tmp_path)
    dispatch._mine_shard([("S1", "D1", "asthma")], ner._config(), "cuda:2", in_worker=True)
    assert calls == [("configure_logging", tmp_path, "cuda:2"), "pdeathsig", ("construct_ner", "cuda:2")]


# --- _set_parent_death_signal ---------------------------------------------------


def test_mine_shard_logs_the_traceback_before_propagating(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shard failure leaves its traceback in the WORKER log, not only a pickled exception.

    The parent sees nothing but ``future.result()`` re-raising, so without this the worker file
    simply stops mid-shard: DAG runs 8 and 9 both died on ``IndexError: string index out of
    range`` with no traceback anywhere on disk to name the frame.
    """

    class BoomNER:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def extract_batch(self, _texts: Sequence[str]) -> list[list[Mention]]:
            raise IndexError("string index out of range")

    monkeypatch.setattr(dispatch, "DiseaseNER", BoomNER)
    with _captured_logs() as lines, pytest.raises(IndexError):
        dispatch._mine_shard([("S1", "D1", "asthma"), ("S2", "D2", "hives")], _ner("asthma")._config(), "cuda:1")
    assert any("ner_shard_failed: device = cuda:1 items = 2" in line for line in lines)


def test_mine_multi_gpu_names_the_device_whose_shard_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The task log attributes a shard failure to its device and points at the worker-log dir."""

    class FakeFuture:
        def __init__(self, device: str) -> None:
            self._device = device

        def result(self) -> list[tuple[str, str, list[Mention]]]:
            if self._device == "cuda:1":
                raise IndexError("string index out of range")
            return []

    class FakePool:
        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def submit(self, _function: Any, _shard: list[Any], _config: dict[str, Any], device: str, **_kwargs: Any) -> FakeFuture:
            return FakeFuture(device)

    monkeypatch.setattr(dispatch, "ProcessPoolExecutor", lambda **_kwargs: FakePool())
    ner = DiseaseNER(gazetteer={"asthma": "disease"}, workdir=tmp_path)
    with _captured_logs() as lines, pytest.raises(IndexError):
        dispatch._mine_multi_gpu([("S1", "D1", "asthma"), ("S2", "D2", "diabetes")], ner, ("cuda:0", "cuda:1"))
    assert any("ner_dispatch_failed: device = cuda:1 items = 1" in line for line in lines)
    assert any(str(tmp_path / WORKER_LOG_SUBDIR) in line for line in lines)


def test_pdeathsig_is_a_noop_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch.sys, "platform", "darwin")
    boom = lambda *_args: (_ for _ in ()).throw(AssertionError("CDLL must not load off-Linux"))
    monkeypatch.setattr(dispatch.ctypes, "CDLL", boom)
    dispatch._set_parent_death_signal()  # returns without touching ctypes


def _fake_libc(prctl_rc: int = 0) -> Any:
    calls: list[tuple[Any, ...]] = []

    class FakeLibc:
        @staticmethod
        def prctl(*args: Any) -> int:
            calls.append(args)
            return prctl_rc

    return FakeLibc(), calls


def test_pdeathsig_arms_sigkill_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    libc, calls = _fake_libc(prctl_rc=0)
    monkeypatch.setattr(dispatch.ctypes, "CDLL", lambda *_a, **_k: libc)
    monkeypatch.setattr(dispatch.os, "getppid", lambda: 4321)
    dispatch._set_parent_death_signal()
    assert calls == [(1, dispatch.signal.SIGKILL, 0, 0, 0)]  # PR_SET_PDEATHSIG = 1


def test_pdeathsig_warns_and_returns_when_prctl_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    libc, _calls = _fake_libc(prctl_rc=-1)
    monkeypatch.setattr(dispatch.ctypes, "CDLL", lambda *_a, **_k: libc)
    exited: list[int] = []
    monkeypatch.setattr(dispatch.os, "_exit", lambda code: exited.append(code))
    dispatch._set_parent_death_signal()  # no exit: the worker still functions without reaping
    assert exited == []


def test_pdeathsig_warns_and_returns_when_prctl_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch.sys, "platform", "linux")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("no libc here")

    monkeypatch.setattr(dispatch.ctypes, "CDLL", _boom)
    dispatch._set_parent_death_signal()


def test_pdeathsig_exits_when_parent_already_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    """The set-after-death race: parent died BEFORE prctl armed, so the kernel delivers nothing."""
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    libc, _calls = _fake_libc(prctl_rc=0)
    monkeypatch.setattr(dispatch.ctypes, "CDLL", lambda *_a, **_k: libc)
    monkeypatch.setattr(dispatch.os, "getppid", lambda: 1)
    exited: list[int] = []
    monkeypatch.setattr(dispatch.os, "_exit", lambda code: exited.append(code))
    dispatch._set_parent_death_signal()
    assert exited == [1]


def test_dispatch_announces_the_worker_log_directory_and_prunes_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The parent names the worker-log dir in the task log and prunes stale files before spawning.

    Worker narration lives in files the Airflow task log never shows, so without this line there
    is no pointer to it, and nothing else ever retires those per-process files.
    """
    announced: list[Any] = []
    monkeypatch.setattr(dispatch, "prune_worker_logs", lambda _workdir: 3)
    monkeypatch.setattr(dispatch, "stats", lambda _log, event, **fields: announced.append((event, fields)))

    class FakeFuture:
        def result(self) -> list[tuple[str, str, list[Mention]]]:
            return []

    class FakePool:
        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def submit(self, _function: Any, _shard: list[Any], _config: dict[str, Any], _device: str, **_kwargs: Any) -> FakeFuture:
            return FakeFuture()

    monkeypatch.setattr(dispatch, "ProcessPoolExecutor", lambda **_kwargs: FakePool())
    ner = DiseaseNER(gazetteer={"asthma": "disease"}, workdir=tmp_path)
    dispatch._mine_multi_gpu([("S1", "D1", "asthma")], ner, ("cuda:0",))
    assert announced == [("ner_worker_logs", {"path": str(tmp_path / WORKER_LOG_SUBDIR), "pruned": 3})]


def test_dispatch_submits_shards_with_the_in_worker_flag_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every ``pool.submit`` opts the CHILD into the logging redirect (and only the child)."""
    submitted: list[dict[str, Any]] = []

    class FakeFuture:
        def result(self) -> list[tuple[str, str, list[Mention]]]:
            return []

    class FakePool:
        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def submit(self, _function: Any, _shard: list[Any], _config: dict[str, Any], _device: str, **kwargs: Any) -> FakeFuture:
            submitted.append(kwargs)
            return FakeFuture()

    monkeypatch.setattr(dispatch, "ProcessPoolExecutor", lambda **_kwargs: FakePool())
    ner = DiseaseNER(gazetteer={"asthma": "disease"}, workdir=tmp_path)
    dispatch._mine_multi_gpu([("S1", "D1", "asthma"), ("S2", "D2", "diabetes")], ner, ("cuda:0", "cuda:1"))
    assert submitted == [{"in_worker": True}] * 2


def test_announce_worker_logs_without_a_workdir_says_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No workdir: there is no worker-log directory to name or prune (the None branch)."""
    announced: list[Any] = []
    monkeypatch.setattr(dispatch, "stats", lambda _log, event, **_fields: announced.append(event))
    dispatch._announce_worker_logs(None)
    assert announced == []


def test_four_device_sharding_creates_four_distinct_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sufficiently large workload schedules one independent shard per visible device."""
    submitted: list[tuple[list[Any], str]] = []

    class FakeFuture:
        def result(self) -> list[tuple[str, str, list[Mention]]]:
            return []

    class FakePool:
        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def submit(self, _function: Any, shard: list[Any], _config: dict[str, Any], device: str, **_kwargs: Any) -> FakeFuture:
            submitted.append((shard, device))
            return FakeFuture()

    monkeypatch.setattr(dispatch, "ProcessPoolExecutor", lambda **_kwargs: FakePool())
    ner = _ner("asthma")
    items = [("S", f"D{i}", "asthma " * (i + 1)) for i in range(8)]
    dispatch._mine_multi_gpu(items, ner, ("cuda:0", "cuda:1", "cuda:2", "cuda:3"))
    assert [device for _shard, device in submitted] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert sorted(item[1] for shard, _device in submitted for item in shard) == [f"D{i}" for i in range(8)]
    assert all(shard for shard, _device in submitted)


# --- mine_by_position: (set_id, doc_id) is not a unique work-item key --------------


def test_mine_by_position_gives_duplicate_documents_their_own_mentions() -> None:
    """Two sections of ONE SPL document must not share a mining-map entry.

    Regression guard for run 12: ``doc_id`` is the document id, so the pair collided, one item
    received the other's mentions, and those offsets — relative to a different text — failed
    ``attach_qualifiers_with_scores``' sentence-bounds contract two seconds after a 2h51m mine.
    """
    items = [("SET-A", "DOC-A", "asthma"), ("SET-A", "DOC-A", "diabetes"), ("SET-B", "DOC-B", "asthma")]
    ner = _ner("asthma", "diabetes")

    def mine(batch: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        # Receives the ordinal-suffixed proxies; keys its results by whatever it is handed.
        return {(set_id, doc_id): ner.extract(text) for set_id, doc_id, text in batch}

    mined = mine_by_position(items, ner, mine, None)
    assert [[mention.text for mention in row] for row in mined] == [["asthma"], ["diabetes"], ["asthma"]]


def test_mine_by_position_deduplicates_identical_texts_through_the_cache_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical texts are still mined once: the rekeying leaves the cache/dedup seam intact."""
    items = [("SET-A", "DOC-A", "asthma"), ("SET-A", "DOC-A", "asthma"), ("SET-A", "DOC-B", "diabetes")]
    ner = _ner("asthma", "diabetes")
    monkeypatch.setattr(dispatch, "ner_cache_material", lambda _backend: ("model", "b3deadbeef", "fingerprint"))
    mined_texts: list[str] = []

    def mine(batch: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        mined_texts.extend(text for _set_id, _doc_id, text in batch)
        return {(set_id, doc_id): ner.extract(text) for set_id, doc_id, text in batch}

    cache = _FakeCache()
    out = mine_by_position(items, ner, mine, cache)  # type: ignore[arg-type]
    assert sorted(mined_texts) == ["asthma", "diabetes"]  # one representative per distinct text
    assert [[mention.text for mention in row] for row in out] == [["asthma"], ["asthma"], ["diabetes"]]
    assert (cache.get_calls, cache.put_calls) == (1, 1)


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
        return [Mention(text=text, start=0, end=len(text), type="Disease", score=0.9)]

    monkeypatch.setattr(ner, "extract", fake_extract)
    return calls


def _sequential_mine(ner: DiseaseNER):
    def mine(items: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        return {(item[0], item[1]): ner.extract(item[2]) for item in items}

    return mine


def test_mentions_fit_rejects_an_entry_mined_from_another_text() -> None:
    """The mention contract is the cheap proof that a cached entry belongs to THIS text."""
    text = "contraindicated in asthma"
    start = text.index("asthma")
    assert _mentions_fit(text, [Mention("asthma", start, start + 6, "Disease", 1.0)])
    assert _mentions_fit(text, [])  # an empty entry is a valid "no mentions" result, not a miss
    assert not _mentions_fit(text, [Mention("asthma", 589, 605, "Disease", 1.0)])  # run 13: past the end
    assert not _mentions_fit(text, [Mention("asthma", 0, 6, "Disease", 1.0)])  # in bounds, wrong surface
    assert not _mentions_fit(text, [Mention("asthma", -1, start + 6, "Disease", 1.0)])  # negative start


def test_mine_with_cache_reminds_when_a_cached_entry_does_not_fit_the_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale entry is a MISS: the text is re-mined and the entry overwritten.

    Regression guard for run 13 — a cached entry mined from a whitespace variant of the section
    served offsets past the end of the requesting text (589..605 into 568 chars), which the
    shaper's sentence-bounds contract turned into a failed task after 2h51m of mining.
    """
    ner = _production_ner(tmp_path)
    calls = _counting_extract(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma")]
    first = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert calls == ["asthma"]

    key = next(iter(cache.store))
    stale = [Mention("asthma", 589, 605, "Disease", 1.0).to_dict()]
    cache.store[key] = stale
    calls.clear()
    second = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert calls == ["asthma"]  # re-mined instead of served
    assert second == first  # and the correct mentions are back
    assert cache.store[key] != stale  # the stale entry was overwritten


def test_mine_with_cache_separates_texts_that_share_a_folded_cache_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Whitespace variants share one cache key but need their own offsets, so each is mined.

    Keying folds whitespace while offsets index the raw text; deduplicating representatives by key
    would hand one variant the other's mentions.
    """
    ner = _production_ner(tmp_path)
    calls = _counting_extract(ner, monkeypatch)
    cache = _FakeCache()
    items = [("S1", "D1", "asthma  in adults"), ("S2", "D2", "asthma in adults")]
    out = mine_with_cache(items, ner, _sequential_mine(ner), cache)  # type: ignore[arg-type]
    assert sorted(calls) == ["asthma  in adults", "asthma in adults"]
    assert [mention.text for mention in out[("S1", "D1")]] == ["asthma  in adults"]
    assert [mention.text for mention in out[("S2", "D2")]] == ["asthma in adults"]
    assert len(cache.store) == 1  # one folded key: last write wins, _mentions_fit re-checks it


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
