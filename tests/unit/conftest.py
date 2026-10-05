"""Shared fixtures for the Milestone-5 assertion-aggregation tests.

Runs the *real* extractors on the tiny pipeline fixtures so every assertion test aggregates
genuine interim tables (not hand-rolled mocks). ``ctx`` carries the lexical disease baseline
loaded from the ontology fixture, exactly as ``run_pipeline`` does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dakp_pipeline.extract import drugsfda_products, ema_registry, faers_ascii, spl_xml
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


@pytest.fixture
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.fixture
def disease_map() -> dict[str, dict[str, str]]:
    """Lexical disease baseline loaded from the ontology fixture (mirrors run_pipeline)."""
    frame = pl.read_csv(FIXTURE_ROOT / "ontology" / "disease_map.tsv", separator="\t")
    mapping: dict[str, dict[str, str]] = {}
    for rec in frame.iter_rows(named=True):
        text = str(rec.get("text", "") or "").strip()
        if text:
            mapping[text] = {
                "curie": str(rec.get("curie", "") or ""),
                "name": str(rec.get("name", text) or text),
                "category": str(rec.get("category", "Disease") or "Disease"),
            }
    return mapping


@pytest.fixture
def ctx(tmp_path: Path, disease_map: dict[str, dict[str, str]]) -> TaskContext:
    context = TaskContext(workdir=tmp_path / "work", fixture_root=FIXTURE_ROOT, params={"disease_map": disease_map})
    Workdir(context.workdir).create()
    return context


@pytest.fixture
def dailymed_refs(ctx: TaskContext) -> list[ArtifactRef]:
    return spl_xml.extract([_ref(FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz")], ctx)


@pytest.fixture
def drugsfda_refs(ctx: TaskContext) -> list[ArtifactRef]:
    return drugsfda_products.extract(
        [_ref(FIXTURE_ROOT / "drugsfda" / "drugsfda_products.tsv"), _ref(FIXTURE_ROOT / "drugsfda" / "drugsfda_applications.tsv")], ctx
    )


@pytest.fixture
def ema_refs(ctx: TaskContext) -> list[ArtifactRef]:
    return ema_registry.extract([_ref(FIXTURE_ROOT / "ema" / "medicines-output-medicines-report_en.xlsx")], ctx)


@pytest.fixture
def faers_refs(ctx: TaskContext) -> list[ArtifactRef]:
    """FAERS 24Q3 cases *without* the DELETE file, so all three cases (incl. Placebo) survive."""
    names = ("DEMO24Q3.txt", "DRUG24Q3.txt", "INDI24Q3.txt", "REAC24Q3.txt")
    return faers_ascii.extract([_ref(FIXTURE_ROOT / "faers" / name) for name in names], ctx)


@pytest.fixture(scope="session")
def dakp_build():
    """The Airflow ``dakp_build`` DAG module, imported lazily at execution (not collection).

    pytest-xdist collects the whole suite in *every* worker; importing the Airflow DAG at test-module
    load made all 24 workers pay ~1s of ``import airflow`` during collection. Importing it here defers
    that cost to the one worker that actually runs a DAG test, keeping per-worker collection cheap.
    """
    from dakp_pipeline.dags import dakp_build as _dag_build

    return _dag_build


class _PoolRecorder:
    """What a shaper did with its dispatch pool: how many pools, and what each ``mine`` received."""

    def __init__(self) -> None:
        self.pools = 0
        self.mined: list[tuple[int, tuple[str, ...]]] = []

    def __eq__(self, other: object) -> bool:
        """Compare against the ``mined`` list so tests read as ``assert recorder == [(2, (...))]``."""
        return self.mined == other

    def __repr__(self) -> str:
        return f"_PoolRecorder(pools={self.pools}, mined={self.mined})"


@pytest.fixture
def as_worker(monkeypatch: pytest.MonkeyPatch) -> Callable[[str, Any], None]:
    """Put this process in a pool worker's shoes: a claimed device slot plus its handed config.

    ``_init_worker`` also redirects logging and arms the parent-death signal, neither of which a unit
    test wants inside the pytest process, so only the state it would have set is installed here.
    ``monkeypatch`` restores all three globals at teardown, so worker state never leaks between tests.
    """

    def install(slot: str, config: Any = None) -> None:
        from dakp_pipeline.assertions import ner_dispatch

        monkeypatch.setattr(ner_dispatch, "_WORKER_SLOT", slot)
        monkeypatch.setattr(ner_dispatch, "_WORKER_CONFIG", dict(config or {}))
        monkeypatch.setattr(ner_dispatch, "_WORKER_BACKEND", None)

    return install


@pytest.fixture
def fake_dispatch_pool(monkeypatch: pytest.MonkeyPatch) -> Callable[..., _PoolRecorder]:
    """Patch a module's ``dispatch_pool`` seam with an in-process fake that never spawns a worker.

    ``install(module, mine_fn)`` returns a :class:`_PoolRecorder` whose ``mined`` is the list of
    ``(item_count, devices)`` calls made THROUGH THE POOL and whose ``pools`` counts the pool objects
    the shaper created. Those are two different quantities and the gap between them is the point:
    ``mine_with_cache`` calls ``mine`` once per cache-put batch, so a per-batch pool shows
    ``pools == len(mined)`` while one run-scoped pool shows ``pools == 1``.
    ``mine_fn(items, ner)`` supplies the canned result the real pool would have mined.

    The fake reproduces the real eligibility rule exactly (yields ``None`` for an offline backend or
    an empty device list) so the shaper's sequential fallback stays exercised by the same tests that
    exercise dispatch.
    """

    def install(module: Any, mine_fn: Callable[[Sequence[Any], Any], dict[tuple[str, str], Any]]) -> _PoolRecorder:
        recorder = _PoolRecorder()

        @contextmanager
        def _fake(ner: Any, devices: Sequence[str] | None) -> Iterator[Any]:
            slots = tuple(devices or ())
            if not slots or ner._offline:
                yield None
                return
            recorder.pools += 1

            class _FakePool:
                def mine(self, items: Sequence[Any]) -> dict[tuple[str, str], Any]:
                    recorder.mined.append((len(items), slots))
                    return mine_fn(items, ner)

            yield _FakePool()

        monkeypatch.setattr(module, "dispatch_pool", _fake)
        return recorder

    return install
