"""Unit tests for the DailyMed evidence cache and the shape-stage "already done" skip.

``load_or_build_dailymed_evidence`` persists the built :class:`DailyMedEvidence` as a store
artifact keyed by the consumed interim tables (+ builder version), so the second shape task of
a run deserializes instead of re-scanning ``spl_sections.parquet``. ``cached_shape_outputs`` /
``write_assertion_table`` together back the shape tasks' skip: identical input ids + config
fingerprint return the previously registered output refs.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.assertions import evidence, row_for
from dakp_pipeline.assertions.evidence import (
    EVIDENCE_OPERATION,
    build_dailymed_evidence,
    cached_shape_outputs,
    load_or_build_dailymed_evidence,
    shape_config_fingerprint,
    shape_operation_inputs,
    write_assertion_table,
)
from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir

# --- DailyMed evidence cache ----------------------------------------------------


def test_load_or_build_dailymed_evidence_caches_the_second_call(
    ctx: TaskContext, dailymed_refs: list[ArtifactRef], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = load_or_build_dailymed_evidence(dailymed_refs, ctx)
    assert first == build_dailymed_evidence(dailymed_refs)  # cache write never changes content
    assert first.indication_docs  # real evidence, not an empty fallback

    def boom(_inputs: object) -> None:
        raise AssertionError("build_dailymed_evidence must not run on a cache hit")

    monkeypatch.setattr(evidence, "build_dailymed_evidence", boom)
    second = load_or_build_dailymed_evidence(dailymed_refs, ctx)
    assert second == first  # pickle round-trip equality


def test_load_or_build_dailymed_evidence_without_ctx_always_builds(dailymed_refs: list[ArtifactRef], monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real_build = evidence.build_dailymed_evidence

    def counting_build(inputs: object) -> object:
        nonlocal calls
        calls += 1
        return real_build(inputs)  # type: ignore[arg-type]

    monkeypatch.setattr(evidence, "build_dailymed_evidence", counting_build)
    load_or_build_dailymed_evidence(dailymed_refs)  # no ctx
    load_or_build_dailymed_evidence(dailymed_refs, None)
    assert calls == 2


def test_load_or_build_dailymed_evidence_without_consumed_tables_just_builds(
    ctx: TaskContext, drugsfda_refs: list[ArtifactRef], monkeypatch: pytest.MonkeyPatch
) -> None:
    # No spl_*.parquet among the inputs: the graceful-degradation path never touches the store.
    calls = 0
    real_build = evidence.build_dailymed_evidence

    def counting_build(inputs: object) -> object:
        nonlocal calls
        calls += 1
        return real_build(inputs)  # type: ignore[arg-type]

    monkeypatch.setattr(evidence, "build_dailymed_evidence", counting_build)
    load_or_build_dailymed_evidence(drugsfda_refs, ctx)
    load_or_build_dailymed_evidence(drugsfda_refs, ctx)
    assert calls == 2


def test_load_or_build_dailymed_evidence_rebuilds_when_the_cached_artifact_vanished(
    ctx: TaskContext, dailymed_refs: list[ArtifactRef], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = load_or_build_dailymed_evidence(dailymed_refs, ctx)

    calls = 0
    real_build = evidence.build_dailymed_evidence

    def counting_build(inputs: object) -> object:
        nonlocal calls
        calls += 1
        return real_build(inputs)  # type: ignore[arg-type]

    # Deleting the registered pickle prunes the index entry: the next call rebuilds + re-registers.
    store = ArtifactStore(Workdir(ctx.workdir))
    consumed = sorted(ref.blake3 for ref in dailymed_refs if ref.uri.name in evidence._EVIDENCE_TABLES)
    cached = store.find_by_operation(EVIDENCE_OPERATION, [*consumed, evidence._EVIDENCE_VERSION_INPUT])
    assert cached is not None
    cached[0].uri.unlink()

    monkeypatch.setattr(evidence, "build_dailymed_evidence", counting_build)
    rebuilt = load_or_build_dailymed_evidence(dailymed_refs, ctx)
    assert calls == 1
    assert rebuilt == first


# --- shape-stage "already done" skip ---------------------------------------------

_OPERATION = "shape_approved_treats"


def test_shape_config_fingerprint_is_stable_and_config_sensitive(ctx: TaskContext, disease_map: dict[str, dict[str, str]]) -> None:
    fp = shape_config_fingerprint(ctx)
    assert fp.startswith("b3:")
    assert shape_config_fingerprint(ctx) == fp  # deterministic

    changed_map = TaskContext(
        workdir=ctx.workdir, fixture_root=ctx.fixture_root, params={"disease_map": {**disease_map, "new disease": {"curie": "MONDO:1"}}}
    )
    assert shape_config_fingerprint(changed_map) != fp

    keywords = TaskContext(
        workdir=ctx.workdir, fixture_root=ctx.fixture_root, params={**dict(ctx.params), "contraindication_keywords": re.compile("renal")}
    )
    assert shape_config_fingerprint(keywords) != fp


def test_shape_operation_inputs_append_the_config_fingerprint(ctx: TaskContext, dailymed_refs: list[ArtifactRef]) -> None:
    ids = shape_operation_inputs(dailymed_refs, ctx)
    assert ids[: len(dailymed_refs)] == [ref.blake3 for ref in dailymed_refs]
    assert ids[-1] == shape_config_fingerprint(ctx)


def test_cached_shape_outputs_returns_refs_registered_by_write_assertion_table(ctx: TaskContext, dailymed_refs: list[ArtifactRef]) -> None:
    written = write_assertion_table("approved_treats_assertions", [], dailymed_refs, ctx, operation=_OPERATION)
    assert len(written) == 1

    cached = cached_shape_outputs(_OPERATION, dailymed_refs, ctx)
    assert cached is not None
    assert [ref.blake3 for ref in cached] == [ref.blake3 for ref in written]
    assert cached[0].uri == written[0].uri
    assert cached[0].uri.exists()

    # A changed config fingerprint misses (the table would be reshaped).
    other = TaskContext(workdir=ctx.workdir, fixture_root=ctx.fixture_root, params={"disease_map": {"other disease": {"curie": "MONDO:2"}}})
    assert cached_shape_outputs(_OPERATION, dailymed_refs, other) is None

    # force bypasses the skip even when everything matches.
    forced = TaskContext(workdir=ctx.workdir, fixture_root=ctx.fixture_root, params={**dict(ctx.params), "force": True})
    assert cached_shape_outputs(_OPERATION, dailymed_refs, forced) is None

    # Changed input ids miss too.
    assert cached_shape_outputs(_OPERATION, dailymed_refs[1:], ctx) is None


def test_cached_shape_outputs_misses_when_the_assertion_schema_changes(
    ctx: TaskContext, dailymed_refs: list[ArtifactRef], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_assertion_table("approved_treats_assertions", [], dailymed_refs, ctx, operation=_OPERATION)
    assert cached_shape_outputs(_OPERATION, dailymed_refs, ctx) is not None

    # A schema drift (e.g. a new evidence column) busts the skip: the stale TSV lacks the
    # columns the freshly generated Tablassert configs reference.
    drifted = {**schemas.ASSERTION_TABLES, "approved_treats_assertions": [*schemas.ASSERTION_TABLES["approved_treats_assertions"], "new_column"]}
    monkeypatch.setattr(schemas, "ASSERTION_TABLES", drifted)
    assert cached_shape_outputs(_OPERATION, dailymed_refs, ctx) is None


def test_cached_shape_outputs_prunes_when_the_table_was_deleted(ctx: TaskContext, dailymed_refs: list[ArtifactRef], tmp_path: Path) -> None:
    del tmp_path  # ctx owns the workdir
    written = write_assertion_table("approved_treats_assertions", [], dailymed_refs, ctx, operation=_OPERATION)
    written[0].uri.unlink()
    assert cached_shape_outputs(_OPERATION, dailymed_refs, ctx) is None


def test_write_assertion_table_coerces_number_of_cases_float_cells(ctx: TaskContext, dailymed_refs: list[ArtifactRef]) -> None:
    """A float-formatted count cell ("55.0") reaches the Tablassert handoff TSV as "55".

    Tablassert claims ``number_of_cases`` onto the integer-ranged Biolink slot only for
    integer-formatted cells; the v1.4.0 release shipped three ``applied_to_treat`` edges whose
    counts fell into ``has_supporting_studies`` descriptions because the cells were floats.
    """
    row = row_for(
        "faers_applied_to_treat_assertions", subject_text="drug", predicate="biolink:applied_to_treat", object_text="disease", number_of_cases="55.0"
    )
    written = write_assertion_table("faers_applied_to_treat_assertions", [row], dailymed_refs, ctx, operation=_OPERATION)
    # All-string schema: assert the TSV cell TEXT Tablassert will read (read_table would infer
    # a lone "55" column back as Int64).
    back = pl.read_csv(written[0].uri, separator="\t", infer_schema_length=0)
    assert back["number_of_cases"].to_list() == ["55"]
