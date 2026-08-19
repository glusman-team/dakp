"""Edge-case tests for the ``dakp_pipeline.translator`` regression guardrails.

Covers the two branches the fixture/positive tests miss: a family invariant whose
``knowledge_level`` is ``None`` (the guard skips the check), reached by monkeypatching a
custom family into ``FAMILY_INVARIANTS``; and ``check_assertion_tables`` skipping refs whose
stem is not a DAKP assertion table.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline import translator as regression
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.translator import check_assertion_tables, check_rows

_DAKP = "infores:multiomics-drugapprovals"


def test_check_rows_skips_unconstrained_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    # A family with neither a clinical_approval_status nor a knowledge_level constraint:
    # both guards short-circuit (the knowledge_level-None branch is otherwise unreachable).
    custom = regression.FamilyInvariant("biolink:custom", frozenset({"infores:dailymed"}), None, None)
    patched = dict(regression.FAMILY_INVARIANTS)
    patched["biolink:custom"] = custom
    monkeypatch.setattr(regression, "FAMILY_INVARIANTS", patched)

    row = {
        "predicate": "biolink:custom",
        "primary_knowledge_source": _DAKP,
        "upstream_resource_ids": "infores:dailymed",
        "clinical_approval_status": "anything at all",
        "knowledge_level": "whatever",
    }
    report = check_rows([row])
    assert report.ok is True
    assert report.violations == []
    assert report.families_seen == ["biolink:custom"]
    assert report.row_count == 1


def test_check_assertion_tables_skips_non_assertion_refs(tmp_path: Path) -> None:
    path = tmp_path / "not_an_assertion_table.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    ref = ArtifactRef(uri=path, blake3="b3:" + "0" * 64, media_type="text/tab-separated-values")
    report = check_assertion_tables([ref])
    assert report.row_count == 0
    assert report.ok is True
    assert report.families_seen == []


def test_check_assertion_tables_streams_parquet_refs(tmp_path: Path) -> None:
    # Parquet interims stream through the lazy engine (the batched reader is TSV-only).
    columns = schemas.columns_for("contraindication_assertions")
    row = dict.fromkeys(columns, "")
    row.update(
        {
            "predicate": "biolink:contraindicated_in",
            "knowledge_level": "knowledge_assertion",
            "primary_knowledge_source": _DAKP,
            "upstream_resource_ids": "infores:dailymed",
        }
    )
    path = tmp_path / "contraindication_assertions.parquet"
    schemas.write_parquet(pl.DataFrame([row], schema=columns), path)
    ref = ArtifactRef(uri=path, blake3="b3:" + "4" * 64, media_type=schemas.PARQUET_MEDIA_TYPE)

    report = check_assertion_tables([ref])
    assert report.ok is True
    assert report.row_count == 1
    assert report.families_seen == ["biolink:contraindicated_in"]
