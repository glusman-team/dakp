"""Regression tests for the FAERS observed-use NER bypass.

FAERS indication values are short report fields, not prose for biomedical NER. Observed-use
shaping therefore keeps the lexical disease-map path and leaves misses as text-first values for
Tablassert's downstream intervention/object mapping.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
import pytest

from dakp_pipeline.assertions.observed_uses import ObservedUsesShaper, build_observed_use_rows
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.ner.ner import DiseaseNER


def _cases(*indications: str) -> pl.DataFrame:
    return pl.DataFrame(
        {"drugname": ["DrugX"] * len(indications), "indication": list(indications), "primaryid": list(range(1, len(indications) + 1))}
    )


def test_dictionary_hit_keeps_the_lexical_path() -> None:
    disease_map = {"back pain": {"curie": "MONDO:1", "name": "back pain", "category": "Disease"}}
    rows = build_observed_use_rows(_cases("back pain"), disease_map)
    assert rows[0]["object_text"] == "back pain"
    assert rows[0]["object_curie"] == "MONDO:1"


def test_dictionary_match_is_not_replaced_by_ner_or_shortened() -> None:
    # A substring disease-map match remains the existing fast condition mapping. In particular,
    # FAERS does not ask NER to shorten a value such as ``back pain`` to ``pain``.
    disease_map = {"pain": {"curie": "HP:0012531", "name": "pain", "category": "Disease"}}
    rows = build_observed_use_rows(_cases("back pain"), disease_map, ner=DiseaseNER(gazetteer={"pain": "disease"}))
    assert rows[0]["object_text"] == "pain"
    assert rows[0]["object_curie"] == "HP:0012531"


def test_dictionary_miss_is_left_as_text_for_downstream_mapping() -> None:
    rows = build_observed_use_rows(_cases("acute kidney injury"), {})
    assert rows[0]["object_text"] == "acute kidney injury"
    assert rows[0]["object_curie"] == ""
    assert rows[0]["object_category"] == "Disease"


def test_faers_indications_never_invoke_ner(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_extract(_self: DiseaseNER, _text: str, **_kwargs: Any) -> Any:
        raise AssertionError("FAERS indication was sent to NER")

    monkeypatch.setattr(DiseaseNER, "extract", fail_extract)
    rows = build_observed_use_rows(_cases("Migraine prophylaxis", "acute kidney injury"), {}, ner=DiseaseNER(gazetteer={"migraine": "disease"}))
    assert [row["object_text"] for row in rows] == ["Migraine prophylaxis", "acute kidney injury"]


def test_shaper_ignores_injected_ner_for_faers(ctx: TaskContext, faers_refs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_extract(_self: DiseaseNER, _text: str, **_kwargs: Any) -> Any:
        raise AssertionError("FAERS indication was sent to NER")

    monkeypatch.setattr(DiseaseNER, "extract", fail_extract)
    injected_ctx = dataclasses.replace(ctx, params={**ctx.params, "ner": DiseaseNER(gazetteer={"migraine": "disease"})})
    refs = ObservedUsesShaper().transform(faers_refs, injected_ctx)
    assert len(refs) == 1
    assert refs[0].uri.name == "faers_applied_to_treat_assertions.tsv"
