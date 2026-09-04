"""Tests for the NER object-cleanup channel in ``dakp_pipeline.assertions.observed_uses``.

Covers: single-mention resolution of dictionary-miss FAERS indications (object text/name/
category from the mention), the conservative raw passthrough for zero/several/unknown-type
mentions, once-per-distinct-string mining (sequential + multi-GPU dispatch), and the shaper's
injected NER resolution.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
import pytest

from dakp_pipeline.assertions import observed_uses
from dakp_pipeline.assertions.observed_uses import ObservedUsesShaper, _mine_indication_mentions, _ner_object, build_observed_use_rows
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.ner.lexical import Mention
from dakp_pipeline.ner.ner import DiseaseNER


def _cases(*indications: str) -> pl.DataFrame:
    return pl.DataFrame(
        {"drugname": ["DrugX"] * len(indications), "indication": list(indications), "primaryid": list(range(1, len(indications) + 1))}
    )


# --- the single-mention resolution rule ------------------------------------------


def test_single_disease_mention_resolves_object() -> None:
    # 'Migraine prophylaxis' is no placeholder (the anchored stoplist needs the bare term) and the
    # dictionary misses; the single NER mention supplies the clean object.
    ner = DiseaseNER(gazetteer={"migraine": "disease"})
    rows = build_observed_use_rows(_cases("Migraine prophylaxis"), {}, ner=ner)
    assert len(rows) == 1
    assert rows[0]["object_text"] == "migraine"
    assert rows[0]["object_name"] == "migraine"
    assert rows[0]["object_category"] == "Disease"
    assert rows[0]["number_of_cases"] == "1"


def test_single_phenotype_mention_gets_phenotype_category() -> None:
    ner = DiseaseNER(gazetteer={"tremor": "phenotype"})
    rows = build_observed_use_rows(_cases("Severe tremor"), {}, ner=ner)
    assert rows[0]["object_text"] == "tremor"
    assert rows[0]["object_category"] == "PhenotypicFeature"


def test_multiple_mentions_keep_raw_passthrough() -> None:
    # A conjunction names two conditions; picking one would be a guess -> raw passthrough.
    ner = DiseaseNER(gazetteer={"migraine": "disease", "epilepsy": "disease"})
    rows = build_observed_use_rows(_cases("migraine and epilepsy"), {}, ner=ner)
    assert rows[0]["object_text"] == "migraine and epilepsy"
    assert rows[0]["object_category"] == "Disease"


def test_no_mention_keeps_raw_passthrough() -> None:
    ner = DiseaseNER(gazetteer={"migraine": "disease"})
    rows = build_observed_use_rows(_cases("zzz unknown condition"), {}, ner=ner)
    assert rows[0]["object_text"] == "zzz unknown condition"


def test_dictionary_hit_keeps_the_lexical_path() -> None:
    # Dictionary matches never consult the NER channel.
    ner = DiseaseNER(gazetteer={"migraine": "disease"})
    disease_map = {"back pain": {"curie": "MONDO:1", "name": "back pain", "category": "Disease"}}
    rows = build_observed_use_rows(_cases("back pain"), disease_map, ner=ner)
    assert rows[0]["object_text"] == "back pain"
    assert rows[0]["object_curie"] == "MONDO:1"


def test_stoplisted_indications_are_never_mined(monkeypatch: pytest.MonkeyPatch) -> None:
    ner = DiseaseNER(gazetteer={"migraine": "disease"})

    def fail_extract(text: str, **kwargs: Any) -> Any:
        raise AssertionError(f"stoplisted text mined: {text}")

    monkeypatch.setattr(DiseaseNER, "extract", fail_extract)
    rows = build_observed_use_rows(_cases("Product used for unknown indication"), {}, ner=ner)
    assert rows == []


# --- _ner_object unit edges ---------------------------------------------------------


def test_ner_object_unknown_mention_type_returns_none() -> None:
    mentions = {"x": [Mention(text="foo", start=0, end=3, type="organism", score=0.9)]}
    assert _ner_object("x", mentions) is None


def test_ner_object_blank_mention_text_returns_none() -> None:
    mentions = {"x": [Mention(text="!!!", start=0, end=3, type="disease", score=0.9)]}
    assert _ner_object("x", mentions) is None


def test_ner_object_without_map_returns_none() -> None:
    assert _ner_object("migraine", None) is None


def test_ner_object_lookup_miss_returns_none() -> None:
    mentions = {"migraine": [Mention(text="migraine", start=0, end=7, type="disease", score=0.9)]}
    assert _ner_object("epilepsy", mentions) is None


# --- mining dispatch ------------------------------------------------------------------


def test_mine_indication_mentions_empty_list() -> None:
    assert _mine_indication_mentions([], DiseaseNER(gazetteer={"migraine": "disease"}), None) == {}


def test_mine_indication_mentions_sequential_offline() -> None:
    mined = _mine_indication_mentions(["migraine prophylaxis"], DiseaseNER(gazetteer={"migraine": "disease"}), None)
    assert [m.text for m in mined["migraine prophylaxis"]] == ["migraine"]


def test_production_ner_dispatches_multi_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production NER + devices + >1 distinct miss string: mining goes through _mine_multi_gpu."""
    ner = DiseaseNER(offline=False, gazetteer={"migraine": "disease", "epilepsy": "disease"})

    called: list[dict[str, Any]] = []

    def fake_multi_gpu(work_items: Any, ner_arg: Any, devs: Any) -> dict[tuple[str, str], Any]:
        called.append({"items": len(work_items), "devices": tuple(devs)})
        offline = DiseaseNER(gazetteer=ner_arg._gazetteer)
        return {(s, d): offline.extract(t) for s, d, t in work_items}

    monkeypatch.setattr(observed_uses, "_mine_multi_gpu", fake_multi_gpu)

    rows = build_observed_use_rows(_cases("migraine prophylaxis", "epilepsy surgery"), {}, ner=ner, devices=("cuda:0", "cuda:1"))
    assert called == [{"items": 2, "devices": ("cuda:0", "cuda:1")}]
    by_object = {row["object_text"]: row for row in rows}
    assert by_object["migraine"]["object_category"] == "Disease"
    assert by_object["epilepsy"]["object_category"] == "Disease"


def test_offline_ner_never_dispatches_even_with_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    ner = DiseaseNER(offline=True, gazetteer={"migraine": "disease"})
    monkeypatch.setattr(observed_uses, "_mine_multi_gpu", lambda *args: (_ for _ in ()).throw(AssertionError("must not dispatch")))
    rows = build_observed_use_rows(_cases("migraine prophylaxis"), {}, ner=ner, devices=("cuda:0",))
    assert rows[0]["object_text"] == "migraine"


def test_distinct_indications_are_mined_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two raw variants normalize to the same string -> one mining call.
    seen: list[str] = []
    ner = DiseaseNER(gazetteer={"migraine": "disease"})
    original_extract = DiseaseNER.extract

    def counting_extract(self: DiseaseNER, text: str, **kwargs: Any) -> Any:
        seen.append(text)
        return original_extract(self, text, **kwargs)

    monkeypatch.setattr(DiseaseNER, "extract", counting_extract)
    build_observed_use_rows(_cases("Migraine prophylaxis", "migraine  prophylaxis"), {}, ner=ner)
    assert seen == ["migraine prophylaxis"]


# --- shaper NER resolution --------------------------------------------------------------


def test_shaper_uses_injected_ner(ctx: TaskContext, faers_refs: Any) -> None:
    ner = DiseaseNER(gazetteer={"migraine": "disease"})
    injected_ctx = dataclasses.replace(ctx, params={**ctx.params, "ner": ner})
    refs = ObservedUsesShaper().transform(faers_refs, injected_ctx)
    assert len(refs) == 1
    assert refs[0].uri.name == "faers_applied_to_treat_assertions.tsv"
