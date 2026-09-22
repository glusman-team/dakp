"""Error/edge branches of the NER training-data export (:mod:`dakp_pipeline.ner_export`).

Complements the happy-path suite in ``test_ner_export.py``: gold validation, unreadable
or schema-drifted interim tables, defensive row-skipping branches, the RelMedNER model
mirror's unused projection helpers, mention localization gaps, qualifier attachment,
the :func:`export` entry point's loud error paths and cache skip, and the CLI's
reference-extractor helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dakp_pipeline import ner_export
from dakp_pipeline.assertions.contexts import attach_qualifiers_with_scores, patient_clause_contexts
from dakp_pipeline.cli import extract_fixture_sources
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.lexical import Mention

#: Same repo-root convention as ``tests/unit/conftest.py``.
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _write_parquet(path: Path, frame: pl.DataFrame) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return _ref(path)


# --- RelMedNER model-mirror projection helpers ------------------------------------------


def test_choice_field_to_output() -> None:
    choice = ner_export.ChoiceField(value="daily", choices=["daily", "weekly"])
    assert choice.to_output() == {"value": "daily", "choices": ["daily", "weekly"]}


def test_structure_field_to_value_both_shapes() -> None:
    plain = ner_export.StructureField(name="n", value="v", description=None)
    nested = ner_export.StructureField(name="n", value=ner_export.ChoiceField(value="a", choices=["a"]), description="d")
    assert plain.to_value() == "v"
    assert nested.to_value() == {"value": "a", "choices": ["a"]}


def test_describe_and_structures_out() -> None:
    described = ner_export.describe([ner_export.Description(key="k", description="d")])
    assert described == {"k": "d"}
    assert ner_export.describe(None) == {}
    assert ner_export.describe([]) == {}
    example = ner_export.TrainingExample(
        text="t",
        structures=[
            ner_export.Structure(
                name="s",
                fields=[
                    ner_export.StructureField(name="a", value="1", description="first"),
                    ner_export.StructureField(name="b", value="2", description=None),
                ],
            ),
            ner_export.Structure(name="undocumented", fields=[ner_export.StructureField(name="c", value="3", description=None)]),
        ],
    )
    out = example.structures_out()
    assert out["json_descriptions"] == {"s": {"a": "first"}}
    assert out["json_structures"] == [{"s": {"a": "1", "b": "2"}}, {"undocumented": {"c": "3"}}]


# --- gold path + gold validation ---------------------------------------------------------


def test_gold_path_missing_is_loud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ner_export, "Path", lambda _raw: tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="NER gold benchmark is missing"):
        ner_export.gold_path()


def _gold(tmp_path: Path, payload: dict[str, Any], name: str = "ner_gold.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_VALID_GOLD: dict[str, Any] = {
    "schema_version": ner_export.GOLD_SCHEMA_VERSION,
    "annotation_policy": "mined from assertion tables",
    "cases": [{"text": "Drug treats asthma"}],
}


def test_load_gold_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NER gold benchmark is missing"):
        ner_export._load_gold(tmp_path / "absent.json")


def test_load_gold_not_json(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text("not json {", encoding="utf-8")
    with pytest.raises(ValueError, match="not readable JSON"):
        ner_export._load_gold(path)


def test_load_gold_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        ner_export._load_gold(path)


def test_load_gold_bad_schema_version(tmp_path: Path) -> None:
    path = _gold(tmp_path, {**_VALID_GOLD, "schema_version": "dakp.ner.gold.v0"})
    with pytest.raises(ValueError, match="schema_version"):
        ner_export._load_gold(path)


def test_load_gold_missing_annotation_policy(tmp_path: Path) -> None:
    path = _gold(tmp_path, {k: v for k, v in _VALID_GOLD.items() if k != "annotation_policy"})
    with pytest.raises(ValueError, match="annotation_policy"):
        ner_export._load_gold(path)


def test_load_gold_empty_cases(tmp_path: Path) -> None:
    path = _gold(tmp_path, {**_VALID_GOLD, "cases": []})
    with pytest.raises(ValueError, match="non-empty cases list"):
        ner_export._load_gold(path)


# --- interim-table readers ----------------------------------------------------------------


def test_read_dailymed_sections_falls_back_to_full_read(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.parquet"
    pl.DataFrame({"spl_document_id": ["d1"], "section_text": ["text"]}).write_parquet(legacy)
    frame = ner_export.read_dailymed_sections(legacy)
    assert frame is not None
    assert frame.height == 1


def test_read_dailymed_sections_unreadable_returns_none(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.parquet"
    garbage.write_bytes(b"definitely not parquet")
    assert ner_export.read_dailymed_sections(garbage) is None


def test_read_ema_registry_falls_back_to_full_read(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.parquet"
    pl.DataFrame({"therapeutic_indication": ["treats x"]}).write_parquet(legacy)
    frame = ner_export.read_ema_registry(legacy)
    assert frame is not None
    assert frame.height == 1


def test_read_ema_registry_unreadable_returns_none(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.parquet"
    garbage.write_bytes(b"definitely not parquet")
    assert ner_export.read_ema_registry(garbage) is None


# --- candidate-row selection edges --------------------------------------------------------


def test_singleton_subjects_missing_columns_is_empty() -> None:
    assert ner_export._singleton_subjects(pl.DataFrame({"x": [1]})) == {}


def test_singleton_subjects_skips_blank_document_and_blanks_multi_ingredient() -> None:
    table = pl.DataFrame({"spl_document_id": ["", "d1", "d1", "d2"], "active_ingredient_name": ["Drug", "Aspirin", "Ibuprofen", "Standalone"]})
    subjects = ner_export._singleton_subjects(table)
    assert subjects == {"d2": "Standalone"}


def test_select_dailymed_rows_skips_unknown_loinc_and_blank_text() -> None:
    table = pl.DataFrame(
        {
            "spl_document_id": ["d1", "d2", "d3"],
            "loinc_code": ["34070-3", "99999-9", "34070-3"],
            "section_text": ["real text", "wrong section", "   "],
            "active_ingredient_name": ["Drug", "Drug", "Drug"],
        }
    )
    rows = ner_export.select_dailymed_rows(table)
    assert len(rows) == 1
    assert rows[0]["source_document_id"] == "d1"


def test_select_ema_rows_skips_blank_indication() -> None:
    table = pl.DataFrame(
        {
            "therapeutic_indication": ["treats x", "  "],
            "active_substance": ["Drug", "Drug"],
            "inn": ["", ""],
            "ema_product_number": ["EMAV", "EMAV"],
            "medicine_url": ["", ""],
        }
    )
    rows = ner_export.select_ema_rows(table)
    assert len(rows) == 1
    assert rows[0]["text"] == "treats x"


def test_reduce_faers_frame_without_indication_column_is_identity() -> None:
    table = pl.DataFrame({"drugname": ["Drug"]})
    assert ner_export.reduce_faers_frame(table) is table


def test_quarter_urls_without_quarter_column_is_empty() -> None:
    assert ner_export._quarter_urls(pl.DataFrame({"x": [1]})) == {}


def test_nonblank_indications_without_indication_column_is_identity() -> None:
    table = pl.DataFrame({"drugname": ["Drug"]})
    assert ner_export._nonblank_indications(table) is table


def test_select_faers_rows_skips_blank_indication() -> None:
    table = pl.DataFrame({"indication": ["headache", ""], "primaryid": ["1", "2"], "quarter": ["24Q2", "24Q2"]})
    rows = ner_export.select_faers_rows(table)
    assert len(rows) == 1
    assert rows[0]["text"] == "headache"


# --- localization + qualifier relations ---------------------------------------------------


def test_localize_drops_mentions_outside_every_sentence_span() -> None:
    text = "Drug treats asthma."
    stray = Mention(text="asthma", start=100, end=106, type="disease", score=1.0)
    sentence_of, localized = ner_export._localize(text, [stray])
    assert localized == []
    assert sentence_of(stray) is None


def test_localize_drops_mentions_crossing_sentence_boundaries() -> None:
    """A mention spanning two sentences cannot be made sentence-relative; it is excluded
    from the relation build (entity slot kept) instead of tripping the attachment bounds
    guard on real mined data."""
    text = "Drug treats asthma. Patients tolerate it well."
    crosser = Mention(text="asthma. Patients", start=12, end=28, type="disease", score=1.0)
    sentence_of, localized = ner_export._localize(text, [crosser])
    assert localized == []
    assert sentence_of(crosser) is None


def test_qualifier_relations_fire_when_one_host_matches() -> None:
    sentence = "Drug treats asthma in adult male patients."
    objects = [Mention(text="asthma", start=12, end=18, type="disease", score=1.0)]
    qualifiers = [Mention(text="male", start=27, end=31, type="BiologicalSex", score=0.9)]

    def sentence_of(mention: Mention) -> str:
        return sentence

    attached = attach_qualifiers_with_scores(objects, qualifiers, sentence_of)[0]
    assert attached == {0: {"sex_text": "male"}}
    relations = ner_export._qualifier_relations(objects, qualifiers, sentence_of)
    assert [(relation.name, relation.fields[0].value, relation.fields[1].value, relation.evidence) for relation in relations] == [
        ("sex_qualifier", "asthma", "male", "mined")
    ]


def test_patient_clause_contexts_skips_unmapped_sentences() -> None:
    objects = [Mention(text="asthma", start=0, end=6, type="disease", score=1.0)]
    clause = patient_clause_contexts(objects, lambda mention: None)
    assert clause.contexts == {}
    assert clause.context_only == {}
    assert clause.ambiguous == {}


# --- export() entry point ------------------------------------------------------------------


def _table_refs(tmp_path: Path, *, dailymed: bool = True, faers: bool = True, ema: bool = True) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    if dailymed:
        refs.append(
            _write_parquet(
                tmp_path / "interim" / "dailymed" / "spl_documents.parquet",
                pl.DataFrame(
                    {
                        "spl_document_id": ["d1"],
                        "loinc_code": ["34070-3"],
                        "section_text": ["Drug is contraindicated in asthma."],
                        "active_ingredient_name": ["Drug"],
                    }
                ),
            )
        )
    if faers:
        refs.append(
            _write_parquet(
                tmp_path / "interim" / "faers" / "cases.parquet",
                pl.DataFrame({"quarter": ["24Q2"], "primaryid": ["1"], "drugname": ["Drug"], "indication": ["asthma"]}),
            )
        )
    if ema:
        refs.append(
            _write_parquet(
                tmp_path / "interim" / "ema" / "ema_registry.parquet",
                pl.DataFrame(
                    {
                        "therapeutic_indication": ["treatment of asthma"],
                        "active_substance": ["Drug"],
                        "inn": [""],
                        "ema_product_number": ["EMAV"],
                        "medicine_url": ["https://ema.example"],
                    }
                ),
            )
        )
    return refs


def _export_ctx(tmp_path: Path) -> TaskContext:
    context = TaskContext(workdir=tmp_path / "work", fixture_root=FIXTURE_ROOT, params={})
    from dakp_pipeline.paths import Workdir

    Workdir(context.workdir).create()
    return context


def test_export_missing_dailymed_table_raises(tmp_path: Path) -> None:
    refs = _table_refs(tmp_path, dailymed=False)
    with pytest.raises(RuntimeError, match="missing DailyMed interim table"):
        ner_export.export(refs, _export_ctx(tmp_path))


def test_export_missing_faers_table_raises(tmp_path: Path) -> None:
    refs = _table_refs(tmp_path, faers=False)
    with pytest.raises(RuntimeError, match="missing FAERS case table"):
        ner_export.export(refs, _export_ctx(tmp_path))


def test_export_missing_ema_table_raises(tmp_path: Path) -> None:
    refs = _table_refs(tmp_path, ema=False)
    with pytest.raises(RuntimeError, match="missing EMA registry table"):
        ner_export.export(refs, _export_ctx(tmp_path))


def test_export_unreadable_dailymed_table_raises(tmp_path: Path) -> None:
    garbage = tmp_path / "interim" / "dailymed" / "spl_documents.parquet"
    garbage.parent.mkdir(parents=True)
    garbage.write_bytes(b"not parquet")
    refs = _table_refs(tmp_path, dailymed=False)
    refs.insert(0, _ref(garbage))
    with pytest.raises(RuntimeError, match="unreadable DailyMed interim table"):
        ner_export.export(refs, _export_ctx(tmp_path))


def test_export_unusable_faers_table_raises(tmp_path: Path) -> None:
    refs = _table_refs(tmp_path, faers=False)
    refs.append(_write_parquet(tmp_path / "interim" / "faers" / "cases.parquet", pl.DataFrame({"x": [1]})))
    with pytest.raises(RuntimeError, match="unusable FAERS case table"):
        ner_export.export(refs, _export_ctx(tmp_path))


def test_export_unreadable_ema_table_raises(tmp_path: Path) -> None:
    refs = _table_refs(tmp_path, ema=False)
    garbage = tmp_path / "interim" / "ema" / "ema_registry.parquet"
    garbage.parent.mkdir(parents=True)
    garbage.write_bytes(b"not parquet")
    refs.append(_ref(garbage))
    with pytest.raises(RuntimeError, match="unreadable EMA registry"):
        ner_export.export(refs, _export_ctx(tmp_path))


def test_export_skips_when_cache_has_the_operation_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    refs = _table_refs(tmp_path)
    cache_dir = tmp_path / "cached"
    cache_dir.mkdir()
    cached: list[ArtifactRef] = []
    for name in (ner_export.MANIFEST_FILENAME, ner_export.EXAMPLES_AVRO_FILENAME):
        path = cache_dir / name
        path.write_bytes(b"cached")
        cached.append(_ref(path))
    monkeypatch.setattr(ArtifactStore, "find_by_operation", lambda self, operation, inputs: cached)
    produced = ner_export.export(refs, _export_ctx(tmp_path))
    assert produced == cached


# --- CLI reference-extractor helper ---------------------------------------------------------


def test_extract_fixture_sources_runs_the_reference_extractors(ctx: TaskContext) -> None:
    refs = extract_fixture_sources(ctx, FIXTURE_ROOT)
    names = {ref.uri.name for ref in refs}
    assert "spl_documents.parquet" in names
    assert any(name.startswith("cases") for name in names)
