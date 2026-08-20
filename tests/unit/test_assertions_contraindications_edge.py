"""Edge-case tests for ``dakp_pipeline.assertions.contraindications`` (drive to 100% branch coverage).

Targets: ``default_ner`` fallback when the fixture lacks an ontology (and when fixture_root is
None); a contraindication set with no active ingredient; multi-ingredient (combination-product)
sets skipped by the singleton-ingredient discipline in both mining passes; a blank mined span; ingredient rows with
missing fields / duplicates in the shared evidence cache; a second observation of the same
(subject, object) pair unioning support; the empty-scores ``_max_score`` guard; the shaper honoring / ignoring an injected
``params["ner"]``; multi-GPU dispatch (LPT sharding, ``_mine_shard`` worker, ``_mine_multi_gpu``
orchestrator, ``_resolve_devices`` CUDA guard, and the ``devices`` param on
``build_contraindication_rows``). Inputs are tiny parquet tables built in tmp so no heavy NER deps
are needed.

**Pass 2 tests** cover: the sentence keyword filter (``_split_sentences`` / ``_contraindication_sentences``);
embedded contraindication provenance from indication sections (``SETID#34067-9``); false-positive
prevention (indication-only diseases are NOT mined); no-regression on contraindication-only sets;
a fake monkeypatched GLiNER mock for the production path; and ``_mine_two_passes_multi_gpu``
2+2 parallel dispatch.
"""

from __future__ import annotations

import re
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from dakp_pipeline.assertions.contraindications import (
    CONTRAINDICATION_GPUS,
    DEFAULT_CONTRA_KEYWORDS,
    ContraindicationsShaper,
    ContraWorkItem,
    EvidenceSpan,
    _accumulate,
    _classify_mention,
    _classify_mentions,
    _contraindication_sentences,
    _finalize_row,
    _max_score,
    _mention_local_span,
    _mine_multi_gpu,
    _mine_shard,
    _mine_two_passes_multi_gpu,
    _resolve_devices,
    _resolve_keywords,
    _sentence_spans,
    _shard_by_text_length,
    _spawn_safe_main,
    _split_sentences,
    _work_item_evidence,
    _work_item_parts,
    build_contraindication_rows,
    default_ner,
)
from dakp_pipeline.assertions.evidence import build_dailymed_evidence
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.ner import DiseaseNER, Mention
from dakp_pipeline.paths import Workdir

CONTRA_LOINC = "34070-3"
INDICATION_LOINC = "34067-9"
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _sections(tmp_path: Path, rows: list[tuple[str, str, str]]) -> ArtifactRef:
    """spl_sections.parquet from (spl_set_id, spl_document_id, text) rows (all contraindication)."""
    frame = pl.DataFrame(
        {
            "spl_set_id": [r[0] for r in rows],
            "spl_document_id": [r[1] for r in rows],
            "clean_text": [r[2] for r in rows],
            "loinc_code": [CONTRA_LOINC for _ in rows],
        }
    )
    path = tmp_path / "spl_sections.parquet"
    frame.write_parquet(path)
    return _ref(path)


def _mixed_sections(tmp_path: Path, rows: list[tuple[str, str, str, str]]) -> ArtifactRef:
    """spl_sections.parquet from (spl_set_id, spl_document_id, loinc_code, text) rows.

    Unlike :func:`_sections`, each row carries its own LOINC code so a single file can mix
    indication (34067-9) and contraindication (34070-3) sections.
    """
    frame = pl.DataFrame(
        {
            "spl_set_id": [r[0] for r in rows],
            "spl_document_id": [r[1] for r in rows],
            "loinc_code": [r[2] for r in rows],
            "clean_text": [r[3] for r in rows],
        }
    )
    path = tmp_path / "spl_sections.parquet"
    frame.write_parquet(path)
    return _ref(path)


def _ingredients(tmp_path: Path, rows: list[tuple[str, str, str, str]]) -> ArtifactRef:
    """spl_ingredients.parquet from (role, spl_set_id, ingredient_name, ingredient_unii) rows."""
    frame = pl.DataFrame(
        {
            "role": [r[0] for r in rows],
            "spl_set_id": [r[1] for r in rows],
            "ingredient_name": [r[2] for r in rows],
            "ingredient_unii": [r[3] for r in rows],
        }
    )
    path = tmp_path / "spl_ingredients.parquet"
    frame.write_parquet(path)
    return _ref(path)


def test_work_item_parts_accepts_legacy_tuple() -> None:
    """The tuple-like fallback in ``_work_item_parts`` reads plain (set_id, doc_id, text) tuples."""
    assert _work_item_parts(("SET-A", "DOC-A", "text")) == ("SET-A", "DOC-A", "text")


class _BlankNER(DiseaseNER):
    """A backend that extracts a single whitespace-only span (to exercise the blank-skip)."""

    def extract(self, text: str) -> list[Mention]:
        return [Mention(text="   ", start=0, end=3, type="disease", score=1.0)]


def _ctx(tmp_path: Path, params: Mapping[str, Any]) -> TaskContext:
    context = TaskContext(workdir=tmp_path / "work", fixture_root=FIXTURE_ROOT, params=params)
    Workdir(context.workdir).create()
    return context


# --- default_ner fallbacks ------------------------------------------------------


def test_default_ner_fixture_without_ontology_falls_back_to_embedded(tmp_path: Path) -> None:
    # fixture_root exists but has no ontology/disease_map.tsv -> embedded-gazetteer backend.
    ner = default_ner(tmp_path)
    assert isinstance(ner, DiseaseNER)
    # The embedded gazetteer still recognizes common terms.
    assert [m.text for m in ner.extract("asthma")] == ["asthma"]


def test_default_ner_none_fixture_uses_embedded() -> None:
    ner = default_ner(None)
    assert isinstance(ner, DiseaseNER)
    assert [m.text for m in ner.extract("headache")] == ["headache"]


# --- set without an active ingredient is skipped --------------------------------


def test_contraindication_set_without_active_ingredient_is_skipped(tmp_path: Path) -> None:
    sections = _sections(tmp_path, [("SET-X", "SET-X#d", "asthma"), ("SET-Y", "SET-Y#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-Y", "DrugY", "UNII:Y")])  # SET-X has none
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    # SET-X is skipped (no active ingredient); only SET-Y -> DrugY mines a row.
    assert [r["subject_text"] for r in rows] == ["DrugY"]
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["object_curie"] == ""


# --- singleton-ingredient discipline (legacy selectActiveIngredientSingletons.pl) ------------


def test_multi_ingredient_contraindication_set_is_skipped(tmp_path: Path) -> None:
    # SET-COMBO has TWO actives (a combination product): pairing the mention with each component
    # would over-attribute the contraindication, so the set is never mined. The singleton SET-S
    # still mines its row.
    sections = _sections(tmp_path, [("SET-COMBO", "SET-COMBO#d", "asthma"), ("SET-S", "SET-S#d", "asthma")])
    ingredients = _ingredients(
        tmp_path,
        [("active", "SET-COMBO", "ComponentA", "UNII:A"), ("active", "SET-COMBO", "ComponentB", "UNII:B"), ("active", "SET-S", "DrugS", "UNII:S")],
    )
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    assert [r["subject_text"] for r in rows] == ["DrugS"]
    assert rows[0]["supporting_spl_sets"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-S"
    assert rows[0]["supporting_spl_evidence"] == "dailymed:SET-S"


def test_multi_ingredient_indication_set_is_skipped_in_pass_2(tmp_path: Path) -> None:
    # Pass 2 obeys the same singleton rule: a combination product's indication section is not mined.
    sections = _mixed_sections(tmp_path, [("SET-COMBO", "SET-COMBO#34067-9", INDICATION_LOINC, "It is contraindicated in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-COMBO", "ComponentA", "UNII:A"), ("active", "SET-COMBO", "ComponentB", "UNII:B")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    assert build_contraindication_rows([sections, ingredients], ner) == []


# --- blank mined span is skipped ------------------------------------------------


def test_blank_mined_span_is_skipped(tmp_path: Path) -> None:
    sections = _sections(tmp_path, [("SET-Y", "SET-Y#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-Y", "DrugY", "UNII:Y")])
    assert build_contraindication_rows([sections, ingredients], _BlankNER()) == []  # whitespace-only -> skipped


# --- mined mention text is case-normalized --------------------------------------


def test_mined_mention_case_is_normalized(tmp_path: Path) -> None:
    # The same ingredient mentions the disease with different casing across SPL sections; the mined
    # mention text is canonicalized (normalize_text) so the case variants collapse to one object
    # instead of fragmenting into asthma / Asthma / ASTHMA rows.
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "Asthma"), ("SET-B", "SET-B#d", "contraindicated in ASTHMA")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugY", "UNII:Y"), ("active", "SET-B", "DrugY", "UNII:Y")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})
    rows = build_contraindication_rows([sections, ingredients], ner)
    assert len(rows) == 1  # both sets aggregate to the single (DrugY, asthma) pair
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["subject_text"] == "DrugY"


# --- DailyMedEvidence.active_ingredients_by_set: missing fields + duplicates ----


def test_active_ingredients_skip_missing_fields_inactive_and_duplicates(tmp_path: Path) -> None:
    ingredients = _ingredients(
        tmp_path,
        [
            ("active", "", "NoSet", "UNII:1"),  # missing set_id -> skipped
            ("active", "SET", "", "UNII:2"),  # missing name -> skipped
            ("inactive", "SET", "Inactive", "UNII:3"),  # not active -> skipped
            ("active", "SET", "DrugY", "UNII:Y"),  # kept
            ("active", "SET", "DrugY", "UNII:Y"),  # exact duplicate -> skipped
            ("active", "SET", "drugy", "UNII:Y"),  # case-insensitive duplicate key -> skipped
            ("active", "SET", "OtherDrug", "UNII:Z"),  # kept
        ],
    )
    evidence = build_dailymed_evidence([ingredients])
    assert evidence.active_ingredients_by_set == {"SET": [("DrugY", "UNII:Y"), ("OtherDrug", "UNII:Z")]}  # sorted, deduped
    assert evidence.set_ingredient == {"SET": ("DrugY", "UNII:Y")}  # first active ingredient retained for treatment fallback


def test_active_ingredients_empty_without_ingredients_table(tmp_path: Path) -> None:
    sections = _sections(tmp_path, [("SET-Y", "SET-Y#d", "asthma")])
    assert build_dailymed_evidence([sections]).active_ingredients_by_set == {}  # no spl_ingredients.parquet present
    assert build_dailymed_evidence([]).active_ingredients_by_set == {}


# --- second observation of the same pair unions support -------------------------


def test_second_observation_of_same_pair_unions_support(tmp_path: Path) -> None:
    # Two sets share the SAME active ingredient and both mention 'asthma' -> one aggregated row.
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "asthma"), ("SET-B", "SET-B#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X"), ("active", "SET-B", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    assert len(rows) == 1
    row = rows[0]
    assert row["subject_text"] == "DrugX"
    assert row["object_text"] == "asthma"
    assert row["object_curie"] == ""
    assert (
        row["supporting_spl_sets"]
        == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A|https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-B"
    )  # both observations unioned
    assert row["source_score"] == "1"


# --- _max_score guard -----------------------------------------------------------


def test_max_score_empty_and_nonempty() -> None:
    assert _max_score([]) == ""  # defensive guard: no scores -> empty string
    assert _max_score([0.5, 0.9, 0.7]) == "0.9"
    assert _max_score([1.0]) == "1"


# --- shaper: injected / ignored ner param ---------------------------------------


def test_shaper_uses_injected_ner_param(tmp_path: Path) -> None:
    from dakp_pipeline.extract import spl_xml

    ctx = _ctx(tmp_path, {"ner": DiseaseNER(gazetteer={"asthma": "disease", "liver disease": "disease"})})
    refs = spl_xml.extract([_ref(FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz")], ctx)
    out = ContraindicationsShaper().transform(refs, ctx)
    assert len(out) == 1
    # The injected gazetteer mines BOTH contraindication sections.
    from dakp_pipeline.io import schemas

    subjects = sorted(schemas.read_table(out[0].uri)["subject_text"].to_list())
    assert subjects == ["Examplestatin", "Ibuprofen"]


def test_shaper_ignores_non_backend_ner_param_and_falls_back(tmp_path: Path) -> None:
    from dakp_pipeline.extract import spl_xml
    from dakp_pipeline.io import schemas

    # A non-DiseaseNER "ner" param is ignored; the shaper falls back to default_ner(fixture_root).
    ctx = _ctx(tmp_path, {"ner": "not a backend"})
    refs = spl_xml.extract([_ref(FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz")], ctx)
    out = ContraindicationsShaper().transform(refs, ctx)
    frame = schemas.read_table(out[0].uri)
    assert frame.height == 1  # fixture gazetteer -> Ibuprofen -> asthma only
    assert frame.row(0, named=True)["subject_text"] == "Ibuprofen"


# --- multi-GPU dispatch: LPT sharding -------------------------------------------


def test_shard_by_text_length_balances_by_text_length() -> None:
    """LPT scheduling distributes items so the total text length per shard is balanced."""
    items = [(f"S{i}", f"D{i}", "x" * length) for i, length in enumerate([100, 50, 30, 10])]
    shards = _shard_by_text_length(items, 2)
    loads = [sum(len(item[2]) for item in shard) for shard in shards]
    # LPT assigns 100 -> shard 0, 50 -> shard 1, 30 -> shard 1 (80<100), 10 -> shard 1 (90<100).
    assert loads == [100, 90]


def test_shard_by_text_length_returns_n_shards_even_if_empty() -> None:
    """Always returns exactly n shards (some may be empty when n > len(items))."""
    shards = _shard_by_text_length([], 3)
    assert len(shards) == 3
    assert all(shard == [] for shard in shards)


def test_shard_by_text_length_preserves_all_items() -> None:
    """Every item appears exactly once across all shards."""
    items = [(f"S{i}", f"D{i}", f"text-{i}") for i in range(6)]
    shards = _shard_by_text_length(items, 3)
    all_items = [item for shard in shards for item in shard]
    assert sorted(all_items) == sorted(items)


# --- multi-GPU dispatch: _mine_shard worker -------------------------------------


def test_mine_shard_extracts_mentions_from_each_text() -> None:
    """_mine_shard reconstructs a DiseaseNER from config and extracts mentions for each text."""
    ner = DiseaseNER(gazetteer={"asthma": "disease"})
    config = ner._config()
    shard = [("SET-A", "DOC-A", "patient has asthma"), ("SET-B", "DOC-B", "no disease here")]
    results = _mine_shard(shard, config, "cpu")
    assert len(results) == 2
    assert results[0][0] == "SET-A"  # set_id preserved
    assert results[0][1] == "DOC-A"  # doc_id preserved
    assert [m.text for m in results[0][2]] == ["asthma"]
    assert results[1][2] == []  # no mentions


def test_mine_shard_empty_shard_returns_empty_list() -> None:
    """An empty shard yields an empty result list (no texts to mine)."""
    ner = DiseaseNER(gazetteer={"asthma": "disease"})
    assert _mine_shard([], ner._config(), "cpu") == []


# --- multi-GPU dispatch: _mine_multi_gpu orchestrator --------------------------


def test_mine_multi_gpu_collects_mentions_from_all_workers() -> None:
    """_mine_multi_gpu shards work and collects mentions from every worker into one map."""
    ner = DiseaseNER(gazetteer={"asthma": "disease", "diabetes": "disease"})
    items = [("SET-A", "DOC-A", "asthma"), ("SET-B", "DOC-B", "diabetes")]
    results = _mine_multi_gpu(items, ner, ("cpu", "cpu"))
    assert set(results.keys()) == {("SET-A", "DOC-A"), ("SET-B", "DOC-B")}
    assert [m.text for m in results[("SET-A", "DOC-A")]] == ["asthma"]
    assert [m.text for m in results[("SET-B", "DOC-B")]] == ["diabetes"]


# --- multi-GPU dispatch: build_contraindication_rows devices param ---------------


def test_devices_ignored_for_offline_ner(tmp_path: Path) -> None:
    """The multi-GPU path is skipped for offline NERs even when devices are given — output
    is identical to sequential."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "asthma"), ("SET-B", "SET-B#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X"), ("active", "SET-B", "DrugX", "UNII:X")])
    ner = DiseaseNER(offline=True, gazetteer={"asthma": "disease"})
    rows_seq = build_contraindication_rows([sections, ingredients], ner)
    rows_multi = build_contraindication_rows([sections, ingredients], ner, devices=("cuda:0", "cuda:1"))
    assert rows_seq == rows_multi  # offline NER -> sequential regardless of devices


def test_devices_single_work_item_falls_back_to_sequential(tmp_path: Path) -> None:
    """Only one work item -> no benefit from multi-GPU, so sequential is used."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})
    rows = build_contraindication_rows([sections, ingredients], ner, devices=("cuda:0", "cuda:1"))
    assert len(rows) == 1
    assert rows[0]["object_text"] == "asthma"


# --- _resolve_devices: CUDA guard for multi-GPU ---------------------------------


def test_resolve_devices_returns_none_for_offline_ner() -> None:
    """Offline NER never triggers multi-GPU — the gazetteer is CPU-only."""
    assert _resolve_devices(DiseaseNER(offline=True)) is None


def test_resolve_devices_returns_gpus_when_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production NER + 4 visible CUDA devices -> the full hardcoded GPU list."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_50", "sm_60", "sm_75"])
    assert _resolve_devices(DiseaseNER(offline=False)) == CONTRAINDICATION_GPUS


def test_resolve_devices_caps_at_visible_device_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-GPU host (e.g. the laptop) gets only cuda:0 — never a missing cuda:N."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_60"])
    assert _resolve_devices(DiseaseNER(offline=False)) == ("cuda:0",)


def test_resolve_devices_returns_none_when_no_arch_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA visible but torch has no kernels for any GPU's arch (e.g. a cu128 build on the
    P100s) — sequential CPU fallback instead of dispatching workers that crash on first use."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    assert _resolve_devices(DiseaseNER(offline=False)) is None


def test_resolve_devices_keeps_only_arch_supported_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed fleet: only devices whose arch is compiled into torch are dispatched."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 6) if index == 0 else (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_86"])
    assert _resolve_devices(DiseaseNER(offline=False)) == ("cuda:0",)


def test_resolve_devices_skips_device_whose_capability_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device that errors on capability query counts as unsupported; the rest still dispatch."""
    import torch

    def capability(index: int) -> tuple[int, int]:
        if index == 0:
            raise RuntimeError("CUDA error")
        return (8, 6)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_86"])
    assert _resolve_devices(DiseaseNER(offline=False)) == ("cuda:1",)


def test_resolve_devices_returns_none_when_cuda_reports_zero_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_available() True but device_count() == 0 (driver edge) falls back to sequential."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert _resolve_devices(DiseaseNER(offline=False)) is None


def test_resolve_devices_returns_none_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production NER but no CUDA -> sequential fallback (CI / non-GPU hosts)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_devices(DiseaseNER(offline=False)) is None


def test_resolve_devices_returns_none_when_torch_unimportable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production NER but torch not importable -> sequential fallback."""
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    assert _resolve_devices(DiseaseNER(offline=False)) is None


# --- multi-GPU dispatch: production NER + devices -> _mine_multi_gpu is called ------


def test_build_rows_dispatches_to_multi_gpu_for_production_ner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Production NER + devices + >1 item: build_contraindication_rows calls _mine_multi_gpu."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "asthma"), ("SET-B", "SET-B#d", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X"), ("active", "SET-B", "DrugY", "UNII:Y")])
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease"})

    called: list[tuple[int, tuple[str, ...]]] = []

    def fake_multi_gpu(work_items: list, ner_arg: DiseaseNER, devices: tuple[str, ...]) -> dict:
        called.append((len(work_items), tuple(devices)))
        # Use an offline clone to avoid GLiNER loading in tests.
        offline = DiseaseNER(gazetteer=ner_arg._gazetteer)
        return {(s, d): offline.extract(t) for s, d, t in work_items}

    import dakp_pipeline.assertions.contraindications as contra_mod

    monkeypatch.setattr(contra_mod, "_mine_multi_gpu", fake_multi_gpu)

    rows = build_contraindication_rows([sections, ingredients], ner, devices=("cuda:0", "cuda:1"))
    assert called == [(2, ("cuda:0", "cuda:1"))]  # dispatched with 2 items across 2 devices
    assert len(rows) == 2
    assert {r["subject_text"] for r in rows} == {"DrugX", "DrugY"}


def test_shaper_logs_and_dispatches_when_multi_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The shaper logs and passes devices to build_contraindication_rows when CUDA is available."""
    from dakp_pipeline.extract import spl_xml
    from dakp_pipeline.io import schemas

    ctx = _ctx(tmp_path, {"ner": DiseaseNER(offline=False, gazetteer={"asthma": "disease", "liver disease": "disease"})})
    refs = spl_xml.extract([_ref(FIXTURE_ROOT / "dailymed" / "dailymed_spl.xml.gz")], ctx)

    # Force the multi-GPU path: _resolve_devices returns GPUs, _mine_multi_gpu avoids GPU work.
    import dakp_pipeline.assertions.contraindications as contra_mod

    monkeypatch.setattr(contra_mod, "_resolve_devices", lambda _ner: CONTRAINDICATION_GPUS)
    monkeypatch.setattr(
        contra_mod, "_mine_multi_gpu", lambda items, ner, devs: {(s, d): DiseaseNER(gazetteer=ner._gazetteer).extract(t) for s, d, t in items}
    )

    out = ContraindicationsShaper().transform(refs, ctx)
    assert len(out) == 1
    subjects = sorted(schemas.read_table(out[0].uri)["subject_text"].to_list())
    assert subjects == ["Examplestatin", "Ibuprofen"]


# --- Pass 2: sentence keyword filter helpers -----------------------------------


def test_split_sentences_basic() -> None:
    """Sentence splitter handles periods and semicolons + whitespace."""
    assert _split_sentences("Hello. World.") == ["Hello.", "World."]
    assert _split_sentences("A; B; C") == ["A;", "B;", "C"]
    assert _split_sentences("No boundaries here") == ["No boundaries here"]
    assert _split_sentences("") == []
    assert _split_sentences("Trailing.") == ["Trailing."]


def test_contraindication_sentences_keeps_only_keyword_sentences() -> None:
    """Only sentences matching contraindication keywords survive the filter."""
    text = "DrugX is indicated for hypertension. It is contraindicated in patients with asthma. DrugX is also indicated for diabetes."
    result = _contraindication_sentences(text, DEFAULT_CONTRA_KEYWORDS)
    assert "contraindicated" in result.lower()
    assert "asthma" in result.lower()
    assert "hypertension" not in result.lower()
    assert "diabetes" not in result.lower()


def test_contraindication_sentences_empty_when_no_keywords() -> None:
    """Indication-only text yields empty filtered text (no Pass 2 work)."""
    text = "DrugX is indicated for hypertension and type 2 diabetes mellitus."
    assert _contraindication_sentences(text, DEFAULT_CONTRA_KEYWORDS) == ""


def test_contraindication_sentences_catches_various_phrasings() -> None:
    """Multiple contraindication phrasings trigger the filter."""
    for phrase in [
        "Should not be used in patients with heart failure.",
        "Avoid use in severe renal impairment.",
        "Not recommended in pregnancy.",
        "Do not use in active bleeding.",
        "Use is contraindicated in hepatic impairment.",
    ]:
        assert _contraindication_sentences(phrase, DEFAULT_CONTRA_KEYWORDS) != "", f"Failed for: {phrase}"


# --- Pass 2: contraindication mined from indication section --------------------


def test_conditional_contraindication_mined_from_indication_section_keeps_original_evidence(tmp_path: Path) -> None:
    """Pass 2 maps filtered NER offsets back to the original sentence and context."""
    sections = _mixed_sections(
        tmp_path,
        [
            (
                "SET-CONTEXT",
                "SET-CONTEXT#34067-9",
                INDICATION_LOINC,
                "DrugX is indicated for hypertension. It is contraindicated for treatment of hypertension in patients with asthma.",
            )
        ],
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-CONTEXT", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease", "hypertension": "disease"})
    rows = build_contraindication_rows([sections, ingredients], ner)
    assert len(rows) == 1
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["disease_context_text"] == "hypertension"
    assert rows[0]["evidence_text"] == "It is contraindicated for treatment of hypertension in patients with asthma."
    assert rows[0]["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-CONTEXT#34067-9"


def test_contraindication_mined_from_indication_section(tmp_path: Path) -> None:
    """Pass 2: a contraindication embedded in the indication section is mined with #34067-9 provenance."""
    sections = _mixed_sections(
        tmp_path,
        [
            (
                "SET-A",
                "SET-A#34067-9",
                INDICATION_LOINC,
                "DrugX is indicated for hypertension. It is contraindicated in patients with asthma. DrugX is also indicated for diabetes.",
            )
        ],
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease", "hypertension": "disease", "diabetes": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    by_object = {r["object_text"] for r in rows}

    # asthma is in a contraindication-context sentence -> mined.
    assert "asthma" in by_object
    # hypertension and diabetes are in indication-context sentences -> NOT mined.
    assert "hypertension" not in by_object
    assert "diabetes" not in by_object

    # Provenance: the indication section document (34067-9).
    asthma_row = next(r for r in rows if r["object_text"] == "asthma")
    assert asthma_row["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#34067-9"
    assert asthma_row["supporting_spl_sets"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A"


def test_indication_only_diseases_not_mined_as_contraindications(tmp_path: Path) -> None:
    """Pass 2: indication text with no contraindication keywords yields no rows."""
    sections = _mixed_sections(
        tmp_path, [("SET-A", "SET-A#34067-9", INDICATION_LOINC, "DrugX is indicated for hypertension and type 2 diabetes mellitus.")]
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"hypertension": "disease", "type 2 diabetes mellitus": "disease"})

    assert build_contraindication_rows([sections, ingredients], ner) == []


def test_both_passes_mine_different_sections_of_same_set(tmp_path: Path) -> None:
    """Pass 1 (contraindication section) and Pass 2 (indication section) both contribute."""
    sections = _mixed_sections(
        tmp_path,
        [
            ("SET-A", "SET-A#34070-3", CONTRA_LOINC, "Contraindicated in patients with asthma."),
            ("SET-A", "SET-A#34067-9", INDICATION_LOINC, "DrugX is indicated for hypertension. It is contraindicated in patients with diabetes."),
        ],
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease", "diabetes": "disease", "hypertension": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    by_object = {r["object_text"] for r in rows}
    assert by_object == {"asthma", "diabetes"}  # both passes contribute
    assert "hypertension" not in by_object  # indication-only -> excluded

    # Provenance: asthma from 34070-3 (Pass 1), diabetes from 34067-9 (Pass 2).
    by_obj = {r["object_text"]: r for r in rows}
    assert by_obj["asthma"]["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#34070-3"
    assert by_obj["diabetes"]["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#34067-9"


def test_no_regression_contraindication_only_sections(tmp_path: Path) -> None:
    """A set with only contraindication sections (no indication section) is unaffected by Pass 2."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#34070-3", "asthma")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner)
    assert len(rows) == 1
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#34070-3"


# --- Gold-style semantic fixture matrix ----------------------------------------


@pytest.mark.parametrize(
    ("label", "terms", "expected_object", "expected_context"),
    [
        ("Contraindicated in patients with asthma.", {"asthma": "disease"}, "asthma", ""),
        (
            "Contraindicated for treatment of hypertension in patients with asthma.",
            {"hypertension": "disease", "asthma": "disease"},
            "asthma",
            "hypertension",
        ),
        ("Contraindicated for asthma.", {"asthma": "disease"}, "asthma", ""),
        ("Contraindicated for treatment of hypertension in patients receiving warfarin.", {"hypertension": "disease"}, None, None),
        ("None known for asthma.", {"asthma": "disease"}, None, None),
        ("Asthma is not contraindicated.", {"asthma": "disease"}, None, None),
        ("Avoid use in patients with asthma.", {"asthma": "disease"}, None, None),
        (
            "Contraindicated for treatment of hypertension in patients with asthma and diabetes.",
            {"hypertension": "disease", "asthma": "disease", "diabetes": "disease"},
            None,
            None,
        ),
    ],
    ids=[
        "direct",
        "explicit-disease-context",
        "blank-context",
        "medication-context",
        "none-known",
        "explicit-negation",
        "warning-only",
        "and-context",
    ],
)
def test_gold_semantic_fixture_matrix(
    tmp_path: Path, label: str, terms: dict[str, str], expected_object: str | None, expected_context: str | None
) -> None:
    """Small precision-first matrix for direct, conditional, negative, and medication language."""
    sections = _sections(tmp_path, [("SET-GOLD", "SET-GOLD#34070-3", label)])
    ingredients = _ingredients(tmp_path, [("active", "SET-GOLD", "DrugGold", "UNII:GOLD")])
    rows = build_contraindication_rows([sections, ingredients], DiseaseNER(gazetteer=terms))
    if expected_object is None:
        assert rows == []
        return
    assert len(rows) == 1
    row = rows[0]
    assert row["object_text"] == expected_object
    assert row["disease_context_text"] == expected_context
    assert row["evidence_text"] == label


# --- Pass 2: configurable keywords ---------------------------------------------


def test_custom_keywords_filter_indication_section(tmp_path: Path) -> None:
    """A custom keyword pattern (via the ``keywords`` param) controls which sentences are mined."""
    import re

    custom = re.compile(r"\bnever\s+give\b", re.IGNORECASE)
    sections = _mixed_sections(
        tmp_path, [("SET-A", "SET-A#34067-9", INDICATION_LOINC, "DrugX is indicated for asthma. Never give to patients with diabetes.")]
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease", "diabetes": "disease"})

    rows = build_contraindication_rows([sections, ingredients], ner, keywords=custom)
    by_object = {r["object_text"] for r in rows}
    assert "diabetes" in by_object  # custom keyword matched
    assert "asthma" not in by_object  # default keywords would NOT have matched, custom didn't either


# --- Fake GLiNER mock for the production path -----------------------------------


class _TermScanningGLiNERModel:
    """Fake GLiNER model that returns entities for known disease terms found in text.

    Unlike a fixed-prediction mock, this model scans the input text for known disease terms
    and returns them as entities. This lets tests verify that the sentence filter prevents
    indication-context diseases from ever reaching the model.
    """

    def __init__(self, terms: dict[str, str]) -> None:
        self._terms = terms  # lowercase term -> entity type
        self.calls: list[str] = []

    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5) -> list[dict[str, Any]]:
        self.calls.append(text)
        preds: list[dict[str, Any]] = []
        lower = text.lower()
        for term, etype in sorted(self._terms.items(), key=lambda x: -len(x[0])):  # longest first
            idx = 0
            while True:
                found = lower.find(term, idx)
                if found == -1:
                    break
                preds.append({"text": text[found : found + len(term)], "start": found, "end": found + len(term), "label": etype, "score": 0.8})
                idx = found + len(term)
        return preds


def _install_term_scanning_gliner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terms: dict[str, str]) -> _TermScanningGLiNERModel:
    """Install a fake gliner module whose ``predict_entities`` scans text for known disease terms.

    Also patches ``ensure_model`` so the production NER backend can load without network access.
    Returns the fake model so tests can inspect ``.calls`` to verify which texts reached GLiNER.
    """
    model = _TermScanningGLiNERModel(terms)
    module = types.ModuleType("gliner")
    module.GLiNER = type("GLiNER", (), {"from_pretrained": staticmethod(lambda *a, **kw: model)})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)

    import dakp_pipeline.ner.ner as ner_module
    from dakp_pipeline.ner.model_cache import ModelRef

    def _fake_ensure_model(model_id: str, **kwargs: Any) -> ModelRef:
        return ModelRef(model_id=model_id, source="huggingface", path=tmp_path, b3="b3:deadbeef", manifest=tmp_path / "manifest.json")

    monkeypatch.setattr(ner_module, "ensure_model", _fake_ensure_model)
    return model


def test_production_ner_mines_contraindication_from_indication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Production NER (fake GLiNER) + sentence filter: embedded contraindication is mined,
    indication-only diseases are NOT (sentence filter prevents them from reaching GLiNER)."""
    disease_terms = {"asthma": "disease", "hypertension": "disease", "diabetes": "disease"}
    model = _install_term_scanning_gliner(monkeypatch, tmp_path, disease_terms)

    sections = _mixed_sections(
        tmp_path,
        [
            (
                "SET-A",
                "SET-A#34067-9",
                INDICATION_LOINC,
                "DrugX is indicated for hypertension. It is contraindicated in patients with asthma. DrugX is also indicated for diabetes.",
            )
        ],
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])

    # Production NER with empty gazetteer — relies on GLiNER (fake) for extraction.
    ner = DiseaseNER(offline=False, gazetteer={}, cache_dir=tmp_path)
    rows = build_contraindication_rows([sections, ingredients], ner)

    by_object = {r["object_text"] for r in rows}
    assert "asthma" in by_object  # contraindication context -> mined
    assert "hypertension" not in by_object  # indication context -> filtered out before GLiNER
    assert "diabetes" not in by_object  # indication context -> filtered out before GLiNER

    # The fake GLiNER only received the contraindication-context sentence — not the full text.
    all_glimer_input = " ".join(model.calls)
    assert "contraindicated" in all_glimer_input.lower()
    assert "hypertension" not in all_glimer_input.lower()  # sentence filter prevented it


# --- _mine_two_passes_multi_gpu: 2+2 parallel dispatch -------------------------


def test_mine_two_passes_multi_gpu_splits_devices_and_collects() -> None:
    """_mine_two_passes_multi_gpu shards both passes across half the GPUs each and collects results."""
    ner = DiseaseNER(gazetteer={"asthma": "disease", "diabetes": "disease"})
    work_p1 = [("SET-A", "DOC-A", "asthma")]
    work_p2 = [("SET-B", "DOC-B", "diabetes")]
    results = _mine_two_passes_multi_gpu(work_p1, work_p2, ner, ("cpu", "cpu"))

    assert set(results.keys()) == {("SET-A", "DOC-A"), ("SET-B", "DOC-B")}
    assert [m.text for m in results[("SET-A", "DOC-A")]] == ["asthma"]
    assert [m.text for m in results[("SET-B", "DOC-B")]] == ["diabetes"]


def test_mine_two_passes_no_pass2_falls_back_to_single() -> None:
    """When Pass 2 has no work items, _mine_two_passes_multi_gpu delegates to _mine_multi_gpu."""
    ner = DiseaseNER(gazetteer={"asthma": "disease"})
    work_p1 = [("SET-A", "DOC-A", "asthma")]
    results = _mine_two_passes_multi_gpu(work_p1, [], ner, ("cpu", "cpu"))
    assert [m.text for m in results[("SET-A", "DOC-A")]] == ["asthma"]


def test_mine_two_passes_no_pass1_falls_back_to_single() -> None:
    """When Pass 1 has no work items, all GPUs go to Pass 2."""
    ner = DiseaseNER(gazetteer={"diabetes": "disease"})
    work_p2 = [("SET-B", "DOC-B", "diabetes")]
    results = _mine_two_passes_multi_gpu([], work_p2, ner, ("cpu", "cpu"))
    assert [m.text for m in results[("SET-B", "DOC-B")]] == ["diabetes"]


def test_build_rows_dispatches_two_passes_for_production_ner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Production NER + devices + both passes have work: build_contraindication_rows calls _mine_two_passes_multi_gpu."""
    sections = _mixed_sections(
        tmp_path,
        [
            ("SET-A", "SET-A#34070-3", CONTRA_LOINC, "asthma"),
            ("SET-B", "SET-B#34067-9", INDICATION_LOINC, "contraindicated in patients with diabetes"),
        ],
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X"), ("active", "SET-B", "DrugY", "UNII:Y")])
    ner = DiseaseNER(offline=False, gazetteer={"asthma": "disease", "diabetes": "disease"})

    called: list[dict[str, Any]] = []

    def fake_two_pass(w1, w2, ner_arg, devs):
        called.append({"p1": len(w1), "p2": len(w2), "devices": tuple(devs)})
        offline = DiseaseNER(gazetteer=ner_arg._gazetteer)
        return {(s, d): offline.extract(t) for s, d, t in w1 + w2}

    import dakp_pipeline.assertions.contraindications as contra_mod

    monkeypatch.setattr(contra_mod, "_mine_two_passes_multi_gpu", fake_two_pass)

    rows = build_contraindication_rows([sections, ingredients], ner, devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"))
    assert called == [{"p1": 1, "p2": 1, "devices": ("cuda:0", "cuda:1", "cuda:2", "cuda:3")}]
    assert {r["subject_text"] for r in rows} == {"DrugX", "DrugY"}


# --- gate-closure tests: keyword param resolution + Pass 2 ingredient guard ------


def test_resolve_keywords_passes_through_compiled_pattern(tmp_path: Path) -> None:
    pattern = re.compile(r"never use")
    assert _resolve_keywords(_ctx(tmp_path, {"contraindication_keywords": pattern})) is pattern


def test_resolve_keywords_compiles_string_param_case_insensitive(tmp_path: Path) -> None:
    pattern = _resolve_keywords(_ctx(tmp_path, {"contraindication_keywords": "never use"}))
    assert pattern.search("NEVER USE in pregnancy") is not None


def test_indication_set_without_active_ingredient_is_skipped_in_pass_2(tmp_path: Path) -> None:
    """Pass 2: an indication section whose set has no active ingredient is never mined."""
    sections = _mixed_sections(tmp_path, [("SET-B", "SET-B#34067-9", INDICATION_LOINC, "It is contraindicated in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-OTHER", "DrugY", "UNII:Y")])
    ner = DiseaseNER(gazetteer={"asthma": "disease"})

    assert build_contraindication_rows([sections, ingredients], ner) == []


def test_spawn_safe_main_swaps_script_main_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Script-style __main__ (no spec, e.g. the airflow CLI): spawn is redirected to a
    side-effect-free module and the original spec (None) is restored afterwards."""
    fake_main = SimpleNamespace(__spec__=None)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    with _spawn_safe_main():
        assert fake_main.__spec__ is not None
        assert fake_main.__spec__.name == "dakp_pipeline.logging_setup"
    assert fake_main.__spec__ is None


def test_spawn_safe_main_leaves_module_main_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module-style __main__ (has a spec): spawn already imports by name — nothing changes."""
    spec = SimpleNamespace(name="some_module")
    fake_main = SimpleNamespace(__spec__=spec)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    with _spawn_safe_main():
        assert fake_main.__spec__ is spec
    assert fake_main.__spec__ is spec


# --- ContraWorkItem legacy tuple protocol --------------------------------------


def test_contra_work_item_keeps_legacy_tuple_protocol() -> None:
    """Legacy workers/tests treat work items as ``(set_id, doc_id, text)`` 3-tuples, so
    indexing, ``len()``, and unpacking must keep behaving like the old tuple interface."""
    item = ContraWorkItem("SET-A", "DOC-A", "mined text", "full source text", ())
    assert item[0] == "SET-A"
    assert item[1] == "DOC-A"
    assert item[2] == "mined text"
    assert len(item) == 3
    assert list(item) == ["SET-A", "DOC-A", "mined text"]


# --- _sentence_spans: source-offset preservation --------------------------------


def test_sentence_spans_preserve_source_offsets() -> None:
    """Span offsets must index into the ORIGINAL text: leading whitespace is skipped, and a
    trailing fragment without a sentence boundary still becomes the final span."""
    spans = _sentence_spans("  First.  Second without boundary")
    assert [(s.start, s.end, s.text) for s in spans] == [(2, 8, "First."), (10, 33, "Second without boundary")]
    assert _sentence_spans("   ") == []  # whitespace-only text yields no spans


# --- legacy tuple work items (pre-ContraWorkItem interface) ---------------------


def test_legacy_tuple_item_classifies_by_section_default() -> None:
    """Legacy ``(set_id, doc_id, text)`` tuples predate ContraWorkItem and stand in for a
    dedicated 34070-3 section: evidence is the whole mined text, and the mention is accepted
    on section context even when the sentence carries no explicit trigger word."""
    text = "Avoid use in asthma."
    mention = Mention(text="asthma", start=13, end=19, type="disease", score=1.0)
    decision = _classify_mention(("SET-A", "DOC-A", text), mention)
    assert decision.accepted
    assert decision.trigger == "contraindication_section"
    assert decision.evidence_text == text


def test_work_item_evidence_resolves_spans_and_falls_back() -> None:
    """Evidence recovery walks the joined spans (skipping non-overlapping ones), returns the
    ORIGINAL sentence for the overlapping span, falls back to the full source text when no
    span contains the mention, and uses the whole text for legacy tuple items."""
    source = "First clean. Second with asthma. Trailing provenance."
    item = ContraWorkItem(
        "SET-A",
        "DOC-A",
        "First clean. Second with asthma.",
        source,
        (EvidenceSpan(0, 12, 0, 12, "First clean."), EvidenceSpan(13, 32, 40, 59, "Second with asthma.")),
    )
    # Mention inside the SECOND joined span: the first span is skipped without overlap.
    in_second = Mention(text="asthma", start=25, end=31, type="disease", score=1.0)
    assert _work_item_evidence(item, in_second) == "Second with asthma."
    # Mention past every span: falls back to the full source text.
    orphan = Mention(text="asthma", start=33, end=39, type="disease", score=1.0)
    assert _work_item_evidence(item, orphan) == source
    # Legacy tuple items carry no spans: the whole mined text is the evidence.
    assert _work_item_evidence(("SET-A", "DOC-A", " plain text "), in_second) == "plain text"


def test_mention_local_span_maps_overlap_and_returns_none_without_any() -> None:
    """A mapped mention yields ``(sentence, local start, local end, source start)`` offsets;
    a mention overlapping NO span maps to None so qualifier logic can skip it safely."""
    item = ContraWorkItem("SET-A", "DOC-A", "xxxx asthma", "src asthma text", (EvidenceSpan(5, 11, 20, 26, "asthma"),))
    overlapping = Mention(text="asthma", start=5, end=11, type="disease", score=1.0)
    assert _mention_local_span(item, overlapping) == ("asthma", 0, 6, 20)
    # Disjoint mention: the loop finds no overlapping span -> None.
    disjoint = Mention(text="xxxx", start=0, end=4, type="disease", score=1.0)
    assert _mention_local_span(item, disjoint) is None


def test_classify_mentions_keeps_unmapped_mentions_out_of_qualifier_grouping() -> None:
    """A mention that maps to no source span keeps its local decision but cannot join the
    per-sentence patient-clause grouping (there is no sentence to group on)."""
    spans = (EvidenceSpan(0, 7, 30, 37, "asthma."),)
    item = ContraWorkItem("SET-A", "DOC-A", "asthma. diabetes.", "asthma. diabetes.", spans)
    mapped = Mention(text="asthma", start=0, end=6, type="disease", score=1.0)
    unmapped = Mention(text="diabetes", start=8, end=16, type="disease", score=1.0)
    decisions = _classify_mentions(item, [mapped, unmapped])
    # Both keep their section-context acceptance; the unmapped one just skips grouping.
    assert [(d.accepted, d.trigger, d.context_text) for d in decisions] == [
        (True, "contraindication_section", ""),
        (True, "contraindication_section", ""),
    ]


# --- medication-context classification ------------------------------------------


def test_mention_after_medication_marker_is_not_medication_only_context(tmp_path: Path) -> None:
    """A disease named AFTER the medication marker (``concomitant ... asthma``) is the
    contraindicated condition itself — the medication_only_context rejection only applies to
    diseases that precede the marker, so this edge must survive."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "Contraindicated with concomitant use in asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    rows = build_contraindication_rows([sections, ingredients], DiseaseNER(gazetteer={"asthma": "disease"}))
    assert len(rows) == 1
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["disease_context_text"] == ""


def test_context_qualifier_withheld_when_sentence_has_medication_marker(tmp_path: Path) -> None:
    """A companion medication in the patient clause (``receiving warfarin``) disqualifies the
    sentence from contributing a disease_context_qualifier (the marker is present, so both
    diseases survive local classification first) — but the base asthma edge keeps its evidence."""
    sections = _sections(
        tmp_path, [("SET-A", "SET-A#d", "Contraindicated for treatment of hypertension in patients with asthma receiving warfarin.")]
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"hypertension": "disease", "asthma": "disease"})
    rows = build_contraindication_rows([sections, ingredients], ner)
    # hypertension becomes context_only_medication (rejected); only the asthma edge remains.
    assert len(rows) == 1
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["disease_context_text"] == ""
    assert rows[0]["evidence_text"] == "Contraindicated for treatment of hypertension in patients with asthma receiving warfarin."


# --- Pass 2: soft language alone is not a hard trigger ---------------------------


def test_indication_section_soft_language_alone_yields_no_edge(tmp_path: Path) -> None:
    """Pass 2 requires HARD prohibition language: ``avoid use in`` admits the sentence to the
    keyword filter, but without a hard trigger the mined disease is rejected (no_hard_trigger)
    and no contraindication edge is asserted."""
    sections = _mixed_sections(
        tmp_path, [("SET-A", "SET-A#34067-9", INDICATION_LOINC, "DrugX is indicated for hypertension. Avoid use in patients with asthma.")]
    )
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"asthma": "disease", "hypertension": "disease"})
    assert build_contraindication_rows([sections, ingredients], ner) == []


# --- context qualifier loop: intro / blank / non-disease / rejected-object guards -


def test_context_qualifier_requires_intro_phrase(tmp_path: Path) -> None:
    """A disease before the patient clause only becomes a context qualifier when an intro
    phrase (``for treatment of``, ...) marks it as the treated condition; without one, both
    diseases stay separate unconditional contraindication objects."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "Hypertension: contraindicated in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"hypertension": "disease", "asthma": "disease"})
    rows = build_contraindication_rows([sections, ingredients], ner)
    assert {(r["object_text"], r["disease_context_text"]) for r in rows} == {("hypertension", ""), ("asthma", "")}


def test_context_qualifier_rejected_when_mention_starts_inside_intro(tmp_path: Path) -> None:
    """If the candidate context mention starts INSIDE the intro phrase itself, the intro cannot
    vouch for it (``intro.end() > context_start``) -> no qualifier is assigned and both
    diseases remain separate objects."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "Contraindicated when used for hypertension in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"used for hypertension": "disease", "asthma": "disease"})
    rows = build_contraindication_rows([sections, ingredients], ner)
    assert {(r["object_text"], r["disease_context_text"]) for r in rows} == {("used for hypertension", ""), ("asthma", "")}


def test_context_qualifier_skipped_when_context_normalizes_blank() -> None:
    """A before-marker mention whose normalized text is empty (punctuation-only) cannot become
    a qualifier: the loop bails out and the object edge survives untouched."""
    sentence = "Contraindicated for treatment of --- in patients with asthma."
    spans = (EvidenceSpan(0, len(sentence), 0, len(sentence), sentence),)
    item = ContraWorkItem("SET-A", "DOC-A", sentence, sentence, spans)
    context = Mention(text="---", start=33, end=36, type="disease", score=1.0)
    asthma = Mention(text="asthma", start=54, end=60, type="disease", score=1.0)
    decisions = _classify_mentions(item, [context, asthma])
    assert [(d.accepted, d.trigger, d.context_text) for d in decisions] == [(True, "contraindicated", ""), (True, "contraindicated", "")]


def test_context_qualifier_rejects_non_disease_context(tmp_path: Path) -> None:
    """Biolink's disease_context_qualifier is disease-ranged: a phenotype (headache) in the
    context slot is rejected as qualifier material while the asthma edge stays unconditional."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "Contraindicated for treatment of headache in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"headache": "phenotype", "asthma": "disease"})
    rows = build_contraindication_rows([sections, ingredients], ner)
    assert len(rows) == 1
    assert rows[0]["object_text"] == "asthma"
    assert rows[0]["disease_context_text"] == ""


def test_context_not_attached_to_rejected_object(tmp_path: Path) -> None:
    """When the after-marker object mention itself is rejected (soft-safety-only language in a
    34070-3 section), the context mention is still withheld as context_only — but no qualifier
    upgrade is attempted on the rejected edge, and nothing is mined."""
    sections = _sections(tmp_path, [("SET-A", "SET-A#d", "Avoid for treatment of hypertension in patients with asthma.")])
    ingredients = _ingredients(tmp_path, [("active", "SET-A", "DrugX", "UNII:X")])
    ner = DiseaseNER(gazetteer={"hypertension": "disease", "asthma": "disease"})
    assert build_contraindication_rows([sections, ingredients], ner) == []


# --- _accumulate: blank evidence is not unioned ----------------------------------


def test_accumulate_skips_blank_evidence_text() -> None:
    """Blank evidence must not enter the evidence union — the sorted-pipe evidence column may
    only contain real sentences, while support/scores still accumulate."""
    aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
    mention = Mention(text="asthma", start=0, end=6, type="disease", score=0.9)
    _accumulate(aggregated, "SET-A", "DOC-A", "DrugX", "UNII:X", "asthma", mention, evidence_text="   ")
    agg = next(iter(aggregated.values()))
    assert agg["evidence_texts"] == []
    assert agg["sets"] == ["SET-A"]
    assert agg["scores"] == [0.9]


def test_accumulate_sanitizes_pipe_delimiters_in_label_prose() -> None:
    """Regression: mined label sentences legitimately contain ``|`` bullets and line breaks (real
    DailyMed warnings prose crashed ``shape_contraindication_tables`` when the pipe reached the
    sorted-pipe evidence encoder). Free-form text is sanitized, not rejected."""
    aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
    mention = Mention(text="asthma", start=0, end=6, type="disease", score=0.9)
    _accumulate(
        aggregated,
        "SET-A",
        "DOC-A",
        "DrugX",
        "UNII:X",
        "asthma",
        mention,
        evidence_text="Do not use■Prohibited use for ethanol allergy| When using this product\navoid open flames",
        context_text="patients with\tasthma|severe",
    )
    row = _finalize_row(next(iter(aggregated.values())))
    assert row["evidence_text"] == "Do not use■Prohibited use for ethanol allergy When using this product avoid open flames"
    assert row["disease_context_text"] == "patients with asthma severe"
    assert "|" not in row["evidence_text"]
