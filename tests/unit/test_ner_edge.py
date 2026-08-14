"""Edge-case tests for ``dakp_pipeline.ner.ner`` (drive to 100% branch coverage).

Targets the production (GLiNER) path without network: a fake ``gliner``
module is injected into ``sys.modules`` and ``ensure_model`` is stubbed, so ``_load_model``'s
successful-import branch and ``_merge_model_spans`` (gazetteer-wins-on-overlap, type filtering,
GLiNER recall) are fully exercised. The missing-dep branch skips when ``gliner`` is installed.
"""

from __future__ import annotations

import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, ClassVar

import pytest

from dakp_pipeline.ner import ner as ner_module
from dakp_pipeline.ner.dictionary import Gazetteer
from dakp_pipeline.ner.model_cache import ModelRef
from dakp_pipeline.ner.ner import (
    _DEFAULT_WORD_BUDGET,
    _GLINER_TOKEN,
    DEFAULT_MODEL,
    DiseaseNER,
    Mention,
    _cuda_device_supported,
    _install_message,
    _model_device,
    _overlaps_any,
    _sort_key,
    _token_budget,
    _trim_hedges,
    _windows,
)

# --- helpers -------------------------------------------------------------------


def test_install_message_names_module_and_command() -> None:
    message = _install_message("gliner")
    assert "gliner" in message
    assert "uv sync" in message


# --- _model_device: CUDA when available, CPU fallbacks -------------------------


def test_model_device_selects_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_50", "sm_60", "sm_75"])
    assert _model_device() == "cuda"


def test_model_device_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _model_device() == "cpu"


def test_model_device_cpu_when_torch_unimportable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    assert _model_device() == "cpu"


def test_model_device_falls_back_to_cpu_when_arch_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA visible but torch has no kernels for the GPU arch (e.g. a cu128 build on a
    P100/sm_60) — CPU instead of crashing on the first CUDA call."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    assert _model_device() == "cpu"


# --- _cuda_device_supported: arch-list gate for one device ----------------------


def test_cuda_device_supported_matches_compiled_arch_list(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_60", "sm_75"])
    assert _cuda_device_supported(torch, 0) is True
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    assert _cuda_device_supported(torch, 0) is False


def test_cuda_device_supported_false_when_capability_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device that errors on capability query counts as unsupported, not a crash."""
    import torch

    def _raise(index: int) -> tuple[int, int]:
        raise RuntimeError("CUDA driver error")

    monkeypatch.setattr(torch.cuda, "get_device_capability", _raise)
    assert _cuda_device_supported(torch, 0) is False


def test_sort_key_and_overlaps_helpers() -> None:
    mention = Mention(text="asthma", start=1, end=7, type="disease", score=1.0)
    assert _sort_key(mention) == (1, 7, "disease", "asthma")
    assert _overlaps_any(0, 5, [(3, 8)]) is True
    assert _overlaps_any(0, 3, [(3, 8)]) is False  # touching, not overlapping
    assert _overlaps_any(8, 9, [(3, 8)]) is False
    assert _overlaps_any(0, 5, []) is False


def test_mention_is_frozen() -> None:
    mention = Mention(text="asthma", start=0, end=6, type="disease", score=1.0)
    with pytest.raises(FrozenInstanceError):
        mention.score = 0.5  # type: ignore[misc]


# --- _load_model: missing dep raises a clear install error ---------------------


def test_load_model_raises_clear_error_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # Block the gliner import (None in sys.modules raises ImportError) so the missing-dep
    # branch is exercised deterministically whether or not gliner is installed.
    monkeypatch.setitem(sys.modules, "gliner", None)
    backend = DiseaseNER(offline=False)
    with pytest.raises(ner_module.NERDependencyError, match=r"uv sync"):
        backend.extract("contraindicated in severe hepatic impairment")


# --- _load_model + _merge_model_spans: fake gliner (no network) ----------------


class _FakeGLiNERModel:
    def __init__(self, predictions: list[dict[str, Any]]) -> None:
        self._predictions = predictions
        self.calls: list[tuple[str, list[str], float]] = []

    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.0) -> list[dict[str, Any]]:
        self.calls.append((text, labels, threshold))
        return self._predictions


class _FakeGLiNER:
    loaded_from: ClassVar[list[str]] = []
    loaded_map_location: ClassVar[list[str]] = []
    model = _FakeGLiNERModel([])

    @staticmethod
    def from_pretrained(path: str, map_location: str = "cpu") -> _FakeGLiNERModel:
        _FakeGLiNER.loaded_from.append(path)
        _FakeGLiNER.loaded_map_location.append(map_location)
        return _FakeGLiNER.model


def _install_fake_gliner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, predictions: list[dict[str, Any]], model: Any = None) -> None:
    _FakeGLiNER.loaded_from = []
    _FakeGLiNER.loaded_map_location = []
    _FakeGLiNER.model = model if model is not None else _FakeGLiNERModel(predictions)
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)

    def _fake_ensure_model(model_id: str, **kwargs: Any) -> ModelRef:
        return ModelRef(model_id=model_id, source="huggingface", path=tmp_path, b3="b3:deadbeef", manifest=tmp_path / "manifest.json")

    monkeypatch.setattr(ner_module, "ensure_model", _fake_ensure_model)


def test_production_merge_gazetteer_wins_and_gliner_adds_recall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # "asthma" is in the gazetteer; "porphyria" is out-of-gazetteer (GLiNER recall);
    # "aspirin" is predicted CHEMICAL and must be filtered (not a disease/phenotype).
    predictions = [
        {"start": 0, "end": 6, "label": "DISEASE", "score": 0.9},  # overlaps gazetteer 'asthma' -> suppressed
        {"start": 7, "end": 16, "label": "DISEASE", "score": 0.8},  # 'porphyria' -> added (recall)
        {"start": 17, "end": 24, "label": "CHEMICAL", "score": 0.7},  # filtered by type
    ]
    _install_fake_gliner(monkeypatch, tmp_path, predictions)

    backend = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, model_id="acme/ner", threshold=0.42, cache_dir=tmp_path)
    mentions = backend.extract("asthma porphyria aspirin")

    assert [(m.text, m.type, m.notes) for m in mentions] == [("asthma", "disease", "exact"), ("porphyria", "disease", "gliner")]
    porphyria = mentions[1]
    assert porphyria.normalized == "porphyria"
    assert porphyria.score == 0.8
    # The model was loaded from the cached content path, with the contraindication labels + threshold.
    assert _FakeGLiNER.loaded_from == [str(tmp_path)]
    _text, labels, threshold = _FakeGLiNER.model.calls[0]
    assert labels == ["disease", "phenotype"]
    assert threshold == 0.42

    # A second extract reuses the cached model (from_pretrained called exactly once).
    backend.extract("asthma")
    assert len(_FakeGLiNER.loaded_from) == 1


def test_production_with_empty_gazetteer_is_model_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner(monkeypatch, tmp_path, [{"start": 0, "end": 9, "label": "phenotype", "score": 0.6}])
    backend = DiseaseNER(offline=False, gazetteer={})
    mentions = backend.extract("porphyria")
    assert [(m.text, m.type, m.notes) for m in mentions] == [("porphyria", "phenotype", "gliner")]


def test_load_model_returns_cached_model_without_reimport() -> None:
    sentinel = object()
    backend = DiseaseNER(offline=False)
    backend._model = sentinel  # pre-cache the model
    # With the model already cached, _load_model returns it without importing gliner.
    assert backend._load_model() is sentinel
    assert "gliner" not in sys.modules


def test_default_model_id_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeGLiNER.loaded_from = []
    _FakeGLiNER.model = _FakeGLiNERModel([])
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    seen: list[str] = []

    def _recording_ensure_model(model_id: str, **kwargs: Any) -> ModelRef:
        seen.append(model_id)
        return ModelRef(model_id=model_id, source="huggingface", path=tmp_path, b3="b3:x", manifest=tmp_path / "m.json")

    monkeypatch.setattr(ner_module, "ensure_model", _recording_ensure_model)
    DiseaseNER(offline=False).extract("asthma")
    assert seen == [DEFAULT_MODEL]


# --- population-descriptor filter: subject populations are not mentions ---------


def test_production_population_descriptor_spans_are_filtered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GLiNER loves to tag 'women of childbearing potential' as a phenotype; the blocklist drops it."""
    text = "Contraindicated in women of childbearing potential."
    start, end = text.index("women"), len(text) - 1  # 'women of childbearing potential'
    _install_fake_gliner(monkeypatch, tmp_path, [{"start": start, "end": end, "label": "phenotype", "score": 0.9}])
    backend = DiseaseNER(offline=False, gazetteer={})
    assert backend.extract(text) == []


def test_population_filter_only_drops_exact_population_phrases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner(monkeypatch, tmp_path, [{"start": 0, "end": 9, "label": "disease", "score": 0.9}])
    backend = DiseaseNER(offline=False, gazetteer={})
    assert [m.text for m in backend.extract("porphyria cases")] == ["porphyria"]


# --- specificity merge: the most specific span wins on containment --------------


def _span(text: str, phrase: str, label: str, score: float) -> dict[str, Any]:
    """A fake GLiNER prediction for ``phrase`` at its (unique) offsets in ``text``."""
    start = text.index(phrase)
    return {"start": start, "end": start + len(phrase), "label": label, "score": score}


def test_model_span_containing_a_gazetteer_span_supersedes_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """'pulmonary hypertension' beats the gazetteer's bare 'hypertension' — the regression this
    merge exists for. The model supplies the boundary; the gazetteer keeps the type (GLiNER is
    the documented source of disease<->phenotype confusion, see ner/BENCHMARK.md)."""
    text = "Contraindicated in patients with severe pulmonary hypertension."
    _install_fake_gliner(monkeypatch, tmp_path, [_span(text, "pulmonary hypertension", "phenotype", 0.87)])

    mentions = DiseaseNER(offline=False, gazetteer={"hypertension": "disease"}).extract(text)

    assert [(m.text, m.type, m.notes) for m in mentions] == [("pulmonary hypertension", "disease", "gliner:extends")]
    assert mentions[0].score == pytest.approx(0.87)
    assert text[mentions[0].start : mentions[0].end] == "pulmonary hypertension"
    assert mentions[0].normalized == "pulmonary hypertension"


def test_unconfident_specific_span_abstains_instead_of_emitting_the_generic_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Below the acceptance floor the backend returns NOTHING. It must not fall back to
    'hypertension' — that would assert a broader contraindication than the label supports."""
    text = "Contraindicated in patients with severe pulmonary hypertension."
    _install_fake_gliner(monkeypatch, tmp_path, [_span(text, "pulmonary hypertension", "disease", 0.42)])

    backend = DiseaseNER(offline=False, gazetteer={"hypertension": "disease"}, threshold=0.35, accept_threshold=0.5)
    assert backend.extract(text) == []


def test_plain_model_span_below_accept_floor_is_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A free-standing candidate generated above `threshold` but below `accept_threshold` is
    abstained on rather than asserted."""
    _install_fake_gliner(monkeypatch, tmp_path, [{"start": 0, "end": 9, "label": "disease", "score": 0.4}])
    backend = DiseaseNER(offline=False, gazetteer={}, threshold=0.35, accept_threshold=0.5)
    assert backend.extract("porphyria only") == []
    # ...and the same span clears a floor set below its score.
    _install_fake_gliner(monkeypatch, tmp_path, [{"start": 0, "end": 9, "label": "disease", "score": 0.4}])
    lenient = DiseaseNER(offline=False, gazetteer={}, threshold=0.35, accept_threshold=0.35)
    assert [m.text for m in lenient.extract("porphyria only")] == ["porphyria"]


@pytest.mark.parametrize(
    ("sentence", "phrase", "term"),
    [
        ("Contraindicated in patients with a recent myocardial infarction.", "a recent myocardial infarction", "myocardial infarction"),
        ("Contraindicated in patients with a history of peptic ulcer disease.", "a history of peptic ulcer disease", "peptic ulcer disease"),
        ("Do not use in patients with known hypersensitivity.", "known hypersensitivity", "hypersensitivity"),
    ],
)
def test_hedge_prefixes_are_trimmed_so_they_never_over_extend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sentence: str, phrase: str, term: str
) -> None:
    """The over-extension that made overlap-extension too risky to ship (ner/BENCHMARK.md):
    temporal/evidential hedges are trimmed, so the model span collapses onto the gazetteer term
    instead of superseding it with a longer, meaningless boundary."""
    _install_fake_gliner(monkeypatch, tmp_path, [_span(sentence, phrase, "disease", 0.95)])
    mentions = DiseaseNER(offline=False, gazetteer={term: "disease"}).extract(sentence)
    assert [(m.text, m.notes) for m in mentions] == [(term, "exact")]


def test_span_of_only_hedge_tokens_is_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = "Contraindicated in patients with asthma."
    _install_fake_gliner(monkeypatch, tmp_path, [_span(text, "patients with", "disease", 0.9)])
    assert DiseaseNER(offline=False, gazetteer={}).extract(text) == []


def test_population_descriptor_revealed_by_trimming_is_still_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """'in pregnant women' is not a population phrase until the leading 'in' comes off, so the
    filter has to run again on the trimmed surface."""
    text = "Contraindicated in pregnant women."
    _install_fake_gliner(monkeypatch, tmp_path, [_span(text, "in pregnant women", "phenotype", 0.9)])
    assert DiseaseNER(offline=False, gazetteer={}).extract(text) == []


def test_span_covering_several_gazetteer_terms_is_a_conjunction_not_a_qualifier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A model span swallowing two gazetteer terms is 'asthma or hypertension', not one more
    specific disease — both gazetteer spans stand and the model span is discarded."""
    text = "Contraindicated in asthma or hypertension."
    _install_fake_gliner(monkeypatch, tmp_path, [_span(text, "asthma or hypertension", "disease", 0.99)])
    mentions = DiseaseNER(offline=False, gazetteer={"asthma": "disease", "hypertension": "disease"}).extract(text)
    assert [(m.text, m.notes) for m in mentions] == [("asthma", "exact"), ("hypertension", "exact")]


def test_partial_overlap_still_goes_to_the_gazetteer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Neither span contains the other, so there is no specificity gain to bank — the
    high-precision gazetteer span wins, exactly as before this merge existed."""
    text = "congestive heart failure risk"
    _install_fake_gliner(monkeypatch, tmp_path, [_span(text, "failure risk", "phenotype", 0.99)])
    mentions = DiseaseNER(offline=False, gazetteer={"heart failure": "disease"}).extract(text)
    assert [(m.text, m.type, m.notes) for m in mentions] == [("heart failure", "disease", "exact")]


def test_overlapping_model_spans_resolve_longest_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Model-vs-model: the longer span wins even when the shorter one scores higher."""
    text = "severe pulmonary hypertension noted"
    predictions = [_span(text, "hypertension", "disease", 0.95), _span(text, "pulmonary hypertension", "disease", 0.6)]
    _install_fake_gliner(monkeypatch, tmp_path, predictions)
    mentions = DiseaseNER(offline=False, gazetteer={}).extract(text)
    assert [(m.text, m.notes) for m in mentions] == [("pulmonary hypertension", "gliner")]


def test_equal_length_overlapping_model_spans_break_ties_deterministically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = "alpha beta gamma"
    _install_fake_gliner(monkeypatch, tmp_path, [_span(text, "alpha beta", "disease", 0.6), _span(text, "beta gamma", "disease", 0.9)])
    assert [m.text for m in DiseaseNER(offline=False, gazetteer={}).extract(text)] == ["beta gamma"]  # equal length -> higher score

    # A full tie falls back to the leftmost span, so repeated runs agree.
    _install_fake_gliner(monkeypatch, tmp_path, [_span(text, "alpha beta", "disease", 0.9), _span(text, "beta gamma", "disease", 0.9)])
    assert [m.text for m in DiseaseNER(offline=False, gazetteer={}).extract(text)] == ["alpha beta"]


def test_trim_hedges_returns_none_for_a_tokenless_span() -> None:
    """A span that is pure whitespace has no GLiNER tokens to walk — dropped, not crashed."""
    assert _trim_hedges("a   b", 1, 4) is None
    assert _trim_hedges("recent asthma", 0, 13) == (7, 13)
    assert _trim_hedges("asthma flare", 0, 12) == (0, 12)  # no leading hedge -> untouched


def test_offline_mode_keeps_its_gazetteer_granularity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Deliberate: offline mode has no model, so it still returns the generic gazetteer term.
    Specificity is a production-mode capability; offline stays the deterministic baseline."""
    text = "Contraindicated in patients with severe pulmonary hypertension."
    mentions = DiseaseNER(offline=True, gazetteer={"hypertension": "disease"}).extract(text)
    assert [(m.text, m.notes) for m in mentions] == [("hypertension", "exact")]


# --- cross-window rejoin: hard splits can cut a multiword mention in two --------


_SPLIT_FILLER = " ".join(f"word{i}" for i in range(39))  # puts 'myasthenia' flush at a hard-split boundary
_SPLIT_PROBE = f"{_SPLIT_FILLER} myasthenia gravis must be excluded"


class _WindowRoutedFakeModel:
    """Fake GLiNER whose predictions depend on which window it is given (window-relative
    offsets, like the real model). The first route whose key occurs in the window wins."""

    def __init__(self, routes: dict[str, list[dict[str, Any]]]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.0) -> list[dict[str, Any]]:
        self.calls.append(text)
        for key, predictions in self.routes.items():
            if key in text:
                return predictions
        return []


def _split_probe_span(window: str, phrase: str) -> tuple[int, int]:
    start = window.index(phrase)
    return start, start + len(phrase)


def test_production_straddling_spans_rejoin_across_hard_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """'myasthenia | gravis' cut by a hard split comes back as ONE 'myasthenia gravis' span."""
    windows = _windows(_SPLIT_PROBE, 20)
    assert len(windows) == 3
    assert windows[2][0] > windows[1][0] + len(windows[1][1])  # hard-split gap
    left_start, left_end = _split_probe_span(windows[1][1], "myasthenia")
    assert left_end == len(windows[1][1])  # flush at the window end
    right_start, right_end = _split_probe_span(windows[2][1], "gravis")
    assert right_start == 0  # flush at the next window start

    model = _WindowRoutedFakeModel(
        {
            "myasthenia": [{"start": left_start, "end": left_end, "label": "disease", "score": 0.9}],
            "gravis": [{"start": right_start, "end": right_end, "label": "phenotype", "score": 0.6}],
        }
    )
    _install_fake_gliner(monkeypatch, tmp_path, [], model=model)

    mentions = DiseaseNER(offline=False, gazetteer={}, chunk_words=20).extract(_SPLIT_PROBE)
    assert [(m.text, m.type) for m in mentions] == [("myasthenia gravis", "disease")]  # higher-scoring side's type
    assert mentions[0].score == pytest.approx(0.9)
    assert _SPLIT_PROBE[mentions[0].start : mentions[0].end] == "myasthenia gravis"


def test_production_straddling_rejoin_takes_higher_scoring_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    windows = _windows(_SPLIT_PROBE, 20)
    left_start, left_end = _split_probe_span(windows[1][1], "myasthenia")
    right_start, right_end = _split_probe_span(windows[2][1], "gravis")
    model = _WindowRoutedFakeModel(
        {
            "myasthenia": [{"start": left_start, "end": left_end, "label": "disease", "score": 0.6}],
            "gravis": [{"start": right_start, "end": right_end, "label": "phenotype", "score": 0.95}],
        }
    )
    _install_fake_gliner(monkeypatch, tmp_path, [], model=model)
    mentions = DiseaseNER(offline=False, gazetteer={}, chunk_words=20).extract(_SPLIT_PROBE)
    assert [(m.text, m.type) for m in mentions] == [("myasthenia gravis", "phenotype")]
    assert mentions[0].score == pytest.approx(0.95)


def test_production_straddling_rejoin_tie_keeps_left_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    windows = _windows(_SPLIT_PROBE, 20)
    left_start, left_end = _split_probe_span(windows[1][1], "myasthenia")
    right_start, right_end = _split_probe_span(windows[2][1], "gravis")
    model = _WindowRoutedFakeModel(
        {
            "myasthenia": [{"start": left_start, "end": left_end, "label": "disease", "score": 0.8}],
            "gravis": [{"start": right_start, "end": right_end, "label": "phenotype", "score": 0.8}],
        }
    )
    _install_fake_gliner(monkeypatch, tmp_path, [], model=model)
    mentions = DiseaseNER(offline=False, gazetteer={}, chunk_words=20).extract(_SPLIT_PROBE)
    assert [(m.text, m.type) for m in mentions] == [("myasthenia gravis", "disease")]


def test_production_single_edge_span_is_kept_without_merge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only one side of a hard-split boundary predicted -> no merge, the span survives as-is."""
    windows = _windows(_SPLIT_PROBE, 20)
    left_start, left_end = _split_probe_span(windows[1][1], "myasthenia")
    right_start, right_end = _split_probe_span(windows[2][1], "gravis")

    left_only = _WindowRoutedFakeModel({"myasthenia": [{"start": left_start, "end": left_end, "label": "disease", "score": 0.9}]})
    _install_fake_gliner(monkeypatch, tmp_path, [], model=left_only)
    mentions = DiseaseNER(offline=False, gazetteer={}, chunk_words=20).extract(_SPLIT_PROBE)
    assert [(m.text, m.type) for m in mentions] == [("myasthenia", "disease")]

    right_only = _WindowRoutedFakeModel({"gravis": [{"start": right_start, "end": right_end, "label": "phenotype", "score": 0.6}]})
    _install_fake_gliner(monkeypatch, tmp_path, [], model=right_only)
    mentions = DiseaseNER(offline=False, gazetteer={}, chunk_words=20).extract(_SPLIT_PROBE)
    assert [(m.text, m.type) for m in mentions] == [("gravis", "phenotype")]


def test_production_no_merge_across_contiguous_sentence_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sentence-piece windows tile gap-free, so flush spans on both sides stay separate mentions."""
    text = "Asthma attack. Porphyria noted."
    windows = _windows(text, 3)
    assert len(windows) == 2
    assert windows[1][0] == windows[0][0] + len(windows[0][1])  # contiguous boundary
    model = _WindowRoutedFakeModel(
        {
            "Asthma": [{"start": 0, "end": len(windows[0][1]), "label": "disease", "score": 0.9}],
            "Porphyria": [{"start": 0, "end": 9, "label": "disease", "score": 0.9}],
        }
    )
    _install_fake_gliner(monkeypatch, tmp_path, [], model=model)
    mentions = DiseaseNER(offline=False, gazetteer={}, chunk_words=3).extract(text)
    assert [(m.text, m.type) for m in mentions] == [("Asthma attack. ", "disease"), ("Porphyria", "disease")]


# --- windowing: GLiNER truncates long inputs, so long texts are predicted per window ---


class _WindowAwareFakeModel:
    """Fake GLiNER that predicts every occurrence of ``needle`` inside the window it is given
    (offsets window-relative, exactly like the real model)."""

    def __init__(self, needle: str, label: str = "disease", score: float = 0.9) -> None:
        self.needle = needle
        self.label = label
        self.score = score
        self.calls: list[str] = []

    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.0) -> list[dict[str, Any]]:
        self.calls.append(text)
        entities: list[dict[str, Any]] = []
        index = 0
        while (found := text.find(self.needle, index)) != -1:
            entities.append({"start": found, "end": found + len(self.needle), "label": self.label, "score": self.score})
            index = found + len(self.needle)
        return entities


_LONG_FILLER = "The label states the drug is contraindicated under the circumstances described here. " * 10


def test_production_long_text_is_windowed_and_offsets_remap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = _LONG_FILLER + "Patients with porphyria must not receive it."
    model = _WindowAwareFakeModel("porphyria")
    _install_fake_gliner(monkeypatch, tmp_path, [], model=model)

    backend = DiseaseNER(offline=False, gazetteer={}, chunk_words=25)
    mentions = backend.extract(text)

    # Windowed, not one truncated call; windows tile the section exactly and each fits the budget.
    assert len(model.calls) > 1
    assert "".join(model.calls) == text
    assert all(len(_GLINER_TOKEN.findall(call)) <= 25 for call in model.calls)

    # The late mention (past the first window's 384-token-style truncation point) is found and
    # its offsets remap into full-text coordinates.
    assert [(m.text, m.type, m.notes) for m in mentions] == [("porphyria", "disease", "gliner")]
    mention = mentions[0]
    assert text[mention.start : mention.end] == "porphyria"
    assert mention.start > len(model.calls[0])
    assert mention.normalized == "porphyria"
    assert mention.score == 0.9


def test_production_short_text_is_a_single_full_text_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = _WindowAwareFakeModel("porphyria")
    _install_fake_gliner(monkeypatch, tmp_path, [], model=model)
    text = "porphyria only"
    DiseaseNER(offline=False, gazetteer={}).extract(text)
    assert model.calls == [text]  # nothing ≤ budget changes vs the pre-windowing behavior


def test_production_gazetteer_wins_in_later_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = _LONG_FILLER + "Patients with asthma must not receive it."
    model = _WindowAwareFakeModel("asthma")
    _install_fake_gliner(monkeypatch, tmp_path, [], model=model)

    backend = DiseaseNER(offline=False, gazetteer={"asthma": "disease"}, chunk_words=25)
    mentions = backend.extract(text)

    assert len(model.calls) > 1
    # Gazetteer span wins on overlap even when the GLiNER span comes from a later window.
    assert [(m.text, m.type, m.notes) for m in mentions] == [("asthma", "disease", "exact")]


def test_production_oversize_single_sentence_is_hard_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    filler = " ".join(f"word{i}" for i in range(60))  # one sentence, no terminal punctuation
    text = filler + " porphyria"
    model = _WindowAwareFakeModel("porphyria")
    _install_fake_gliner(monkeypatch, tmp_path, [], model=model)

    backend = DiseaseNER(offline=False, gazetteer={}, chunk_words=20)
    mentions = backend.extract(text)

    assert len(model.calls) > 1
    assert model.calls == [window for _, window in _windows(text, 20)]
    assert all(len(_GLINER_TOKEN.findall(call)) <= 20 for call in model.calls)
    assert [(m.text, m.notes) for m in mentions] == [("porphyria", "gliner")]
    assert text[mentions[0].start : mentions[0].end] == "porphyria"


# --- _windows: exact-substring tiling, budgets, fallbacks -------------------------------


def test_windows_single_window_when_text_fits() -> None:
    text = "Contraindicated in asthma and headache."
    assert _windows(text, _DEFAULT_WORD_BUDGET) == [(0, text)]


def test_windows_blank_text_yields_nothing() -> None:
    assert _windows("", 384) == []
    assert _windows("   \t\n ", 384) == []


def test_windows_tile_the_text_exactly_and_fit_the_budget() -> None:
    text = "Asthma attack. " * 40 + "Porphyria noted. " * 40
    windows = _windows(text, 25)
    assert len(windows) > 1
    assert "".join(window for _, window in windows) == text
    offset = 0
    for start, window in windows:
        assert start == offset
        assert text[start : start + len(window)] == window
        assert len(_GLINER_TOKEN.findall(window)) <= 25
        offset += len(window)


def test_windows_hard_split_oversize_sentence() -> None:
    text = " ".join(f"word{i}" for i in range(100))  # one sentence, no terminal punctuation
    windows = _windows(text, 30)
    assert len(windows) == 4  # 100 tokens -> 30 + 30 + 30 + 10
    for start, window in windows:
        assert text[start : start + len(window)] == window
        assert len(_GLINER_TOKEN.findall(window)) <= 30
    assert all(f"word{i}" in "".join(window for _, window in windows) for i in range(100))


def test_windows_counts_gliner_tokens_not_whitespace_words() -> None:
    # 3 whitespace words but 6 GLiNER tokens (each punctuation glyph is its own token), so a
    # budget of 4 must still split — GLiNER's own truncation counts these tokens.
    assert len(_windows("x, y; z.", 4)) == 2


def test_windows_falls_back_to_single_piece_for_untileable_text() -> None:
    # Leading punctuation, dangling terminals and gaps break the sentence tiling; the whole text
    # becomes one piece and still yields one (windowed) exact substring when it fits the budget.
    for text in (".", "!abc.", "a. .", "a. .b"):
        assert _windows(text, _DEFAULT_WORD_BUDGET) == [(0, text)]


# --- _token_budget: override > model config > default ------------------------------------


def test_token_budget_override_wins_and_clamps() -> None:
    assert _token_budget(None, 25) == 25
    assert _token_budget(None, 1) == 1
    assert _token_budget(None, 0) == 1
    assert _token_budget(None, -5) == 1


def test_token_budget_reads_model_config_max_len() -> None:
    class _Model:
        class config:
            max_len = 512

    assert _token_budget(_Model(), None) == 512


def test_token_budget_falls_back_to_default() -> None:
    assert _token_budget(object(), None) == _DEFAULT_WORD_BUDGET  # no .config attribute

    class _NoMaxLen:
        config = None

    assert _token_budget(_NoMaxLen(), None) == _DEFAULT_WORD_BUDGET

    class _NonInt:
        class config:
            max_len = "nope"

    assert _token_budget(_NonInt(), None) == _DEFAULT_WORD_BUDGET

    class _TooSmall:
        class config:
            max_len = 0

    assert _token_budget(_TooSmall(), None) == _DEFAULT_WORD_BUDGET


# --- device parameter: pin GLiNER to a specific GPU for multi-GPU dispatch ---------


def test_device_param_pins_model_to_specified_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An explicit ``device`` kwarg flows through to ``map_location`` at model load time."""
    _install_fake_gliner(monkeypatch, tmp_path, [])
    backend = DiseaseNER(offline=False, device="cuda:2", cache_dir=tmp_path)
    backend.extract("some text")
    assert _FakeGLiNER.loaded_map_location == ["cuda:2"]


def test_device_none_falls_back_to_model_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When ``device`` is None, ``_load_model`` uses ``_model_device()`` (auto-detect)."""
    _install_fake_gliner(monkeypatch, tmp_path, [])
    backend = DiseaseNER(offline=False, cache_dir=tmp_path)  # device=None
    backend.extract("some text")
    # _model_device() returns "cpu" in CI (no CUDA) or "cuda" when available.
    assert _FakeGLiNER.loaded_map_location[0] in ("cpu", "cuda")


# --- _config: serializable construction kwargs for multi-process workers ------------


def test_config_returns_serializable_construction_kwargs(tmp_path: Path) -> None:
    gaz = Gazetteer({"asthma": "disease"})
    backend = DiseaseNER(
        offline=False,
        gazetteer=gaz,
        model_id="acme/ner",
        threshold=0.42,
        accept_threshold=0.66,
        chunk_words=128,
        cache_dir=tmp_path,
        workdir=tmp_path / "work",
        device="cuda:1",
    )
    config = backend._config()
    assert config["offline"] is False
    assert config["model_id"] == "acme/ner"
    assert config["threshold"] == 0.42
    # Must round-trip: the multi-GPU workers rebuild from this dict, and a missing key would
    # silently run all four P100s at the default floor while the parent ran tuned.
    assert config["accept_threshold"] == 0.66
    assert config["chunk_words"] == 128
    assert config["cache_dir"] == tmp_path
    assert config["workdir"] == tmp_path / "work"
    assert config["gazetteer"] is gaz
    # device is deliberately excluded — the caller sets it per-worker.
    assert "device" not in config


def test_config_can_reconstruct_equivalent_backend(tmp_path: Path) -> None:
    """A DiseaseNER built from ``_config()`` + a ``device`` produces the same mentions."""
    original = DiseaseNER(offline=True, gazetteer={"asthma": "disease"}, device="cuda:3")
    reconstructed = DiseaseNER(device="cuda:0", **original._config())
    text = "patient has asthma"
    assert [m.text for m in reconstructed.extract(text)] == [m.text for m in original.extract(text)]
    assert reconstructed._offline == original._offline
