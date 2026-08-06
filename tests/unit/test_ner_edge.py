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
from dakp_pipeline.ner.model_cache import ModelRef
from dakp_pipeline.ner.ner import (
    _DEFAULT_WORD_BUDGET,
    _GLINER_TOKEN,
    DEFAULT_MODEL,
    DiseaseNER,
    Mention,
    _install_message,
    _model_device,
    _overlaps_any,
    _sort_key,
    _token_budget,
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
    assert _model_device() == "cuda"


def test_model_device_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _model_device() == "cpu"


def test_model_device_cpu_when_torch_unimportable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    assert _model_device() == "cpu"


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
