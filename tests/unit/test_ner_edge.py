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
from dakp_pipeline.ner.ner import DEFAULT_MODEL, DiseaseNER, Mention, _install_message, _model_device, _overlaps_any, _sort_key

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


def _install_fake_gliner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, predictions: list[dict[str, Any]]) -> None:
    _FakeGLiNER.loaded_from = []
    _FakeGLiNER.loaded_map_location = []
    _FakeGLiNER.model = _FakeGLiNERModel(predictions)
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
