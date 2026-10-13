"""Unit tests for the gliner2 2.1.0 boundary-attribute CUDA device shim.

The shim (``dakp_pipeline.ner._gliner25_compat``) swaps in a device-corrected copy of
``BoundaryExtractor._attach_entity_attributes`` only when the live source shows the exact
buggy pairing (CPU-resident logits indexed with accelerator-device row indices). These tests
pin: the vendored copy carries the fix and nothing else, detection no-ops on already-fixed or
foreign sources, detection applies to a buggy source, and the vendored copy reproduces the
upstream multi-label attribute scoring behavior on CPU tensors.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dakp_pipeline.ner import _gliner25_compat as compat
from dakp_pipeline.ner._gliner25_compat import (
    _BUGGY_INDEX,
    _FIXED_INDEX,
    _LOGITS_TO_CPU,
    _fixed_attach_entity_attributes,
    apply_boundary_attribute_device_fix,
)


def _source_of_buggy_method(tmp_path: Path) -> types.ModuleType:
    """Build a file-backed fake engine module whose method matches the 2.1.0 buggy source."""
    source = (
        "import torch\n"
        "\n"
        "\n"
        "class BoundaryExtractor:\n"
        "    def _attach_entity_attributes(self):\n"
        "        logits = self.boundary_head()[0]\n"
        "        logits = (logits / self.boundary_settings.pair_temperature).float().cpu()\n"
        "        values = logits[\n"
        f"            {_BUGGY_INDEX}\n"
        "            column,\n"
        "        ]\n"
        "        return values\n"
    )
    return _module_from_source(tmp_path, source)


def _source_of_fixed_method(tmp_path: Path) -> types.ModuleType:
    source = (
        "import torch\n"
        "\n"
        "\n"
        "class BoundaryExtractor:\n"
        "    def _attach_entity_attributes(self):\n"
        "        logits = self.boundary_head()[0]\n"
        "        logits = (logits / self.boundary_settings.pair_temperature).float().cpu()\n"
        "        values = logits[\n"
        f"            {_FIXED_INDEX}\n"
        "            column,\n"
        "        ]\n"
        "        return values\n"
    )
    return _module_from_source(tmp_path, source)


def _module_from_source(tmp_path: Path, source: str) -> types.ModuleType:
    path = tmp_path / "engine_fake.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_gliner2(monkeypatch: pytest.MonkeyPatch, engine_module: Any) -> None:
    gliner2 = types.ModuleType("gliner2")
    gliner2.__path__ = []  # mark as a package for import machinery
    models = types.ModuleType("gliner2.models")
    models.__path__ = []  # type: ignore[attr-defined]
    boundary = types.ModuleType("gliner2.models.boundary")
    boundary.__path__ = []  # type: ignore[attr-defined]
    boundary.engine = engine_module  # type: ignore[attr-defined]
    gliner2.models = models  # type: ignore[attr-defined]
    models.boundary = boundary  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", gliner2)
    monkeypatch.setitem(sys.modules, "gliner2.models", models)
    monkeypatch.setitem(sys.modules, "gliner2.models.boundary", boundary)
    monkeypatch.setitem(sys.modules, "gliner2.models.boundary.engine", engine_module)


@pytest.fixture(autouse=True)
def _bind_torch() -> None:
    """The vendored method resolves `torch` from the shim module's globals (bind-light module);
    tests bind the real torch the same way apply() does."""
    import torch

    compat.torch = torch


def test_vendored_copy_differs_from_upstream_only_in_the_device_argument() -> None:
    source = inspect.getsource(_fixed_attach_entity_attributes)
    assert _FIXED_INDEX in source
    assert _BUGGY_INDEX not in source
    # The deliberate CPU move stays; only the index tensor's device changes.
    assert _LOGITS_TO_CPU in source


def test_apply_noops_when_boundary_engine_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_gliner2(monkeypatch, types.SimpleNamespace())  # engine module without the class
    assert apply_boundary_attribute_device_fix() is False


def test_apply_noops_on_already_fixed_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = _source_of_fixed_method(tmp_path)
    original = engine.BoundaryExtractor._attach_entity_attributes
    _install_fake_gliner2(monkeypatch, engine)
    assert apply_boundary_attribute_device_fix() is False
    assert engine.BoundaryExtractor._attach_entity_attributes is original


def test_apply_patches_buggy_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = _source_of_buggy_method(tmp_path)
    _install_fake_gliner2(monkeypatch, engine)
    assert apply_boundary_attribute_device_fix() is True
    assert engine.BoundaryExtractor._attach_entity_attributes is _fixed_attach_entity_attributes


def test_vendored_method_reproduces_multi_label_attribute_scoring() -> None:
    """CPU end-to-end over the vendored copy: two attribute rows, one retained span, both
    attribute labels above threshold -> the group is attached with per-label confidences."""
    import torch

    # OrderedDict matches the vendored signature (gliner2 passes a decode-ordered mapping).
    decoded: OrderedDict[str, list[dict[str, Any]]] = OrderedDict({"Disease": [{"text": "migraine"}]})
    item = decoded["Disease"][0]
    core = {
        # dim 1 spans all entity query fields (biolink:Disease + the two attribute prompt fields);
        # attribute decoding index_selects the prompt rows (ids 1 and 2) out of it.
        "query_states": torch.zeros(1, 3, 8),
        "query_mask": torch.ones(1, 3),
        "text_states": torch.zeros(1, 1, 8),
        "text_mask": torch.ones(1, 1),
    }
    specs = [
        {"field_name": "biolink:Disease", "task_type": "entities"},
        # qualify_labels=True (DAKP's setting) adds each attribute label to the entity schema
        # under "<group>: <label>"; those prompt fields are what attribute decoding scores.
        {"field_name": "assertion_context: indication", "task_type": "entities"},
        {"field_name": "assertion_context: contraindication", "task_type": "entities"},
    ]
    groups = {"assertion_context": SimpleNamespace(labels=["indication", "contraindication"], applies_to=None, multi_label=True, threshold=0.5)}
    metadata = {
        "entity_attribute_groups": groups,
        "entity_attribute_prompt_labels": {"indication": "assertion_context: indication", "contraindication": "assertion_context: contraindication"},
    }
    token_coordinates = {id(item): (0, 1)}

    class _StubHead:
        @staticmethod
        def score_explicit_spans(*_args: object) -> list[torch.Tensor]:
            # rows = attribute labels (indication, contraindication), cols = unique pairs.
            return [torch.tensor([[2.0], [1.0]])]

    class _StubEngine:
        boundary_head = _StubHead()
        boundary_settings = SimpleNamespace(pair_temperature=1.0)

    _fixed_attach_entity_attributes(_StubEngine(), 0, core, specs, metadata, decoded, token_coordinates)
    attached = item["assertion_context"]
    assert [(entry["label"], round(entry["confidence"], 2)) for entry in attached] == [
        ("indication", 0.88),
        ("contraindication", 0.73),
    ]  # PT018: single-comparison assertion over the ordered pair list
