"""Edge-case tests for the ``dakp_pipeline.translator.rig`` dependency-free YAML serializer.

The committed RIG content never contains bools, numbers, ``null``, or empty containers, so the
serializer's scalar/container edge branches are exercised here with a synthetic mapping passed
to :func:`rig_yaml` (which accepts an arbitrary rig mapping).
"""

from __future__ import annotations

from typing import Any

import pytest

from dakp_pipeline.translator.rig import generate_rig, rig_yaml


def test_rig_yaml_serializer_scalar_and_container_edges() -> None:
    custom: dict[str, Any] = {
        "bool_true": True,
        "bool_false": False,
        "an_int": 42,
        "a_float": 2.5,
        "a_none": None,
        "empty_dict": {},
        "empty_list": [],
        "items": [{"first_empty": {}, "scalar": "x"}, {"first_scalar": "y", "later_empty": []}, "plain scalar item"],
    }
    text = rig_yaml(custom)

    assert "bool_true: true" in text
    assert "bool_false: false" in text
    assert "an_int: 42" in text
    assert "a_float: 2.5" in text
    assert "a_none: null" in text
    assert "empty_dict: {}" in text
    assert "empty_list: []" in text
    # sequence item whose FIRST value is an empty container -> inline after the dash
    assert "- first_empty: {}" in text
    assert '  scalar: "x"' in text
    # sequence item with a scalar first value and a later empty-container key
    assert '- first_scalar: "y"' in text
    assert "  later_empty: []" in text
    # a bare scalar sequence item
    assert '- "plain scalar item"' in text


def test_rig_yaml_custom_round_trips_when_yaml_available() -> None:
    yaml = pytest.importorskip("yaml")
    custom: dict[str, Any] = {"flag": True, "count": 3, "nothing": None, "empty": {}, "list": []}
    assert yaml.safe_load(rig_yaml(custom)) == custom


def test_rig_yaml_default_still_matches_generated_rig() -> None:
    assert rig_yaml() == rig_yaml(generate_rig())
