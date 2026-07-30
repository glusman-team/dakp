"""Edge-case tests for the ``dakp_pipeline.tablassert.configs`` YAML emitter.

The generated Graph/table configs contain no bools, numbers, or empty strings that wrap to
nothing, so the emitter's scalar branches (``_yaml_scalar`` bool/int/float) and the empty-input
branch of ``_wrap`` are exercised directly here.
"""

from __future__ import annotations

from dakp_pipeline.tablassert import configs


def test_yaml_scalar_emits_bool_int_and_float() -> None:
    text = configs._dump_yaml({"flag_t": True, "flag_f": False, "count": 3, "ratio": 2.5})
    assert "flag_t: true" in text
    assert "flag_f: false" in text
    assert "count: 3" in text
    assert "ratio: 2.5" in text


def test_wrap_empty_string_yields_no_lines() -> None:
    # Empty input -> `current` stays falsy -> the trailing append is skipped.
    assert configs._wrap("") == []
