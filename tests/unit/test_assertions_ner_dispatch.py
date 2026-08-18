"""Unit tests for the shared NER dispatch plumbing (assertions/ner_dispatch.py).

Covers the generalized multi-pass GPU dispatch (mine_passes_multi_gpu / _group_devices)
beyond what the contraindication shaper's historical two-pass tests exercise. All tests
run offline with gazetteer-only DiseaseNERs on "cpu" devices (the spawn pool is real).
"""

from __future__ import annotations

from dakp_pipeline.assertions.ner_dispatch import _group_devices, default_ner, mine_passes_multi_gpu
from dakp_pipeline.ner.ner import DiseaseNER


def _ner(*terms: str) -> DiseaseNER:
    return DiseaseNER(gazetteer=dict.fromkeys(terms, "disease"))


# --- mine_passes_multi_gpu ------------------------------------------------------


def test_mine_passes_multi_gpu_all_passes_empty() -> None:
    """No work in any pass: empty result map, no pool dispatched."""
    assert mine_passes_multi_gpu([[], []], _ner("asthma"), ("cpu", "cpu")) == {}


def test_mine_passes_multi_gpu_single_nonempty_pass_uses_multi_gpu() -> None:
    """One nonempty pass: all devices go to it (the _mine_multi_gpu fallback)."""
    results = mine_passes_multi_gpu([[], [("SET-B", "DOC-B", "diabetes")]], _ner("diabetes"), ("cpu", "cpu"))
    assert [m.text for m in results[("SET-B", "DOC-B")]] == ["diabetes"]


def test_mine_passes_multi_gpu_splits_devices_across_three_passes() -> None:
    """Three passes share the device list contiguously; every pass's mentions are collected."""
    passes = [[("SET-A", "DOC-A", "asthma")], [("SET-B", "DOC-B", "diabetes")], [("SET-C", "DOC-C", "epilepsy")]]
    results = mine_passes_multi_gpu(passes, _ner("asthma", "diabetes", "epilepsy"), ("cpu", "cpu", "cpu"))
    assert set(results.keys()) == {("SET-A", "DOC-A"), ("SET-B", "DOC-B"), ("SET-C", "DOC-C")}
    assert [m.text for m in results[("SET-C", "DOC-C")]] == ["epilepsy"]


def test_mine_passes_multi_gpu_one_device_two_passes_shares_it() -> None:
    """Fewer devices than passes: the empty group falls back to the first device."""
    passes = [[("SET-A", "DOC-A", "asthma")], [("SET-B", "DOC-B", "diabetes")]]
    results = mine_passes_multi_gpu(passes, _ner("asthma", "diabetes"), ("cpu",))
    assert [m.text for m in results[("SET-A", "DOC-A")]] == ["asthma"]
    assert [m.text for m in results[("SET-B", "DOC-B")]] == ["diabetes"]


# --- _group_devices --------------------------------------------------------------


def test_group_devices_even_split() -> None:
    assert _group_devices(["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 2) == [["cuda:0", "cuda:1"], ["cuda:2", "cuda:3"]]


def test_group_devices_remainder_goes_to_earlier_groups() -> None:
    assert _group_devices(["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 3) == [["cuda:0", "cuda:1"], ["cuda:2"], ["cuda:3"]]


def test_group_devices_never_empty() -> None:
    """k > len(devices): groups that would be empty defensively take the first device."""
    assert _group_devices(["cuda:0"], 3) == [["cuda:0"], ["cuda:0"], ["cuda:0"]]


# --- default_ner -----------------------------------------------------------------


def test_default_ner_without_fixture_root_uses_embedded_gazetteer() -> None:
    ner = default_ner(None)
    assert ner._offline
    assert [m.text for m in ner.extract("patients with asthma")] == ["asthma"]
