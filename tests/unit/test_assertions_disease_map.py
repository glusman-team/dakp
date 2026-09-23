"""Direct unit tests for :func:`dakp_pipeline.assertions.match_diseases`.

The shapers exercise `match_diseases` only through the lexical baseline join, so the
contract edges (empty keys, case folding, non-matching keys) live here: the empty-key
skip and the mixed-case `key.lower()` branch are distinct code paths the coverage gate
must see exercised.
"""

from __future__ import annotations

from dakp_pipeline.assertions import match_diseases


def test_empty_key_is_skipped() -> None:
    """A disease-map entry with an empty key can never match (and must not raise or
    match everything, since "" is a substring of every text)."""
    disease_map = {
        "": {"curie": "MONDO:0000001", "name": "Empty", "category": "Disease"},
        "asthma": {"curie": "MONDO:0004976", "name": "asthma", "category": "Disease"},
    }
    matches = match_diseases("Chronic asthma noted", disease_map)
    assert [match["text"] for match in matches] == ["asthma"]


def test_mixed_case_key_matches_case_insensitively() -> None:
    """Mixed-case keys are lowered before the substring test, so they still match."""
    disease_map = {"Pulmonary Hypertension": {"curie": "HP:0002092", "name": "Pulmonary hypertension", "category": "PhenotypicFeature"}}
    matches = match_diseases("History of pulmonary hypertension", disease_map)
    assert matches == [{"text": "Pulmonary Hypertension", "curie": "HP:0002092", "name": "Pulmonary hypertension", "category": "PhenotypicFeature"}]


def test_non_matching_key_yields_no_matches() -> None:
    assert match_diseases("Headache and nausea", {"asthma": {"curie": "MONDO:0004976", "name": "asthma", "category": "Disease"}}) == []


def test_empty_text_never_matches() -> None:
    assert match_diseases("", {"asthma": {"curie": "MONDO:0004976", "name": "asthma", "category": "Disease"}}) == []
