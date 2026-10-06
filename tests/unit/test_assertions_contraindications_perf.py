"""Equivalence tests for the contraindication aggregate performance path.

The aggregate tail of :mod:`dakp_pipeline.assertions.contraindications` localizes every mined
mention against the work item's evidence spans. The performance rewrite must keep the exact
first-overlapping-span semantics, per-sentence qualifier grouping, and row output. These tests
pin: first-overlap precedence with overlapping spans; duplicate/equal mention values mapping
identically through the shared per-item memo; unknown offsets falling back (and memoizing the
miss); adjacent split spans mapping into their own source sentence; and end-to-end row output
with duplicate mentions plus per-sentence qualifier attachment.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.assertions.contraindications import (
    ContraWorkItem,
    EvidenceSpan,
    _mention_local_span,
    _work_item_evidence,
    build_contraindication_rows,
)
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.ner.ner import DiseaseNER, Mention

CONTRA_LOINC = "34070-3"


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


# --- first-overlap precedence ----------------------------------------------------


def test_first_overlapping_span_wins_for_overlapping_spans() -> None:
    mention = Mention("asthma", 10, 16, "Disease", 1.0)
    # Both spans overlap the mention; the FIRST one (source offset 100) must win, exactly as
    # the original linear scan did.
    spans = (EvidenceSpan(8, 21, 100, 113, "severe asthma"), EvidenceSpan(12, 24, 200, 212, "asthma attack"))
    item = ContraWorkItem("SET", "DOC", "mined text", "source text", spans)
    assert _mention_local_span(item, mention) == ("severe asthma", 2, 8, 100)
    assert _work_item_evidence(item, mention) == "severe asthma"
    memo: dict[Mention, EvidenceSpan | None] = {}
    assert _mention_local_span(item, mention, memo) == ("severe asthma", 2, 8, 100)
    assert _work_item_evidence(item, mention, memo) == "severe asthma"


# --- duplicate / equal mention values --------------------------------------------


def test_equal_mention_values_share_one_span_answer() -> None:
    item = ContraWorkItem("SET", "DOC", "asthma risk noted", "asthma risk noted", (EvidenceSpan(0, 17, 0, 17, "asthma risk noted"),))
    first = Mention("asthma", 0, 6, "Disease", 1.0)
    duplicate = Mention("asthma", 0, 6, "Disease", 1.0)  # equal value, distinct instance
    memo: dict[Mention, EvidenceSpan | None] = {}
    assert _mention_local_span(item, first, memo) == _mention_local_span(item, duplicate, memo)
    assert _mention_local_span(item, duplicate, memo) == _mention_local_span(item, duplicate)
    assert _work_item_evidence(item, duplicate, memo) == _work_item_evidence(item, first)


def test_mention_outside_all_spans_falls_back_and_memoizes_the_miss() -> None:
    item = ContraWorkItem("SET", "DOC", "joined text", "  source text  ", (EvidenceSpan(0, 11, 5, 16, "joined text"),))
    stray = Mention("rash", 40, 44, "Disease", 0.7)
    assert _mention_local_span(item, stray) is None
    assert _work_item_evidence(item, stray) == "source text"
    memo: dict[Mention, EvidenceSpan | None] = {}
    assert _mention_local_span(item, stray, memo) is None
    assert _work_item_evidence(item, stray, memo) == "source text"
    assert memo[stray] is None  # misses are memoized too: no repeated scans


# --- adjacent split spans ---------------------------------------------------------


def test_adjacent_split_spans_map_into_their_own_sentence() -> None:
    text = "It is contraindicated in asthma. Avoid in hypotension."
    first_text = "It is contraindicated in asthma."
    second_text = "Avoid in hypotension."
    second_start = text.index(second_text)
    spans = (
        EvidenceSpan(0, len(first_text), 0, len(first_text), first_text),
        EvidenceSpan(second_start, second_start + len(second_text), second_start, second_start + len(second_text), second_text),
    )
    item = ContraWorkItem("SET", "DOC", text, text, spans)
    second_mention = Mention("hypotension", text.index("hypotension"), text.index("hypotension") + 11, "Disease", 1.0)
    mapped = _mention_local_span(item, second_mention)
    assert mapped is not None
    sentence, start, end, source_start = mapped
    assert (sentence, source_start) == (second_text, second_start)
    assert sentence[start:end] == "hypotension"
    assert _work_item_evidence(item, second_mention) == second_text
    # The first sentence's mention still maps to the first span.
    first_mention = Mention("asthma", text.index("asthma"), text.index("asthma") + 6, "Disease", 1.0)
    assert _work_item_evidence(item, first_mention) == first_text


# --- end-to-end aggregate equivalence ---------------------------------------------


def test_duplicate_mentions_and_per_sentence_qualifiers_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Duplicate object mentions aggregate once; qualifiers attach only within their sentence."""
    import dakp_pipeline.assertions.contraindications as contra_mod

    text = "It is contraindicated in asthma. Do not use in women with hypotension."
    sections = _sections(tmp_path, [("SET-Q", "SET-Q#d", text)])
    ingredients = _ingredients(tmp_path, [("active", "SET-Q", "DrugQ", "UNII:Q")])

    asthma = (text.index("asthma"), text.index("asthma") + 6)
    women = (text.index("women"), text.index("women") + 5)
    hypo = (text.index("hypotension"), text.index("hypotension") + 11)
    # The duplicate asthma mention is value-equal to the first (distinct instance), so the
    # object membership short-circuit must drop exactly one of the two.
    mentions = [
        Mention("asthma", asthma[0], asthma[1], "Disease", 0.9),
        Mention("asthma", asthma[0], asthma[1], "Disease", 0.9),
        Mention("women", women[0], women[1], "BiologicalSex", 0.8),
        Mention("hypotension", hypo[0], hypo[1], "Disease", 0.95),
    ]
    monkeypatch.setattr(contra_mod, "extract_contraindication_diseases", lambda _text, _ner: list(mentions))

    ner = DiseaseNER(gazetteer={"asthma": "disease"})
    rows = build_contraindication_rows([sections, ingredients], ner)
    by_object = {r["object_text"]: r for r in rows}
    assert set(by_object) == {"asthma", "hypotension"}
    assert by_object["asthma"].get("sex_text", "") == ""
    assert by_object["hypotension"]["sex_text"] == "women"
    assert by_object["asthma"]["ner_confidence_score"] == "0.9"
    assert by_object["hypotension"]["ner_confidence_score"] == "0.95"
    # Deterministic, ordered output; a second run is byte-equal.
    again = build_contraindication_rows([sections, ingredients], ner)
    assert rows == again
    assert [r["object_text"] for r in rows] == ["asthma", "hypotension"]
