"""Unit tests for approved-treatment assertion aggregation (Milestone 5).

Covers the explicit rule (NDA-bearing pair → Drugs@FDA ingredient mapping → DailyMed approval +
SPL indication support), the FAERS-primary vs DailyMed-fallback candidate sources, join edge cases
(NDA without approval, NDA without Drugs@FDA mapping, multi-NDA dedup), provenance columns,
determinism, and the end-to-end shaper TSV output.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import pytest

import dakp_pipeline.assertions.approved_treats as approved_treats
from dakp_pipeline.assertions.approved_treats import ApprovedTreatsShaper, build_approved_treats_rows
from dakp_pipeline.assertions.evidence import (
    DailyMedEvidence,
    FDAApprovalIndex,
    build_dailymed_evidence,
    build_drugsfda_ingredient_map,
    find_faers_cases,
)
from dakp_pipeline.io import schemas
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.ner import Mention

# --- the rule, driven by FAERS NDA-bearing pairs --------------------------------


def _rows_by_object(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["object_text"]: row for row in rows}


def test_faers_rule_keeps_approved_pairs_and_filters_unmapped(
    faers_refs: list[ArtifactRef], dailymed_refs: list[ArtifactRef], drugsfda_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]
) -> None:
    cases = find_faers_cases(faers_refs)
    ev = build_dailymed_evidence(dailymed_refs)
    mapping = build_drugsfda_ingredient_map(drugsfda_refs)

    rows = build_approved_treats_rows(cases, ev, mapping, disease_map)
    by_obj = _rows_by_object(rows)

    # Examplestatin (NDA 012345) and Ibuprofen (NDA 017977) are approved; Placebo (099999, not in
    # Drugs@FDA) is filtered out.
    assert set(by_obj) == {"hypercholesterolemia", "headache"}

    statin = by_obj["hypercholesterolemia"]
    assert statin["subject_text"] == "Examplestatin"
    assert statin["subject_curie"] == "UNII:QFX8B1R4QF"  # DailyMed-provided UNII
    assert statin["object_curie"] == "MONDO:0005154"
    assert statin["FDA_regulatory_approvals"] == "NDA012345"  # legacy display form: application type + number
    assert statin["supporting_spl_evidence"] == "dailymed:SETID-EXAMPLESTATIN-001"  # legacy set-CURIE form
    assert statin["supporting_spl_sets"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SETID-EXAMPLESTATIN-001"
    assert statin["supporting_spl_documents"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SETID-EXAMPLESTATIN-001#34067-9"

    # FAERS reported the brand 'Advil'; the subject is the canonical DailyMed ingredient.
    ibuprofen = by_obj["headache"]
    assert ibuprofen["subject_text"] == "Ibuprofen"
    assert ibuprofen["subject_curie"] == "UNII:WK2XYI10QM"
    assert ibuprofen["object_curie"] == "HP:0002315"
    assert ibuprofen["predicate"] == "biolink:treats"
    assert "dailymed:SETID-IBUPROFEN-002" in ibuprofen["edge_evidence"]
    assert "faers:" not in ibuprofen["edge_evidence"]
    assert ibuprofen["supporting_faers_records"]
    assert ibuprofen["supporting_faers_urls"]


def test_faers_pair_without_drugsfda_mapping_is_not_approved(disease_map: dict[str, dict[str, str]]) -> None:
    # NDA 207500 maps in Drugs@FDA but has NO DailyMed approval -> not approved-treats.
    ev = DailyMedEvidence(
        approval_sets={"12345": {"SET-A"}},
        approval_display={"12345": "012345"},
        set_ingredient={"SET-A": ("Examplestatin", "UNII:QFX8B1R4QF")},
        indication_docs={"SET-A": [("SET-A#34067-9", "hypercholesterolemia")]},
    )
    mapping = {"12345": {"EXAMPLESTATIN"}, "207500": {"REALDRUG"}}
    cases = pl.DataFrame({"nda": ["207500"], "indication": ["hypercholesterolemia"], "drugname": ["Realdrug"], "ingredient": ["Realdrug"]})

    assert build_approved_treats_rows(cases, ev, mapping, disease_map) == []


def test_faers_pair_without_spl_indication_support_is_not_approved(disease_map: dict[str, dict[str, str]]) -> None:
    # NDA maps in Drugs@FDA and has a DailyMed approval, but the approved set has NO indication
    # section -> no SPL indication support -> not approved-treats.
    ev = DailyMedEvidence(
        approval_sets={"12345": {"SET-A"}},
        approval_display={"12345": "012345"},
        set_ingredient={"SET-A": ("Examplestatin", "UNII:QFX8B1R4QF")},
        indication_docs={},  # approval exists, but no indication section
    )
    mapping = {"12345": {"EXAMPLESTATIN"}}
    cases = pl.DataFrame({"nda": ["012345"], "indication": ["hypercholesterolemia"], "drugname": ["Examplestatin"], "ingredient": ["Examplestatin"]})

    assert build_approved_treats_rows(cases, ev, mapping, disease_map) == []


def test_multi_nda_for_same_subject_object_aggregates_approvals(disease_map: dict[str, dict[str, str]]) -> None:
    # Two distinct NDAs, same ingredient + indication -> ONE row with both approvals aggregated.
    ev = DailyMedEvidence(
        approval_sets={"12345": {"SET-A"}, "99998": {"SET-B"}},
        approval_display={"12345": "012345", "99998": "099998"},
        set_ingredient={"SET-A": ("Examplestatin", "UNII:QFX8B1R4QF"), "SET-B": ("Examplestatin", "UNII:QFX8B1R4QF")},
        indication_docs={"SET-A": [("SET-A#34067-9", "hypercholesterolemia")], "SET-B": [("SET-B#34067-9", "hypercholesterolemia")]},
    )
    mapping = {"12345": {"EXAMPLESTATIN"}, "99998": {"EXAMPLESTATIN"}}
    cases = pl.DataFrame(
        {
            "nda": ["012345", "099998", "012345"],  # third row duplicates the first case's NDA
            "indication": ["hypercholesterolemia"] * 3,
            "drugname": ["Examplestatin"] * 3,
            "ingredient": ["Examplestatin"] * 3,
        }
    )

    rows = build_approved_treats_rows(cases, ev, mapping, disease_map)
    assert len(rows) == 1
    row = rows[0]
    # No index, and DailyMed's bare display value names no application type: both numbers are
    # dropped rather than shipped as values nobody can resolve.
    assert row["FDA_regulatory_approvals"] == ""

    # With the Drugs@FDA register, each number resolves to its FDA form.
    index = FDAApprovalIndex({"12345": ("NDA012345",), "99998": ("BLA099998",)})
    expanded = build_approved_treats_rows(cases, ev, mapping, disease_map, approvals=index)
    assert expanded[0]["FDA_regulatory_approvals"] == "BLA099998|NDA012345"
    assert (
        row["supporting_spl_sets"]
        == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A|https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-B"
    )
    assert (
        row["supporting_spl_documents"]
        == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-A#34067-9|https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=SET-B#34067-9"
    )


def test_faers_placeholder_indication_is_not_an_approval_claim(disease_map: dict[str, dict[str, str]]) -> None:
    # Every gate passes for this NDA (Drugs@FDA mapping, DailyMed approval, indication section),
    # but the FAERS indication is a usage-context placeholder, not a condition -> no row.
    ev = DailyMedEvidence(
        approval_sets={"12345": {"SET-A"}},
        approval_display={"12345": "012345"},
        set_ingredient={"SET-A": ("Examplestatin", "UNII:QFX8B1R4QF")},
        indication_docs={"SET-A": [("SET-A#34067-9", "hypercholesterolemia")]},
    )
    mapping = {"12345": {"EXAMPLESTATIN"}}
    cases = pl.DataFrame(
        {
            "nda": ["012345", "012345"],
            "indication": ["Product used for unknown indication", "hypercholesterolemia"],
            "drugname": ["Examplestatin"] * 2,
            "ingredient": ["Examplestatin"] * 2,
        }
    )

    rows = build_approved_treats_rows(cases, ev, mapping, disease_map)
    assert [row["object_text"] for row in rows] == ["hypercholesterolemia"]  # placeholder dropped, real pair kept


# --- DailyMed fallback (used when no FAERS case table is present) ---------------


def test_dailymed_fallback_when_no_faers_cases(
    dailymed_refs: list[ArtifactRef], drugsfda_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]
) -> None:
    ev = build_dailymed_evidence(dailymed_refs)
    mapping = build_drugsfda_ingredient_map(drugsfda_refs)

    rows = build_approved_treats_rows(None, ev, mapping, disease_map)
    by_obj = _rows_by_object(rows)

    # Examplestatin->hypercholesterolemia; Ibuprofen->headache + pain. Omeprazole excluded (NDA
    # 022329 not in Drugs@FDA; its indication is also not in the dictionary).
    assert set(by_obj) == {"hypercholesterolemia", "headache", "pain"}
    assert by_obj["pain"]["subject_text"] == "Ibuprofen"
    assert by_obj["pain"]["FDA_regulatory_approvals"] == "NDA017977"
    assert by_obj["pain"]["supporting_spl_evidence"] == "dailymed:SETID-IBUPROFEN-002"


# --- provenance columns + determinism -------------------------------------------


def test_provenance_columns_are_fixed(
    faers_refs: list[ArtifactRef], dailymed_refs: list[ArtifactRef], drugsfda_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]
) -> None:
    rows = build_approved_treats_rows(
        find_faers_cases(faers_refs), build_dailymed_evidence(dailymed_refs), build_drugsfda_ingredient_map(drugsfda_refs), disease_map
    )
    assert rows
    for row in rows:
        assert row["clinical_approval_status"] == "approved_for_condition"
        assert row["knowledge_level"] == "knowledge_assertion"
        assert row["agent_type"] == "manual_validation_of_automated_agent"
        assert row["primary_knowledge_source"] == "infores:drugapprovals-kp"
        assert row["upstream_resource_ids"] == "infores:dailymed|infores:faers"
        assert row["subject_category"] == "ChemicalEntity"


def test_row_ordering_is_deterministic(
    faers_refs: list[ArtifactRef], dailymed_refs: list[ArtifactRef], drugsfda_refs: list[ArtifactRef], disease_map: dict[str, dict[str, str]]
) -> None:
    args = (find_faers_cases(faers_refs), build_dailymed_evidence(dailymed_refs), build_drugsfda_ingredient_map(drugsfda_refs), disease_map)
    first = build_approved_treats_rows(*args)
    second = build_approved_treats_rows(*args)
    assert first == second
    keys = [(r["subject_text"], r["object_text"]) for r in first]
    assert keys == sorted(keys)


# --- end-to-end shaper output ---------------------------------------------------


def test_shaper_writes_uncompressed_tsv_with_contract_columns(
    dailymed_refs: list[ArtifactRef], drugsfda_refs: list[ArtifactRef], ctx: TaskContext
) -> None:
    refs = ApprovedTreatsShaper().transform([*dailymed_refs, *drugsfda_refs], ctx)
    assert len(refs) == 1
    out = refs[0]
    assert out.uri.name == "approved_treats_assertions.tsv"
    assert out.uri.suffix == ".tsv"

    frame = schemas.read_table(out.uri)
    assert frame.columns == schemas.APPROVED_TREATS_COLUMNS
    assert frame.height == 3  # DailyMed-fallback path (no FAERS case table among the inputs)
    # Uncompressed: plain-text header is the first line.
    assert out.uri.read_bytes().startswith(b"subject_text\t")


# --- run-scoped preparation cache (per-text invariants across many candidates) ---


def _host_mention(text: str) -> Mention:
    return Mention(text=text, start=0, end=len(text), type="Disease", score=1.0)


class _NullPreparationCache:
    """Replaces the run-scoped cache with per-call recomputation (the pre-fix behavior)."""

    def prepare(self, text: str, disease_map: Mapping[str, Mapping[str, str]]) -> approved_treats._SectionPreparation:
        positive_text = approved_treats._positive_context_text(text)
        return approved_treats._SectionPreparation(
            normalized_text=approved_treats.normalize_text(positive_text),
            dictionary_matches=tuple(approved_treats.match_diseases(positive_text, disease_map)),
        )

    def sentence_spans(self, text: str) -> list[tuple[int, int, str]]:
        return approved_treats._sentence_spans(text)


def test_preparation_cache_prepares_each_unique_text_once(monkeypatch, disease_map: dict[str, dict[str, str]]) -> None:
    """Section preparation runs once per unique text, never once per candidate x document."""
    section = "Examplestatin improves hypercholesterolemia. Headache has not been established. Treats migraine disorder in adults."
    ndas = [f"9{i:04d}" for i in range(1, 9)]
    ev = DailyMedEvidence(
        approval_sets={nda: {f"SET-{nda}"} for nda in ndas},
        approval_display={nda: nda for nda in ndas},
        set_ingredient={f"SET-{nda}": ("Examplestatin", "UNII:QFX8B1R4QF") for nda in ndas},
        indication_docs={f"SET-{nda}": [(f"SET-{nda}#34067-9", section)] for nda in ndas},
    )
    mapping = {nda: {"EXAMPLESTATIN"} for nda in ndas}
    cases = pl.DataFrame(
        {
            "nda": ndas,
            "indication": ["hypercholesterolemia"] * len(ndas),
            "drugname": ["Examplestatin"] * len(ndas),
            "ingredient": ["Examplestatin"] * len(ndas),
        }
    )

    real_positive_text = approved_treats._positive_context_text
    prepared_texts: list[str] = []

    def counting_positive_text(text: str) -> str:
        prepared_texts.append(text)
        return real_positive_text(text)

    monkeypatch.setattr(approved_treats, "_positive_context_text", counting_positive_text)
    rows = build_approved_treats_rows(cases, ev, mapping, disease_map)

    assert rows, "sanity: candidates must be corroborated"
    unique_texts = {section, *(sentence for _s, _e, sentence in approved_treats._sentence_spans(section))}
    assert len(prepared_texts) == len(unique_texts)  # 4, NOT candidates x (1 section + 3 sentences)


def test_rows_identical_with_and_without_preparation_cache(monkeypatch, disease_map: dict[str, dict[str, str]]) -> None:
    """Cached and uncached aggregation produce identical rows (lexical + negation + normalize channels)."""
    section = "Treats hypercholesterolemia in adults. Migraine disorder not recommended. Also treats migraine headaches in adults."
    ndas = ["11111", "22222", "33333"]
    ev = DailyMedEvidence(
        approval_sets={nda: {f"SET-{nda}"} for nda in ndas},
        approval_display={nda: nda for nda in ndas},
        set_ingredient={f"SET-{nda}": ("Examplestatin", "UNII:QFX8B1R4QF") for nda in ndas},
        indication_docs={f"SET-{nda}": [(f"SET-{nda}#34067-9", section), (f"SET-{nda}#34067-9b", "Migraine disorder clinic.")] for nda in ndas},
    )
    mapping = {nda: {"EXAMPLESTATIN"} for nda in ndas}
    indications = ["hypercholesterolemia", "migraine headaches", "migraine disorder", "Migraine-Disorder"]
    cases = pl.DataFrame(
        {
            "nda": [nda for nda in ndas for _ in indications],
            "indication": indications * len(ndas),
            "drugname": ["Examplestatin"] * (len(indications) * len(ndas)),
            "ingredient": ["Examplestatin"] * (len(indications) * len(ndas)),
        }
    )

    original = approved_treats._positive_context_text
    preparations: list[str] = []

    def counted(text: str) -> str:
        preparations.append(text)
        return original(text)

    monkeypatch.setattr(approved_treats, "_positive_context_text", counted)
    rows_cached = build_approved_treats_rows(cases, ev, mapping, disease_map)
    cached_count = len(preparations)
    preparations.clear()
    monkeypatch.setattr(approved_treats, "_PreparationCache", _NullPreparationCache)
    rows_uncached = build_approved_treats_rows(cases, ev, mapping, disease_map)

    assert rows_cached
    assert rows_cached == rows_uncached
    assert len(preparations) > cached_count, "the uncached seam must actually recompute preparation"


@pytest.mark.parametrize(
    ("section_text", "object_text", "object_curie", "section_mentions", "expected"),
    [
        # lexical verbatim, no dictionary entry anywhere
        ("Treats migraine headaches in adults.", "migraine headaches", "", [], True),
        # different spelling/case/punctuation: the normalized-text channel matches
        ("Treats Migraine-Headaches, chronic form.", "MIGRAINE    headaches!", "", [], True),
        # NER mention word-contained IN the more specific candidate (model channel)
        ("Hormone receptor positive breast cancer.", "hormone receptor positive breast cancer", "", [_host_mention("breast cancer")], True),
        # NER mention naming a condition that survives only in a negated sentence
        ("Headache not indicated. Treats migraine disorder.", "headache", "", [_host_mention("headache")], False),
        # same CURIE, different surface spelling (dictionary CURIE-equality channel)
        ("Migraine disorder responds to therapy.", "hemicrania", "MONDO:0005439", [], True),
        # different CURIE and no textual overlap: no channel corroborates
        ("Migraine disorder responds to therapy.", "episodic pain", "MONDO:9999999", [], False),
    ],
)
def test_section_mentions_condition_with_preparation_matches_without(
    section_text: str, object_text: str, object_curie: str, section_mentions: list, expected: bool
) -> None:
    local_map = {"migraine disorder": {"curie": "MONDO:0005439", "name": "Migraine", "category": "Disease"}}
    cand = {"object_text": object_text, "object_curie": object_curie}
    cache = approved_treats._PreparationCache()
    uncached = approved_treats._section_mentions_condition(section_text, cand, local_map, section_mentions)
    prepared = cache.prepare(section_text, local_map)
    cached = approved_treats._section_mentions_condition(section_text, cand, local_map, section_mentions, preparation=prepared)
    repeated = approved_treats._section_mentions_condition(
        section_text, cand, local_map, section_mentions, preparation=cache.prepare(section_text, local_map)
    )
    assert cached == repeated == uncached == expected
