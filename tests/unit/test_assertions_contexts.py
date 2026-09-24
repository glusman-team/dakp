from __future__ import annotations

import pytest

from dakp_pipeline.assertions.approved_treats import _candidate_mention
from dakp_pipeline.assertions.contexts import (
    ASSERTION_CONTEXTS,
    PREVENTION_CUE,
    _template_host_indices,
    assertion_context,
    attach_qualifiers,
    context_predicate,
    patient_clause_contexts,
)
from dakp_pipeline.ner.lexical import Mention
from dakp_pipeline.ner.ner import _spans_from_result


def mention(text: str, start: int, end: int, typ: str, score: float = 1.0) -> Mention:
    return Mention(text=text, start=start, end=end, type=typ, score=score)


def test_context_matrix_and_every_prevention_cue() -> None:
    cues = (
        "prophylaxis",
        "prophylactic",
        "prevention of",
        "to prevent",
        "preventing",
        "immunization",
        "immunisation",
        "vaccine",
        "post-exposure prophylaxis",
        "PEP",
        "PrEP",
        "reduce the risk of",
    )
    for cue in cues:
        assert PREVENTION_CUE.search(cue)
        assert assertion_context("dailymed", "34067-9", f"For {cue} migraine") == "prevention"
        assert assertion_context("faers", "indication", f"For {cue} migraine") == "observed_prevention"
    assert assertion_context("dailymed", "34067-9", "Treatment of migraine") == "indication"
    assert assertion_context("faers", "indication", "Treatment of migraine") == "indication"
    for section in ("34070-3", "34066-1", "43685-7", "34071-1", "42232-9", "contraindications", "boxed_warning", "warnings"):
        assert assertion_context("dailymed", section, "vaccine migraine") == "contraindication"
    assert set(ASSERTION_CONTEXTS) == {"indication", "contraindication", "prevention", "observed_prevention"}


def test_context_errors_and_deferred_predicate_mapping() -> None:
    with pytest.raises(ValueError, match="source"):
        assertion_context("other", "34067-9", "x")
    with pytest.raises(ValueError, match="section_kind"):
        assertion_context("dailymed", "unknown", "x")
    with pytest.raises(ValueError, match="source"):
        context_predicate("indication", "other")
    with pytest.raises(ValueError, match="context"):
        context_predicate("unknown", "dailymed")
    assert context_predicate("indication", "dailymed") == "biolink:treats"
    assert context_predicate("prevention", "dailymed") == "biolink:treats"
    assert context_predicate("observed_prevention", "faers") == "biolink:applied_to_treat"
    assert context_predicate("contraindication", "faers") == "biolink:contraindicated_in"


def test_sparse_qualifier_attachment_withheld_cases() -> None:
    text = "In patients with asthma, drug is used in women."
    objs = [mention("asthma", 17, 23, "Disease")]
    qualifiers = [mention("women", 41, 46, "BiologicalSex")]
    sentence_of = lambda m: text
    assert attach_qualifiers(objs, qualifiers, sentence_of) == {0: {"sex_text": "women"}}
    assert attach_qualifiers(objs, [mention("women", 41, 46, "BiologicalSex", 0.4)], sentence_of) == {}
    assert attach_qualifiers(objs, [mention("women", 43, 48, "BiologicalSex")], lambda _m: None) == {}
    two = [*objs, mention("migraine", 25, 32, "Disease")]
    assert attach_qualifiers(two, qualifiers, sentence_of) == {}
    assert attach_qualifiers(objs, [mention("asthma", 17, 23, "AnatomicalEntity")], sentence_of) == {}
    assert (
        attach_qualifiers(
            [mention("asthma", 30, 36, "Disease")],
            [mention("hypertension", 0, 11, "PhenotypicFeature")],
            lambda _m: "Treatment of hypertension in patients with asthma",
        )
        == {}
    )


def test_patient_disease_context_and_attribute_abstention_payload() -> None:
    text = "For treatment of hypertension in patients with asthma."
    context = [mention("hypertension", 17, 29, "Disease")]
    host = [mention("asthma", 47, 53, "Disease")]
    assert attach_qualifiers(host, context, lambda _m: text) == {}
    assert (
        _spans_from_result({"entities": {"biolink:Disease": [{"start": 0, "end": 6, "confidence": 1.0, "assertion_context": []}]}}, "asthma")[
            0
        ].context_model
        == ""
    )
    assert (
        _spans_from_result(
            {
                "entities": {
                    "biolink:Disease": [
                        {
                            "start": 0,
                            "end": 6,
                            "confidence": 1.0,
                            "assertion_context": [{"label": "indication", "confidence": 0.4}, {"label": "prevention", "confidence": 0.8}],
                        }
                    ]
                }
            },
            "asthma",
        )[0].context_model
        == "prevention"
    )


def test_sentence_relative_offsets_are_position_independent() -> None:
    """Qualifiers must attach identically when the same sentence is embedded at any document offset.

    This guards the prior false negative where document-relative spans were compared directly with
    sentence-relative marker geometry.
    """
    sentence = "Drug is used in women."
    objects = [mention("Drug", 0, 4, "Disease")]
    qualifiers = [mention("women", 17, 22, "BiologicalSex")]
    assert attach_qualifiers(objects, qualifiers, lambda _mention: sentence) == {0: {"sex_text": "women"}}
    shifted = [mention(item.text, item.start + 500, item.end + 500, item.type, item.score) for item in (*objects, *qualifiers)]
    with pytest.raises(ValueError, match="sentence-relative"):
        attach_qualifiers(shifted[:1], shifted[1:], lambda _mention: sentence)


def test_context_faers_alias_and_qualifier_tie_breaking() -> None:
    """FAERS prevention must retain observed context, and repeated fields keep the higher score.

    These guards protect the source-specific context branch and deterministic sparse-field update
    without crossing the deferred ``prevents`` predicate boundary.
    """
    assert assertion_context(" FAERS ", "faers_indication", "PEP for migraine") == "observed_prevention"
    with pytest.raises(ValueError, match="section_kind"):
        assertion_context("faers", "warnings", "x")
    text = "Drug is used in women twice daily."
    host = [mention("Drug", 0, 4, "Disease")]
    qualifiers = [mention("women", 17, 22, "BiologicalSex", 0.9), mention("women", 17, 22, "BiologicalSex", 0.8)]
    assert attach_qualifiers(host, qualifiers, lambda _m: text) == {0: {"sex_text": "women"}}

    # A template with two post-marker objects is ambiguous even though the marker itself is clear.
    sentence = "For patients with asthma and migraine"
    two_hosts = [
        mention("asthma", sentence.index("asthma"), sentence.index("asthma") + 6, "Disease"),
        mention("migraine", sentence.index("migraine"), sentence.index("migraine") + 7, "Disease"),
    ]
    assert attach_qualifiers(two_hosts, [mention("women", 0, 5, "BiologicalSex")], lambda _m: sentence) == {}


def test_attachment_template_and_rejection_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "For treatment of hypertension in patients with asthma."
    context = [mention("hypertension", 17, 29, "Disease")]
    host = [mention("asthma", 47, 53, "Disease")]
    assert attach_qualifiers([mention("asthma", 47, 53, "PhenotypicFeature")], context, lambda _m: text) == {}
    with pytest.raises(ValueError, match="sentence-relative"):
        attach_qualifiers(host, [mention("hypertension", 17, 29, "Disease")], lambda _m: "")
    with pytest.raises(ValueError, match="sentence-relative"):
        attach_qualifiers(host, [mention("hypertension", 17, 29, "Disease")], lambda _m: "unrelated")
    assert attach_qualifiers([], [mention("women", 0, 5, "BiologicalSex")], lambda _m: "In patients with asthma") == {}
    # A patient template with no post-marker object must not invent a host for the qualifier.
    assert attach_qualifiers([], [mention("women", 0, 5, "BiologicalSex")], lambda _m: "In patients with") == {}
    # Disease mentions after the marker are hosts, not disease-context qualifiers; a disease
    # mention overlapping the marker is also rejected rather than guessed into the context slot.
    assert (
        attach_qualifiers([mention("asthma", 18, 24, "Disease")], [mention("hypertension", 0, 11, "Disease")], lambda _m: "For patients with asthma")
        == {}
    )
    assert (
        attach_qualifiers([mention("asthma", 18, 24, "Disease")], [mention("patients with", 3, 16, "Disease")], lambda _m: "For patients with asthma")
        == {}
    )
    # A disease typed as a context qualifier without the patient-template marker is withheld.
    assert (
        attach_qualifiers(
            [mention("asthma", 25, 31, "Disease")], [mention("hypertension", 0, 11, "Disease")], lambda _m: "Treatment of hypertension and asthma"
        )
        == {}
    )
    assert _template_host_indices("Treatment has no patient marker", [], lambda _m: "") == []
    template = "For patients with asthma and migraine"
    template_objects = [
        mention("asthma", template.index("asthma"), template.index("asthma") + 6, "Disease"),
        mention("migraine", template.index("migraine"), template.index("migraine") + 7, "Disease"),
    ]
    assert _template_host_indices(template, template_objects, lambda _m: template) == []
    monkeypatch.setattr("dakp_pipeline.assertions.contexts._template_host_indices", lambda *_args: [0, 1])
    assert attach_qualifiers([mention("asthma", 24, 30, "Disease")], [mention("women", 0, 5, "BiologicalSex")], lambda _m: template) == {}
    sentence = "For patients with asthma and migraine"
    hosts = [
        mention("asthma", sentence.index("asthma"), sentence.index("asthma") + 6, "Disease"),
        mention("migraine", sentence.index("migraine"), sentence.index("migraine") + 7, "Disease"),
    ]
    assert attach_qualifiers(hosts, [mention("women", 0, 5, "BiologicalSex")], lambda _m: sentence) == {}
    assert _candidate_mention("No matching condition.", {"object_text": "", "object_category": "Disease"}, 0) is None
    assert _candidate_mention("No matching condition.", {"object_text": "migraine", "object_category": "Disease"}, 0) is None


def test_patient_clause_contexts_skips_mentions_without_a_sentence() -> None:
    """A ``None`` sentence cannot join per-sentence grouping, so the mention is skipped."""
    clause = patient_clause_contexts([mention("asthma", 0, 6, "Disease")], lambda _m: None)
    assert clause.contexts == {}
    assert clause.context_only == {}
    assert clause.ambiguous == {}


def test_junk_wording_qualifiers_are_withheld() -> None:
    """Dosage-form frequency spans and anamnesis temporal spans are junk wordings.

    The v1.13.0 release audit showed these wordings resolve to drug products, chemical dosage
    forms, and history-of Phenomenon concepts, so they are withheld at attachment time, before
    entity resolution. A legitimate wording in the same sentence still attaches.
    """
    sentence = "Medical history noted; drug treats asthma twice daily short-term."
    host = [mention("asthma", sentence.index("asthma"), sentence.index("asthma") + 6, "Disease")]
    qualifiers = [
        mention("oral tablet", 0, 11, "frequency_qualifier"),
        mention("Extended Release Oral Capsule", 0, 28, "frequency_qualifier"),
        mention("twice daily", sentence.index("twice daily"), sentence.index("twice daily") + 11, "frequency_qualifier"),
        mention("Medical history", 0, 15, "temporal_context_qualifier"),
        mention("H/O: hypertension", 0, 17, "temporal_context_qualifier"),
        mention("short-term", sentence.index("short-term"), sentence.index("short-term") + 10, "temporal_context_qualifier"),
    ]
    assert attach_qualifiers(host, qualifiers, lambda _m: sentence) == {0: {"frequency_text": "twice daily", "temporal_context_text": "short-term"}}


def test_junk_wording_regex_never_matches_legit_wordings() -> None:
    """Genuine frequency/temporal wordings pass the denylist untouched."""
    from dakp_pipeline.assertions.contexts import _QUALIFIER_WORDING_DENYLIST

    freq = _QUALIFIER_WORDING_DENYLIST["frequency_qualifier"]
    for legit in ("once daily", "twice a week", "every 6 hours", "at bedtime", "3 times daily", "as needed"):
        assert freq.search(legit) is None, legit
    for junk in ("oral tablet", "tablets", "injection", "topical cream", "Extended Release Oral Capsule", "oral suspension", "suppository"):
        assert freq.search(junk), junk
    temporal = _QUALIFIER_WORDING_DENYLIST["temporal_context_qualifier"]
    for legit in ("preoperative", "short-term", "chronic", "7 days", "during treatment", "postoperative"):
        assert temporal.search(legit) is None, legit
    for junk in ("medical history", "history", "prior therapy", "previous therapy", "H/O: hypertension", "documentation", "documents"):
        assert temporal.search(junk), junk
