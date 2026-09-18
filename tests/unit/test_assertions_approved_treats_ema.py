"""Tests for the EMA union in approved-treatment assertion aggregation.

Covers the per-(active substance, MeSH therapeutic-area term) fan-out from the EMA interim
registry table: the INN fallback when "Active substance" is empty, skips for subject-less or
object-less rows, deterministic aggregation/dedup of EMA product numbers and EPAR URLs, the
``infores:ema`` provenance, and the end-to-end shaper output over the committed fixture xlsx.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.assertions.approved_treats import (
    ApprovedTreatsShaper,
    _ema_indication_items,
    build_approved_treats_rows,
    build_ema_treats_rows,
    build_epar_treats_rows,
)
from dakp_pipeline.assertions.evidence import DailyMedEvidence
from dakp_pipeline.extract.ema_registry import EMA_REGISTRY_COLUMNS
from dakp_pipeline.extract.ema_registry import extract as ema_extract
from dakp_pipeline.io import schemas
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.ner.lexical import Mention
from dakp_pipeline.ner.ner import DiseaseNER

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_EMA_FIXTURE = _FIXTURE_ROOT / "ema" / "medicines-output-medicines-report_en.xlsx"


def _ema_frame(rows: list[dict[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=dict.fromkeys(EMA_REGISTRY_COLUMNS, pl.Utf8))


def _registry_row(**overrides: str) -> dict[str, str]:
    row = {
        "medicine_name": "KemSu",
        "ema_product_number": "EMEA/H/C/006395",
        "category": "Human",
        "medicine_status": "Authorised",
        "inn": "sufentanil/ketamine",
        "active_substance": "sufentanil;ketamine",
        "therapeutic_area_mesh": "Pain;Shock",
        "therapeutic_indication": "KemSu is indicated for ...",
        "medicine_url": "https://www.ema.europa.eu/en/medicines/human/EPAR/kemsu",
    }
    row.update(overrides)
    return row


# --- the fan-out + provenance contract -------------------------------------------


def test_ema_rows_fan_out_per_substance_and_mesh_term(disease_map: dict[str, dict[str, str]]) -> None:
    rows = build_ema_treats_rows(_ema_frame([_registry_row()]), disease_map)
    by_pair = _rows_by_pair(rows)
    assert set(by_pair) == {("sufentanil", "Pain"), ("sufentanil", "Shock"), ("ketamine", "Pain"), ("ketamine", "Shock")}

    row = by_pair[("ketamine", "Pain")]
    assert row["predicate"] == "biolink:treats"
    assert row["subject_category"] == "ChemicalEntity"
    assert row["subject_curie"] == ""  # text-first: the EMA export carries no substance ids
    assert row["FDA_regulatory_approvals"] == "EMEA/H/C/006395"
    assert row["supporting_spl_sets"] == ""  # EMA rows have no SPL evidence
    assert row["supporting_spl_documents"] == "https://www.ema.europa.eu/en/medicines/human/EPAR/kemsu"
    assert row["clinical_approval_status"] == "approved_for_condition"
    assert row["knowledge_level"] == "knowledge_assertion"
    assert row["agent_type"] == "manual_validation_of_automated_agent"
    assert row["primary_knowledge_source"] == "infores:multiomics-drugapprovals"
    assert row["upstream_resource_ids"] == "infores:ema"


def test_inn_fallback_when_active_substance_empty(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame([_registry_row(active_substance="", inn="teclistamab", therapeutic_area_mesh="Multiple Myeloma")])
    rows = build_ema_treats_rows(frame, disease_map)
    assert [(row["subject_text"], row["object_text"]) for row in rows] == [("teclistamab", "Multiple Myeloma")]


def test_rows_without_subject_or_mesh_term_are_skipped(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame(
        [
            _registry_row(active_substance="", inn=""),  # no subject at all
            _registry_row(therapeutic_area_mesh=""),  # no object
            _registry_row(therapeutic_area_mesh="Pain;"),  # trailing separator yields one term
        ]
    )
    rows = build_ema_treats_rows(frame, disease_map)
    assert [(row["subject_text"], row["object_text"]) for row in rows] == [("ketamine", "Pain"), ("sufentanil", "Pain")]


def test_aggregation_dedups_and_sorts_product_numbers_and_urls(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame(
        [
            _registry_row(ema_product_number="EMEA/H/C/000002", active_substance="mitapivat sulfate", therapeutic_area_mesh="Anemia, Hemolytic"),
            _registry_row(ema_product_number="EMEA/H/C/000001", active_substance="mitapivat sulfate", therapeutic_area_mesh="Anemia, Hemolytic"),
            _registry_row(  # exact duplicate pair + product number -> deduplicated away
                ema_product_number="EMEA/H/C/000001",
                active_substance="mitapivat sulfate",
                therapeutic_area_mesh="Anemia, Hemolytic",
                medicine_url="https://www.ema.europa.eu/en/medicines/human/EPAR/other",
            ),
        ]
    )
    rows = build_ema_treats_rows(frame, disease_map)
    assert len(rows) == 1
    assert rows[0]["FDA_regulatory_approvals"] == "EMEA/H/C/000001|EMEA/H/C/000002"
    assert rows[0]["supporting_spl_documents"] == (
        "https://www.ema.europa.eu/en/medicines/human/EPAR/kemsu|https://www.ema.europa.eu/en/medicines/human/EPAR/other"
    )


def test_object_curie_resolves_via_the_disease_map(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame([_registry_row(active_substance="ibuprofen", therapeutic_area_mesh="headache")])
    row = build_ema_treats_rows(frame, disease_map)[0]
    assert row["object_curie"] == "HP:0002315"  # lexical baseline match
    assert row["object_category"] == "PhenotypicFeature"


# --- union with the FDA path + end-to-end shaper output --------------------------


def test_build_approved_treats_rows_unions_ema_rows_sorted(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame([_registry_row(active_substance="teclistamab", therapeutic_area_mesh="Multiple Myeloma")])
    rows = build_approved_treats_rows(None, DailyMedEvidence(), {}, disease_map, ema_registry=frame)
    assert [(row["subject_text"], row["object_text"]) for row in rows] == [("teclistamab", "Multiple Myeloma")]
    assert rows[0]["upstream_resource_ids"] == "infores:ema"


def test_shaper_unions_fixture_registry_into_the_assertion_tsv(ctx: TaskContext) -> None:
    ema_refs = ema_extract([ArtifactRef(uri=_EMA_FIXTURE, blake3=hash_file(_EMA_FIXTURE), media_type="application/octet-stream")], ctx)

    refs = ApprovedTreatsShaper().transform(ema_refs, ctx)
    assert len(refs) == 1
    frame = schemas.read_table(refs[0].uri)
    assert frame.columns == schemas.APPROVED_TREATS_COLUMNS

    rows = frame.to_dicts()
    # Pyrukynd (1 substance x 2 MeSH) + Twinrix Adult (2 substances x 3 MeSH) + Tecvayli (INN x 1 MeSH).
    assert frame.height == 9
    assert {row["upstream_resource_ids"] for row in rows} == {"infores:ema"}
    by_pair = {(row["subject_text"], row["object_text"]): row for row in rows}
    pyrukynd = by_pair[("mitapivat sulfate", "Anemia, Hemolytic")]
    assert pyrukynd["FDA_regulatory_approvals"] == "EMEA/H/C/005540"
    assert pyrukynd["supporting_spl_documents"] == "https://www.ema.europa.eu/en/medicines/human/EPAR/pyrukynd"
    assert ("teclistamab", "Multiple Myeloma") in by_pair  # INN fallback row survives
    keys = [(row["subject_text"], row["object_text"]) for row in rows]
    assert keys == sorted(keys)  # deterministic ordering


# --- Phase 2: EPAR indication mining (infores:epar) ------------------------------


def _rows_by_pair(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["subject_text"], row["object_text"]): row for row in rows}


def _ner(*terms: str) -> DiseaseNER:
    """A deterministic offline NER over an ad-hoc gazetteer (the fake-backend pattern)."""
    return DiseaseNER(offline=True, gazetteer=dict.fromkeys(terms, "disease"))


def _mined_map(frame: pl.DataFrame, ner: DiseaseNER) -> dict[tuple[str, str], list[Mention]]:
    """Mine every registry row's indication text the way the shared pool does."""
    return {(key, doc): ner.extract(text) for key, doc, text in _ema_indication_items(frame)}


def test_epar_rows_mined_from_indication_text(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame([_registry_row(therapeutic_indication="KemSu is indicated for the management of severe pain.")])
    rows = build_epar_treats_rows(frame, _mined_map(frame, _ner("severe pain", "pain")), disease_map)
    by_pair = _rows_by_pair(rows)
    # Subject fan-out (both combo substances); the maximal gazetteer span wins the object.
    assert set(by_pair) == {("sufentanil", "severe pain"), ("ketamine", "severe pain")}

    row = by_pair[("ketamine", "severe pain")]
    assert row["predicate"] == "biolink:treats"
    assert row["subject_category"] == "ChemicalEntity"
    assert row["FDA_regulatory_approvals"] == "EMEA/H/C/006395"
    assert row["supporting_spl_sets"] == ""
    assert row["supporting_spl_documents"] == "https://www.ema.europa.eu/en/medicines/human/EPAR/kemsu"
    assert row["clinical_approval_status"] == "approved_for_condition"
    assert row["knowledge_level"] == "knowledge_assertion"
    assert row["agent_type"] == "manual_validation_of_automated_agent"
    assert row["primary_knowledge_source"] == "infores:multiomics-drugapprovals"
    assert row["upstream_resource_ids"] == "infores:epar"


def test_epar_qualifiers_are_populated_from_the_indication_sentence(disease_map: dict[str, dict[str, str]]) -> None:
    """Non-host mentions in the SAME sentence become sparse qualifiers on the EPAR row.

    EPAR indication text is qualifier-rich prose ("in adult patients", "in paediatric patients
    aged 6 years and older"), so the mined rows run the identical
    :func:`~dakp_pipeline.assertions.contexts.attach_qualifiers_with_scores` floor/host rules the
    DailyMed rows use rather than emitting bare pairs. Without this the qualifier columns would
    ship empty for every EMA-derived edge.
    """
    frame = _ema_frame([_registry_row(therapeutic_indication="KemSu is indicated for the treatment of severe pain in adult patients.")])
    ner = DiseaseNER(offline=True, gazetteer={"severe pain": "disease", "adult patients": "PopulationOfIndividualOrganisms"})
    rows = build_epar_treats_rows(frame, _mined_map(frame, ner), disease_map)

    row = _rows_by_pair(rows)[("ketamine", "severe pain")]
    assert row["population_context_text"] == "adult patients"
    # The qualifier never doubles as the object: only the disease-typed span is a host.
    assert row["object_text"] == "severe pain"


def test_epar_qualifiers_do_not_leak_across_sentences(disease_map: dict[str, dict[str, str]]) -> None:
    """Rows are built sentence by sentence, so one indication never inherits another's qualifier.

    An EPAR routinely lists several indications in one cell. Mining the cell as a single blob
    would attach "adult patients" from sentence two onto the disease from sentence one —
    fabricated provenance. Sentence-localized mention offsets are what prevent it, and this is
    also why the qualifier slots joined ``UUID_FIELDS`` (a qualified row must stay a DISTINCT
    edge from the unqualified one, never merged by ``uuid_on_collision``).
    """
    frame = _ema_frame(
        [
            _registry_row(
                active_substance="ketamine",
                therapeutic_indication="KemSu is indicated for severe pain. It is also indicated for shock in adult patients.",
            )
        ]
    )
    ner = DiseaseNER(offline=True, gazetteer={"severe pain": "disease", "shock": "disease", "adult patients": "PopulationOfIndividualOrganisms"})
    rows = build_epar_treats_rows(frame, _mined_map(frame, ner), disease_map)
    by_pair = _rows_by_pair(rows)

    assert by_pair[("ketamine", "severe pain")]["population_context_text"] == ""
    assert by_pair[("ketamine", "shock")]["population_context_text"] == "adult patients"


def test_epar_row_key_falls_back_to_url_then_row_index() -> None:
    """``_ema_row_key`` keeps mined mentions addressable when the registry row lacks ids.

    A trimmed registry can miss the product number (draft medicine) and even the EPAR URL; the
    row index is the last-resort key so the mined pool still round-trips. Cycles through the
    fallback chain so each branch stays covered.
    """
    from dakp_pipeline.assertions.approved_treats import _ema_row_key

    assert _ema_row_key({"ema_product_number": "EMEA/H/C/000001", "medicine_url": "u"}, 3) == "ema:EMEA/H/C/000001"
    assert _ema_row_key({"ema_product_number": "", "medicine_url": "https://epar"}, 3) == "ema:https://epar"
    assert _ema_row_key({"ema_product_number": "", "medicine_url": ""}, 3) == "ema:row-3"


def test_epar_skips_negated_sentences_and_blank_mentions(disease_map: dict[str, dict[str, str]]) -> None:
    """EPAR sentences carrying the SPL disclaimers are skipped entirely; blank-normalized hosts are dropped."""
    frame = _ema_frame(
        [_registry_row(active_substance="ketamine", therapeutic_indication="KemSu is indicated for severe pain. It is not indicated for shock.")]
    )
    rows = build_epar_treats_rows(frame, _mined_map(frame, _ner("severe pain", "shock")), disease_map)
    assert [(row["subject_text"], row["object_text"]) for row in rows] == [("ketamine", "severe pain")]

    # A host mention that normalizes to nothing (punctuation-only span) is dropped, not emitted.
    # Note the sentence-localization re-derives the span text from the sentence, so the blank
    # must be a real punctuation-only span of the indication text itself.
    text = "KemSu . pain"
    blank_frame = _ema_frame([_registry_row(active_substance="ketamine", therapeutic_indication=text)])
    blank = Mention(text=".", start=text.index("."), end=text.index(".") + 1, type="Disease", score=1.0)
    assert build_epar_treats_rows(blank_frame, {("ema:EMEA/H/C/006395", "indication"): [blank]}, disease_map) == []


def test_epar_keeps_the_highest_context_model_and_qualifier_scores(disease_map: dict[str, dict[str, str]]) -> None:
    """Aggregation across sentences keeps the strongest context-model score and qualifier candidate.

    Two sentences can assert the same pair (a treatment sentence and a prevention sentence); the
    context classifier score decides which ``assertion_context_model`` wins, and when the same
    qualifier slot is attached twice the better (score, text) tuple replaces the earlier one.
    """
    from dakp_pipeline.ner.lexical import Mention

    text = "KemSu treats severe pain in elderly patients. KemSu prevents severe pain in adult patients."
    frame = _ema_frame([_registry_row(active_substance="ketamine", therapeutic_indication=text)])
    mentions = [
        Mention(
            text="severe pain",
            start=text.index("severe pain"),
            end=text.index("severe pain") + len("severe pain"),
            type="Disease",
            score=0.9,
            context_model="treatment",
            context_model_score=0.4,
        ),
        Mention(
            text="elderly patients",
            start=text.index("elderly patients"),
            end=text.index("elderly patients") + len("elderly patients"),
            type="PopulationOfIndividualOrganisms",
            score=0.5,
        ),
        Mention(
            text="severe pain",
            start=text.rindex("severe pain"),
            end=text.rindex("severe pain") + len("severe pain"),
            type="Disease",
            score=0.9,
            context_model="prevention",
            context_model_score=0.8,
        ),
        # Second sentence's qualifier ties on score but loses the text tie-break ("adult" < "elderly"):
        # aggregation keeps the first one — the qualifier never regresses even when the context
        # model score does.
        Mention(
            text="adult patients",
            start=text.rindex("adult patients"),
            end=text.rindex("adult patients") + len("adult patients"),
            type="PopulationOfIndividualOrganisms",
            score=0.5,
        ),
    ]
    rows = build_epar_treats_rows(frame, {("ema:EMEA/H/C/006395", "indication"): mentions}, disease_map)

    # Neither sentence carries a prevention cue, so both share the "indication" context and
    # aggregate into ONE row: the context-model score is REPLACED by the higher one (prevention,
    # 0.8) while the qualifier tie-break ("adult" > "aged" at equal 0.5) keeps the FIRST
    # qualifier — evidence aggregation is max-per-field, never first-or-last-wins wholesale.
    assert [row["assertion_context"] for row in rows] == ["indication"]
    row = rows[0]
    assert row["assertion_context_model"] == "prevention"
    assert row["assertion_context_model_score"] == "0.8"
    assert row["population_context_text"] == "elderly patients"  # the smaller-text candidate did not replace it


def test_epar_inn_fallback_when_active_substance_empty(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame(
        [_registry_row(active_substance="", inn="teclistamab", therapeutic_indication="Tecvayli is indicated for relapsed multiple myeloma.")]
    )
    rows = build_epar_treats_rows(frame, _mined_map(frame, _ner("multiple myeloma")), disease_map)
    assert [(row["subject_text"], row["object_text"]) for row in rows] == [("teclistamab", "multiple myeloma")]


def test_epar_skips_empty_indication_and_subjectless_rows(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame(
        [
            _registry_row(therapeutic_indication=""),  # nothing to mine
            _registry_row(active_substance="", inn="", therapeutic_indication="Indicated for pain."),  # no subject
        ]
    )
    assert build_epar_treats_rows(frame, _mined_map(frame, _ner("pain")), disease_map) == []


def test_epar_dedups_across_medicines_sharing_a_substance(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame(
        [
            _registry_row(ema_product_number="EMEA/H/C/000002", active_substance="ibuprofen", therapeutic_indication="Indicated for pain."),
            _registry_row(ema_product_number="EMEA/H/C/000001", active_substance="ibuprofen", therapeutic_indication="Indicated for pain."),
        ]
    )
    rows = build_epar_treats_rows(frame, _mined_map(frame, _ner("pain")), disease_map)
    assert len(rows) == 1
    assert rows[0]["FDA_regulatory_approvals"] == "EMEA/H/C/000001|EMEA/H/C/000002"  # sorted, deduped


def test_epar_object_curie_resolves_via_the_disease_map(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame([_registry_row(active_substance="ibuprofen", therapeutic_indication="Indicated for headache.")])
    row = build_epar_treats_rows(frame, _mined_map(frame, _ner("headache")), disease_map)[0]
    assert row["object_text"] == "headache"
    assert row["object_curie"] == "HP:0002315"
    assert row["object_category"] == "PhenotypicFeature"


def test_epar_drops_mentions_that_normalize_to_nothing(disease_map: dict[str, dict[str, str]]) -> None:
    class _PunctNER(DiseaseNER):
        def extract(self, text: str) -> list[Mention]:
            return [Mention(text="---", start=0, end=3, type="disease", score=1.0)]

    frame = _ema_frame([_registry_row(therapeutic_indication="Indicated for something.")])
    assert build_epar_treats_rows(frame, _mined_map(frame, _PunctNER(offline=True)), disease_map) == []


def test_build_approved_treats_rows_unions_mesh_and_mined_ema_rows(disease_map: dict[str, dict[str, str]]) -> None:
    frame = _ema_frame(
        [_registry_row(active_substance="mitapivat sulfate", therapeutic_area_mesh="Anemia, Hemolytic", therapeutic_indication="Indicated for pain.")]
    )
    rows = build_approved_treats_rows(None, DailyMedEvidence(), {}, disease_map, ema_registry=frame, ner=_ner("pain"))
    by_pair = _rows_by_pair(rows)
    # The MeSH-area row (infores:ema) and the mined indication row (infores:epar) key separately.
    assert by_pair[("mitapivat sulfate", "Anemia, Hemolytic")]["upstream_resource_ids"] == "infores:ema"
    assert by_pair[("mitapivat sulfate", "pain")]["upstream_resource_ids"] == "infores:epar"
    keys = [(row["subject_text"], row["object_text"], row["upstream_resource_ids"]) for row in rows]
    assert keys == sorted(keys)


def test_transform_mines_indications_with_the_injected_ner(ctx: TaskContext, disease_map: dict[str, dict[str, str]]) -> None:
    ema_refs = ema_extract([ArtifactRef(uri=_EMA_FIXTURE, blake3=hash_file(_EMA_FIXTURE), media_type="application/octet-stream")], ctx)
    ner_ctx = TaskContext(
        workdir=ctx.workdir,
        fixture_root=ctx.fixture_root,
        params={"disease_map": disease_map, "ner": _ner("pyruvate kinase deficiency", "multiple myeloma")},
    )

    refs = ApprovedTreatsShaper().transform(ema_refs, ner_ctx)
    rows = schemas.read_table(refs[0].uri).to_dicts()
    by_pair = {(row["subject_text"], row["object_text"]): row for row in rows}
    assert len(rows) == 11  # 9 MeSH-area rows (infores:ema) + 2 mined rows (infores:epar)
    mined = by_pair[("mitapivat sulfate", "pyruvate kinase deficiency")]
    assert mined["upstream_resource_ids"] == "infores:epar"
    assert mined["FDA_regulatory_approvals"] == "EMEA/H/C/005540"
    assert mined["supporting_spl_documents"] == "https://www.ema.europa.eu/en/medicines/human/EPAR/pyrukynd"
    assert by_pair[("teclistamab", "multiple myeloma")]["upstream_resource_ids"] == "infores:epar"  # INN fallback subject
