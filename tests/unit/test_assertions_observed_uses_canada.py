"""Tests for the Canada Vigilance pass in ``dakp_pipeline.assertions.observed_uses``.

CV observations join the SAME ``faers_applied_to_treat_assertions`` table as FAERS with
source-partitioned provenance. These tests pin: the ingredient subject (brand fallback), the
distinct-report case count, the corroboration-derived ``clinical_approval_status``, the
``infores:canada-vigilance|infores:dailymed`` upstream chain, row separation from FAERS rows
for the same triple, and the Translator regression contract.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.assertions.observed_uses import ApprovedTreatsIndex, _canada_vigilance_rows, build_observed_use_rows
from dakp_pipeline.extract.canada_vigilance import CV_INDICATIONS_COLUMNS
from dakp_pipeline.sources.canada_vigilance import CANADA_VIGILANCE_EXTRACTS_URL
from dakp_pipeline.translator import check_rows


def _cv_frame(rows: list[dict[str, str]]) -> pl.DataFrame:
    return pl.DataFrame([{**dict.fromkeys(CV_INDICATIONS_COLUMNS, ""), **row} for row in rows], schema=dict.fromkeys(CV_INDICATIONS_COLUMNS, pl.Utf8))


def _observation(report_id: str, ingredient: str, indication: str, drugname: str = "KEMPLAR") -> dict[str, str]:
    return {
        "source_record_id": f"cv:{report_id}:1:{ingredient or drugname}",
        "report_id": report_id,
        "report_drug_id": "1",
        "drug_product_id": "10",
        "drugname": drugname,
        "ingredient": ingredient,
        "indication": indication,
    }


def test_canada_rows_use_ingredient_subject_and_report_counts() -> None:
    """The subject is the active ingredient; number_of_cases counts DISTINCT report ids."""
    frame = _cv_frame(
        [
            _observation("100001", "methotrexate", "rheumatoid arthritis"),
            _observation("100001", "methotrexate", "rheumatoid arthritis"),  # same report: one case
            _observation("100002", "methotrexate", "rheumatoid arthritis"),
        ]
    )
    rows = _canada_vigilance_rows(frame, {}, None)
    assert len(rows) == 1
    row = rows[0]
    assert row["subject_text"] == "methotrexate"
    assert row["object_text"] == "rheumatoid arthritis"
    assert row["number_of_cases"] == "2"
    assert sorted(row["case_ids"].split("|")) == ["100001", "100002"]
    assert row["assertion_context"] == "indication"
    assert row["knowledge_level"] == "statistical_association"
    assert row["upstream_resource_ids"] == "infores:canada-vigilance|infores:dailymed"
    assert row["supporting_faers_records"].startswith("cv:")
    assert row["supporting_faers_urls"] == CANADA_VIGILANCE_EXTRACTS_URL
    assert row["primary_knowledge_source"] == "infores:drugapprovals-kp"


def test_canada_rows_fall_back_to_brand_subject() -> None:
    """A product the ingredients member does not cover keeps the brand name as the subject."""
    frame = _cv_frame([_observation("100001", "", "migraine", drugname="GHOSTDRUG")])
    rows = _canada_vigilance_rows(frame, {}, None)
    assert len(rows) == 1
    assert rows[0]["subject_text"] == "GHOSTDRUG"


def test_canada_status_derives_from_the_approved_pair_index() -> None:
    """approved_for_condition when the pair matches (normalized), off_label_use otherwise."""
    frame = _cv_frame([_observation("100001", "methotrexate", "rheumatoid arthritis"), _observation("100002", "methotrexate", "migraine")])
    rows = _canada_vigilance_rows(frame, {}, ApprovedTreatsIndex(pairs=frozenset({("methotrexate", "rheumatoid arthritis")})))
    by_object = {row["object_text"]: row for row in rows}
    assert by_object["rheumatoid arthritis"]["clinical_approval_status"] == "approved_for_condition"
    assert by_object["migraine"]["clinical_approval_status"] == "off_label_use"
    # Degraded mode: no approved table -> not_provided (never an off-label claim).
    degraded = _canada_vigilance_rows(frame, {}, None)
    assert {row["clinical_approval_status"] for row in degraded} == {"not_provided"}


def test_canada_and_faers_rows_for_same_triple_stay_separate(disease_map: dict[str, dict[str, str]]) -> None:
    """Same (subject, object) from both sources: two rows, one upstream chain each."""
    faers_cases = pl.DataFrame(
        {"primaryid": ["1", "2"], "drugname": ["Methotrexate", "Methotrexate"], "indication": ["rheumatoid arthritis", "rheumatoid arthritis"]}
    )
    cv_frame = _cv_frame([_observation("100001", "methotrexate", "rheumatoid arthritis")])
    rows = build_observed_use_rows(faers_cases, disease_map, cv_indications=cv_frame)
    by_upstream = {row["upstream_resource_ids"]: row for row in rows}
    assert set(by_upstream) == {"infores:faers|infores:dailymed", "infores:canada-vigilance|infores:dailymed"}
    # Deterministic total order: (subject, object, context, upstream) — the raw-spelling FAERS
    # row ("Methotrexate") precedes the CV row ("methotrexate").
    assert [row["subject_text"] for row in rows] == ["Methotrexate", "methotrexate"]
    assert [row["upstream_resource_ids"] for row in rows] == ["infores:faers|infores:dailymed", "infores:canada-vigilance|infores:dailymed"]


def test_translator_accepts_canada_observed_use_row() -> None:
    """The regression contract accepts the CV chain and still rejects foreign upstreams."""
    frame = _cv_frame([_observation("100001", "methotrexate", "rheumatoid arthritis")])
    row = _canada_vigilance_rows(frame, {}, None)[0]
    report = check_rows([row])
    assert report.ok

    foreign = dict(row, upstream_resource_ids="infores:medi")
    assert not check_rows([foreign]).ok
