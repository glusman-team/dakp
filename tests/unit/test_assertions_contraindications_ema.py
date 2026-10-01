"""EU SmPC pass tests for ``dakp_pipeline.assertions.contraindications`` (Pass 4).

The SmPC pass joins ``smpc_sections.parquet`` (extract.ema_smpc) to ``ema_registry.parquet``
on ``ema_product_number`` and mines EU product-information sections through the SAME
work-item pool, acceptance rules, and aggregation machinery as the DailyMed passes. These
tests pin the pass-specific contracts:

- provenance: ``infores:epar`` upstream, EMA product number in ``FDA_regulatory_approvals``,
  SmPC document URL in ``supporting_spl_documents``, DailyMed-shaped columns empty (including
  ``edge_evidence``, whose ``dailymed:`` CURIE prefix would mislabel an EU document);
- the singleton rule carries over: multi-substance products and products absent from the
  authorised-medicines registry contribute no subject, hence no rows (INN fallback applies);
- section semantics match the DailyMed passes: dedicated 4.3 sections mine in full,
  indication sections are keyword-filtered AND hard-trigger-only, warnings sections are
  hard-trigger-only (soft caution language never becomes an edge);
- a DailyMed row and an EU row for the same (subject, object, context) stay SEPARATE rows
  (the aggregation key partitions by source, so neither provenance chain is diluted);
- the Translator regression contract accepts the EU chain for ``contraindicated_in``.

Inputs are tiny parquet tables built in tmp so no heavy NER deps are needed.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dakp_pipeline.assertions import KL_ASSERTION
from dakp_pipeline.assertions.contraindications import _smpc_work_items, build_contraindication_rows
from dakp_pipeline.extract.ema_registry import EMA_REGISTRY_COLUMNS
from dakp_pipeline.extract.ema_smpc import SMPC_SECTIONS_COLUMNS
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef
from dakp_pipeline.ner.ner import DiseaseNER
from dakp_pipeline.translator import check_rows

NER = DiseaseNER(gazetteer={"asthma": "disease"})
PRODUCT = "EMEA/H/C/000796"
SMPD_URL = f"https://www.ema.europa.eu/en/documents/product-information/{PRODUCT}-product-information_en.pdf"


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


SectionRow = tuple[str, str, str] | tuple[str, str, str, str]


def _sections_ref(tmp_path: Path, rows: list[SectionRow]) -> ArtifactRef:
    """smpc_sections.parquet from (product_number, section_kind, section_text[, url]) rows."""

    def row_url(row: SectionRow) -> str:
        # Real crawls carry one document URL per product-document; an explicit fourth element
        # models a second document version (distinct URL, same product + kind).
        return str(row[3]) if len(row) > 3 else SMPD_URL

    frame = pl.DataFrame(
        {
            "source_record_id": [f"{r[0]}::{r[1]}" for r in rows],
            "ema_product_number": [r[0] for r in rows],
            "medicine_name": ["Medicine"] * len(rows),
            "section_kind": [r[1] for r in rows],
            "section_title": ["Section"] * len(rows),
            "section_text": [r[2] for r in rows],
            "document_url": [row_url(r) for r in rows],
            "last_updated_date": ["2024-01-01"] * len(rows),
            "pdf_path": [f"store/smpc/x{i}.pdf" for i in range(len(rows))],
        }
    )
    frame = frame.select(SMPC_SECTIONS_COLUMNS)
    path = tmp_path / "smpc_sections.parquet"
    frame.write_parquet(path)
    return _ref(path)


def _registry_ref(tmp_path: Path, rows: list[tuple[str, str, str]]) -> ArtifactRef:
    """ema_registry.parquet from (product_number, active_substance, inn) rows."""
    frame = pl.DataFrame(
        {
            "medicine_name": ["Medicine"] * len(rows),
            "ema_product_number": [r[0] for r in rows],
            "category": ["Human"] * len(rows),
            "medicine_status": ["Authorised"] * len(rows),
            "inn": [r[2] for r in rows],
            "active_substance": [r[1] for r in rows],
            "therapeutic_area_mesh": [""] * len(rows),
            "therapeutic_indication": [""] * len(rows),
            "medicine_url": ["https://epar"] * len(rows),
        }
    ).select(EMA_REGISTRY_COLUMNS)
    path = tmp_path / "ema_registry.parquet"
    frame.write_parquet(path)
    return _ref(path)


def _smpc_inputs(tmp_path: Path, sections: list[SectionRow], registry: list[tuple[str, str, str]]) -> list[ArtifactRef]:
    return [_sections_ref(tmp_path, sections), _registry_ref(tmp_path, registry)]


def _dailymed_inputs(tmp_path: Path, set_id: str = "SET-A") -> list[ArtifactRef]:
    sections = tmp_path / "spl_sections.parquet"
    pl.DataFrame(
        {"spl_set_id": [set_id], "spl_document_id": ["DOC-A"], "clean_text": ["Contraindicated in patients with asthma."], "loinc_code": ["34070-3"]}
    ).write_parquet(sections)
    ingredients = tmp_path / "spl_ingredients.parquet"
    pl.DataFrame({"role": ["active"], "spl_set_id": [set_id], "ingredient_name": ["Cladribine"], "ingredient_unii": ["UNII:GZ6"]}).write_parquet(
        ingredients
    )
    return [_ref(sections), _ref(ingredients)]


def test_smpc_contraindication_row_provenance(tmp_path: Path) -> None:
    """A dedicated 4.3 section mines in full; the EU row carries the approved-treats EMA provenance."""
    inputs = _smpc_inputs(
        tmp_path,
        [(PRODUCT, "contraindications", "This medicinal product is contraindicated in patients with asthma.")],
        [(PRODUCT, "Histamine dihydrochloride", "")],
    )
    rows = build_contraindication_rows(inputs, NER)
    assert len(rows) == 1
    row = rows[0]
    assert row["subject_text"] == "Histamine dihydrochloride"
    assert row["subject_curie"] == ""  # no UNII: EMA substance text only
    assert row["subject_category"] == "ChemicalEntity"
    assert row["predicate"] == "biolink:contraindicated_in"
    assert row["object_text"] == "asthma"
    assert row["assertion_context"] == "contraindication"
    assert row["FDA_regulatory_approvals"] == PRODUCT
    assert row["supporting_spl_documents"] == SMPD_URL
    assert row["supporting_spl_sets"] == ""
    assert row["supporting_spl_evidence"] == ""
    assert row["edge_evidence"] == ""  # a dailymed: CURIE would mislabel an EU document
    assert row["upstream_resource_ids"] == "infores:epar"
    assert row["primary_knowledge_source"] == "infores:drugapprovals-kp"
    assert row["knowledge_level"] == KL_ASSERTION


def test_smpc_multi_substance_product_skipped(tmp_path: Path) -> None:
    """Combination products have no single attributable subject (singleton rule)."""
    inputs = _smpc_inputs(
        tmp_path, [(PRODUCT, "contraindications", "Contraindicated in patients with asthma.")], [(PRODUCT, "Histamine; Serotonin", "")]
    )
    assert build_contraindication_rows(inputs, NER) == []


def test_smpc_unmapped_product_skipped(tmp_path: Path) -> None:
    """A product absent from the authorised-medicines registry has no resolvable subject."""
    inputs = _smpc_inputs(
        tmp_path, [(PRODUCT, "contraindications", "Contraindicated in patients with asthma.")], [("EMEA/H/C/999999", "Histamine", "")]
    )
    assert build_contraindication_rows(inputs, NER) == []


def test_smpc_inn_fallback_subject(tmp_path: Path) -> None:
    """An empty active_substance cell falls back to the INN (approved-treats convention)."""
    inputs = _smpc_inputs(tmp_path, [(PRODUCT, "contraindications", "Contraindicated in patients with asthma.")], [(PRODUCT, "", "Histamine")])
    rows = build_contraindication_rows(inputs, NER)
    assert len(rows) == 1
    assert rows[0]["subject_text"] == "Histamine"


def test_smpc_indication_section_keyword_and_hard_trigger(tmp_path: Path) -> None:
    """Indication sections are keyword-filtered (Pass 2 semantics) AND hard-trigger-only."""
    inputs = _smpc_inputs(
        tmp_path,
        [
            (
                PRODUCT,
                "indications",
                "Treatment of patients with asthma.\n\nThis medicinal product must not be used in patients with uncontrolled asthma.",
            )
        ],
        [(PRODUCT, "Histamine", "")],
    )
    rows = build_contraindication_rows(inputs, NER)
    assert len(rows) == 1
    # Only the keyword sentence ("must not be used") is mined; plain indication prose is not.
    assert "asthma" in rows[0]["object_text"]
    assert "Treatment of patients" not in rows[0]["evidence_text"]
    assert "must not be used" in rows[0]["evidence_text"]


def test_smpc_warnings_soft_language_is_not_an_edge(tmp_path: Path) -> None:
    """Warnings sections are hard-trigger-only (Pass 3 semantics): soft caution never becomes an edge."""
    inputs = _smpc_inputs(
        tmp_path,
        [
            (PRODUCT, "warnings", "Asthma may worsen; monitor patients carefully.", SMPD_URL.replace("_en.pdf", "-v1_en.pdf")),
            (PRODUCT, "warnings", "Do not use in patients with asthma during pregnancy.", SMPD_URL.replace("_en.pdf", "-v2_en.pdf")),
        ],
        [(PRODUCT, "Histamine", "")],
    )
    rows = build_contraindication_rows(inputs, NER)
    assert len(rows) == 1
    assert "pregnancy" in rows[0]["evidence_text"]
    assert "monitor" not in rows[0]["evidence_text"]


def test_smpc_explicit_negation_is_not_an_edge(tmp_path: Path) -> None:
    inputs = _smpc_inputs(
        tmp_path, [(PRODUCT, "warnings", "There is no known contraindication in patients with asthma.")], [(PRODUCT, "Histamine", "")]
    )
    assert build_contraindication_rows(inputs, NER) == []


def test_dailymed_and_smpc_rows_for_same_triple_stay_separate(tmp_path: Path) -> None:
    """Same (subject-free) observation from both label worlds: two rows, one upstream chain each."""
    inputs = [
        *_dailymed_inputs(tmp_path),
        *_smpc_inputs(tmp_path, [(PRODUCT, "contraindications", "Contraindicated in patients with asthma.")], [(PRODUCT, "Cladribine", "")]),
    ]
    rows = build_contraindication_rows(inputs, NER)
    assert len(rows) == 2
    by_upstream = {row["upstream_resource_ids"]: row for row in rows}
    assert set(by_upstream) == {"infores:dailymed", "infores:epar"}
    assert by_upstream["infores:dailymed"]["subject_curie"] == "UNII:GZ6"
    assert by_upstream["infores:dailymed"]["supporting_spl_sets"].startswith("https://dailymed.nlm.nih.gov")
    assert by_upstream["infores:epar"]["subject_curie"] == ""
    assert by_upstream["infores:epar"]["FDA_regulatory_approvals"] == PRODUCT


def test_smpc_work_items_dedup_and_counters(tmp_path: Path) -> None:
    """Duplicate section rows collapse; skipped products are counted, not silently dropped."""
    from dakp_pipeline.assertions.contraindications import DEFAULT_CONTRA_KEYWORDS

    inputs = _smpc_inputs(
        tmp_path,
        [
            (PRODUCT, "contraindications", "Contraindicated in patients with asthma."),
            (PRODUCT, "contraindications", "Contraindicated in patients with asthma."),  # duplicate row
            ("EMEA/H/C/000001", "contraindications", "Contraindicated in patients with asthma."),  # multi-substance
            ("EMEA/H/C/000002", "contraindications", "Contraindicated in patients with asthma."),  # unmapped
            (PRODUCT, "unknown_kind", "irrelevant"),  # unknown section kind
            (PRODUCT, "warnings", "   "),  # empty text
        ],
        [(PRODUCT, "Histamine", ""), ("EMEA/H/C/000001", "A; B", "")],
    )
    items, subjects, counters = _smpc_work_items(inputs, DEFAULT_CONTRA_KEYWORDS)
    assert counters == {"sections": 1, "skipped_multi_substance": 1, "skipped_unmapped": 1}
    assert len(items) == 1
    assert items[0].set_id == f"smpc:{PRODUCT}"
    assert items[0].doc_id == SMPD_URL
    assert subjects == {f"smpc:{PRODUCT}": ("Histamine", "")}


def test_translator_accepts_smpc_contraindication_row(tmp_path: Path) -> None:
    """The regression contract accepts the EU chain and still rejects foreign upstreams."""
    inputs = _smpc_inputs(tmp_path, [(PRODUCT, "contraindications", "Contraindicated in patients with asthma.")], [(PRODUCT, "Histamine", "")])
    row = build_contraindication_rows(inputs, NER)[0]
    report = check_rows([row])
    assert report.ok
    assert report.families_seen == ["biolink:contraindicated_in"]

    foreign = dict(row, upstream_resource_ids="infores:medi")
    assert not check_rows([foreign]).ok
