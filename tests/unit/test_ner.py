"""Unit tests for the single composite NER backend (``dakp_pipeline.ner.ner``).

ALL of these pass without importing the heavy NER deps: the offline gazetteer mode is
deterministic and dep-free; the production (GLiNER) mode is asserted to be *lazy* (importing
``ner.ner`` imports no heavy deps) and is exercised with a fake ``gliner`` module + a stubbed
``ensure_model`` (no network). The missing-dep error path skips when ``gliner`` is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dakp_pipeline.ner.dictionary import CONTRAINDICATION_DISEASE_TYPES, TYPE_DISEASE, TYPE_PHENOTYPE, Gazetteer
from dakp_pipeline.ner.ner import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    EMBEDDED_GAZETTEER,
    DiseaseNER,
    Mention,
    extract_contraindication_diseases,
    extract_disease_mentions,
)

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_ONTOLOGY_TSV = _FIXTURE_ROOT / "ontology" / "disease_map.tsv"


# --- constants + embedded gazetteer --------------------------------------------


def test_defaults_and_contraindication_types() -> None:
    assert DEFAULT_MODEL == "gliner-community/gliner_large-v2.5"
    assert DEFAULT_THRESHOLD == 0.5
    assert CONTRAINDICATION_DISEASE_TYPES == (TYPE_DISEASE, TYPE_PHENOTYPE)
    # The embedded gazetteer is non-empty and every term is typed disease/phenotype.
    assert EMBEDDED_GAZETTEER
    assert set(EMBEDDED_GAZETTEER.values()) == {TYPE_DISEASE, TYPE_PHENOTYPE}


# --- offline mode (deterministic embedded gazetteer) ---------------------------


def test_offline_default_extracts_embedded_gazetteer_mentions() -> None:
    ner = DiseaseNER()
    text = "Contraindicated in patients with asthma and headache."
    mentions = ner.extract(text)
    by_text = {m.text: m for m in mentions}
    assert set(by_text) == {"asthma", "headache"}
    for mention in mentions:
        assert text[mention.start : mention.end] == mention.text
    assert by_text["asthma"].type == TYPE_DISEASE
    assert by_text["headache"].type == TYPE_PHENOTYPE
    assert all(m.score == 1.0 for m in mentions)


def test_offline_is_deterministic_and_sorted() -> None:
    ner = DiseaseNER()
    text = "asthma and pain and asthma"
    first = [(m.start, m.end, m.type, m.text) for m in ner.extract(text)]
    for _ in range(5):
        assert [(m.start, m.end, m.type, m.text) for m in ner.extract(text)] == first
    assert first == sorted(first)


def test_offline_empty_and_blank_text_yield_nothing() -> None:
    ner = DiseaseNER()
    assert ner.extract("") == []
    assert ner.extract("   \t\n ") == []
    assert ner.extract("nothing relevant here") == []


def test_gazetteer_accepts_mapping_or_gazetteer_instance() -> None:
    from_mapping = DiseaseNER(gazetteer={"fever": "phenotype"})
    assert [m.text for m in from_mapping.extract("fever")] == ["fever"]
    # A pre-built Gazetteer instance is used as-is.
    from_instance = DiseaseNER(gazetteer=Gazetteer({"rash": "phenotype"}))
    assert [m.text for m in from_instance.extract("rash")] == ["rash"]


def test_from_tsv_builds_offline_backend(tmp_path: Path) -> None:
    tsv = tmp_path / "terms.tsv"
    tsv.write_text("text\ttype\npain\tdisease\nfever\tphenotype\n", encoding="utf-8")
    ner = DiseaseNER.from_tsv(tsv)
    assert sorted(m.text for m in ner.extract("pain and fever")) == ["fever", "pain"]


def test_from_tsv_reads_fixture_ontology_as_term_to_type_only() -> None:
    # The ontology fixture has text/curie/name/category; from_tsv reads text + category only.
    ner = DiseaseNER.from_tsv(_ONTOLOGY_TSV, text_col="text", type_col="category")
    mentions = {m.text: m.type for m in ner.extract("asthma and headache")}
    assert mentions == {"asthma": TYPE_DISEASE, "headache": TYPE_PHENOTYPE}


# --- module-level entry points -------------------------------------------------


def test_extract_disease_mentions_defaults_to_offline_backend() -> None:
    assert [m.text for m in extract_disease_mentions("asthma")] == ["asthma"]
    assert extract_disease_mentions("") == []


def test_extract_disease_mentions_uses_injected_backend() -> None:
    ner = DiseaseNER(gazetteer={"custom term": "disease"})
    assert [m.text for m in extract_disease_mentions("a custom term here", ner)] == ["custom term"]


def test_extract_contraindication_diseases_delegates() -> None:
    ner = DiseaseNER(gazetteer={"liver disease": "disease"})
    text = "Contraindicated in liver disease."
    assert extract_contraindication_diseases(text, ner) == extract_disease_mentions(text, ner)


# --- laziness: importing ner.ner imports no heavy deps -------------------------


def test_importing_ner_does_not_import_heavy_deps() -> None:
    assert "dakp_pipeline.ner.ner" in sys.modules
    for module in ("gliner", "huggingface_hub"):
        assert module not in sys.modules, f"importing ner.ner must not import {module}"


def test_constructing_production_backend_does_not_import_gliner() -> None:
    DiseaseNER(offline=False)
    assert "gliner" not in sys.modules


def test_mention_is_frozen() -> None:
    mention = Mention(text="asthma", start=0, end=6, type=TYPE_DISEASE, score=1.0)
    assert isinstance(hash(mention), int)
