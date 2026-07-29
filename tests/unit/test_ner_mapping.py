"""Unit tests for the canonical mapping backend interface + mocked fullmap backend.

Covers: fixture-driven construction (pipeline ontology fixture), the fullmap-shaped
resolution output (curie/name/category/taxon/source), a resolve_many round-trip on real
mention strings, normalized (case-insensitive) matching, omission of unresolvable inputs,
determinism, Protocol conformance, and monkeypatch-friendly substitution.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from dakp_pipeline.ner.mapping import DEFAULT_TAXON, MappedTerm, MappingBackend, MockFullmapBackend

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_ONTOLOGY_TSV = _FIXTURE_ROOT / "ontology" / "disease_map.tsv"


# --- fixture-driven construction ----------------------------------------------


def test_mock_backend_loads_from_ontology_fixture() -> None:
    backend = MockFullmapBackend.from_tsv(_ONTOLOGY_TSV)
    assert len(backend) == 5  # 5 distinct resolvable strings in disease_map.tsv


def test_mock_backend_accepts_text_only_frame() -> None:
    # A fullmap export may key on `text`; `name` falls back to the text column.
    frame = pl.DataFrame({"text": ["asthma"], "curie": ["MONDO:0004979"], "category": ["Disease"]})
    backend = MockFullmapBackend.from_frame(frame)
    assert backend.resolve_many(["asthma"])["asthma"][0].curie == "MONDO:0004979"


# --- resolution round-trip -----------------------------------------------------


def test_resolve_many_round_trip_returns_fullmap_shape() -> None:
    backend = MockFullmapBackend.from_tsv(_ONTOLOGY_TSV)
    resolved = backend.resolve_many(["hypercholesterolemia", "headache"])
    assert set(resolved) == {"hypercholesterolemia", "headache"}

    (chol,) = resolved["hypercholesterolemia"]
    assert chol == MappedTerm(curie="MONDO:0005154", name="hypercholesterolemia", category="Disease", taxon=DEFAULT_TAXON, source="MONDO")
    (head,) = resolved["headache"]
    assert head.curie == "HP:0002315"
    assert head.source == "HPO"
    assert head.taxon == DEFAULT_TAXON  # default human taxon when none supplied


def test_resolve_many_is_case_and_punctuation_insensitive() -> None:
    backend = MockFullmapBackend.from_tsv(_ONTOLOGY_TSV)
    resolved = backend.resolve_many(["  Peptic-Ulcer Disease "])
    # Keyed by the exact input string; matched by normalized form.
    assert "  Peptic-Ulcer Disease " in resolved
    assert resolved["  Peptic-Ulcer Disease "][0].curie == "MONDO:0005194"


def test_resolve_many_omits_unresolvable_inputs() -> None:
    backend = MockFullmapBackend.from_tsv(_ONTOLOGY_TSV)
    resolved = backend.resolve_many(["asthma", "not-a-real-term", ""])
    assert set(resolved) == {"asthma"}


def test_resolve_many_empty_input_returns_empty_dict() -> None:
    backend = MockFullmapBackend.from_tsv(_ONTOLOGY_TSV)
    assert backend.resolve_many([]) == {}


def test_taxon_and_source_columns_override_defaults() -> None:
    frame = pl.DataFrame({"name": ["asthma"], "curie": ["MONDO:0004979"], "category": ["Disease"], "taxon": ["NCBITaxon:10090"], "src": ["fullmap"]})
    backend = MockFullmapBackend.from_frame(frame, taxon_col="taxon", source_col="src")
    (term,) = backend.resolve_many(["asthma"])["asthma"]
    assert term.taxon == "NCBITaxon:10090"
    assert term.source == "fullmap"


def test_multiple_terms_per_text_ordered_deterministically() -> None:
    terms = [
        MappedTerm("MONDO:1", "fever", "Disease", DEFAULT_TAXON, "MONDO"),
        MappedTerm("HP:1", "fever", "PhenotypicFeature", DEFAULT_TAXON, "HPO"),
    ]
    backend = MockFullmapBackend(terms)
    got = backend.resolve_many(["fever"])["fever"]
    assert [t.source for t in got] == ["HPO", "MONDO"]  # sorted by (source, category, curie)


# --- determinism ---------------------------------------------------------------


def test_resolve_many_is_deterministic() -> None:
    backend = MockFullmapBackend.from_tsv(_ONTOLOGY_TSV)
    texts = ["pain", "asthma", "headache", "hypercholesterolemia"]
    first = backend.resolve_many(texts)
    for _ in range(5):
        assert backend.resolve_many(texts) == first


# --- protocol conformance & monkeypatchability --------------------------------


def test_mock_backend_satisfies_mapping_backend_protocol() -> None:
    backend = MockFullmapBackend.from_tsv(_ONTOLOGY_TSV)
    assert isinstance(backend, MappingBackend)


def test_backend_is_substitutable_with_a_stub(monkeypatch) -> None:
    """Any object satisfying the Protocol can replace the mock (monkeypatch seam)."""

    class StubBackend:
        def resolve_many(self, texts: Sequence[str]) -> dict[str, list[MappedTerm]]:
            return {t: [MappedTerm("STUB:1", t, "Disease", DEFAULT_TAXON, "stub")] for t in texts}

    stub = StubBackend()
    assert isinstance(stub, MappingBackend)

    # Demonstrate the injection seam: a module-level default can be monkeypatched.
    import dakp_pipeline.ner.mapping as mapping_mod

    monkeypatch.setattr(mapping_mod, "DEFAULT_TAXON", "NCBITaxon:0")
    assert mapping_mod.DEFAULT_TAXON == "NCBITaxon:0"
    assert stub.resolve_many(["x"])["x"][0].curie == "STUB:1"
