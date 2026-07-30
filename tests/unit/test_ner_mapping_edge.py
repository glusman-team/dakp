"""Edge-case tests for ``dakp_pipeline.ner.mapping`` (drive to 100% branch coverage).

Covers the two uncovered ``continue`` branches (a term whose name normalizes to empty in the
constructor; a frame row with neither a name nor a text column) plus adversarial resolution:
unresolvable inputs omitted, normalized matching, determinism/dedup, taxon/source column
handling and CURIE-prefix source inference.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.ner.mapping import DEFAULT_TAXON, MappedTerm, MockFullmapBackend


def _term(name: str, curie: str = "MONDO:1", category: str = "Disease", taxon: str = "", source: str = "MONDO") -> MappedTerm:
    return MappedTerm(curie=curie, name=name, category=category, taxon=taxon, source=source)


# --- constructor: empty-name skip + dedup/ordering ------------------------------


def test_constructor_skips_terms_whose_name_normalizes_to_empty() -> None:
    backend = MockFullmapBackend([_term("!!!"), _term(""), _term("asthma")])
    # The two blank-name terms are dropped; only 'asthma' is resolvable.
    assert len(backend) == 1
    assert backend.resolve_many(["asthma"]) != {}
    assert backend.resolve_many(["!!!"]) == {}


def test_constructor_dedups_and_orders_terms_per_key() -> None:
    backend = MockFullmapBackend([_term("asthma", curie="MONDO:2"), _term("asthma", curie="MONDO:1"), _term("asthma", curie="MONDO:1")])
    resolved = backend.resolve_many(["asthma"])["asthma"]
    assert [t.curie for t in resolved] == ["MONDO:1", "MONDO:2"]  # deduped + sorted


# --- from_frame: missing name/text skip + columns -------------------------------


def test_from_frame_skips_rows_without_name_or_text() -> None:
    frame = pl.DataFrame({"name": ["asthma", "", "   "], "curie": ["MONDO:1", "MONDO:2", "MONDO:3"], "category": ["Disease", "Disease", "Disease"]})
    backend = MockFullmapBackend.from_frame(frame)
    assert len(backend) == 1
    assert backend.resolve_many(["asthma"]) != {}


def test_from_frame_falls_back_to_text_column_when_name_missing() -> None:
    frame = pl.DataFrame(
        {
            "name": ["", "headache"],
            "text": ["asthma", "ignored"],  # used only where name is blank
            "curie": ["MONDO:1", "HP:1"],
            "category": ["Disease", "PhenotypicFeature"],
        }
    )
    backend = MockFullmapBackend.from_frame(frame)
    assert set(backend.resolve_many(["asthma", "headache"])) == {"asthma", "headache"}


def test_from_frame_taxon_and_source_columns_with_defaults() -> None:
    frame = pl.DataFrame(
        {
            "name": ["asthma", "pain"],
            "curie": ["MONDO:1", "UNKNOWN:2"],
            "category": ["Disease", "Disease"],
            "taxon": ["NCBITaxon:10090", ""],  # second falls back to the default taxon
            "src": ["", ""],  # blank -> inferred from the CURIE prefix
        }
    )
    backend = MockFullmapBackend.from_frame(frame, taxon_col="taxon", source_col="src")
    asthma = backend.resolve_many(["asthma"])["asthma"][0]
    pain = backend.resolve_many(["pain"])["pain"][0]
    assert asthma.taxon == "NCBITaxon:10090"
    assert asthma.source == "MONDO"  # inferred from MONDO: prefix
    assert pain.taxon == DEFAULT_TAXON  # blank taxon -> default
    assert pain.source == "fullmap"  # UNKNOWN: prefix -> fullmap


def test_from_frame_custom_default_taxon() -> None:
    frame = pl.DataFrame({"name": ["asthma"], "curie": ["MONDO:1"], "category": ["Disease"]})
    backend = MockFullmapBackend.from_frame(frame, default_taxon="NCBITaxon:custom")
    assert backend.resolve_many(["asthma"])["asthma"][0].taxon == "NCBITaxon:custom"


# --- resolve_many behavior ------------------------------------------------------


def test_resolve_many_omits_unresolvable_and_matches_normalized() -> None:
    backend = MockFullmapBackend([_term("asthma")])
    resolved = backend.resolve_many(["ASTHMA", "unknown", "  asthma  ", ""])
    # Keyed by the EXACT input strings that resolved; case/whitespace-insensitive match.
    assert set(resolved) == {"ASTHMA", "  asthma  "}
    assert "unknown" not in resolved
    assert "" not in resolved


def test_resolve_many_is_deterministic() -> None:
    backend = MockFullmapBackend([_term("asthma", curie="MONDO:2"), _term("asthma", curie="MONDO:1")])
    first = backend.resolve_many(["asthma"])
    for _ in range(5):
        assert backend.resolve_many(["asthma"]) == first
