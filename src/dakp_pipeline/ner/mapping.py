"""Canonical mapping backend interface + a deterministic mocked fullmap backend.

Tablassert/fullmap is the canonical resolver (PLAN.md "Phase 4" step 3 and "Phase 5"):
it resolves mention/entity text to canonical CURIE/name/category/taxon/source metadata.
The real Tablassert wiring lands later; for this milestone DAKP defines the *interface*
that resolver must satisfy (:class:`MappingBackend`, shaped like fullmap ``resolve_many``)
and a deterministic, fixture-driven :class:`MockFullmapBackend` so the whole NER/mapping
layer is testable with no real fullmap/Tablassert present.

Mention generation (:mod:`dakp_pipeline.ner.lexical` / :mod:`dakp_pipeline.ner.candidates`)
stays decoupled from final mapping: a backend is an optional, dependency-injected seam
that can be monkeypatched (any object satisfying :class:`MappingBackend`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

from dakp_pipeline.ner.dictionary import infer_source, normalize_text

# Default taxon for the mock backend (human), matching fullmap's typical disease/phenotype
# resolution. Overridable per-term via a taxon column.
DEFAULT_TAXON = "NCBITaxon:9606"


@dataclass(frozen=True)
class MappedTerm:
    """One canonical resolution result — the fullmap output shape.

    ``curie``/``name``/``category`` are the canonical node; ``taxon`` is the resolved
    taxon (empty string if none); ``source`` is the contributing resource
    (``fullmap``/``MONDO``/``HPO``/...).
    """

    curie: str
    name: str
    category: str
    taxon: str
    source: str


@runtime_checkable
class MappingBackend(Protocol):
    """Canonical text -> term resolver (shaped like fullmap ``resolve_many``).

    Implementations must be deterministic and monkeypatch-friendly. Real Tablassert/
    fullmap will satisfy this later; :class:`MockFullmapBackend` satisfies it now.
    """

    def resolve_many(self, texts: Sequence[str]) -> dict[str, list[MappedTerm]]:
        """Resolve unique mention strings to canonical terms.

        Returns a mapping keyed by the exact input strings that resolved; input strings
        with no resolution are omitted. Results per string are deterministically ordered.
        """
        ...


class MockFullmapBackend:
    """Deterministic, fixture-driven stand-in for Tablassert/fullmap ``resolve_many``.

    Builds a normalized-text -> candidate index from a term table (the pipeline ontology
    fixture today; a real fullmap export later) and resolves by normalized phrase match.
    """

    def __init__(self, terms: Iterable[MappedTerm]) -> None:
        by_text: dict[str, list[MappedTerm]] = {}
        for term in terms:
            key = normalize_text(term.name)
            if not key:
                continue
            by_text.setdefault(key, []).append(term)
        self._by_text: dict[str, tuple[MappedTerm, ...]] = {key: tuple(sorted(set(group), key=_term_sort_key)) for key, group in by_text.items()}

    # -- builders --------------------------------------------------------------
    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        name_col: str = "name",
        curie_col: str = "curie",
        category_col: str = "category",
        taxon_col: str | None = None,
        source_col: str | None = None,
        default_taxon: str = DEFAULT_TAXON,
    ) -> MockFullmapBackend:
        """Build a mock backend from a fullmap-shaped table.

        Uses ``name`` as the resolvable text (falling back to a ``text`` column if present,
        to accept the ontology fixture shape). ``taxon``/``source`` come from optional
        columns; taxon defaults to ``default_taxon`` and source is inferred from the CURIE
        prefix when absent.
        """
        has_text = "text" in frame.columns
        terms: list[MappedTerm] = []
        for row in frame.iter_rows(named=True):
            curie = str(row.get(curie_col) or "").strip()
            name = str(row.get(name_col) or "").strip()
            if not name and has_text:
                name = str(row.get("text") or "").strip()
            if not name:
                continue
            category = str(row.get(category_col) or "").strip()
            taxon = str(row.get(taxon_col) or "").strip() if taxon_col else ""
            source = str(row.get(source_col) or "").strip() if source_col else ""
            terms.append(MappedTerm(curie=curie, name=name, category=category, taxon=taxon or default_taxon, source=source or infer_source(curie)))
        return cls(terms)

    @classmethod
    def from_tsv(cls, path: Path, **kwargs: object) -> MockFullmapBackend:
        """Build a mock backend from an uncompressed fullmap-shaped TSV."""
        return cls.from_frame(pl.read_csv(path, separator="\t"), **kwargs)  # type: ignore[arg-type]

    # -- resolution ------------------------------------------------------------
    def resolve_many(self, texts: Sequence[str]) -> dict[str, list[MappedTerm]]:
        """Resolve unique mention strings by normalized phrase match (deterministic).

        Keyed by the exact input strings that resolved; unresolvable inputs are omitted.
        """
        resolved: dict[str, list[MappedTerm]] = {}
        for text in texts:
            terms = self._by_text.get(normalize_text(text))
            if terms:
                resolved[text] = list(terms)
        return resolved

    def __len__(self) -> int:
        """Number of distinct resolvable normalized strings."""
        return len(self._by_text)


def _term_sort_key(term: MappedTerm) -> tuple[str, str, str, str]:
    return (term.source, term.category, term.curie, term.name)


__all__ = ["DEFAULT_TAXON", "MappedTerm", "MappingBackend", "MockFullmapBackend"]
