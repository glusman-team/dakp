"""Tiny hermetic fullmap builder for the DAKP -> Tablassert -> KGX end-to-end test.

Writes SMALL Tablassert ``classes`` (KGX node records) + ``synonyms`` NDJSON files covering
the DAKP assertion-table vocabulary, then builds a ``fullmap.redb`` via
``tablassert.rs.build_fullmap_db`` — the SAME file formats the Tablassert ``fullmap_db`` test
fixture uses (``tests/test_fullmap.py`` in ``../Tablassert``): a class row is
``{"id": curie, "equivalent_identifiers": [{"identifier": x}, ...]}`` and a synonym row is
``{"curie", "preferred_name", "names": [...], "types": [category], "taxa": [taxon]}`` (bare
category names; Tablassert adds the ``biolink:`` prefix on resolution).

The fullmap maps the assertion-table MENTION TEXT (``subject_text`` / ``object_text``) to
canonical CURIEs + categories so ``tablassert build-kg`` can resolve them:

======================  ==============  =================
mention text            CURIE           category
======================  ==============  =================
Ibuprofen / Advil       CHEBI:5855      SmallMolecule
Examplestatin           CHEBI:1000001   SmallMolecule (fictional stand-in)
hypercholesterolemia    MONDO:0005154   Disease
headache                HP:0002315      PhenotypicFeature
pain                    HP:0012531      PhenotypicFeature
asthma                  MONDO:0004979   Disease
peptic ulcer disease    MONDO:0005011   Disease
fipronil                CHEBI:83394     SmallMolecule     (veterinary fixture)
renal disease           MONDO:0005240   Disease           (veterinary fixture)
======================  ==============  =================

Every species CURIE the assertion tables can write in ``species_context_text`` is indexed too
(:func:`_taxon_terms`): one ``OrganismTaxon`` node per NCBITaxon id in
``dakp_pipeline.assertions.species``, because the ``species_context_qualifier`` slot is
non-nullable and an unresolved species cell drops the whole row.

``Advil`` (the FAERS-reported brand) and ``Ibuprofen`` are both synonyms of ``CHEBI:5855``, so
the FAERS ``Advil``-subject rows canonicalize to the same ibuprofen node as the DailyMed rows.

redb locking note: ``rs.build_fullmap_db`` keeps the built database OPEN (cached, holding
redb's exclusive ``flock``) in the process that calls it. The end-to-end test then runs
``tablassert build-kg`` as a SEPARATE subprocess that must acquire that same flock. To avoid
``Database already open. Cannot acquire lock.``, :func:`build_tiny_fullmap` runs the build in a
short-lived CHILD interpreter that exits (releasing the flock) before build-kg runs. This keeps
the DAKP test free to invoke the real build-kg subprocess while staying hermetic (no network).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from dakp_pipeline.assertions.species import HUMAN_TAXON, SPECIES_LEXICON

# Human taxon for chemicals; ``NCBITaxon:0`` (no taxon) for diseases/phenotypes — mirrors the
# Tablassert fixture convention. Taxon never filters anything here: the species slot lifts the
# human default explicitly (``taxon: null``) and every other encoding keeps the 9606 default,
# which still matches ``TAXON_ID 0`` rows. No DAKP encoding scopes resolution per row by taxon
# (``taxon_from``): the production fullmap cannot support it, and the rationale lives
# beside :data:`_NON_NULLABLE_QUALIFIERS` in ``dakp_pipeline.tablassert``.
_HUMAN = "NCBITaxon:9606"
_NO_TAXON = "NCBITaxon:0"


def _taxon_terms() -> tuple[tuple[str, str, str, str, tuple[str, ...]], ...]:
    """One ``OrganismTaxon`` row per species CURIE the shapers can write.

    The assertion tables carry literal NCBITaxon CURIEs in ``species_context_text`` (the human
    constant on human corpora, the derived veterinary CURIE on animal labels), and the
    ``species_context_qualifier`` slot is NON-nullable: a species cell that resolves nothing drops
    the whole row. So the fixture must resolve every CURIE in
    :data:`dakp_pipeline.assertions.species.SPECIES_LEXICON` plus :data:`HUMAN_TAXON`, keyed off
    those tables instead of a hand-written list that can drift from the lexicon. Each row also
    carries the surface words, mirroring how the real fullmap indexes species names, and
    ``TAXON_ID 0``, mirroring that production ``OrganismTaxon`` rows carry no taxon of their own.
    """
    words_by_curie: dict[str, list[str]] = {}
    for word, curie in SPECIES_LEXICON.items():
        words_by_curie.setdefault(curie, []).append(word)
    rows: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    for curie in sorted({*words_by_curie, HUMAN_TAXON}):
        names = (curie, *sorted(words_by_curie.get(curie, ())))
        rows.append((curie, curie, "OrganismTaxon", _NO_TAXON, names))
    return tuple(rows)


# The DAKP assertion-table vocabulary -> (CURIE, preferred name, category, taxon, extra synonyms).
# ``names`` always includes the mention text the assertion tables carry in subject_text/object_text,
# plus the species CURIEs in species_context_text (see :func:`_taxon_terms`).
TERMS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    ("CHEBI:5855", "ibuprofen", "SmallMolecule", _HUMAN, ("Ibuprofen", "Advil")),
    ("CHEBI:1000001", "examplestatin", "SmallMolecule", _HUMAN, ("Examplestatin",)),
    ("MONDO:0005154", "hypercholesterolemia", "Disease", _NO_TAXON, ("hypercholesterolemia",)),
    ("HP:0002315", "headache", "PhenotypicFeature", _HUMAN, ("headache",)),
    ("HP:0012531", "pain", "PhenotypicFeature", _HUMAN, ("pain",)),
    ("MONDO:0004979", "asthma", "Disease", _NO_TAXON, ("asthma",)),
    ("MONDO:0005011", "peptic ulcer disease", "Disease", _NO_TAXON, ("peptic ulcer disease",)),
    # Animal-label vocabulary (the veterinary SPL fixture): Fipronil treating pain in dogs and
    # cats, contraindicated in renal disease. Both CURIEs come from a production-fullmap probe
    # (``quick_map``, 2026-10-21): ``fipronil`` -> ``CHEBI:83394`` (SmallMolecule) and
    # ``renal disease`` -> ``MONDO:0005240`` (Disease, preferred "kidney disorder"), so the fixture
    # names the same concepts the real vocabulary does. Without them the animal rows resolve no
    # subject/object and drop, which is why the veterinary path needs them indexed.
    ("CHEBI:83394", "fipronil", "SmallMolecule", _HUMAN, ("Fipronil",)),
    ("MONDO:0005240", "kidney disorder", "Disease", _NO_TAXON, ("renal disease",)),
    *_taxon_terms(),
)


def class_row(curie: str, equivalents: list[str]) -> dict[str, Any]:
    """One ``classes`` NDJSON record (KGX node canonicalization), matching the Tablassert fixture."""
    return {"id": curie, "equivalent_identifiers": [{"identifier": x} for x in equivalents]}


def synonym_row(curie: str, preferred_name: str, names: list[str], category: str, taxon: str = _HUMAN) -> dict[str, Any]:
    """One ``synonyms`` NDJSON record (mention text -> CURIE/category), matching the Tablassert fixture."""
    return {"curie": curie, "preferred_name": preferred_name, "names": names, "types": [category], "taxa": [taxon]}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def write_fullmap_inputs(directory: Path) -> tuple[Path, Path]:
    """Write the tiny ``classes.ndjson`` + ``synonyms.ndjson`` for the DAKP terms into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    classes = _write_jsonl(directory / "classes.ndjson", [class_row(curie, [curie]) for curie, *_ in TERMS])
    synonyms = _write_jsonl(
        directory / "synonyms.ndjson",
        [synonym_row(curie, preferred, list(names), category, taxon) for curie, preferred, category, taxon, names in TERMS],
    )
    return classes, synonyms


def build_tiny_fullmap(output: Path) -> Path:
    """Build a tiny ``fullmap.redb`` (sharded) at ``output`` covering the DAKP terms.

    The build runs in a short-lived CHILD interpreter (see module docstring) so the calling
    process never holds redb's exclusive flock — a later ``tablassert build-kg`` subprocess can
    then open the database. Returns ``output`` (the primary redb; sibling ``fullmap.s*.redb``
    shard files are written alongside it).
    """
    classes, synonyms = write_fullmap_inputs(output.parent)
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from tablassert import rs

        rs.build_fullmap_db(Path({str(output)!r}), [Path({str(classes)!r})], [Path({str(synonyms)!r})])
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output.is_file():
        msg = f"tiny fullmap build failed (exit {completed.returncode}): {completed.stderr.strip()[-2000:]}"
        raise RuntimeError(msg)
    return output


__all__ = ["TERMS", "build_tiny_fullmap", "class_row", "synonym_row", "write_fullmap_inputs"]
