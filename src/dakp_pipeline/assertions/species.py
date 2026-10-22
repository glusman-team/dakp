"""Label-type-gated species context for assertion rows.

Every DAKP edge carries a ``species_context_qualifier`` (Biolink: "a taxonomic category of
species in which a relationship expressed in an association took place"), and species is
part of edge identity, so an animal-label assertion can never fold onto the human edge for
the same drug-condition pair.

Derivation, in order (pure; no I/O):

1. **Non-animal sources** (human/homeopathic DailyMed labels, FAERS, Canada Vigilance, EMA)
   are human by construction: ``("NCBITaxon:9606",)``. Species words on a human label
   ("studies in rats") are study context, never the organism the assertion occurred in,
   so they are ignored there.
2. **Animal DailyMed labels** -- document type ``50578-4`` / ``50577-6`` (see
   :data:`dakp_pipeline.extract.spl_xml.ANIMAL_LABEL_TYPE_CODES`) OR any NADA/ANADA/CNADA
   approval on the set (see
   :data:`dakp_pipeline.assertions.evidence.ANIMAL_APPLICATION_PREFIXES`). Production
   ``spl_documents`` carries no type code at all for animal-approval sets, and a new animal
   drug may only be marketed under one of those three application types (21 U.S.C.
   360b(a)), so the application prefix is an authoritative animal signal, not a heuristic.
   Animal labels carry no structured species element, so species comes from the text:

   - the row's own evidence sentence, matched against :data:`SPECIES_LEXICON`; else
   - the label-wide species (the union over the label's mined section texts); else
   - nothing: the row is unattributable and the caller DROPS it (counted loudly), because
     an animal edge of unknown species must never masquerade as a human one.

   A sentence naming several species ("for use in dogs and cats") yields one row per
   species: the label asserts the relationship for each. Human words are never matched on
   animal labels: there they are user-safety warnings ("not for use in humans"), not the
   treated organism.

Why a curated lexicon instead of passing species TEXT to fullmap: batch resolution of
veterinary species words against the production fullmap (``tablassert.fullmap.quick_map``,
2026-10-22) is unreliable for this vocabulary -- ``cattle`` resolves to ``MESH:D002417``
(not NCBITaxon), ``calves`` to a disease, ``hogs`` to a gene, ``kids`` to a disease, and
``cats`` ties on the Felidae family -- while every species-level NCBITaxon CURIE below was
verified to resolve to itself. The lexicon values are therefore literal CURIEs that the
Tablassert ``species_context_qualifier`` column resolves through fullmap unchanged; they
are a closed controlled vocabulary like the human ``NCBITaxon:9606`` constant, not ad hoc
ontology resolution. ``tests/eval/species_lexicon_audit.py`` re-verifies them against a
fullmap.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Final

from dakp_pipeline.extract.spl_xml import is_animal_label_type

#: Species context of every non-animal source row.
HUMAN_TAXON: Final = "NCBITaxon:9606"

#: Curated veterinary species words -> species-level NCBITaxon CURIE. Keys are lowercase
#: surface forms (plurals listed explicitly; matching is whole-word, case-insensitive,
#: longest-first). Deliberately ABSENT: multi-species group words (``poultry``, ``fish``,
#: ``birds``, ``livestock``, ``animals``), words with a dominant non-species reading
#: (``kids``, ``steer``, ``man``), and every human word (see module docstring).
SPECIES_LEXICON: Final[Mapping[str, str]] = {
    # Canis lupus familiaris (dog)
    "dog": "NCBITaxon:9615",
    "dogs": "NCBITaxon:9615",
    "puppy": "NCBITaxon:9615",
    "puppies": "NCBITaxon:9615",
    "canine": "NCBITaxon:9615",
    "canines": "NCBITaxon:9615",
    # Felis catus (domestic cat)
    "cat": "NCBITaxon:9685",
    "cats": "NCBITaxon:9685",
    "kitten": "NCBITaxon:9685",
    "kittens": "NCBITaxon:9685",
    "feline": "NCBITaxon:9685",
    "felines": "NCBITaxon:9685",
    # Bos taurus (domestic cattle)
    "cattle": "NCBITaxon:9913",
    "cow": "NCBITaxon:9913",
    "cows": "NCBITaxon:9913",
    "calf": "NCBITaxon:9913",
    "calves": "NCBITaxon:9913",
    "bovine": "NCBITaxon:9913",
    "heifer": "NCBITaxon:9913",
    "heifers": "NCBITaxon:9913",
    "steers": "NCBITaxon:9913",
    "bull": "NCBITaxon:9913",
    "bulls": "NCBITaxon:9913",
    # Equus caballus (horse)
    "horse": "NCBITaxon:9796",
    "horses": "NCBITaxon:9796",
    "foal": "NCBITaxon:9796",
    "foals": "NCBITaxon:9796",
    "equine": "NCBITaxon:9796",
    "mare": "NCBITaxon:9796",
    "mares": "NCBITaxon:9796",
    "pony": "NCBITaxon:9796",
    "ponies": "NCBITaxon:9796",
    # Sus scrofa (pig)
    "swine": "NCBITaxon:9823",
    "pig": "NCBITaxon:9823",
    "pigs": "NCBITaxon:9823",
    "hog": "NCBITaxon:9823",
    "hogs": "NCBITaxon:9823",
    "piglet": "NCBITaxon:9823",
    "piglets": "NCBITaxon:9823",
    "sow": "NCBITaxon:9823",
    "sows": "NCBITaxon:9823",
    "gilts": "NCBITaxon:9823",
    "porcine": "NCBITaxon:9823",
    # Ovis aries (sheep)
    "sheep": "NCBITaxon:9940",
    "lamb": "NCBITaxon:9940",
    "lambs": "NCBITaxon:9940",
    "ewe": "NCBITaxon:9940",
    "ewes": "NCBITaxon:9940",
    "ovine": "NCBITaxon:9940",
    # Capra hircus (goat)
    "goat": "NCBITaxon:9925",
    "goats": "NCBITaxon:9925",
    "caprine": "NCBITaxon:9925",
    # Gallus gallus (chicken)
    "chicken": "NCBITaxon:9031",
    "chickens": "NCBITaxon:9031",
    "broiler": "NCBITaxon:9031",
    "broilers": "NCBITaxon:9031",
    "hens": "NCBITaxon:9031",
    "laying hens": "NCBITaxon:9031",
    "pullets": "NCBITaxon:9031",
    # Meleagris gallopavo (turkey)
    "turkey": "NCBITaxon:9103",
    "turkeys": "NCBITaxon:9103",
    # Anas platyrhynchos (domestic duck / mallard)
    "duck": "NCBITaxon:8839",
    "ducks": "NCBITaxon:8839",
    # Apis mellifera (honey bee)
    "honey bee": "NCBITaxon:7460",
    "honey bees": "NCBITaxon:7460",
    "honeybee": "NCBITaxon:7460",
    "honeybees": "NCBITaxon:7460",
    # Oryctolagus cuniculus (rabbit)
    "rabbit": "NCBITaxon:9986",
    "rabbits": "NCBITaxon:9986",
    # Mustela putorius furo (domestic ferret)
    "ferret": "NCBITaxon:9669",
    "ferrets": "NCBITaxon:9669",
    # Vicugna pacos (alpaca) / Lama glama (llama)
    "alpaca": "NCBITaxon:30538",
    "alpacas": "NCBITaxon:30538",
    "llama": "NCBITaxon:9844",
    "llamas": "NCBITaxon:9844",
    # Salmonidae (salmonids; US aquaculture labels target the family) / Ictalurus punctatus
    "salmon": "NCBITaxon:8015",
    "salmonids": "NCBITaxon:8015",
    "catfish": "NCBITaxon:7998",
}

_SPECIES_PATTERN: Final = re.compile(
    r"\b(" + "|".join(re.escape(word).replace(r"\ ", r"\s+") for word in sorted(SPECIES_LEXICON, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


def species_in_text(text: str | None) -> frozenset[str]:
    """Distinct NCBITaxon CURIEs named in ``text`` (whole-word lexicon matches).

    Whitespace inside multi-word entries is flexible (``honey  bees``), matching is
    case-insensitive, and longer entries win over their substrings (``laying hens``
    before ``hens``). Empty set for ``None``/blank text or no match.
    """
    if not text:
        return frozenset()
    return frozenset(SPECIES_LEXICON[re.sub(r"\s+", " ", match.lower())] for match in _SPECIES_PATTERN.findall(text))


def label_species(texts: Iterable[str | None]) -> frozenset[str]:
    """Union of :func:`species_in_text` over a label's mined section texts."""
    found: set[str] = set()
    for text in texts:
        found |= species_in_text(text)
    return frozenset(found)


def row_species(label_type: str | None, sentence: str | None, label_wide: frozenset[str]) -> tuple[str, ...]:
    """Species CURIEs one assertion row occurred in; empty tuple means unattributable.

    Args:
        label_type: The DailyMed document-level LOINC type code for the row's label, or
            ``None`` for a non-DailyMed source (FAERS, Canada Vigilance, EMA). Only an
            animal label type (``50578-4``/``50577-6``) leaves the human default.
        sentence: The row's evidence sentence (searched first on animal labels).
        label_wide: :func:`label_species` over the label's mined sections (the fallback
            when the sentence names no species).

    Returns:
        Sorted distinct NCBITaxon CURIEs: ``(HUMAN_TAXON,)`` for every non-animal source;
        on an animal label the sentence's species, else the label-wide species, else
        ``()`` -- the caller drops an empty result and counts it, never defaulting an
        animal row to human.
    """
    if not is_animal_label_type(label_type or ""):
        return (HUMAN_TAXON,)
    return animal_row_species(sentence, label_wide)


def animal_row_species(sentence: str | None, label_wide: frozenset[str]) -> tuple[str, ...]:
    """Species CURIEs for a row known to be on an ANIMAL source; ``()`` = unattributable.

    The row's own evidence sentence first, then the label's mined sections; an animal row
    naming no species anywhere stays empty so the caller drops it (counted loudly), never
    defaulting an animal edge onto the human taxon. Shared by the document-type gate in
    :func:`row_species` and the application-type gate in
    :meth:`~dakp_pipeline.assertions.evidence.DailyMedEvidence.species_context_for`.
    """
    in_sentence = species_in_text(sentence)
    return tuple(sorted(in_sentence or label_wide))


__all__ = ["HUMAN_TAXON", "SPECIES_LEXICON", "animal_row_species", "label_species", "row_species", "species_in_text"]
