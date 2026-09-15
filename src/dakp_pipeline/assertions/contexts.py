"""Pure assertion-context derivation and sparse qualifier attachment.

The model's context attribute is provenance only. Source/section rules remain authoritative;
prevention predicates are deliberately deferred because the pinned Biolink 4.4.4 vocabulary does
not contain ``prevents`` or ``applied_to_prevent``. Rows therefore retain the prevention context
while their existing treats predicates remain unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from dakp_pipeline.logging_setup import logger, stats
from dakp_pipeline.ner.dictionary import OBJECT_TYPES, canonical_type
from dakp_pipeline.ner.lexical import Mention

ASSERTION_CONTEXTS = ("indication", "contraindication", "prevention", "observed_prevention")

# Closed, case-insensitive cues. Longer phrases precede their component words for readability;
# regex matching is deterministic and word bounded.
PREVENTION_CUE = re.compile(
    r"\b(?:post[- ]exposure\s+prophylaxis|reduce\s+the\s+risk\s+of|"
    r"prophylaxis|prophylactic|prevention\s+of|to\s+prevent|preventing|"
    r"immunization|immunisation|vaccine|PEP|PrEP)\b",
    re.IGNORECASE,
)

_DAILYMED_CONTRA = frozenset({"34070-3", "34066-1", "43685-7", "34071-1", "42232-9", "contraindications", "boxed_warning", "warnings", "warning"})
_DAILYMED_INDICATION = frozenset({"34067-9", "indications", "indications_and_usage", "indication"})


def _has_cue(sentence: str) -> bool:
    return bool(PREVENTION_CUE.search(sentence or ""))


def assertion_context(source: str, section_kind: str, sentence: str) -> str:
    """Derive the authoritative context from source section and sentence cues.

    ``source`` is ``dailymed`` or ``faers``. DailyMed indication sections distinguish treatment
    from prevention; dedicated contraindication/warning sections always remain contraindications.
    FAERS indications distinguish observed prevention from an observed indication. Model opinions
    are intentionally not accepted by this function.
    """
    source_key = source.strip().lower()
    section_key = section_kind.strip().lower()
    if source_key not in {"dailymed", "faers"}:
        raise ValueError(f"unknown assertion source: {source!r}")
    if source_key == "dailymed":
        if section_key in _DAILYMED_CONTRA:
            return "contraindication"
        if section_key in _DAILYMED_INDICATION:
            return "prevention" if _has_cue(sentence) else "indication"
    elif section_key in {"indication", "indications", "faers_indication"}:
        return "observed_prevention" if _has_cue(sentence) else "indication"
    raise ValueError(f"unknown assertion section_kind: {section_kind!r}")


def context_predicate(context: str, source: str) -> str:
    """Map a context to the currently emittable predicate.

    Prevention contexts are recorded but use the producing shaper's existing predicate: the
    anticipated ``biolink:prevents`` and ``biolink:applied_to_prevent`` slots are absent from the
    pinned Biolink-model 4.4.4 and are therefore not emitted or tested here.
    """
    source_key = source.strip().lower()
    if source_key not in {"dailymed", "faers"}:
        raise ValueError(f"unknown assertion source: {source!r}")
    if context not in ASSERTION_CONTEXTS:
        raise ValueError(f"unknown assertion context: {context!r}")
    if context == "contraindication":
        return "biolink:contraindicated_in"
    return "biolink:treats" if source_key == "dailymed" else "biolink:applied_to_treat"


def _log_withheld(reason: str, qualifier: Mention, host_count: int = 0) -> None:
    stats(logger, "qualifier_withheld", level="DEBUG", reason=reason, qualifier=qualifier.text, qualifier_type=qualifier.type, host_count=host_count)


def _template_host_indices(sentence: str, objects: Sequence[Mention], sentence_of: Callable[[Mention], str | None]) -> list[int]:
    """Return the sole post-marker host for an explicit patient template, if unambiguous."""
    marker = re.search(
        r"\b(?:in|among|for)\s+(?:patients?|people|individuals|subjects|persons|those)\s+(?:with|having|who\s+(?:have|has))\b",
        sentence,
        re.IGNORECASE,
    )
    if marker is None:
        return []
    after = [i for i, obj in enumerate(objects) if sentence_of(obj) == sentence and obj.start >= marker.end()]
    return after if len(after) == 1 else []


def _attach_qualifiers_with_scores(
    objects: Sequence[Mention], qualifiers: Sequence[Mention], sentence_of: Callable[[Mention], str | None]
) -> tuple[dict[int, dict[str, str]], dict[tuple[int, str], tuple[float, str]]]:
    """Attach sparse qualifier spans to object indices without guessing.

    ``objects`` and ``qualifiers`` must carry offsets relative to the sentence returned by
    ``sentence_of``. Each span must therefore fit within that sentence; an offset in document
    space (or any other space) raises ``ValueError`` rather than silently withholding a qualifier.
    A qualifier must meet the score floor, map to a containing sentence, and have one host in that
    sentence (or an explicit patient template). Overlap is allowed only when the qualifier is
    strictly inside or disjoint from its host; equal/covering spans are withheld as restatements.
    Missing values are represented by omission, not sentinels. Disease mentions are intentionally
    not generic qualifiers: contraindication template classification owns that path.
    """
    attached: dict[int, dict[str, str]] = {}
    field_scores: dict[tuple[int, str], tuple[float, str]] = {}
    ordered_objects = tuple(objects)
    for mention in (*ordered_objects, *qualifiers):
        sentence = sentence_of(mention)
        if sentence is not None and not 0 <= mention.start <= mention.end <= len(sentence):
            raise ValueError("mention offsets must be sentence-relative and within sentence bounds")
    for qualifier in sorted(qualifiers, key=lambda item: (item.start, item.end, item.type, -item.score, item.text)):
        if qualifier.score < 0.5:
            _log_withheld("qualifier_below_accept_floor", qualifier)
            continue
        sentence = sentence_of(qualifier)
        if not sentence:
            _log_withheld("qualifier_outside_sentence", qualifier)
            continue
        candidates = [i for i, obj in enumerate(ordered_objects) if sentence_of(obj) == sentence]
        template_hosts = _template_host_indices(sentence, ordered_objects, sentence_of)
        if len(candidates) != 1 and not template_hosts:
            _log_withheld("ambiguous_qualifier_host", qualifier, len(candidates))
            continue
        host_indices = template_hosts or candidates
        if len(host_indices) != 1:
            _log_withheld("ambiguous_qualifier_host", qualifier, len(host_indices))
            continue
        host_index = host_indices[0]
        host = ordered_objects[host_index]
        overlap = qualifier.start < host.end and host.start < qualifier.end
        strictly_inside = host.start < qualifier.start and qualifier.end < host.end
        if overlap and not strictly_inside:
            _log_withheld("qualifier_restates_object", qualifier)
            continue
        qtype = canonical_type(qualifier.type)
        if qtype in OBJECT_TYPES:
            _log_withheld("context_not_disease", qualifier)
            continue
        field = {
            "AnatomicalEntity": "anatomical_context_text",
            "BiologicalSex": "sex_text",
            "PopulationOfIndividualOrganisms": "population_context_text",
            "OrganismTaxon": "species_context_text",
            "frequency_qualifier": "frequency_text",
            "temporal_context_qualifier": "temporal_context_text",
            "temporal_interval_qualifier": "temporal_context_text",
        }.get(qtype, qtype)
        bucket = attached.setdefault(host_index, {})
        previous = bucket.get(field)
        candidate_score = (qualifier.score, qualifier.text)
        if previous is None or candidate_score > field_scores[(host_index, field)]:
            bucket[field] = qualifier.text
            field_scores[(host_index, field)] = candidate_score
    return attached, field_scores


def attach_qualifiers(
    objects: Sequence[Mention], qualifiers: Sequence[Mention], sentence_of: Callable[[Mention], str | None]
) -> dict[int, dict[str, str]]:
    """Attach sparse qualifiers while keeping the score-free public schema."""
    attached, _scores = _attach_qualifiers_with_scores(objects, qualifiers, sentence_of)
    return attached


def attach_qualifiers_with_scores(
    objects: Sequence[Mention], qualifiers: Sequence[Mention], sentence_of: Callable[[Mention], str | None]
) -> tuple[dict[int, dict[str, str]], dict[tuple[int, str], tuple[float, str]]]:
    """Attach qualifiers and expose score/tie-break metadata for cross-observation merging."""
    return _attach_qualifiers_with_scores(objects, qualifiers, sentence_of)


__all__ = ["ASSERTION_CONTEXTS", "PREVENTION_CUE", "assertion_context", "attach_qualifiers", "attach_qualifiers_with_scores", "context_predicate"]
