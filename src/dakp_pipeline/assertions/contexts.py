"""Pure assertion-context derivation and sparse qualifier attachment.

The model's context attribute is provenance only. Source/section rules remain authoritative;
prevention predicates are deliberately deferred because the pinned Biolink 4.4.4 vocabulary does
not contain ``prevents`` or ``applied_to_prevent``. Rows therefore retain the prevention context
while their existing treats predicates remain unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dakp_pipeline.logging_setup import logger, stats
from dakp_pipeline.ner.dictionary import OBJECT_TYPES, TYPE_DISEASE, canonical_type, normalize_text
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
_EMA_INDICATION = frozenset({"therapeutic_indication", "indication"})


def _has_cue(sentence: str) -> bool:
    return bool(PREVENTION_CUE.search(sentence or ""))


def assertion_context(source: str, section_kind: str, sentence: str) -> str:
    """Derive the authoritative context from source section and sentence cues.

    ``source`` is ``dailymed``, ``faers``, or ``ema``. DailyMed indication sections distinguish
    treatment from prevention; dedicated contraindication/warning sections always remain
    contraindications. FAERS indications distinguish observed prevention from an observed
    indication. EMA registry indications reuse the indication semantics (prevention cue ->
    ``prevention``, else ``indication``). Model opinions are intentionally not accepted by this
    function.
    """
    source_key = source.strip().lower()
    section_key = section_kind.strip().lower()
    if source_key not in {"dailymed", "faers", "ema"}:
        raise ValueError(f"unknown assertion source: {source!r}")
    if source_key == "dailymed":
        if section_key in _DAILYMED_CONTRA:
            return "contraindication"
        if section_key in _DAILYMED_INDICATION:
            return "prevention" if _has_cue(sentence) else "indication"
    elif source_key == "faers" and section_key in {"indication", "indications", "faers_indication"}:
        return "observed_prevention" if _has_cue(sentence) else "indication"
    elif source_key == "ema" and section_key in _EMA_INDICATION:
        return "prevention" if _has_cue(sentence) else "indication"
    raise ValueError(f"unknown assertion section_kind: {section_kind!r}")


def context_predicate(context: str, source: str) -> str:
    """Map a context to the currently emittable predicate.

    Prevention contexts are recorded but use the producing shaper's existing predicate: the
    anticipated ``biolink:prevents`` and ``biolink:applied_to_prevent`` slots are absent from the
    pinned Biolink-model 4.4.4 and are therefore not emitted or tested here.
    """
    source_key = source.strip().lower()
    if source_key not in {"dailymed", "faers", "ema"}:
        raise ValueError(f"unknown assertion source: {source!r}")
    if context not in ASSERTION_CONTEXTS:
        raise ValueError(f"unknown assertion context: {context!r}")
    if context == "contraindication":
        return "biolink:contraindicated_in"
    return "biolink:treats" if source_key in {"dailymed", "ema"} else "biolink:applied_to_treat"


def _log_withheld(reason: str, qualifier: Mention, host_count: int = 0) -> None:
    stats(logger, "qualifier_withheld", level="DEBUG", reason=reason, qualifier=qualifier.text, qualifier_type=qualifier.type, host_count=host_count)


# Wording-level junk filters applied at attachment time, keyed by the model's canonical
# qualifier type. These kill qualifier spans whose SURFACE text is known junk before the text
# ever reaches fullmap resolution; junk CANDIDATES behind a legitimate wording ("eye" also
# matching a specimen concept) are handled at resolution instead, via the qualifier configs'
# ``exclude_prefixes``/``exclude_regex`` (:data:`dakp_pipeline.tablassert._QUALIFIER_EXCLUDE_REGEX`).
# Evidence: v1.13.0 release audit of every distinct resolved qualifier CURIE per slot.
#   frequency: dosage-form product wordings mined as frequency ("oral tablet", "Extended Release
#     Oral Capsule", "injection", "topical cream" resolve to drug products / chemical forms);
#   temporal: anamnesis / documentation wordings ("medical history", "prior therapy",
#     "H/O: hypertension", "documentation") resolve to Phenomenon history-of concepts.
# Genuine frequency wordings ("once daily", "twice a week", "every 6 hours", "at bedtime") and
# temporal wordings ("preoperative", "short-term", "chronic") never match.
_QUALIFIER_WORDING_DENYLIST: dict[str, re.Pattern[str]] = {
    "frequency_qualifier": re.compile(
        r"\b(?:tablets?|capsules?|suppositor(?:y|ies)|injections?|solutions?|suspensions?|creams?|gels?"
        r"|inhalers?|inhalants?|sprays?|patches?)\b|"
        r"\b(?:disintegrating|sublingual|delayed[- ]release|extended[- ]release)\b",
        re.IGNORECASE,
    ),
    "temporal_context_qualifier": re.compile(
        r"\b(?:medical\s+)?history\b|\bprior\s+therap|\bprevious\s+therap|\bh/o\b|\bdocuments?\b|\bdocumentation\b", re.IGNORECASE
    ),
}


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
        deny = _QUALIFIER_WORDING_DENYLIST.get(qtype)
        if deny is not None and deny.search(qualifier.text or ""):
            _log_withheld("qualifier_junk_wording", qualifier)
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


# --- explicit patient-clause disease context ---------------------------------------

PATIENT_WITH_MARKER = re.compile(
    r"\b(?:in|among|for)\s+(?:patients?|people|individuals|subjects|persons|those)\s+"
    r"(?:with|having|who\s+(?:have|has))\b",
    re.IGNORECASE,
)
CONTEXT_INTRO = re.compile(
    r"\b(?:for\s+(?:the\s+)?(?:treatment|management)\s+of|"
    r"(?:when|if)\s+(?:used|given|administered)\s+(?:for|to\s+treat)|"
    r"used\s+in\s+the\s+treatment\s+of)\b",
    re.IGNORECASE,
)
MEDICATION_CONTEXT = re.compile(
    r"\b(?:receiving|taking|administered\s+with|co[- ]?administered|concomitant|"
    r"concurrent\s+(?:use|therapy)|drug[- ]drug\s+interaction)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PatientClause:
    """Result of the explicit patient-clause scan over one group of object mentions.

    ``contexts`` maps an object-mention index to its context disease surface (the explicit
    ``for treatment of A in patients with B`` template). ``context_only`` maps a mention
    index to ``(rejection reason, evidence sentence)`` for a mention consumed as context —
    it described the treated condition, so it is withheld as an object. ``ambiguous`` maps
    object-mention indices to their sentence for the ``A and B`` conjunction case, where a
    scalar qualifier cannot preserve the AND/OR requirement and every member of the
    sentence is dropped rather than falsely asserting either alone.
    """

    contexts: dict[int, str]
    context_only: dict[int, tuple[str, str]]
    ambiguous: dict[int, str]


def patient_clause_contexts(
    objects: Sequence[Mention], sentence_of: Callable[[Mention], str | None], group_of: Callable[[Mention], object] | None = None
) -> PatientClause:
    """Assign explicit disease context to objects via the patient-clause template.

    ``objects`` are object-channel mentions whose offsets are relative to the string
    returned by ``sentence_of`` (the same sentence-local contract as
    :func:`attach_qualifiers_with_scores`); a ``None`` sentence skips the mention — it
    cannot join per-sentence grouping. Mentions group on ``group_of(mention)`` when given
    (the shaper passes the mapped source-sentence origin so repeated identical sentences
    across one mined text stay separate groups) and on the sentence text otherwise.
    Guards, in order: an explicit patient marker (:data:`PATIENT_WITH_MARKER`), exactly one
    before-marker candidate and one after-marker object, an intro phrase
    (:data:`CONTEXT_INTRO`) vouching for the context (never starting inside it), a
    non-blank normalized context, the disease-range rule (Biolink's
    ``disease_context_qualifier`` is disease-ranged), and the companion-medication guard
    (:data:`MEDICATION_CONTEXT`). Shared verbatim by the contraindication shaper and the
    ``ner_export`` training-data rows.
    """
    groups: dict[object, tuple[str, list[tuple[int, int, int]]]] = {}
    for index, mention in enumerate(objects):
        sentence = sentence_of(mention)
        if sentence is None:
            continue
        key = group_of(mention) if group_of is not None else sentence
        members = groups.get(key)
        if members is None:
            members = (sentence, [])
            groups[key] = members
        members[1].append((index, mention.start, mention.end))
    contexts: dict[int, str] = {}
    context_only: dict[int, tuple[str, str]] = {}
    ambiguous: dict[int, str] = {}
    for sentence, members in groups.values():
        members.sort(key=lambda member: (member[1], member[2], member[0]))
        marker = PATIENT_WITH_MARKER.search(sentence)
        if marker is None:
            continue
        before = [member for member in members if member[2] <= marker.start()]
        after = [member for member in members if member[1] >= marker.end()]
        if len(after) > 1:
            for index, _start, _end in members:
                ambiguous[index] = sentence
            continue
        if len(before) != 1 or len(after) != 1:
            continue
        context_index, context_start, context_end = before[0]
        object_index = after[0][0]
        intro = CONTEXT_INTRO.search(sentence, 0, marker.start())
        if intro is None or intro.end() > context_start:
            continue
        context_text = normalize_text(sentence[context_start:context_end])
        if not context_text:
            continue
        if canonical_type(str(objects[context_index].type)) != TYPE_DISEASE:
            context_only[context_index] = ("context_not_disease", sentence)
            continue
        if MEDICATION_CONTEXT.search(sentence):
            context_only[context_index] = ("context_only_medication", sentence)
            continue
        context_only[context_index] = ("context_only", sentence)
        contexts[object_index] = context_text
    return PatientClause(contexts=contexts, context_only=context_only, ambiguous=ambiguous)


__all__ = [
    "ASSERTION_CONTEXTS",
    "CONTEXT_INTRO",
    "MEDICATION_CONTEXT",
    "PATIENT_WITH_MARKER",
    "PREVENTION_CUE",
    "PatientClause",
    "assertion_context",
    "attach_qualifiers",
    "attach_qualifiers_with_scores",
    "context_predicate",
    "patient_clause_contexts",
]
