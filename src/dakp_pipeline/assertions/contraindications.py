"""Contraindication assertion aggregation — text-mined from DailyMed SPL.

Builds ``contraindication_assertions.tsv``: drug→condition contraindication assertions
**mined directly** from DailyMed SPL label text using DAKP's single composite NER backend
(:mod:`dakp_pipeline.ner.ner`). Contraindications are extracted from the label text itself,
which yields better coverage and DailyMed-grounded provenance than an externally-sourced list.

Three-pass mining rule (explicit and tested)
------------------------------------------
**Pass 1** — for every SPL set with a contraindication section (LOINC ``34070-3``) and ≥1
active ingredient, run the NER backend over the full section text to extract disease/phenotype
mentions. All text in a dedicated contraindication section is assumed relevant.

**Pass 2** — for every SPL set with an indications-and-usage section (LOINC ``34067-9``), filter
the section text to sentences containing contraindication-context keywords (``contraindicated``,
``should not be used``, ``not recommended``, …) and run the same NER extraction on the filtered
text. This catches contraindications embedded in the indication section — common in real
DailyMed labels where safety statements appear alongside indication text. The keyword filter
ensures indication-context diseases (``indicated for X``) are NOT mined as contraindications.

**Pass 3** — for every SPL set with a boxed-warning section (LOINC ``34066-1``) or a
warnings/precautions section (``43685-7``, or the legacy split ``34071-1``/``42232-9``), mine the
full section text. Boxed warnings in particular frequently carry absolute contraindications that
never appear in a dedicated ``34070-3`` section. Acceptance is **hard-trigger-only** (the Pass 2
discipline): a warning section's general caution prose supplies no section-level contraindication
context, so a mention becomes an edge only when its sentence carries hard prohibition language.

Mentions from all passes are paired with the set's active ingredient to form a
``biolink:contraindicated_in`` assertion (ingredient = subject, mention = object) — but ONLY for
**singleton-ingredient sets** (the legacy ``selectActiveIngredientSingletons.pl`` discipline): a
combination product's contraindication section applies to the mixture, so pairing a mention with
every one of its actives would over-attribute the contraindication to each component. Sets with
zero or multiple active ingredients are skipped (``skipped_multi_ingredient_sets`` counts the
latter). Rows are aggregated by ``(subject_text, object_text, disease_context_text)``:
``supporting_spl_sets``, ``supporting_spl_documents``, and evidence sentences are unioned within
that key, and ``source_score`` takes the max NER span score.

Ontology mapping is Tablassert-only
-----------------------------------
The **object** is the mined disease MENTION TEXT (``object_text``); ``object_curie`` /
``object_name`` / ``object_category`` are left empty for Tablassert/fullmap to resolve at
``tablassert build-kg`` — DAKP does **not** resolve mentions to ontology CURIEs. The
**subject** (active ingredient) carries its text + UNII straight from the SPL source
(source-provided, not DAKP-mapped).

The NER backend
---------------
The shaper uses an injected ``params["ner"]`` :class:`~dakp_pipeline.ner.ner.DiseaseNER` when
present (tests / production wiring), else builds the deterministic **offline** backend from the
ontology fixture gazetteer (``<fixture_root>/ontology/disease_map.tsv``, read as term→type only
— CURIE columns ignored), falling back to the embedded gazetteer. There is no backend-name
selector. Constructing the backend is import-free, so module import + the test suite run with
no heavy NER deps imported.

Provenance: contraindications are text-mined from DailyMed. The unannotated
``supporting_spl_sets`` / ``supporting_spl_documents`` debug columns carry DailyMed label URLs
(``https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=<spl_set_id>[#<loinc>]``), while
``supporting_spl_evidence`` carries the backing SPL sets as ``dailymed:<spl_set_id>`` CURIEs in
the single column Tablassert encodes as Biolink ``has_evidence`` (the legacy DAKP KG evidence
form; see :func:`~dakp_pipeline.assertions.evidence.spl_evidence_pipe`).
``primary_knowledge_source = infores:multiomics-drugapprovals``,
``upstream_resource_ids = infores:dailymed``, ``agent_type = text_mining_agent`` (the DAKP RIG
lists ``text_mining_agent`` for ``contraindicated_in``), ``knowledge_level = knowledge_assertion``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from dakp_pipeline.assertions import INFORES_DAILYMED, INFORES_DAKP, KL_ASSERTION, row_for
from dakp_pipeline.assertions.evidence import (
    dailymed_document_url,
    dailymed_set_url,
    edge_evidence_pipe,
    load_or_build_dailymed_evidence,
    sorted_pipe,
    spl_evidence_pipe,
    write_assertion_table,
)
from dakp_pipeline.assertions.ner_dispatch import BUILD_HOST_GPUS, _resolve_devices, default_ner, mine_passes_multi_gpu, mine_with_cache
from dakp_pipeline.assertions.ner_dispatch import _mine_multi_gpu as _mine_multi_gpu
from dakp_pipeline.assertions.ner_dispatch import _mine_shard as _mine_shard
from dakp_pipeline.assertions.ner_dispatch import _mine_two_passes_multi_gpu as _mine_two_passes_multi_gpu
from dakp_pipeline.assertions.ner_dispatch import _shard_by_text_length as _shard_by_text_length
from dakp_pipeline.assertions.ner_dispatch import _spawn_safe_main as _spawn_safe_main
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, progress, stats, step
from dakp_pipeline.ner.dictionary import normalize_text
from dakp_pipeline.ner.mention_cache import MentionCache
from dakp_pipeline.ner.ner import DiseaseNER, Mention, extract_contraindication_diseases

_TABLE = "contraindication_assertions"
_PREDICATE = "biolink:contraindicated_in"
#: One INFO progress line per this many mined contraindication sections (GLiNER is the slow step).
_MINING_PROGRESS_EVERY = 500

#: Agent type for text-mined contraindications (matches the DAKP RIG ``contraindicated_in``).
AT_TEXT_MINING = "text_mining_agent"

#: Broad sentence filter used only to avoid sending ordinary indication prose to NER. The
#: stricter positive-trigger classifier below decides whether a mention becomes a hard edge.
#: Configurable at runtime via ``ctx.params["contraindication_keywords"]`` (str or compiled Pattern).
DEFAULT_CONTRA_KEYWORDS: re.Pattern[str] = re.compile(
    r"\b(contraindicat\w*|should\s+not\s+be\s+used|must\s+not\s+(?:be\s+)?used|"
    r"do\s+not\s+use|not\s+recommended|avoid\s+(?:use\s+)?in|prohibit\w*|"
    r"must\s+avoid|use\s+is\s+contraindicat\w*|not\s+for\s+use\s+in)\b",
    re.IGNORECASE,
)

#: Historical name for the shared build-host GPU list (see
#: :data:`~dakp_pipeline.assertions.ner_dispatch.BUILD_HOST_GPUS`).
CONTRAINDICATION_GPUS: tuple[str, ...] = BUILD_HOST_GPUS

#: Hard prohibition language accepted for Pass 2. Soft safety language such as ``avoid`` and
#: ``not recommended`` is deliberately not sufficient for a ``contraindicated_in`` edge.
_HARD_CONTRA_TRIGGER = re.compile(
    r"\b(?:contraindicat\w*|do\s+not\s+use|must\s+not\s+(?:be\s+)?used|"
    r"should\s+not\s+be\s+used|not\s+for\s+use\s+in|never\s+(?:give|use)|prohibit\w*)\b",
    re.IGNORECASE,
)
_EXPLICIT_NEGATION = re.compile(
    r"\b(?:not|never)\s+(?:be\s+)?contraindicat\w*\b|"
    r"\b(?:no|none|without)\s+(?:known\s+)?contraindications?\b|"
    r"\bnone\s+known\b|\bnone\s*(?:[.;!?]|$)",
    re.IGNORECASE,
)
_SOFT_SAFETY_TRIGGER = re.compile(
    r"\b(?:avoid(?:\s+use)?|not\s+recommended|use\s+with\s+caution|warning|precaution|"
    r"monitor(?:ing)?|consider\s+an\s+alternative)\b",
    re.IGNORECASE,
)

#: Section kinds whose mentions become edges ONLY under hard prohibition language: indication
#: sections (Pass 2) and the Pass 3 warning sections, whose general caution prose supplies no
#: section-level contraindication context (unlike a dedicated ``34070-3`` section).
_HARD_TRIGGER_SECTION_KINDS = frozenset({"indications", "boxed_warning", "warnings"})


@dataclass(frozen=True)
class MentionDecision:
    """Conservative local semantic decision for one mined disease mention."""

    accepted: bool
    trigger: str
    evidence_text: str
    context_text: str = ""


_PATIENT_WITH_MARKER = re.compile(
    r"\b(?:in|among|for)\s+(?:patients?|people|individuals|subjects|persons|those)\s+"
    r"(?:with|having|who\s+(?:have|has))\b",
    re.IGNORECASE,
)
_CONTEXT_INTRO = re.compile(
    r"\b(?:for\s+(?:the\s+)?(?:treatment|management)\s+of|"
    r"(?:when|if)\s+(?:used|given|administered)\s+(?:for|to\s+treat)|"
    r"used\s+in\s+the\s+treatment\s+of)\b",
    re.IGNORECASE,
)
_MEDICATION_CONTEXT = re.compile(
    r"\b(?:receiving|taking|administered\s+with|co[- ]?administered|concomitant|"
    r"concurrent\s+(?:use|therapy)|drug[- ]drug\s+interaction)\b",
    re.IGNORECASE,
)


# --- sentence filtering for Pass 2 (indication-section contraindications) ------------


_SENTENCE_BOUNDARY = re.compile(r"[.!?;](?:\s+|$)")


@dataclass(frozen=True)
class SentenceSpan:
    """A sentence/clause with offsets into the original SPL section text."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class EvidenceSpan:
    """Mapping from a mined-text span to its original source sentence."""

    mined_start: int
    mined_end: int
    source_start: int
    source_end: int
    text: str


@dataclass(frozen=True)
class ContraWorkItem:
    """One NER input plus enough mapping to recover local source evidence.

    Pass 1 uses the full section as ``text``. Pass 2 uses a filtered, joined text; its
    ``evidence_spans`` map each joined sentence back to the original indication section so
    NER offsets never become detached from provenance. The tuple-like methods intentionally
    preserve the old ``(set_id, doc_id, text)`` test/worker interface.
    """

    set_id: str
    doc_id: str
    text: str
    source_text: str
    evidence_spans: tuple[EvidenceSpan, ...]
    section_kind: str = "contraindications"

    def __iter__(self) -> Iterator[str]:
        yield self.set_id
        yield self.doc_id
        yield self.text

    def __getitem__(self, index: int) -> str:
        return (self.set_id, self.doc_id, self.text)[index]

    def __len__(self) -> int:
        return 3


def _sentence_spans(text: str) -> list[SentenceSpan]:
    """Split text into deterministic, source-offset-preserving sentence-ish pieces."""
    if not text or not text.strip():
        return []
    spans: list[SentenceSpan] = []
    pos = 0
    # Every loop chunk contains its boundary punctuation and the tail (when present) always
    # starts on a non-whitespace char — the boundary regex consumes all trailing whitespace
    # and re's ``\s`` agrees with ``str.strip`` on exactly which chars are whitespace — so
    # each stripped chunk is provably non-empty (no emptiness guard needed).
    for match in _SENTENCE_BOUNDARY.finditer(text):
        raw = text[pos : match.end()]
        stripped = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        start = pos + leading
        end = start + len(stripped)
        spans.append(SentenceSpan(start=start, end=end, text=text[start:end]))
        pos = match.end()
    if pos < len(text):
        raw = text[pos:]
        stripped = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        start = pos + leading
        end = start + len(stripped)
        spans.append(SentenceSpan(start=start, end=end, text=text[start:end]))
    return spans


def _split_sentences(text: str) -> list[str]:
    """Compatibility wrapper returning sentence text without losing the old API."""
    return [span.text for span in _sentence_spans(text)]


def _work_item_parts(item: ContraWorkItem | tuple[str, str, str]) -> tuple[str, str, str]:
    """Read a new evidence-aware work item or a legacy tuple used by focused tests."""
    if isinstance(item, ContraWorkItem):
        return item.set_id, item.doc_id, item.text
    return item


def _work_item_evidence(item: ContraWorkItem | tuple[str, str, str], mention: Mention) -> str:
    """Return the original sentence containing a mention, or the mined text as a fallback."""
    if not isinstance(item, ContraWorkItem):
        return item[2].strip()
    for span in item.evidence_spans:
        if mention.start < span.mined_end and span.mined_start < mention.end:
            return span.text.strip()
    return item.source_text.strip()


def _mention_local_span(item: ContraWorkItem, mention: Mention) -> tuple[str, int, int, int] | None:
    """Map a mined mention back to ``(source sentence, local start, local end, source start)``."""
    for span in item.evidence_spans:
        if mention.start < span.mined_end and span.mined_start < mention.end:
            start = span.source_start + max(0, mention.start - span.mined_start)
            end = span.source_start + min(mention.end - span.mined_start, span.mined_end - span.mined_start)
            return span.text, max(0, start - span.source_start), min(len(span.text), end - span.source_start), span.source_start
    return None


def _classify_mention(item: ContraWorkItem | tuple[str, str, str], mention: Mention) -> MentionDecision:
    """Apply local polarity and trigger rules before an edge is formed.

    A dedicated contraindication section supplies section-level positive context, but explicit
    negation always wins. Indication-section and warning-section (boxed warning / warnings and
    precautions) candidates require hard prohibition language; broad filter terms (``avoid`` /
    ``not recommended``) are not enough to assert a contraindication.
    """
    evidence_text = _work_item_evidence(item, mention)
    if _EXPLICIT_NEGATION.search(evidence_text):
        return MentionDecision(False, "explicit_negation", evidence_text)
    hard = _HARD_CONTRA_TRIGGER.search(evidence_text)
    medication = _MEDICATION_CONTEXT.search(evidence_text)
    if medication is not None and isinstance(item, ContraWorkItem):
        mapped = _mention_local_span(item, mention)
        patient_with = _PATIENT_WITH_MARKER.search(evidence_text)
        # A disease named only before ``patients receiving/taking <drug>`` is treatment context,
        # not a contraindicated patient disease. The grouped template below preserves a separate
        # patient disease when one is explicitly present.
        if mapped is not None and patient_with is None and mapped[2] <= medication.start():
            return MentionDecision(False, "medication_only_context", evidence_text)
    if isinstance(item, ContraWorkItem) and item.section_kind in _HARD_TRIGGER_SECTION_KINDS:
        if hard is None:
            return MentionDecision(False, "no_hard_trigger", evidence_text)
        return MentionDecision(True, hard.group(0).lower(), evidence_text)
    if hard is not None:
        return MentionDecision(True, hard.group(0).lower(), evidence_text)
    # A dedicated 34070-3 section supplies section-4 context for terse condition lists, but a
    # sentence that contains only soft warning language is not promoted to a hard edge.
    if isinstance(item, ContraWorkItem) and item.section_kind == "contraindications":
        soft = _SOFT_SAFETY_TRIGGER.search(evidence_text)
        if soft is not None:
            return MentionDecision(False, "soft_safety_only", evidence_text)
        return MentionDecision(True, "contraindication_section", evidence_text)
    # Legacy tuple work items represent dedicated contraindication sections.
    return MentionDecision(True, "contraindication_section", evidence_text)


def _classify_mentions(item: ContraWorkItem | tuple[str, str, str], mentions: list[Mention]) -> list[MentionDecision]:
    """Classify mentions and assign only explicit, unambiguous disease context.

    The common template is ``for treatment of A in patients with B``: ``B`` is the
    contraindicated object and ``A`` is the disease context. Multiple conditions in the patient
    clause are withheld because a scalar qualifier cannot preserve an AND/OR requirement.
    Medication markers never produce a disease qualifier.
    """
    decisions = [_classify_mention(item, mention) for mention in mentions]
    if not isinstance(item, ContraWorkItem) or not mentions:
        return decisions

    local: list[tuple[int, str, int, int, int]] = []
    for index, mention in enumerate(mentions):
        mapped = _mention_local_span(item, mention)
        if mapped is not None:
            sentence, start, end, source_start = mapped
            local.append((index, sentence, start, end, source_start))

    by_sentence: dict[tuple[int, str], list[tuple[int, int, int]]] = {}
    for index, sentence, start, end, source_start in local:
        by_sentence.setdefault((source_start, sentence), []).append((index, start, end))

    for (_source_start, sentence), members in by_sentence.items():
        members.sort(key=lambda value: (value[1], value[2], value[0]))
        marker = _PATIENT_WITH_MARKER.search(sentence)
        if marker is None:
            continue
        before = [member for member in members if member[2] <= marker.start()]
        after = [member for member in members if member[1] >= marker.end()]
        if len(after) > 1:
            # ``A and B``/``A or B`` in a patient clause is not representable by one scalar
            # qualifier without changing the logic. Drop these candidate objects rather than
            # falsely asserting either disease alone.
            for index, _start, _end in members:
                decisions[index] = MentionDecision(False, "ambiguous_patient_conjunction", sentence)
            continue
        if len(before) != 1 or len(after) != 1:
            continue
        context_index, context_start, context_end = before[0]
        object_index, _object_start, _object_end = after[0]
        intro = _CONTEXT_INTRO.search(sentence, 0, marker.start())
        if intro is None or intro.end() > context_start:
            continue
        # A companion medication may occur in the same condition clause, but it is never a
        # disease_context_qualifier. Keep the base disease edge and its evidence sentence.
        context_text = normalize_text(sentence[context_start:context_end])
        if not context_text:
            continue
        if str(mentions[context_index].type).lower() != "disease":
            # Biolink's disease_context_qualifier is disease-ranged. Keep the explicit object
            # edge, but do not put a phenotype/symptom into this qualifier slot.
            decisions[context_index] = MentionDecision(False, "context_not_disease", sentence)
            continue
        if _MEDICATION_CONTEXT.search(sentence):
            decisions[context_index] = MentionDecision(False, "context_only_medication", sentence)
            continue
        decisions[context_index] = MentionDecision(False, "context_only", sentence)
        if decisions[object_index].accepted:
            current = decisions[object_index]
            decisions[object_index] = MentionDecision(True, current.trigger, current.evidence_text, context_text)
    return decisions


def _filtered_evidence(text: str, keywords: re.Pattern[str]) -> tuple[str, tuple[EvidenceSpan, ...]]:
    """Join matching source sentences while preserving their original offsets."""
    selected = [span for span in _sentence_spans(text) if keywords.search(span.text)]
    pieces: list[str] = []
    mappings: list[EvidenceSpan] = []
    cursor = 0
    for index, span in enumerate(selected):
        if index:
            pieces.append(" ")
            cursor += 1
        pieces.append(span.text)
        mappings.append(EvidenceSpan(cursor, cursor + len(span.text), span.start, span.end, span.text))
        cursor += len(span.text)
    return "".join(pieces), tuple(mappings)


def _contraindication_sentences(text: str, keywords: re.Pattern[str]) -> str:
    """Return only the contraindication-context sentences from ``text``, space-joined.

    Splits ``text`` into sentences and keeps only those matching ``keywords``. The result is
    suitable for direct NER extraction — GLiNER sees only contraindication-relevant text so
    indication-context diseases are never mined. Returns ``""`` when no sentence matches.
    """
    filtered, _mappings = _filtered_evidence(text, keywords)
    return filtered


def _resolve_keywords(ctx: TaskContext) -> re.Pattern[str]:
    """The contraindication keyword pattern: ``ctx.params["contraindication_keywords"]`` or default.

    Accepts a raw string (compiled case-insensitively) or a pre-compiled ``re.Pattern``.
    Returns :data:`DEFAULT_CONTRA_KEYWORDS` when the param is absent.
    """
    raw = ctx.params.get("contraindication_keywords")
    if raw is None:
        return DEFAULT_CONTRA_KEYWORDS
    if isinstance(raw, re.Pattern):
        return raw
    return re.compile(raw, re.IGNORECASE)


class ContraindicationsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, "shape_contraindications"):
            ner_param = ctx.params.get("ner")
            ner = ner_param if isinstance(ner_param, DiseaseNER) else default_ner(ctx.fixture_root)
            devices = _resolve_devices(ner)
            keywords = _resolve_keywords(ctx)
            if devices:
                stats(logger, "shape_contraindications", dispatch_gpus=len(devices))
            with MentionCache(ctx.workdir) as cache:
                rows = build_contraindication_rows(inputs, ner, devices=devices, keywords=keywords, cache=cache, ctx=ctx)
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_contraindications")


def build_contraindication_rows(
    inputs: Iterable[ArtifactRef],
    ner: DiseaseNER,
    *,
    devices: Sequence[str] | None = None,
    keywords: re.Pattern[str] | None = None,
    cache: MentionCache | None = None,
    ctx: TaskContext | None = None,
) -> list[dict[str, str]]:
    """Mine DailyMed contraindication + indication + warning sections into assertion rows (deterministic).

    **Pass 1:** For each singleton-ingredient SPL set with a contraindication section (LOINC
    ``34070-3``), extract disease/phenotype mentions from the full section text.

    **Pass 2:** For each singleton-ingredient SPL set with an indication section (LOINC
    ``34067-9``), filter the text
    to contraindication-context sentences (via ``keywords`` or :data:`DEFAULT_CONTRA_KEYWORDS`)
    and extract mentions from the filtered text only. This catches contraindications embedded
    in the indication section while avoiding false positives from indication-context diseases.

    **Pass 3:** For each singleton-ingredient SPL set with a boxed-warning section (LOINC
    ``34066-1``) or a warnings/precautions section (``43685-7``, ``34071-1``, ``42232-9``),
    extract mentions from the full section text. Acceptance is hard-trigger-only: a warning
    section's caution prose supplies no section-level contraindication context, so mentions
    become edges only when their sentence carries hard prohibition language.

    Mentions from all passes are paired with the set's single active ingredient and aggregated by
    ``(subject_text, object_text, disease_context_text)``. When ``devices`` is provided
    (production multi-GPU), the passes are dispatched across the GPUs concurrently
    (:func:`~dakp_pipeline.assertions.ner_dispatch.mine_passes_multi_gpu`). When ``cache`` (a
    persistent mention cache) is given, previously mined section texts are served from it via
    :func:`~dakp_pipeline.assertions.ner_dispatch.mine_with_cache` — it only changes WHERE
    mentions come from, never their content. Output is byte-identical regardless of dispatch
    mode or cache state. When ``ctx`` is given, the DailyMed evidence index is served from the
    store when another shape task already built it this run
    (:func:`~dakp_pipeline.assertions.evidence.load_or_build_dailymed_evidence`).
    """
    evidence = load_or_build_dailymed_evidence(inputs, ctx)
    kw = keywords or DEFAULT_CONTRA_KEYWORDS

    # Pass 1 work items: contraindication sections (all text is relevant), retaining the
    # original sentence spans for local evidence recovery.
    work_items_p1: list[ContraWorkItem] = []
    skipped_multi_ingredient = 0
    for set_id in sorted(evidence.contraindication_docs):
        ingredients = evidence.active_ingredients_by_set.get(set_id, [])
        if not ingredients:
            continue
        if len(ingredients) > 1:
            skipped_multi_ingredient += 1
            continue  # combination product: no single attributable subject (singleton rule)
        for doc_id, text in evidence.contraindication_docs[set_id]:
            source_spans = _sentence_spans(text)
            mappings = tuple(EvidenceSpan(span.start, span.end, span.start, span.end, span.text) for span in source_spans)
            work_items_p1.append(ContraWorkItem(set_id, doc_id, text, text, mappings, "contraindications"))

    # Pass 2 work items: indication sections filtered to contraindication-context sentences.
    # The mapping prevents the joined filtered text from destroying source offsets.
    work_items_p2: list[ContraWorkItem] = []
    for set_id in sorted(evidence.indication_docs):
        ingredients = evidence.active_ingredients_by_set.get(set_id, [])
        if not ingredients:
            continue
        if len(ingredients) > 1:
            skipped_multi_ingredient += 1
            continue  # combination product: no single attributable subject (singleton rule)
        for doc_id, text in evidence.indication_docs[set_id]:
            filtered, mappings = _filtered_evidence(text, kw)
            if filtered.strip():
                work_items_p2.append(ContraWorkItem(set_id, doc_id, filtered, text, mappings, "indications"))

    # Pass 3 work items: boxed-warning + warnings/precautions sections (full text, hard-trigger-only
    # acceptance), with the same singleton-ingredient rule and full-text identity mappings as Pass 1.
    work_items_p3: list[ContraWorkItem] = []
    for section_kind, docs in (("boxed_warning", evidence.boxed_warning_docs), ("warnings", evidence.warning_docs)):
        for set_id in sorted(docs):
            ingredients = evidence.active_ingredients_by_set.get(set_id, [])
            if not ingredients:
                continue
            if len(ingredients) > 1:
                skipped_multi_ingredient += 1
                continue  # combination product: no single attributable subject (singleton rule)
            for doc_id, text in docs[set_id]:
                source_spans = _sentence_spans(text)
                mappings = tuple(EvidenceSpan(span.start, span.end, span.start, span.end, span.text) for span in source_spans)
                work_items_p3.append(ContraWorkItem(set_id, doc_id, text, text, mappings, section_kind))

    all_work_items = work_items_p1 + work_items_p2 + work_items_p3
    stats(
        logger,
        "shape_contraindications",
        contraindication_sets=len(evidence.contraindication_docs),
        indication_sets=len(evidence.indication_docs),
        skipped_multi_ingredient_sets=skipped_multi_ingredient,
        pass1_sections=len(work_items_p1),
        pass2_sections=len(work_items_p2),
        pass3_sections=len(work_items_p3),
        sections_to_mine=len(all_work_items),
    )

    # Extract mentions: multi-pass multi-GPU when devices given + production NER + >1 item;
    # else sequential (with periodic progress narration — GLiNER mining is the slow step).
    # mine_with_cache fronts the whole block: hits never reach the miners, only misses are
    # dispatched (each pass is filtered to its own misses, preserving the pass split).
    p1_items, p2_items, p3_items = set(work_items_p1), set(work_items_p2), set(work_items_p3)

    def mine(items: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        if devices and len(items) > 1 and not ner._offline:
            miss_p1 = [item for item in items if item in p1_items]
            miss_p2 = [item for item in items if item in p2_items]
            miss_p3 = [item for item in items if item in p3_items]
            if miss_p3:
                return mine_passes_multi_gpu([miss_p1, miss_p2, miss_p3], ner, devices)
            if miss_p2 and len(devices) >= 2:
                return _mine_two_passes_multi_gpu(miss_p1, miss_p2, ner, devices)
            # No Pass 2/3 work, or too few devices to split — all GPUs on combined work.
            return _mine_multi_gpu(list(items), ner, devices)
        mined_seq: dict[tuple[str, str], list[Mention]] = {}
        for done, item in enumerate(items, start=1):
            set_id, doc_id, text = _work_item_parts(item)
            mined_seq[(set_id, doc_id)] = extract_contraindication_diseases(text, ner)
            progress(logger, "shape_contraindications", done, len(items), every=_MINING_PROGRESS_EVERY)
        return mined_seq

    mined = mine_with_cache(all_work_items, ner, mine, cache)

    # Aggregate mentions into assertion rows keyed by (subject, object, disease context).
    # Work items are singleton-only, so each set contributes exactly one subject ingredient;
    # the local source sentence is retained on the aggregate for the evidence column.
    aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
    mentions_mined = 0
    for item in all_work_items:
        set_id, doc_id, _text = _work_item_parts(item)
        ingredients = evidence.active_ingredients_by_set.get(set_id, [])
        mentions = mined.get((set_id, doc_id), [])
        decisions = _classify_mentions(item, mentions)
        for mention, decision in zip(mentions, decisions, strict=True):
            mentions_mined += 1
            # Canonicalize the mined mention (lowercase / strip punctuation) so case variants
            # (asthma / Asthma / ASTHMA) aggregate to one object instead of fragmenting the rows.
            object_text = normalize_text(mention.text)
            if not object_text or not decision.accepted:
                continue
            for ingredient_name, ingredient_unii in ingredients:
                _accumulate(
                    aggregated,
                    set_id,
                    doc_id,
                    ingredient_name,
                    ingredient_unii,
                    object_text,
                    mention,
                    decision.evidence_text,
                    decision.context_text,
                    evidence.approval_ids_for_sets([set_id]),
                )

    stats(logger, "shape_contraindications", mentions_mined=mentions_mined, assertions=len(aggregated))
    return [_finalize_row(agg) for _key, agg in sorted(aggregated.items())]


def _accumulate(
    aggregated: dict[tuple[str, str, str], dict[str, Any]],
    set_id: str,
    doc_id: str,
    ingredient_name: str,
    ingredient_unii: str,
    object_text: str,
    mention: Mention,
    evidence_text: str = "",
    context_text: str = "",
    approval_ids: Iterable[str] = (),
) -> None:
    """Add one observation to the ``(subject, object, disease-context)`` aggregate.

    The subject carries the SPL-provided ingredient text + UNII; the object and optional context
    are mined mention text with CURIE/name/category left empty for Tablassert/fullmap to resolve.
    """
    # Context is part of the semantic identity: unconditional evidence must never inherit a
    # conditional qualifier (or vice versa) merely because subject/object are equal.
    key = (ingredient_name, object_text, context_text.strip())
    agg = aggregated.setdefault(
        key,
        {
            "subject_text": ingredient_name,
            "subject_curie": ingredient_unii,
            "subject_name": ingredient_name,
            "object_text": object_text,
            "object_curie": "",
            "object_name": "",
            "object_category": "",
            "disease_context_text": context_text.strip(),
            "sets": [],
            "docs": [],
            "approval_ids": [],
            "scores": [],
            "evidence_texts": [],
        },
    )
    agg["sets"].append(set_id)
    agg["docs"].append(doc_id)
    agg["approval_ids"].extend(approval_ids)
    agg["scores"].append(mention.score)
    if evidence_text.strip():
        agg["evidence_texts"].append(evidence_text.strip())


def _finalize_row(agg: dict[str, Any]) -> dict[str, str]:
    return row_for(
        _TABLE,
        subject_text=agg["subject_text"],
        subject_curie=agg["subject_curie"],
        subject_name=agg["subject_name"],
        subject_category="ChemicalEntity",
        predicate=_PREDICATE,
        object_text=agg["object_text"],
        object_curie=agg["object_curie"],
        object_name=agg["object_name"],
        object_category=agg["object_category"],
        disease_context_text=agg.get("disease_context_text", ""),
        evidence_text=sorted_pipe(agg.get("evidence_texts", [])),
        supporting_spl_sets=sorted_pipe(dailymed_set_url(set_id) for set_id in agg["sets"]),
        supporting_spl_documents=sorted_pipe(dailymed_document_url(doc_id) for doc_id in agg["docs"]),
        supporting_spl_evidence=spl_evidence_pipe(agg["sets"], agg["docs"]),
        approval_ids=sorted_pipe(agg.get("approval_ids", [])),
        edge_evidence=edge_evidence_pipe(spl_evidence_pipe(agg["sets"], agg["docs"]).split("|") if spl_evidence_pipe(agg["sets"], agg["docs"]) else []),
        source_score=_max_score(agg["scores"]),
        knowledge_level=KL_ASSERTION,
        agent_type=AT_TEXT_MINING,
        primary_knowledge_source=INFORES_DAKP,
        upstream_resource_ids=INFORES_DAILYMED,
    )


def _max_score(scores: list[float]) -> str:
    """Highest NER span score as a deterministic string ("" if none)."""
    if not scores:
        return ""
    return f"{max(scores):g}"


transform = ContraindicationsShaper().transform

__all__ = [
    "AT_TEXT_MINING",
    "CONTRAINDICATION_GPUS",
    "DEFAULT_CONTRA_KEYWORDS",
    "ContraWorkItem",
    "ContraindicationsShaper",
    "EvidenceSpan",
    "MentionDecision",
    "SentenceSpan",
    "build_contraindication_rows",
    "default_ner",
    "transform",
]
