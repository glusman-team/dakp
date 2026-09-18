"""Approved-treatment assertion aggregation.

Builds ``approved_treats_assertions.tsv``: FDA-approved drug→condition assertions.

Aggregation rule (explicit and tested)
---------------------------------------
An approved-treats row requires an **NDA-bearing drug-indication pair** that satisfies all of:

1. the pair's NDA maps (Drugs@FDA ``products``) to an ingredient — confirming a real FDA
   application;
2. the NDA has a **DailyMed SPL approval** (``spl_approvals``);
3. that approved SPL set has an **indications-and-usage section** (LOINC ``34067-9``) — the
   SPL indication support; and
4. the candidate **condition is corroborated by that section's text** — a lexical
   disease-dictionary match on the section (CURIE match when both sides carry one, else
   normalized-text equality) or a verbatim word-bounded mention of the candidate indication
   text in the section (same normalized-space boundaries as
   :class:`~dakp_pipeline.ner.lexical.LexicalMatcher`) or an **NER mention** whose normalized
   text equals the candidate or is word-contained IN it (the label naming the general
   condition — e.g. ``breast cancer`` — corroborates the more specific FAERS report —
   e.g. ``hormone receptor positive breast cancer``; the reverse direction would assert
   more than the label supports and is never accepted). All three channels ignore sentences
   that disclaim efficacy or indication ("not indicated", "not recommended", "not approved",
   "not been established" — see :func:`_positive_context_text`), so boilerplate like the
   baclofen label's "efficacy ... in stroke, cerebral palsy, and Parkinson's disease has not
   been established" never corroborates the very conditions it disclaims
   (NCATSTranslator/Feedback#1360). Candidates whose condition appears on
   no supporting label are dropped (``dropped_no_label_term_support``) — the legacy
   ``supportInDailyMed`` gate (``ref/legacy/bin/drug2indi2kg.py``).

Candidate drug-indication pairs come from **FAERS** (``cases.parquet`` rows carrying an NDA +
indication) — the primary source, wired into this stage by the DAG. Placeholder/usage-context
FAERS indications ("Product used for unknown indication", bare "Prophylaxis", ...) are dropped
via the shared :func:`~dakp_pipeline.assertions.observed_uses.is_non_disease_indication`
stop-list. When no FAERS case table is present the candidates fall back to DailyMed SPL
indication sections whose text names a dictionary condition or an NER disease/phenotype
mention (rule 4 holds there by construction). Both paths apply the *same* four-part filter above.

The NER backend
---------------
The shaper uses an injected ``params["ner"]`` :class:`~dakp_pipeline.ner.ner.DiseaseNER` when
present (production GLiNER wiring in the DAG), else the deterministic **offline** backend from
:func:`~dakp_pipeline.assertions.ner_dispatch.default_ner`. Indication sections are mined once
per ``(set_id, doc_id)`` — never per candidate — with the same multi-GPU dispatch as the
contraindication shaper (:mod:`~dakp_pipeline.assertions.ner_dispatch`). The mention channels
degrade gracefully: calling :func:`build_approved_treats_rows` without a ``ner`` keeps the
pure lexical behavior (the historical direct-call test surface).

Provenance (``FDA_regulatory_approvals``, ``supporting_spl_sets``, ``supporting_spl_documents``) is
aggregated per ``(subject, object)`` as deduplicated, sorted, pipe-joined lists, restricted to
the sets whose indication text actually mentions the condition (the legacy "SPLs containing
both UNII and CURIE"). ``FDA_regulatory_approvals`` values use the FDA display form
``<application type><number>`` (e.g. ``BLA103795``), expanded from the raw application number
by :class:`~dakp_pipeline.assertions.evidence.FDAApprovalIndex`. The unannotated
``supporting_spl_sets`` / ``supporting_spl_documents`` debug columns carry DailyMed label URLs
(``https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=<spl_set_id>[#<loinc>]``) so the
values are directly clickable links, and ``supporting_spl_evidence`` carries the backing SPL
sets as ``dailymed:<spl_set_id>`` CURIEs in the single column Tablassert encodes as the Biolink
``publications`` slot (see :func:`~dakp_pipeline.assertions.evidence.spl_evidence_pipe`). Subject CURIEs
are populated only where DailyMed already gives a UNII; object CURIEs come from the lexical
disease baseline. Canonical CURIE mapping is a later milestone (text-first).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

import polars as pl

from dakp_pipeline.assertions import (
    AT_MANUAL,
    INFORES_DAILYMED,
    INFORES_DAKP,
    INFORES_FAERS,
    KL_ASSERTION,
    join_pipe,
    match_diseases,
    object_mentions,
    row_for,
)
from dakp_pipeline.assertions.contexts import assertion_context, attach_qualifiers_with_scores
from dakp_pipeline.assertions.evidence import (
    DailyMedEvidence,
    FDAApprovalIndex,
    build_drugsfda_ingredient_map,
    build_fda_approval_index,
    dailymed_document_url,
    dailymed_set_url,
    faers_quarter_urls,
    faers_record_url,
    find_faers_cases,
    load_or_build_dailymed_evidence,
    sorted_pipe,
    spl_evidence_pipe,
    write_assertion_table,
)
from dakp_pipeline.assertions.ner_dispatch import _mine_multi_gpu, _resolve_devices, default_ner, mine_with_cache
from dakp_pipeline.assertions.observed_uses import is_non_disease_indication
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, progress, stats, step
from dakp_pipeline.ner.dictionary import normalize_text, normalize_with_map
from dakp_pipeline.ner.mention_cache import MentionCache
from dakp_pipeline.ner.ner import DiseaseNER, Mention

_TABLE = "approved_treats_assertions"
_PREDICATE = "biolink:treats"
_STATUS = "approved_for_condition"

#: Case-table columns the FAERS candidate path reads (projection keeps production-scale reads cheap).
_FAERS_CASE_COLUMNS = ("nda", "nda_raw", "indication", "ingredient", "drugname", "quarter", "primaryid", "source_record_id")

#: One INFO progress line per this many mined indication sections (GLiNER is the slow step).
_MINING_PROGRESS_EVERY = 500

#: Sentence-level cues for SPL indication-section boilerplate that DISCLAIMS efficacy or
#: indication rather than asserting it — e.g. the baclofen label's "The efficacy of baclofen in
#: stroke, cerebral palsy, and Parkinson's disease has not been established and, therefore, it is
#: not recommended for these conditions." A condition named only in such a sentence is the label
#: saying the drug does NOT treat it, so the sentence must not corroborate a treats candidate
#: (NCATSTranslator/Feedback#1360).
_NEGATION_CUES = re.compile(r"\b(?:not indicated|not recommended|not approved|not been established)\b", re.IGNORECASE)

#: Sentence/bullet boundaries inside an SPL section: blank-or-single newlines separate the
#: section's bulleted lines; sentence-final punctuation followed by whitespace separates prose
#: sentences within a line.
_SENTENCE_BOUNDARY = re.compile(r"(?:\r?\n)+|(?<=[.!?])\s+")


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Return non-empty sentence/bullet spans with offsets into one indication section."""
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        raw = text[start : match.start()]
        if raw.strip():
            left = start + len(raw) - len(raw.lstrip())
            spans.append((left, start + len(raw.rstrip()), text[left : start + len(raw.rstrip())]))
        start = match.end()
    raw = text[start:]
    if raw.strip():
        left = start + len(raw) - len(raw.lstrip())
        spans.append((left, start + len(raw.rstrip()), text[left : start + len(raw.rstrip())]))
    return spans


def _sentence_local(mention: Mention, sentence_start: int, sentence_end: int, sentence: str) -> Mention:
    """Rebase ``mention`` into sentence-relative offsets, CLIPPING it to the sentence.

    A mined span can straddle a sentence boundary: GLiNER predicts over token windows that tile
    the whole section, and :func:`~dakp_pipeline.ner.ner._merge_straddling_spans` deliberately
    re-unions a phrase a hard window split cut. Overlap alone therefore does not make a span
    sentence-local, and a bare offset shift yields a negative ``start`` or an ``end`` past the
    sentence — offsets :func:`~dakp_pipeline.assertions.contexts.attach_qualifiers_with_scores`
    rejects by contract (``ValueError``), which failed run 10's ``shape_treatment_tables`` AFTER
    its 38 minutes of mining had completed and cached. Clipping keeps the part of the span inside
    this sentence, the same policy :func:`dakp_pipeline.assertions.contraindications._mention_local_span`
    applies, and re-slicing the surface keeps ``mention.text == sentence[start:end]`` true.
    """
    start = max(mention.start, sentence_start) - sentence_start
    end = min(mention.end, sentence_end) - sentence_start
    return replace(mention, start=start, end=end, text=sentence[start:end])


def _doc_key(doc_id: str, occurrence: int) -> str:
    """Key the Nth indication section one SPL document contributes (``occurrence`` counts from 0).

    ``doc_id`` is the SPL DOCUMENT id, so a document carrying two indication sections yields two
    work items with the same ``(set_id, doc_id)`` pair — and the mining map is keyed by that pair,
    so the two sections would exchange mentions mined from each other's text (offsets into the
    wrong string; 20,467 real pairs do exactly this). The first section keeps the bare doc_id (the
    historical key) and later ones get an ordinal suffix. This is a ROUTING key only: emitted rows
    and observation records keep the real ``doc_id``, and the mention-cache key is text-derived.
    """
    return doc_id if occurrence == 0 else f"{doc_id}#{occurrence + 1}"


def _candidate_mention(sentence: str, candidate: Mapping[str, str], offset: int) -> Mention | None:
    """Create a lexical host only when the candidate is explicitly present in this sentence."""
    needle = normalize_text(candidate["object_text"])
    if not needle:
        return None
    match = re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", sentence, re.IGNORECASE)
    if match is None:
        normalized_sentence, index_map = normalize_with_map(sentence)
        match = re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", normalized_sentence, re.IGNORECASE)
        if match is None:
            return None
        original_start = index_map[match.start()]
        original_end = index_map[match.end() - 1] + 1
    else:
        original_start, original_end = match.start(), match.end()
    start = offset + original_start
    return Mention(text=sentence[original_start:original_end], start=start, end=offset + original_end, type=candidate["object_category"], score=1.0)


def _model_matches_candidate(mention: Mention, candidate_text: str) -> bool:
    mention_text = normalize_text(mention.text)
    needle = normalize_text(candidate_text)
    return bool(mention_text and (mention_text == needle or f" {mention_text} " in f" {needle} "))


def _indication_observations(
    dailymed: DailyMedEvidence,
    sets: Sequence[str],
    candidate: Mapping[str, str],
    disease_map: Mapping[str, Mapping[str, str]],
    mentions: Mapping[tuple[str, str], list[Mention]] | None,
) -> list[dict[str, Any]]:
    """Preserve the sentence, context, host, and model corroboration for each support document."""
    observations: list[dict[str, Any]] = []
    for set_id in sets:
        for occurrence, (doc_id, text) in enumerate(dailymed.indication_docs.get(set_id, [])):
            doc_mentions = list((mentions or {}).get((set_id, _doc_key(doc_id, occurrence)), []))
            if not _section_mentions_condition(text, candidate, disease_map, doc_mentions):
                continue
            for start, _end, sentence in _sentence_spans(text):
                if not _section_mentions_condition(sentence, candidate, disease_map, doc_mentions):
                    continue
                sentence_end = start + len(sentence)
                sentence_mentions = [m for m in doc_mentions if m.start < sentence_end and start < m.end]
                local_mentions = [_sentence_local(m, start, sentence_end, sentence) for m in sentence_mentions]
                objects = object_mentions(local_mentions)
                host = [m for m in objects if _model_matches_candidate(m, candidate["object_text"])]
                if not host:
                    lexical = _candidate_mention(sentence, candidate, 0)
                    if lexical is not None:
                        host = [lexical]
                qualifiers = [m for m in local_mentions if m not in objects]
                sentence_of = lambda _mention, value=sentence: value
                attached, qualifier_scores = attach_qualifiers_with_scores(host, qualifiers, sentence_of) if host else ({}, {})
                merged_qualifiers: dict[str, str] = {}
                merged_scores: dict[str, tuple[float, str]] = {}
                for (host_index, field), score in qualifier_scores.items():
                    if field not in merged_scores or score > merged_scores[field]:
                        merged_scores[field] = score
                        merged_qualifiers[field] = attached[host_index][field]
                context = assertion_context("dailymed", "34067-9", sentence)
                best_model = max((m for m in host if m.context_model), key=lambda m: (m.context_model_score, m.context_model), default=None)
                observations.append(
                    {
                        "set_id": set_id,
                        "doc_id": doc_id,
                        "context": context,
                        "qualifiers": merged_qualifiers,
                        "qualifier_scores": merged_scores,
                        "model": best_model.context_model if best_model else "",
                        "model_score": best_model.context_model_score if best_model else 0.0,
                    }
                )
    return observations


def _positive_context_text(section_text: str) -> str:
    """The indication section minus every sentence/bullet that disclaims efficacy or indication.

    Splitting is per-sentence, so a disclaimer naming several conditions ("...in stroke, cerebral
    palsy, and Parkinson's disease has not been established...") removes them all, while a
    positive sentence elsewhere in the same section still corroborates its conditions.
    """
    kept = [chunk for chunk in _SENTENCE_BOUNDARY.split(section_text or "") if chunk.strip() and not _NEGATION_CUES.search(chunk)]
    return " ".join(kept)


class ApprovedTreatsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, "shape_approved_treats"):
            disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
            ner_param = ctx.params.get("ner")
            ner = ner_param if isinstance(ner_param, DiseaseNER) else default_ner(ctx.fixture_root)
            devices = _resolve_devices(ner)
            stats(logger, "shape_approved_treats", inputs=len(inputs), disease_map_terms=len(disease_map))
            dailymed = load_or_build_dailymed_evidence(inputs, ctx)
            drugsfda_map = build_drugsfda_ingredient_map(inputs)
            approvals = build_fda_approval_index(inputs)
            faers_cases = find_faers_cases(inputs, columns=_FAERS_CASE_COLUMNS)
            quarter_urls = faers_quarter_urls(inputs)
            with MentionCache(ctx.workdir) as cache:
                rows = build_approved_treats_rows(
                    faers_cases,
                    dailymed,
                    drugsfda_map,
                    disease_map,
                    approvals=approvals,
                    ner=ner,
                    devices=devices,
                    cache=cache,
                    faers_quarter_urls=quarter_urls,
                )
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_approved_treats")


def _mine_indication_mentions(
    dailymed: DailyMedEvidence, ner: DiseaseNER, devices: Sequence[str] | None, cache: MentionCache | None = None
) -> dict[tuple[str, str], list[Mention]]:
    """Mine every indication section ONCE, returning ``{(set_id, doc_key): [mentions]}``.

    Sections are mined per document section — keyed by :func:`_doc_key`, because one SPL document
    can contribute several indication sections and ``(set_id, doc_id)`` alone would collide — and
    shared by both candidate paths (FAERS corroboration + DailyMed fallback), never re-mined per
    candidate. Production runs dispatch across GPUs
    (:func:`~dakp_pipeline.assertions.ner_dispatch._mine_multi_gpu`); the offline gazetteer backend
    runs sequentially with periodic progress narration. When ``cache`` is given, previously mined
    texts are served from the persistent mention cache
    (:func:`~dakp_pipeline.assertions.ner_dispatch.mine_with_cache`). Output is identical
    regardless of dispatch mode or cache state.
    """
    work_items = [
        (set_id, _doc_key(doc_id, occurrence), text)
        for set_id in sorted(dailymed.indication_docs)
        for occurrence, (doc_id, text) in enumerate(dailymed.indication_docs[set_id])
    ]
    if not work_items:
        return {}

    def mine(items: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        if devices and len(items) > 1 and not ner._offline:
            return _mine_multi_gpu(items, ner, devices)
        mined: dict[tuple[str, str], list[Mention]] = {}
        for done, (set_id, doc_id, text) in enumerate(items, start=1):
            mined[(set_id, doc_id)] = ner.extract(text)
            progress(logger, "shape_approved_treats", done, len(items), every=_MINING_PROGRESS_EVERY)
        return mined

    return mine_with_cache(work_items, ner, mine, cache)


def build_approved_treats_rows(
    faers_cases: pl.DataFrame | None,
    dailymed: DailyMedEvidence,
    drugsfda_map: Mapping[str, set[str]],
    disease_map: Mapping[str, Mapping[str, str]],
    *,
    approvals: FDAApprovalIndex | None = None,
    ner: DiseaseNER | None = None,
    devices: Sequence[str] | None = None,
    cache: MentionCache | None = None,
    faers_quarter_urls: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Aggregate approved-treats assertion rows (pure; deterministic ordering).

    When ``ner`` is given, every indication section is mined once and its mentions feed the
    corroboration gate (rule 4) and the DailyMed fallback candidate path; without a backend the
    historical lexical-only behavior is kept. ``cache`` (a persistent mention cache) only
    changes WHERE mentions come from, never their content.

    ``approvals`` expands each application number to its FDA display form (``BLA125514``);
    without it the DailyMed label's own prefixed id is used unchanged.
    """
    approvals = approvals if approvals is not None else FDAApprovalIndex()
    mentions = _mine_indication_mentions(dailymed, ner, devices, cache) if ner is not None else None
    candidates = (
        _faers_candidates(faers_cases, disease_map, faers_quarter_urls)
        if faers_cases is not None
        else _dailymed_candidates(dailymed, disease_map, mentions)
    )

    aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
    candidates_seen = 0
    dropped_no_ingredient_map = 0
    dropped_no_spl_support = 0
    dropped_no_label_term_support = 0
    dropped_no_subject = 0
    for cand in candidates:
        candidates_seen += 1
        norm = cand["norm_nda"]
        if norm not in drugsfda_map:  # (1) NDA must map (Drugs@FDA) to an ingredient
            dropped_no_ingredient_map += 1
            continue
        sets, _docs = dailymed.indication_support(norm)
        if not sets:  # (2)+(3) DailyMed approval AND SPL indication-section support
            dropped_no_spl_support += 1
            continue
        support_sets = _condition_corroborated_sets(dailymed, sets, cand, disease_map, mentions)
        if not support_sets:  # (4) the condition must actually appear on a supporting label
            dropped_no_label_term_support += 1
            continue
        observations = _indication_observations(dailymed, support_sets, cand, disease_map, mentions)
        if not observations:
            dropped_no_label_term_support += 1
            continue
        subject_text, subject_curie = _subject_for_sets(dailymed, support_sets, cand["fallback_subject"])
        if not subject_text:
            dropped_no_subject += 1
            continue
        for observation in observations:
            context = observation["context"]
            key = (subject_text, cand["object_text"], context)
            agg = aggregated.setdefault(
                key,
                {
                    "subject_text": subject_text,
                    "subject_curie": subject_curie,
                    "object_text": cand["object_text"],
                    "object_curie": cand["object_curie"],
                    "object_name": cand["object_name"],
                    "object_category": cand["object_category"],
                    "FDA_regulatory_approvals": [],
                    "sets": [],
                    "docs": [],
                    "faers_source_records": [],
                    "faers_urls": [],
                    "assertion_context": context,
                    "assertion_context_model": "",
                    "assertion_context_model_score": 0.0,
                    "qualifiers": {},
                    "qualifier_scores": {},
                },
            )
            agg["FDA_regulatory_approvals"].extend(approvals.expand(dailymed.approval_display.get(norm) or norm))
            agg["sets"].append(observation["set_id"])
            agg["docs"].append(observation["doc_id"])
            agg["faers_source_records"].extend(cand.get("faers_source_records", []))
            agg["faers_urls"].extend(cand.get("faers_urls", []))
            model_score = float(observation["model_score"])
            if model_score > float(agg["assertion_context_model_score"]):
                agg["assertion_context_model"] = observation["model"]
                agg["assertion_context_model_score"] = model_score
            for field, value in observation["qualifiers"].items():
                score = observation.get("qualifier_scores", {}).get(field, (0.0, value))
                previous = agg["qualifier_scores"].get(field)
                if previous is None or score > previous:
                    agg["qualifier_scores"][field] = score
                    agg["qualifiers"][field] = value
            if not agg["subject_curie"] and subject_curie:
                agg["subject_curie"] = subject_curie
        # object_curie needs no back-fill: it is a deterministic function of object_text (the
        # aggregation key), so every candidate for a key carries the same value already set above.

    stats(
        logger,
        "shape_approved_treats",
        candidates=candidates_seen,
        dropped_no_ingredient_map=dropped_no_ingredient_map,
        dropped_no_spl_support=dropped_no_spl_support,
        dropped_no_label_term_support=dropped_no_label_term_support,
        dropped_no_subject=dropped_no_subject,
        assertions=len(aggregated),
    )
    return [_finalize_row(agg) for _key, agg in sorted(aggregated.items())]


def _finalize_row(agg: dict[str, Any]) -> dict[str, str]:
    return row_for(
        _TABLE,
        subject_text=agg["subject_text"],
        subject_curie=agg["subject_curie"],
        subject_name=agg["subject_text"],
        subject_category="ChemicalEntity",
        predicate=_PREDICATE,
        object_text=agg["object_text"],
        object_curie=agg["object_curie"],
        object_name=agg["object_name"],
        object_category=agg["object_category"],
        assertion_context=agg.get("assertion_context", "indication"),
        FDA_regulatory_approvals=sorted_pipe(agg["FDA_regulatory_approvals"]),
        supporting_spl_sets=sorted_pipe(dailymed_set_url(set_id) for set_id in agg["sets"]),
        supporting_spl_documents=sorted_pipe(dailymed_document_url(doc_id) for doc_id in agg["docs"]),
        supporting_spl_evidence=spl_evidence_pipe(agg["sets"], agg["docs"]),
        edge_evidence=spl_evidence_pipe(agg["sets"], agg["docs"]),
        supporting_faers_records=sorted_pipe(agg.get("faers_source_records", [])),
        supporting_faers_urls=sorted_pipe(agg.get("faers_urls", [])),
        assertion_context_model=agg.get("assertion_context_model", ""),
        assertion_context_model_score=agg.get("assertion_context_model_score", ""),
        **agg.get("qualifiers", {}),
        clinical_approval_status=_STATUS,
        knowledge_level=KL_ASSERTION,
        agent_type=AT_MANUAL,
        primary_knowledge_source=INFORES_DAKP,
        upstream_resource_ids=join_pipe(INFORES_DAILYMED, INFORES_FAERS),
    )


def _condition_corroborated_sets(
    dailymed: DailyMedEvidence,
    sets: list[str],
    cand: Mapping[str, str],
    disease_map: Mapping[str, Mapping[str, str]],
    mentions: Mapping[tuple[str, str], list[Mention]] | None = None,
) -> list[str]:
    """The supporting sets whose indication-section text actually mentions the candidate condition."""
    return [
        set_id
        for set_id in sets
        if any(
            _section_mentions_condition(text, cand, disease_map, (mentions or {}).get((set_id, _doc_key(doc_id, occurrence))))
            for occurrence, (doc_id, text) in enumerate(dailymed.indication_docs[set_id])
        )
    ]


def _section_mentions_condition(
    section_text: str, cand: Mapping[str, str], disease_map: Mapping[str, Mapping[str, str]], mentions: list[Mention] | None = None
) -> bool:
    """True when the indication section names the candidate condition (dictionary, verbatim, or NER).

    A disease-dictionary match on the section counts when it corresponds to the candidate object:
    CURIE equality when both sides carry one, else normalized-text equality. The production
    dictionary baseline is small, so a verbatim word-bounded mention of the candidate indication
    text (normalized space, mirroring :class:`~dakp_pipeline.ner.lexical.LexicalMatcher`) also
    counts — real FAERS indications quoted on the label are not dropped for lack of a dictionary
    entry. Finally, an NER mention of the section corroborates the candidate when their normalized
    texts are equal or the mention is word-contained IN the candidate: the label naming the general
    condition (``breast cancer``) covers the specific FAERS report (``hormone receptor positive
    breast cancer``). The reverse (mention more specific than the candidate) never corroborates —
    and needs no rule: it is already covered by the verbatim check.

    All three channels run against :func:`_positive_context_text` output: sentences that disclaim
    efficacy or indication ("not indicated", "not recommended", "not approved", "not been
    established") are stripped first, so a condition named only in such boilerplate — the label
    saying the drug does NOT treat it — never corroborates (NCATSTranslator/Feedback#1360). An NER
    mention must additionally survive in the positive remainder, since mentions are mined from the
    full (unstripped) section.
    """
    needle = normalize_text(cand["object_text"])
    if not needle:
        return False
    positive_text = _positive_context_text(section_text)
    for match in match_diseases(positive_text, disease_map):
        if cand["object_curie"] and match["curie"]:
            if match["curie"] == cand["object_curie"]:
                return True
        elif normalize_text(match["text"]) == needle:
            return True
    normalized_section = normalize_text(positive_text)
    if f" {needle} " in f" {normalized_section} ":
        return True
    for mention in object_mentions(mentions or []):
        mention_text = normalize_text(mention.text)
        if mention_text and f" {mention_text} " in f" {normalized_section} " and (mention_text == needle or f" {mention_text} " in f" {needle} "):
            return True
    return False


def _subject_for_sets(dailymed: DailyMedEvidence, sets: list[str], fallback: str) -> tuple[str, str]:
    """Subject ingredient (name, UNII) from the first singleton supporting SPL set, else the FAERS fallback.

    Only single-active-ingredient sets are trusted for drug identity (the legacy
    ``selectActiveIngredientSingletons.pl`` discipline): a combination product's label applies to
    the mixture, so adopting one of its several actives would over-attribute the treatment to that
    component. Multi-ingredient sets fall through to the FAERS-reported subject.
    """
    for set_id in sets:  # already sorted deterministically
        ingredients = dailymed.active_ingredients_by_set.get(set_id, [])
        if len(ingredients) == 1:
            return ingredients[0]
    return fallback.strip(), ""


def _object_attrs(text: str, disease_map: Mapping[str, Mapping[str, str]]) -> tuple[str, str, str]:
    """Resolve ``(curie, name, category)`` for an object text via the lexical disease baseline."""
    matches = match_diseases(text, disease_map)
    if matches:
        match = matches[0]
        return match["curie"], match["name"], match["category"]
    return "", text, "Disease"


def _faers_candidates(
    faers_cases: pl.DataFrame, disease_map: Mapping[str, Mapping[str, str]], quarter_urls: Mapping[str, str] | None = None
) -> Iterator[dict[str, Any]]:
    """NDA-bearing FAERS pairs with all contributing report provenance retained.

    Candidate identity remains ``(normalized NDA, indication)`` and the first fallback subject
    wins exactly as before. The grouped rich-row list additionally preserves every contributing
    report/drug line so approved-treat edges can expose the FAERS evidence that supplied them.
    """

    def _text_column(name: str) -> pl.Expr:
        return pl.col(name).fill_null("").cast(pl.Utf8) if name in faers_cases.columns else pl.lit("")

    nda = _text_column("nda")
    nda_raw = _text_column("nda_raw")
    ingredient = _text_column("ingredient")
    drugname = _text_column("drugname")
    indication = _text_column("indication").str.strip_chars()
    quarter = _text_column("quarter")
    primaryid = _text_column("primaryid")
    source_record_id = _text_column("source_record_id")
    normalized_nda = pl.when(nda != "").then(nda).otherwise(nda_raw).str.replace_all(r"[^0-9]+", "").str.strip_chars_start("0")
    pairs = (
        faers_cases.lazy()
        .select(
            normalized_nda.alias("norm_nda"),
            indication.alias("indication"),
            pl.when(ingredient != "").then(ingredient).otherwise(drugname).str.strip_chars().alias("fallback_subject"),
            pl.struct(
                quarter.alias("quarter"), primaryid.alias("primaryid"), source_record_id.alias("source_record_id"), nda_raw.alias("nda_raw")
            ).alias("faers_row"),
        )
        .filter((pl.col("norm_nda") != "") & (pl.col("indication") != ""))
        .group_by("norm_nda", "indication", maintain_order=True)
        .agg(pl.col("fallback_subject").first().alias("fallback_subject"), pl.col("faers_row").unique().alias("faers_rows"))
        .collect()
    )
    for rec in pairs.iter_rows(named=True):
        object_text = str(rec["indication"])
        if is_non_disease_indication(object_text):
            continue  # FAERS placeholder/usage-context indication, not a drug->condition approval claim
        curie, name, category = _object_attrs(object_text, disease_map)
        source_records: set[str] = set()
        urls: set[str] = set()
        for raw in rec.get("faers_rows") or []:
            row = raw or {}
            q = str(row.get("quarter") or "").strip()
            pid = str(row.get("primaryid") or "").strip()
            if q and pid:
                urls.add(faers_record_url(q, dict(quarter_urls or {})))
            source_id = str(row.get("source_record_id") or "").strip()
            if source_id:
                source_records.add(source_id)
        yield {
            "norm_nda": str(rec["norm_nda"]),
            "object_text": object_text,
            "object_curie": curie,
            "object_name": name,
            "object_category": category,
            "fallback_subject": str(rec["fallback_subject"]),
            "faers_source_records": sorted(source_records),
            "faers_urls": sorted(urls),
        }


def _dailymed_candidates(
    dailymed: DailyMedEvidence, disease_map: Mapping[str, Mapping[str, str]], mentions: Mapping[tuple[str, str], list[Mention]] | None = None
) -> Iterator[dict[str, str]]:
    """Fallback candidates: dictionary conditions or NER mentions named in approved SPL indication sections.

    NER mentions the dictionary missed (production GLiNER recall) become text-only candidates
    (CURIE/name/category resolved via :func:`_object_attrs`, empty when unknown). A mention whose
    normalized text equals a dictionary match on the same document is skipped — offline
    (gazetteer) mentions coincide with dictionary matches, so offline candidates are unchanged.
    Only object-channel mentions are considered: qualifier mentions describe an object and must
    never become one (see :func:`~dakp_pipeline.assertions.object_mentions`).
    """
    set_to_ndas: dict[str, set[str]] = {}
    for norm, sets in dailymed.approval_sets.items():
        for set_id in sets:
            set_to_ndas.setdefault(set_id, set()).add(norm)

    seen: set[tuple[str, str]] = set()
    for set_id in sorted(dailymed.indication_docs):
        ndas = sorted(set_to_ndas.get(set_id, ()))
        if not ndas:
            continue
        for occurrence, (doc_id, text) in enumerate(dailymed.indication_docs[set_id]):
            matches = match_diseases(text, disease_map)
            for match in matches:
                for norm in ndas:
                    key = (norm, match["text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "norm_nda": norm,
                        "object_text": match["text"],
                        "object_curie": match["curie"],
                        "object_name": match["name"],
                        "object_category": match["category"],
                        "fallback_subject": "",
                    }
            dictionary_texts = {normalize_text(match["text"]) for match in matches}
            for mention in object_mentions((mentions or {}).get((set_id, _doc_key(doc_id, occurrence)), [])):
                object_text = normalize_text(mention.text)
                if not object_text or object_text in dictionary_texts:
                    continue
                curie, name, category = _object_attrs(object_text, disease_map)
                for norm in ndas:
                    key = (norm, object_text)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "norm_nda": norm,
                        "object_text": object_text,
                        "object_curie": curie,
                        "object_name": name,
                        "object_category": category,
                        "fallback_subject": "",
                    }


transform = ApprovedTreatsShaper().transform

__all__ = ["ApprovedTreatsShaper", "build_approved_treats_rows", "transform"]
