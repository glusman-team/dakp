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
   :class:`~dakp_pipeline.ner.lexical.LexicalMatcher`). Candidates whose condition appears on
   no supporting label are dropped (``dropped_no_label_term_support``) — the legacy
   ``supportInDailyMed`` gate (``ref/legacy/bin/drug2indi2kg.py``).

Candidate drug-indication pairs come from **FAERS** (``cases.parquet`` rows carrying an NDA +
indication) — the primary source, wired into this stage by the DAG. Placeholder/usage-context
FAERS indications ("Product used for unknown indication", bare "Prophylaxis", ...) are dropped
via the shared :func:`~dakp_pipeline.assertions.observed_uses.is_non_disease_indication`
stop-list. When no FAERS case table is present the candidates fall back to DailyMed SPL
indication sections whose text names a dictionary condition (rule 4 holds there by
construction). Both paths apply the *same* four-part filter above.

Provenance (``approval_ids``, ``supporting_spl_sets``, ``supporting_spl_documents``) is
aggregated per ``(subject, object)`` as deduplicated, sorted, pipe-joined lists, restricted to
the sets whose indication text actually mentions the condition (the legacy "SPLs containing
both UNII and CURIE"). ``approval_ids`` values use the legacy display form
``<application type><number>`` (e.g. ``BLA103795``). The unannotated
``supporting_spl_sets`` / ``supporting_spl_documents`` debug columns carry DailyMed label URLs
(``https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=<spl_set_id>[#<loinc>]``) so the
values are directly clickable links, and ``supporting_spl_evidence`` carries the backing SPL
sets as ``dailymed:<spl_set_id>`` CURIEs in the single column Tablassert encodes as Biolink
``has_evidence`` (see :func:`~dakp_pipeline.assertions.evidence.spl_evidence_pipe`). Subject CURIEs
are populated only where DailyMed already gives a UNII; object CURIEs come from the lexical
disease baseline. Canonical CURIE mapping is a later milestone (text-first).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import polars as pl

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_FAERS, KL_ASSERTION, join_pipe, match_diseases, row_for
from dakp_pipeline.assertions.evidence import (
    DailyMedEvidence,
    build_dailymed_evidence,
    build_drugsfda_ingredient_map,
    dailymed_document_url,
    dailymed_set_url,
    find_faers_cases,
    merge_unique,
    normalize_nda,
    sorted_pipe,
    spl_evidence_pipe,
    write_assertion_table,
)
from dakp_pipeline.assertions.observed_uses import is_non_disease_indication
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.ner.dictionary import normalize_text

_TABLE = "approved_treats_assertions"
_PREDICATE = "biolink:treats"
_STATUS = "approved_for_condition"

#: Case-table columns the FAERS candidate path reads (projection keeps production-scale reads cheap).
_FAERS_CASE_COLUMNS = ("nda", "nda_raw", "indication", "ingredient", "drugname")


class ApprovedTreatsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, "shape_approved_treats"):
            disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
            stats(logger, "shape_approved_treats", inputs=len(inputs), disease_map_terms=len(disease_map))
            dailymed = build_dailymed_evidence(inputs)
            drugsfda_map = build_drugsfda_ingredient_map(inputs)
            faers_cases = find_faers_cases(inputs, columns=_FAERS_CASE_COLUMNS)
            rows = build_approved_treats_rows(faers_cases, dailymed, drugsfda_map, disease_map)
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_approved_treats")


def build_approved_treats_rows(
    faers_cases: pl.DataFrame | None, dailymed: DailyMedEvidence, drugsfda_map: Mapping[str, set[str]], disease_map: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    """Aggregate approved-treats assertion rows (pure; deterministic ordering)."""
    candidates = _faers_candidates(faers_cases, disease_map) if faers_cases is not None else _dailymed_candidates(dailymed, disease_map)

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
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
        sets = _condition_corroborated_sets(dailymed, sets, cand, disease_map)
        if not sets:  # (4) the condition must actually appear on a supporting label
            dropped_no_label_term_support += 1
            continue
        docs = merge_unique(doc_id for set_id in sets for doc_id, _text in dailymed.indication_docs[set_id])
        subject_text, subject_curie = _subject_for_sets(dailymed, sets, cand["fallback_subject"])
        if not subject_text:
            dropped_no_subject += 1
            continue
        key = (subject_text, cand["object_text"])
        agg = aggregated.setdefault(
            key,
            {
                "subject_text": subject_text,
                "subject_curie": subject_curie,
                "object_text": cand["object_text"],
                "object_curie": cand["object_curie"],
                "object_name": cand["object_name"],
                "object_category": cand["object_category"],
                "approval_ids": [],
                "sets": [],
                "docs": [],
            },
        )
        agg["approval_ids"].append(dailymed.approval_display.get(norm) or norm)
        agg["sets"].extend(sets)
        agg["docs"].extend(docs)
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
        approval_ids=sorted_pipe(agg["approval_ids"]),
        supporting_spl_sets=sorted_pipe(dailymed_set_url(set_id) for set_id in agg["sets"]),
        supporting_spl_documents=sorted_pipe(dailymed_document_url(doc_id) for doc_id in agg["docs"]),
        supporting_spl_evidence=spl_evidence_pipe(agg["sets"], agg["docs"]),
        clinical_approval_status=_STATUS,
        knowledge_level=KL_ASSERTION,
        agent_type=AT_MANUAL,
        primary_knowledge_source=INFORES_DAKP,
        upstream_resource_ids=join_pipe(INFORES_DAILYMED, INFORES_FAERS),
    )


def _condition_corroborated_sets(
    dailymed: DailyMedEvidence, sets: list[str], cand: Mapping[str, str], disease_map: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """The supporting sets whose indication-section text actually mentions the candidate condition."""
    return [
        set_id for set_id in sets if any(_section_mentions_condition(text, cand, disease_map) for _doc_id, text in dailymed.indication_docs[set_id])
    ]


def _section_mentions_condition(section_text: str, cand: Mapping[str, str], disease_map: Mapping[str, Mapping[str, str]]) -> bool:
    """True when the indication section names the candidate condition (dictionary or verbatim).

    A disease-dictionary match on the section counts when it corresponds to the candidate object:
    CURIE equality when both sides carry one, else normalized-text equality. The production
    dictionary baseline is small, so a verbatim word-bounded mention of the candidate indication
    text (normalized space, mirroring :class:`~dakp_pipeline.ner.lexical.LexicalMatcher`) also
    counts — real FAERS indications quoted on the label are not dropped for lack of a dictionary
    entry.
    """
    needle = normalize_text(cand["object_text"])
    if not needle:
        return False
    for match in match_diseases(section_text, disease_map):
        if cand["object_curie"] and match["curie"]:
            if match["curie"] == cand["object_curie"]:
                return True
        elif normalize_text(match["text"]) == needle:
            return True
    normalized_section = normalize_text(section_text)
    return f" {needle} " in f" {normalized_section} "


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


def _faers_candidates(faers_cases: pl.DataFrame, disease_map: Mapping[str, Mapping[str, str]]) -> Iterator[dict[str, str]]:
    """NDA-bearing FAERS drug-indication pairs (deduplicated by NDA + indication)."""
    seen: set[tuple[str, str]] = set()
    for rec in faers_cases.iter_rows(named=True):
        norm = normalize_nda(rec.get("nda") or rec.get("nda_raw"))
        indication = str(rec.get("indication") or "").strip()
        if not norm or not indication:
            continue
        if is_non_disease_indication(indication):
            continue  # FAERS placeholder/usage-context indication, not a drug->condition approval claim
        key = (norm, indication)
        if key in seen:
            continue
        seen.add(key)
        curie, name, category = _object_attrs(indication, disease_map)
        yield {
            "norm_nda": norm,
            "object_text": indication,
            "object_curie": curie,
            "object_name": name,
            "object_category": category,
            "fallback_subject": str(rec.get("ingredient") or rec.get("drugname") or "").strip(),
        }


def _dailymed_candidates(dailymed: DailyMedEvidence, disease_map: Mapping[str, Mapping[str, str]]) -> Iterator[dict[str, str]]:
    """Fallback candidates: dictionary conditions named in approved SPL indication sections."""
    set_to_ndas: dict[str, set[str]] = {}
    for norm, sets in dailymed.approval_sets.items():
        for set_id in sets:
            set_to_ndas.setdefault(set_id, set()).add(norm)

    seen: set[tuple[str, str]] = set()
    for set_id in sorted(dailymed.indication_docs):
        ndas = sorted(set_to_ndas.get(set_id, ()))
        if not ndas:
            continue
        for _doc_id, text in dailymed.indication_docs[set_id]:
            for match in match_diseases(text, disease_map):
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


transform = ApprovedTreatsShaper().transform

__all__ = ["ApprovedTreatsShaper", "build_approved_treats_rows", "transform"]
