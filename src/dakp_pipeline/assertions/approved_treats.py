"""Approved-treatment assertion aggregation.

Builds ``approved_treats_assertions.tsv``: FDA-approved drug→condition assertions.

Aggregation rule (explicit and tested)
---------------------------------------
An approved-treats row requires an **NDA-bearing drug-indication pair** that satisfies all of:

1. the pair's NDA maps (Drugs@FDA ``products``) to an ingredient — confirming a real FDA
   application;
2. the NDA has a **DailyMed SPL approval** (``spl_approvals``); and
3. that approved SPL set has an **indications-and-usage section** (LOINC ``34067-9``) — the
   SPL indication support.

Candidate drug-indication pairs come from **FAERS** (``cases.parquet`` rows carrying an NDA +
indication) — the primary source, wired into this stage by the DAG. Placeholder/usage-context
FAERS indications ("Product used for unknown indication", bare "Prophylaxis", ...) are dropped
via the shared :func:`~dakp_pipeline.assertions.observed_uses.is_non_disease_indication`
stop-list. When no FAERS case table is present the candidates fall back to DailyMed SPL
indication sections whose text names a dictionary condition. Both paths apply the *same*
three-part filter above.

EMA interim registry rows (``ema_registry.parquet``, when present among the inputs) union into the
same table in two shapes:

* **MeSH-area rows** (``infores:ema``) — one row per ``(active substance, MeSH therapeutic-area
  term)`` pair from each Authorised/Human medicine (substance falls back to the INN when the
  export's "Active substance" cell is empty);
* **EPAR indication rows** (``infores:epar``) — the export's free-text "Therapeutic indication"
  column mined with the composite DiseaseNER (:mod:`dakp_pipeline.ner.ner`): one row per
  ``(active substance, disease/phenotype mention)`` pair. The mined object is the normalized
  mention text, so these key separately from the MeSH-area rows. Mining runs only when the EMA
  registry is among the inputs — FDA-only runs never construct a NER backend.

Both EMA shapes carry the EMA product number as ``approval_ids`` and the EPAR ``medicine_url``
as supporting documents. FDA- and EMA-derived rows keep their own ``upstream_resource_ids``;
the union is re-sorted so output stays deterministic.

Provenance (``approval_ids``, ``supporting_spl_sets``, ``supporting_spl_documents``) is
aggregated per ``(subject, object)`` as deduplicated, sorted, pipe-joined lists. SPL set evidence
is emitted as ``dailymed:<spl_set_id>`` CURIEs to match the deployed Translator ingest contract. Subject CURIEs
are populated only where DailyMed already gives a UNII; object CURIEs come from the lexical
disease baseline. Canonical CURIE mapping is a later milestone (text-first).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import polars as pl

from dakp_pipeline.assertions import (
    AT_MANUAL,
    INFORES_DAILYMED,
    INFORES_DAKP,
    INFORES_EMA,
    INFORES_EPAR,
    INFORES_FAERS,
    KL_ASSERTION,
    join_pipe,
    match_diseases,
    row_for,
)
from dakp_pipeline.assertions.contraindications import default_ner
from dakp_pipeline.assertions.evidence import (
    DailyMedEvidence,
    build_dailymed_evidence,
    build_drugsfda_ingredient_map,
    dailymed_set_curie,
    find_faers_cases,
    find_table,
    normalize_nda,
    sorted_pipe,
    write_assertion_table,
)
from dakp_pipeline.assertions.observed_uses import is_non_disease_indication
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.ner.dictionary import normalize_text
from dakp_pipeline.ner.ner import DiseaseNER, extract_disease_mentions

_TABLE = "approved_treats_assertions"
_PREDICATE = "biolink:treats"
_STATUS = "approved_for_condition"

#: Case-table columns the FAERS candidate path reads (projection keeps production-scale reads cheap).
_FAERS_CASE_COLUMNS = ("nda", "nda_raw", "indication", "ingredient", "drugname")

#: Interim parquet filename the EMA registry extractor emits (read when present among the inputs).
_EMA_REGISTRY_FILENAME = "ema_registry.parquet"


class ApprovedTreatsShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, "shape_approved_treats"):
            disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
            stats(logger, "shape_approved_treats", inputs=len(inputs), disease_map_terms=len(disease_map))
            dailymed = build_dailymed_evidence(inputs)
            drugsfda_map = build_drugsfda_ingredient_map(inputs)
            faers_cases = find_faers_cases(inputs, columns=_FAERS_CASE_COLUMNS)
            ema_registry = find_table(inputs, _EMA_REGISTRY_FILENAME)
            # The NER backend (for EPAR indication mining) is built ONLY when EMA rows are among
            # the inputs: FDA-only runs never touch the NER — no GLiNER weights needed. Injected
            # ``params["ner"]`` wins (tests / the production DAG wiring); otherwise the
            # deterministic offline gazetteer backend is used.
            ner: DiseaseNER | None = None
            if ema_registry is not None:
                ner_param = ctx.params.get("ner")
                ner = ner_param if isinstance(ner_param, DiseaseNER) else default_ner(ctx.fixture_root)
            rows = build_approved_treats_rows(faers_cases, dailymed, drugsfda_map, disease_map, ema_registry=ema_registry, ner=ner)
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_approved_treats")


def build_approved_treats_rows(
    faers_cases: pl.DataFrame | None,
    dailymed: DailyMedEvidence,
    drugsfda_map: Mapping[str, set[str]],
    disease_map: Mapping[str, Mapping[str, str]],
    *,
    ema_registry: pl.DataFrame | None = None,
    ner: DiseaseNER | None = None,
) -> list[dict[str, str]]:
    """Aggregate approved-treats assertion rows (pure; deterministic ordering).

    FDA rows follow the three-part rule below; EMA registry rows (when ``ema_registry`` is given)
    union in per-source rows keyed the same way — MeSH-area rows always, EPAR indication-mined
    rows when a ``ner`` backend is also given — and the combined output is sorted by
    ``(subject_text, object_text, upstream_resource_ids)``.
    """
    candidates = _faers_candidates(faers_cases, disease_map) if faers_cases is not None else _dailymed_candidates(dailymed, disease_map)

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    candidates_seen = 0
    dropped_no_ingredient_map = 0
    dropped_no_spl_support = 0
    dropped_no_subject = 0
    for cand in candidates:
        candidates_seen += 1
        norm = cand["norm_nda"]
        if norm not in drugsfda_map:  # (1) NDA must map (Drugs@FDA) to an ingredient
            dropped_no_ingredient_map += 1
            continue
        sets, docs = dailymed.indication_support(norm)
        if not sets:  # (2)+(3) DailyMed approval AND SPL indication-section support
            dropped_no_spl_support += 1
            continue
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
        dropped_no_subject=dropped_no_subject,
        assertions=len(aggregated),
    )
    fda_rows = [_finalize_row(agg) for _key, agg in sorted(aggregated.items())]
    ema_rows = build_ema_treats_rows(ema_registry, disease_map) if ema_registry is not None else []
    epar_rows = build_epar_treats_rows(ema_registry, ner, disease_map) if ema_registry is not None and ner is not None else []
    return sorted(fda_rows + ema_rows + epar_rows, key=_row_sort_key)


def _row_sort_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    """Total order over the FDA+EMA union (per-source rows keep their own provenance)."""
    return (row["subject_text"], row["object_text"], row["upstream_resource_ids"])


def build_ema_treats_rows(ema_registry: pl.DataFrame, disease_map: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    """EMA approved-treats rows: one per ``(active substance, MeSH therapeutic-area term)`` pair.

    Every row of the interim registry table is an Authorised, Human centrally-approved medicine
    (the extractor filtered everything else out). The subject fan-out splits the semicolon-joined
    ``active_substance`` cell (falling back to ``inn`` when empty); the object fan-out splits
    ``therapeutic_area_mesh``. ``approval_ids`` aggregates the EMA product numbers and
    ``supporting_spl_documents`` the EPAR medicine URLs — deduplicated, sorted, pipe-joined.
    """
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in ema_registry.iter_rows(named=True):
        substance_cell = str(rec.get("active_substance") or "").strip() or str(rec.get("inn") or "").strip()
        for substance in _split_semicolons(substance_cell):
            for mesh_term in _split_semicolons(str(rec.get("therapeutic_area_mesh") or "")):
                key = (substance, mesh_term)
                agg = aggregated.setdefault(key, {"approval_ids": [], "docs": []})
                agg["approval_ids"].append(str(rec.get("ema_product_number") or "").strip())
                agg["docs"].append(str(rec.get("medicine_url") or "").strip())

    stats(logger, "shape_approved_treats", ema_medicines=ema_registry.height, ema_assertions=len(aggregated))
    rows: list[dict[str, str]] = []
    for (substance, mesh_term), agg in sorted(aggregated.items()):
        curie, name, category = _object_attrs(mesh_term, disease_map)
        rows.append(
            row_for(
                _TABLE,
                subject_text=substance,
                subject_curie="",
                subject_name=substance,
                subject_category="ChemicalEntity",
                predicate=_PREDICATE,
                object_text=mesh_term,
                object_curie=curie,
                object_name=name,
                object_category=category,
                approval_ids=sorted_pipe(agg["approval_ids"]),
                supporting_spl_documents=sorted_pipe(agg["docs"]),
                clinical_approval_status=_STATUS,
                knowledge_level=KL_ASSERTION,
                agent_type=AT_MANUAL,
                primary_knowledge_source=INFORES_DAKP,
                upstream_resource_ids=INFORES_EMA,
            )
        )
    return rows


def _split_semicolons(cell: str) -> list[str]:
    """Split a semicolon-joined EMA cell into its stripped, non-empty values."""
    return [part.strip() for part in cell.split(";") if part.strip()]


def build_epar_treats_rows(ema_registry: pl.DataFrame, ner: DiseaseNER, disease_map: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    """EPAR approved-treats rows: one per ``(active substance, mined indication mention)`` pair.

    The export's free-text ``therapeutic_indication`` is mined with the composite DiseaseNER; the
    object is the normalized mention text (``object_curie`` resolved via the lexical disease
    baseline, as with the MeSH-area rows). Rows carry ``infores:epar`` provenance, the EMA
    product number as ``approval_ids``, and the EPAR ``medicine_url`` as supporting documents —
    deduplicated, sorted, pipe-joined. Rows without indication text (or without any subject)
    are skipped; mentions that normalize to nothing are dropped.
    """
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    mentions_mined = 0
    for rec in ema_registry.iter_rows(named=True):
        text = str(rec.get("therapeutic_indication") or "").strip()
        if not text:
            continue
        substance_cell = str(rec.get("active_substance") or "").strip() or str(rec.get("inn") or "").strip()
        subjects = _split_semicolons(substance_cell)
        if not subjects:
            continue
        for mention in extract_disease_mentions(text, ner):
            object_text = normalize_text(mention.text)
            if not object_text:
                continue
            mentions_mined += 1
            for substance in subjects:
                key = (substance, object_text)
                agg = aggregated.setdefault(key, {"approval_ids": [], "docs": []})
                agg["approval_ids"].append(str(rec.get("ema_product_number") or "").strip())
                agg["docs"].append(str(rec.get("medicine_url") or "").strip())

    stats(logger, "shape_approved_treats", epar_medicines=ema_registry.height, epar_mentions=mentions_mined, epar_assertions=len(aggregated))
    rows: list[dict[str, str]] = []
    for (substance, object_text), agg in sorted(aggregated.items()):
        curie, name, category = _object_attrs(object_text, disease_map)
        rows.append(
            row_for(
                _TABLE,
                subject_text=substance,
                subject_curie="",
                subject_name=substance,
                subject_category="ChemicalEntity",
                predicate=_PREDICATE,
                object_text=object_text,
                object_curie=curie,
                object_name=name,
                object_category=category,
                approval_ids=sorted_pipe(agg["approval_ids"]),
                supporting_spl_documents=sorted_pipe(agg["docs"]),
                clinical_approval_status=_STATUS,
                knowledge_level=KL_ASSERTION,
                agent_type=AT_MANUAL,
                primary_knowledge_source=INFORES_DAKP,
                upstream_resource_ids=INFORES_EPAR,
            )
        )
    return rows


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
        supporting_spl_sets=sorted_pipe(dailymed_set_curie(set_id) for set_id in agg["sets"]),
        supporting_spl_documents=sorted_pipe(agg["docs"]),
        clinical_approval_status=_STATUS,
        knowledge_level=KL_ASSERTION,
        agent_type=AT_MANUAL,
        primary_knowledge_source=INFORES_DAKP,
        upstream_resource_ids=join_pipe(INFORES_DAILYMED, INFORES_FAERS),
    )


def _subject_for_sets(dailymed: DailyMedEvidence, sets: list[str], fallback: str) -> tuple[str, str]:
    """Subject ingredient (name, UNII) from the first supporting SPL set, else the FAERS fallback."""
    for set_id in sets:  # already sorted deterministically
        if set_id in dailymed.set_ingredient:
            return dailymed.set_ingredient[set_id]
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

__all__ = ["ApprovedTreatsShaper", "build_approved_treats_rows", "build_ema_treats_rows", "build_epar_treats_rows", "transform"]
