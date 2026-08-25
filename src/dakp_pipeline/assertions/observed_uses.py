"""FAERS observed-use (applied_to_treat) assertion aggregation.

Builds ``faers_applied_to_treat_assertions.tsv``: real-world drug→condition uses observed in
FAERS adverse-event reports, without any approval claim.

Aggregation rule (explicit and tested)
---------------------------------------
FAERS case rows (``cases.parquet``) are aggregated by ``(drugname, indication)``; ``case_count``
is the number of **distinct cases** (``primaryid``) reporting that pair (falls back to row count
when ``primaryid`` is absent). The FAERS ``knowledge_level`` label is preserved from the first
rebuild (``statistical_association``).

``clinical_approval_status`` cross-references the pair against the approved-treats table (the
legacy postprocess rule, ``ref/legacy/bin/dakp-postprocess2jsonlBL.py``): ``approved_for_condition``
when the same (drug, condition) pair has a ``biolink:treats`` row, else ``off_label_use`` — the
observed use is real-world but not label-approved. ``not_provided`` is emitted only when no
approved-treats table was available to check against (degraded mode). All three values are
biolink-valid ``ClinicalApprovalStatusEnum`` members (the legacy ``observed_use`` label never was
one — the DINGO ingest already coerced it to ``not_provided`` — and would emit biolink-invalid
edges now that Tablassert >= 8.2 emits the field first-class). The observed-use meaning stays on
the edge via ``predicate = applied_to_treat`` + ``knowledge_level = observation`` (config override).

Pair matching is case/punctuation-insensitive normalized text on both sides
(:func:`~dakp_pipeline.ner.dictionary.normalize_text`). Limitation: observed-uses subjects are raw
FAERS drugnames while approved-treats subjects are DailyMed ingredient text, so name variants
("Advil" vs "Ibuprofen") MISS and read as ``off_label_use`` for actually-approved pairs — the
same caveat the legacy pipeline carried.

Object cleanup (the NER channel)
--------------------------------
``indi_pt`` is messy free text: qualifiers (``Migraine prophylaxis``) and symptom-vs-disease
confusion pollute ``object_text`` and the blanket ``Disease`` category. When a backend is
injected (``params["ner"]``, production GLiNER wiring in the DAG; else the deterministic
offline gazetteer from :func:`~dakp_pipeline.assertions.ner_dispatch.default_ner`), every
stop-list-passing indication the dictionary misses is mined — deduplicated by normalized text,
multi-GPU dispatched in production. An indication yielding EXACTLY ONE disease/phenotype
mention resolves to that mention: ``object_text``/``object_name`` = the normalized mention text
and ``object_category`` from the mention type (``disease``→Disease, ``phenotype``→PhenotypicFeature).
Zero or several mentions (no unambiguous head condition) keep the raw passthrough. Aggregation
keys and ``case_count`` semantics are untouched — only object attributes improve. Offline the
gazetteer is word-bounded where :func:`~dakp_pipeline.assertions.match_diseases` is
plain-substring, so every gazetteer hit is already a dictionary hit and offline output is
byte-identical to the lexical baseline.

Provenance: DAKP aggregates FAERS primary observations with DailyMed support; FAERS is the
primary upstream source, DailyMed the supporting one. Object CURIEs come from the lexical disease
baseline; subjects carry no CURIE (FAERS gives no drug id here). Canonical mapping is later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_FAERS, join_pipe, match_diseases, row_for
from dakp_pipeline.assertions.evidence import (
    FDAApprovalIndex,
    build_fda_approval_index,
    faers_quarter_urls,
    faers_record_url,
    find_faers_cases,
    find_table,
    normalize_nda,
    sorted_pipe,
    write_assertion_table,
)
from dakp_pipeline.assertions.ner_dispatch import _mine_multi_gpu, _resolve_devices, default_ner, mine_with_cache
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, progress, stats, step
from dakp_pipeline.ner.dictionary import normalize_text
from dakp_pipeline.ner.mention_cache import MentionCache
from dakp_pipeline.ner.ner import DiseaseNER, Mention

_TABLE = "faers_applied_to_treat_assertions"
_PREDICATE = "biolink:applied_to_treat"
#: The (drug, condition) pair has an approved-treats row: the observed use IS the approved use.
_STATUS_APPROVED = "approved_for_condition"
#: No approved-treats row for the pair: the FAERS use is observed but not label-approved (the
#: legacy postprocess off-label signal).
_STATUS_OFF_LABEL = "off_label_use"
#: No approved-treats table was available to check against (degraded mode; the only status that
#: carries no off-label information). All three values are ``ClinicalApprovalStatusEnum`` members.
_STATUS_NOT_PROVIDED = "not_provided"
_KNOWLEDGE_LEVEL = "statistical_association"

#: One INFO progress line per this many mined indication strings (GLiNER is the slow step).
_MINING_PROGRESS_EVERY = 500

#: NER mention type -> biolink-ish object category for mention-resolved FAERS objects.
_MENTION_TYPE_CATEGORIES = {"disease": "Disease", "phenotype": "PhenotypicFeature", "phenotypicfeature": "PhenotypicFeature"}

# FAERS ``indi_pt`` is free text and carries non-disease usage-context values that name no real
# condition (placeholders like "Product used for unknown indication", generic procedures like a
# bare "Prophylaxis"/"Chemotherapy", and reporting artifacts like "Medication error"). These are
# not drug->condition observations and would otherwise default to bogus "Disease" objects (~41% of
# the case-weighted rows in a real quarter). The list is the union of the legacy DAKP FAERS
# stop-lists (ref/legacy FAERS/bin/drug2indi.pl + listCases.pl + caseList2uses.pl), extended
# with MedDRA product-issue PTs seen in production (`Contraindicated product administered`,
# `Product administered to patient of inappropriate age`, …). Specific
# conditions are untouched: "Migraine prophylaxis" or "Hormone receptor positive HER2 negative
# breast cancer" do NOT match (the anchored generic terms require the whole string; the phrase
# terms target the placeholder wording only).
_NON_DISEASE_INDICATION_RE = re.compile(
    r"unknown indication|unapproved indication|off[- ]label|ill-defined|adverse drug reaction"
    r"|evidence based treatment|medication error|not applicable"
    r"|product used for|product use in|product use issue|drug use in|product dose|product prescribing"
    r"|product storage|product availab|product quality|product misuse|product origin unknown"
    r"|product (?:administ|prescrib|dispens)\w*|contraindicated product"
    r"|accidental exposure|exposure during pregnancy"
    r"|\Aprophylaxis\Z|\Apremedication\Z|\Achemotherapy\Z|\Adrug therapy\Z|\Asupplementation therapy\Z",
    re.IGNORECASE,
)


def is_non_disease_indication(indication: str) -> bool:
    """True when a FAERS indication is a placeholder/usage-context value naming no real condition."""
    return bool(_NON_DISEASE_INDICATION_RE.search(indication.strip()))


class ObservedUsesShaper:
    def transform(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        with step(logger, "shape_faers_applied_to_treat"):
            disease_map: dict[str, dict[str, str]] = ctx.params.get("disease_map", {})  # type: ignore[assignment]
            ner_param = ctx.params.get("ner")
            ner = ner_param if isinstance(ner_param, DiseaseNER) else default_ner(ctx.fixture_root)
            devices = _resolve_devices(ner)
            stats(logger, "shape_faers_applied_to_treat", inputs=len(inputs), disease_map_terms=len(disease_map))
            # Projection: only the three columns the aggregation needs (the production case table
            # is tens of millions of rows wide; reading all 17 columns wastes gigabytes).
            faers_cases = find_faers_cases(inputs, columns=("drugname", "indication", "primaryid", "nda", "nda_raw", "quarter", "source_record_id"))
            approved = find_table(inputs, "approved_treats_assertions.tsv")
            approved_pairs = _approved_pair_index(approved) if approved is not None else None
            approvals = build_fda_approval_index(inputs)
            with MentionCache(ctx.workdir) as cache:
                rows = build_observed_use_rows(
                    faers_cases,
                    disease_map,
                    approved_pairs,
                    approvals=approvals,
                    ner=ner,
                    devices=devices,
                    cache=cache,
                    faers_quarter_urls=faers_quarter_urls(inputs),
                )
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_faers_applied_to_treat")


def _mine_indication_mentions(
    texts: list[str], ner: DiseaseNER, devices: Sequence[str] | None, cache: MentionCache | None = None
) -> dict[str, list[Mention]]:
    """Mine distinct dictionary-miss indication strings, returning ``{normalized_text: [mentions]}``.

    Deduplication happens upstream (``texts`` are distinct normalized strings), so each unique
    string is mined exactly once — the production case table's millions of rows collapse to a
    bounded set of distinct indications. Production runs dispatch across GPUs
    (:func:`~dakp_pipeline.assertions.ner_dispatch._mine_multi_gpu`); the offline gazetteer
    backend runs sequentially with periodic progress narration. When ``cache`` is given,
    previously mined texts are served from the persistent mention cache
    (:func:`~dakp_pipeline.assertions.ner_dispatch.mine_with_cache`).
    """
    if not texts:
        return {}
    work_items = [(text, text, text) for text in texts]

    def mine(items: Sequence[Any]) -> dict[tuple[str, str], list[Mention]]:
        if devices and len(items) > 1 and not ner._offline:
            return _mine_multi_gpu(items, ner, devices)
        mined: dict[tuple[str, str], list[Mention]] = {}
        for done, (set_id, doc_id, text) in enumerate(items, start=1):
            mined[(set_id, doc_id)] = ner.extract(text)
            progress(logger, "shape_faers_applied_to_treat", done, len(items), every=_MINING_PROGRESS_EVERY)
        return mined

    mined_all = mine_with_cache(work_items, ner, mine, cache)
    return {set_id: mentions for (set_id, _doc_id), mentions in mined_all.items()}


def _ner_object(indication: str, indication_mentions: Mapping[str, list[Mention]] | None) -> dict[str, str] | None:
    """The single clean disease/phenotype mention for a dictionary-miss indication, else None.

    Only an unambiguous single mention resolves the object — zero mentions (nothing found) or
    several (a conjunction like ``rheumatoid arthritis and diabetes``) keep the raw passthrough
    rather than guess at one head condition.
    """
    if indication_mentions is None:
        return None
    mentions = indication_mentions.get(normalize_text(indication), [])
    if len(mentions) != 1:
        return None
    mention = mentions[0]
    category = _MENTION_TYPE_CATEGORIES.get(str(mention.type).lower())
    text = normalize_text(mention.text)
    if category is None or not text:
        return None
    return {"text": text, "curie": "", "name": text, "category": category}


def _approved_pair_index(approved: pl.DataFrame) -> set[tuple[str, str]]:
    """Normalized ``(subject_text, object_text)`` pairs of the approved-treats table.

    Matching is normalized text on both sides because the two tables spell drugs differently
    (observed-uses subjects are raw FAERS drugnames; approved-treats subjects are DailyMed
    ingredient text), so name variants can still miss — see the module docstring.
    """
    pairs: set[tuple[str, str]] = set()
    for rec in approved.iter_rows(named=True):
        subject = normalize_text(str(rec.get("subject_text") or ""))
        obj = normalize_text(str(rec.get("object_text") or ""))
        if subject and obj:
            pairs.add((subject, obj))
    return pairs


def build_observed_use_rows(
    faers_cases: pl.DataFrame | None,
    disease_map: Mapping[str, Mapping[str, str]],
    approved_pairs: set[tuple[str, str]] | None = None,
    *,
    approvals: FDAApprovalIndex | None = None,
    ner: DiseaseNER | None = None,
    devices: Sequence[str] | None = None,
    cache: MentionCache | None = None,
    faers_quarter_urls: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Aggregate FAERS drug-indication case counts into applied-to-treat rows (deterministic).

    The distinct-case counting runs as a Polars group-by (never Python row iteration), so the
    full-production case table (tens of millions of rows) aggregates in a few seconds with
    bounded memory. Semantics are unchanged: per (drugname, indication) pair the count is the
    number of distinct non-empty primaryids plus one per anonymous row (the legacy
    ``_row{index}`` fallback made every primaryid-less row its own observation), and row-count
    when the frame has no primaryid column at all.

    ``approved_pairs`` is the normalized (subject, object) pair set of the approved-treats table
    (:func:`_approved_pair_index`), or ``None`` when that table is unavailable — in which case
    every row degrades to ``clinical_approval_status = not_provided``.

    When ``ner`` is given, stop-list-passing indications the dictionary misses are mined once per
    distinct normalized string (:func:`_mine_indication_mentions`); a single unambiguous mention
    supplies the object text/name/category (:func:`_ner_object`). ``cache`` (a persistent mention
    cache) only changes WHERE mentions come from, never their content.

    ``approvals`` expands the FAERS application numbers, which FAERS records with both the
    application-type prefix and the leading zeros stripped (``125514``), back to the FDA form
    every other source uses (``BLA125514``). Without it the bare FAERS number is emitted.
    """
    if faers_cases is None:
        return []
    approvals = approvals if approvals is not None else FDAApprovalIndex()

    def _text_column(name: str) -> pl.Expr:
        return pl.col(name).fill_null("").cast(pl.Utf8) if name in faers_cases.columns else pl.lit("")

    primaryid = _text_column("primaryid").str.strip_chars()
    normalized = (
        faers_cases.lazy()
        .select(
            _text_column("drugname").str.strip_chars().alias("drugname"),
            _text_column("indication").str.strip_chars().alias("indication"),
            primaryid.alias("primaryid"),
            _text_column("nda").alias("nda"),
            _text_column("nda_raw").alias("nda_raw"),
            _text_column("quarter").alias("quarter"),
            _text_column("source_record_id").alias("source_record_id"),
        )
        .filter((pl.col("drugname") != "") & (pl.col("indication") != ""))
        .with_columns(pl.struct(["primaryid", "nda", "nda_raw", "quarter", "source_record_id"]).alias("faers_row"))
        .group_by("drugname", "indication")
        .agg(
            pl.col("primaryid").filter(pl.col("primaryid") != "").n_unique().alias("distinct_cases"),
            pl.col("primaryid").filter(pl.col("primaryid") == "").len().alias("anon_rows"),
            pl.col("faers_row").unique().alias("faers_rows"),
        )
        .collect()
        .sort("drugname", "indication")
    )
    pairs = normalized

    # Mine the distinct stop-list-passing, dictionary-miss indication strings once (NER channel).
    indication_mentions: dict[str, list[Mention]] | None = None
    if ner is not None:
        miss_texts = sorted(
            {
                normalize_text(str(rec["indication"]))
                for rec in pairs.iter_rows(named=True)
                if not is_non_disease_indication(str(rec["indication"])) and not match_diseases(str(rec["indication"]), disease_map)
            }
            - {""}
        )
        indication_mentions = _mine_indication_mentions(miss_texts, ner, devices, cache)

    rows: list[dict[str, str]] = []
    pairs_seen = 0
    stoplist_drops = 0
    ner_resolved = 0
    for rec in pairs.iter_rows(named=True):
        pairs_seen += 1
        drug = str(rec["drugname"])
        indication = str(rec["indication"])
        if is_non_disease_indication(indication):
            stoplist_drops += 1
            continue  # FAERS placeholder/usage-context indication, not a drug->condition observation
        matches = match_diseases(indication, disease_map)
        if matches:
            obj = matches[0]
        else:
            mention_obj = _ner_object(indication, indication_mentions)
            if mention_obj is not None:
                ner_resolved += 1
            obj = mention_obj or {"text": indication, "curie": "", "name": indication, "category": "Disease"}
        source_records: set[str] = set()
        evidence_urls: set[str] = set()
        approval_values_by_norm: dict[str, set[str]] = {}
        for raw in rec.get("faers_rows") or []:
            row = raw or {}
            q = str(row.get("quarter") or "").strip()
            pid = str(row.get("primaryid") or "").strip()
            if q and pid:
                evidence_urls.add(faers_record_url(q, dict(faers_quarter_urls or {})))
            source_id = str(row.get("source_record_id") or "").strip()
            if source_id:
                source_records.add(source_id)
            raw_nda = str(row.get("nda_raw") or row.get("nda") or "").strip()
            norm_nda = normalize_nda(raw_nda)
            if norm_nda:
                approval_values_by_norm.setdefault(norm_nda, set()).add(raw_nda)
        # One expansion per DISTINCT application number: the FAERS spellings of a number
        # (``125514``/``0125514``) all normalize to the same key, and the index answers with the
        # FDA display form(s) for that key.
        approval_values = approvals.expand_all(min(values) for values in approval_values_by_norm.values())
        if approved_pairs is None:
            status = _STATUS_NOT_PROVIDED
        elif (normalize_text(drug), normalize_text(obj["text"])) in approved_pairs:
            status = _STATUS_APPROVED
        else:
            status = _STATUS_OFF_LABEL
        rows.append(
            row_for(
                _TABLE,
                subject_text=drug,
                subject_curie="",
                subject_name=drug,
                subject_category="ChemicalEntity",
                predicate=_PREDICATE,
                object_text=obj["text"],
                object_curie=obj["curie"],
                object_name=obj["name"],
                object_category=obj["category"],
                case_count=int(rec["distinct_cases"]) + int(rec["anon_rows"]),
                FDA_regulatory_approvals=sorted_pipe(approval_values),
                edge_evidence="",
                supporting_faers_records=sorted_pipe(source_records),
                supporting_faers_urls=sorted_pipe(evidence_urls),
                clinical_approval_status=status,
                knowledge_level=_KNOWLEDGE_LEVEL,
                agent_type=AT_MANUAL,
                primary_knowledge_source=INFORES_DAKP,
                upstream_resource_ids=join_pipe(INFORES_FAERS, INFORES_DAILYMED),
            )
        )
    stats(logger, "shape_faers_applied_to_treat", pairs=pairs_seen, stoplist_drops=stoplist_drops, ner_resolved=ner_resolved, assertions=len(rows))
    return rows


transform = ObservedUsesShaper().transform

__all__ = ["ObservedUsesShaper", "build_observed_use_rows", "is_non_disease_indication", "transform"]
