"""FAERS observed-use (applied_to_treat) assertion aggregation.

Builds ``faers_applied_to_treat_assertions.tsv``: real-world drug→condition uses observed in
FAERS adverse-event reports, without any approval claim.

Aggregation rule (explicit and tested)
---------------------------------------
Each distinct indication string is first resolved to its object (disease-map match or raw
passthrough); FAERS case rows (``cases.parquet``) are then aggregated by
``(drugname, resolved object_text)`` — the edge-identity key — so wordings that resolve to the
same object merge into ONE row. ``number_of_cases`` is the number of **distinct cases**
(``primaryid``) across all merged wordings (falls back to row count when ``primaryid`` is
absent), and the provenance columns are the deduplicated, sorted, pipe-joined union of the
merged wordings' evidence. ``case_ids`` carries the exact per-case token set behind that count
(one token per distinct case) so Tablassert's merge dedup (``uuid_on_collision: merge``, >= 16.6)
can recompute ``number_of_cases`` as the union size when cross-spelling rows resolve to one
CURIE and fold into a single edge. The FAERS ``knowledge_level`` label is preserved from the first
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

Pair matching runs both sides through the same normalization chain
(:func:`_pair_key`: the textnorm chain then gazetteer normalization), so dosage/form junk,
brand aliases (:data:`~dakp_pipeline.textnorm.BRAND_ALIASES`), punctuation, and casing
cannot make ONE pair answer asymmetrically: every spelling variant of one real pair derives
ONE ``clinical_approval_status``, and Tablassert's ``uuid_on_collision`` first-wins merge
never sees a conflicting scalar (v1.13.0's foldreport recorded 3,283 such conflicts, driven
by pre-textnorm drugname junk). Residual free-text spelling variants beyond the alias table
(non-English INN spellings, typos) can still miss and read as ``off_label_use`` for
actually-approved pairs — the same caveat the legacy pipeline carried.

FAERS text handling
-------------------
FAERS ``drugname`` values are passed through as text-first intervention subjects. Tablassert
tries to resolve those raw drug names to intervention concepts during the downstream fullmap
build. FAERS ``indi_pt`` values are not sent through the biomedical NER model: known conditions
still use the fast lexical disease-map baseline, while unknown conditions remain raw text for
Tablassert's object resolution.

Provenance: DAKP aggregates FAERS primary observations with DailyMed support; FAERS is the
primary upstream source, DailyMed the supporting one. Object CURIEs come from the lexical disease
baseline; subjects carry no CURIE (FAERS gives no drug id here). Canonical mapping is later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import polars as pl

from dakp_pipeline.assertions import AT_MANUAL, INFORES_DAILYMED, INFORES_DAKP, INFORES_FAERS, join_pipe, match_diseases, row_for
from dakp_pipeline.assertions.contexts import assertion_context
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
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.ner.dictionary import normalize_text
from dakp_pipeline.textnorm import defaers_text, defaersify

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
            stats(logger, "shape_faers_applied_to_treat", inputs=len(inputs), disease_map_terms=len(disease_map))
            # Projection: only the three columns the aggregation needs (the production case table
            # is tens of millions of rows wide; reading all 17 columns wastes gigabytes).
            faers_cases = find_faers_cases(inputs, columns=("drugname", "indication", "primaryid", "nda", "nda_raw", "quarter", "source_record_id"))
            approved = find_table(inputs, "approved_treats_assertions.tsv")
            approved_pairs = _approved_pair_index(approved) if approved is not None else None
            approvals = build_fda_approval_index(inputs)
            rows = build_observed_use_rows(
                faers_cases, disease_map, approved_pairs, approvals=approvals, faers_quarter_urls=faers_quarter_urls(inputs)
            )
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_faers_applied_to_treat")


def _approved_pair_index(approved: pl.DataFrame) -> set[tuple[str, str]]:
    """Normalized ``(subject_text, object_text)`` pairs of the approved-treats table.

    Matching is normalized text on both sides because the two tables spell drugs differently
    (observed-uses subjects are raw FAERS drugnames; approved-treats subjects are DailyMed
    ingredient text). Both sides run through :func:`_pair_key` — the SAME chain the
    observed-uses lookup uses (:func:`~dakp_pipeline.textnorm.defaers_text` followed by the
    gazetteer :func:`~dakp_pipeline.ner.dictionary.normalize_text`) — so dosage/form junk,
    brand aliases, and punctuation differences cannot make ONE pair asymmetric: every
    spelling variant of one real pair derives ONE status, and Tablassert's
    ``uuid_on_collision`` first-wins merge then never sees a conflicting
    ``clinical_approval_status`` scalar (v1.13.0 foldreport recorded 3,283 such conflicts,
    driven by pre-textnorm drugname junk). Residual free-text spelling variants beyond the
    alias table (e.g. non-English INN spellings, typos) can still asymmetrically miss and
    read as ``off_label_use`` — the same caveat the legacy pipeline carried.
    """
    pairs: set[tuple[str, str]] = set()
    for rec in approved.iter_rows(named=True):
        subject = _pair_key(str(rec.get("subject_text") or ""))
        obj = _pair_key(str(rec.get("object_text") or ""))
        if subject and obj:
            pairs.add((subject, obj))
    return pairs


def _pair_key(text: str) -> str:
    """Canonical (drug, condition) lookup key: textnorm chain, then gazetteer normalization.

    Single source of truth for both the approved-pair index and the observed-uses status
    lookup; keeping it identical on both sides is what makes the derived status canonical
    across spelling variants.
    """
    return normalize_text(defaers_text(text))


def build_observed_use_rows(
    faers_cases: pl.DataFrame | None,
    disease_map: Mapping[str, Mapping[str, str]],
    approved_pairs: set[tuple[str, str]] | None = None,
    *,
    approvals: FDAApprovalIndex | None = None,
    faers_quarter_urls: Mapping[str, str] | None = None,
    # Kept for source compatibility with callers from before the FAERS NER bypass. These
    # parameters are deliberately ignored: FAERS fields are never sent to NER.
    ner: object | None = None,
    devices: object | None = None,
    cache: object | None = None,
) -> list[dict[str, str]]:
    """Aggregate FAERS drug-indication case counts into applied-to-treat rows (deterministic).

    Indication strings are resolved to their object (disease-map match or raw passthrough) BEFORE
    aggregation. FAERS text is never sent through the biomedical NER model; the distinct-case
    counting then runs as ONE Polars
    group-by on ``(drugname, object_text)`` (never Python row iteration), so the
    full-production case table (tens of millions of rows) aggregates in a few seconds with
    bounded memory. Resolution-first makes the rows unique on the edge-identity key: two raw
    indication wordings that resolve to the same object text (dictionary-key casing, or one
    NER-normalized mention) merge into a single assertion whose evidence columns are the
    deduplicated, sorted, pipe-joined union of the merged wordings' provenance (the
    :func:`~dakp_pipeline.assertions.evidence.sorted_pipe` convention). The merged
    ``number_of_cases`` stays EXACT — the number of distinct non-empty primaryids across ALL merged
    wordings plus one per anonymous row (the legacy ``_row{index}`` fallback made every
    primaryid-less row its own observation), and row-count when the frame has no primaryid
    column at all — so a case that listed the drug under two merged wordings counts ONCE,
    where summing per-wording counts would double-count it.

    ``case_ids`` makes that exactness survive the one merge the shaper cannot see: two rows
    with DIFFERENT resolved ``object_text`` spellings can still resolve to the same CURIE in
    Tablassert's fullmap pass, where ``uuid_on_collision: merge`` (Tablassert >= 16.6) folds
    them into one edge and recomputes ``number_of_cases`` as the union size of the merged
    ``supporting_case_ids`` lists. The cell therefore carries one token per counted case — the
    primaryid for identified cases, ``anon:<source_record_id>`` for primaryid-less rows, padded
    with per-group synthetic ``anon:row:`` tokens so ``len(case_ids) == number_of_cases`` exactly —
    and is pipe-joined like every other multivalued cell.

    ``approved_pairs`` is the normalized (subject, object) pair set of the approved-treats table
    (:func:`_approved_pair_index`), or ``None`` when that table is unavailable — in which case
    every row degrades to ``clinical_approval_status = not_provided``.

    ``approvals`` expands the FAERS application numbers, which FAERS records with both the
    application-type prefix and the leading zeros stripped (``125514``), back to the FDA form
    every other source uses (``BLA125514``). Without it the bare FAERS number is emitted.
    """
    if faers_cases is None:
        return []
    del ner, devices, cache
    approvals = approvals if approvals is not None else FDAApprovalIndex()

    def _text_column(name: str) -> pl.Expr:
        return pl.col(name).fill_null("").cast(pl.Utf8) if name in faers_cases.columns else pl.lit("")

    primaryid = _text_column("primaryid").str.strip_chars()
    cases = (
        faers_cases.lazy()
        .select(
            defaersify(_text_column("drugname").str.strip_chars()).alias("drugname"),
            defaersify(_text_column("indication").str.strip_chars()).alias("indication"),
            primaryid.alias("primaryid"),
            _text_column("nda").alias("nda"),
            _text_column("nda_raw").alias("nda_raw"),
            _text_column("quarter").alias("quarter"),
            _text_column("source_record_id").alias("source_record_id"),
        )
        .filter((pl.col("drugname") != "") & (pl.col("indication") != ""))
        .with_columns(pl.struct(["primaryid", "nda", "nda_raw", "quarter", "source_record_id"]).alias("faers_row"))
    )

    # Resolve each distinct stop-list-passing indication to its object BEFORE aggregation.
    indications = sorted(set(cases.select("indication").unique().collect().get_column("indication").to_list()) - {""})
    resolution: dict[str, dict[str, str]] = {}
    misses: list[str] = []
    stoplist_drops = 0
    for indication in indications:
        if is_non_disease_indication(indication):
            stoplist_drops += 1
            continue  # FAERS placeholder/usage-context indication, not a drug->condition observation
        matches = match_diseases(indication, disease_map)
        if matches:
            resolution[indication] = matches[0]
        else:
            misses.append(indication)

    for indication in misses:
        resolution[indication] = {"text": indication, "curie": "", "name": indication, "category": "Disease"}

    # Canonical object attributes per resolved object text (first indication in sorted order
    # wins), so dictionary-key casing variants stay deterministic when merged.
    canonical: dict[str, dict[str, str]] = {}
    for indication in sorted(resolution):
        canonical.setdefault(resolution[indication]["text"], resolution[indication])
    mapping_rows: list[dict[str, str]] = []
    for indication in sorted(resolution):
        obj = canonical[resolution[indication]["text"]]
        mapping_rows.append(
            {
                "indication": indication,
                "object_text": obj["text"],
                "object_curie": obj["curie"],
                "object_name": obj["name"],
                "object_category": obj["category"],
            }
        )
    mapping = pl.DataFrame(
        mapping_rows,
        schema={"indication": pl.Utf8, "object_text": pl.Utf8, "object_curie": pl.Utf8, "object_name": pl.Utf8, "object_category": pl.Utf8},
    ).with_columns(
        pl.col("indication")
        .map_elements(lambda value: assertion_context("faers", "indication", str(value or "")), return_dtype=pl.Utf8)
        .alias("assertion_context")
    )
    pairs = (
        cases.join(mapping.lazy(), on="indication", how="inner")  # stop-listed indications carry no mapping entry
        .group_by("drugname", "object_text", "assertion_context")
        .agg(
            pl.col("object_curie").first(),
            pl.col("object_name").first(),
            pl.col("object_category").first(),
            pl.col("indication").first().alias("context_indication"),
            pl.col("primaryid").filter(pl.col("primaryid") != "").n_unique().alias("distinct_cases"),
            pl.col("primaryid").filter(pl.col("primaryid") == "").len().alias("anon_rows"),
            pl.col("faers_row").unique().alias("faers_rows"),
        )
        .collect()
        .sort("drugname", "object_text")
    )

    rows: list[dict[str, str]] = []
    for rec in pairs.iter_rows(named=True):
        drug = str(rec["drugname"])
        obj = {
            "text": str(rec["object_text"]),
            "curie": str(rec["object_curie"]),
            "name": str(rec["object_name"]),
            "category": str(rec["object_category"]),
        }
        source_records: set[str] = set()
        evidence_urls: set[str] = set()
        approval_values_by_norm: dict[str, set[str]] = {}
        case_ids: set[str] = set()
        anon_records: set[str] = set()
        for raw in rec.get("faers_rows") or []:
            row = raw or {}
            q = str(row.get("quarter") or "").strip()
            pid = str(row.get("primaryid") or "").strip()
            if q and pid:
                evidence_urls.add(faers_record_url(q, dict(faers_quarter_urls or {})))
            source_id = str(row.get("source_record_id") or "").strip()
            if source_id:
                source_records.add(source_id)
            if pid:
                case_ids.add(pid)
            elif source_id:
                anon_records.add(source_id)
            raw_nda = str(row.get("nda_raw") or row.get("nda") or "").strip()
            norm_nda = normalize_nda(raw_nda)
            if norm_nda:
                approval_values_by_norm.setdefault(norm_nda, set()).add(raw_nda)
        # One expansion per DISTINCT application number: the FAERS spellings of a number
        # (``125514``/``0125514``) all normalize to the same key, and the index answers with the
        # FDA display form(s) for that key.
        approval_values = approvals.expand_all(min(values) for values in approval_values_by_norm.values())
        # ``anon_rows`` counts RAW primaryid-less rows while ``anon_records`` dedups their
        # source_record_ids (and an id-less row leaves no token at all), so pad with per-group
        # synthetic tokens — unique across rows that could merge downstream because the group
        # key is embedded — keeping len(case_ids) == number_of_cases exact.
        anon_tokens = {f"anon:{record}" for record in anon_records}
        pad = int(rec["anon_rows"]) - len(anon_tokens)
        anon_tokens.update(f"anon:row:{drug}:{obj['text']}:{index}" for index in range(max(pad, 0)))
        if approved_pairs is None:
            status = _STATUS_NOT_PROVIDED
        elif (_pair_key(drug), _pair_key(obj["text"])) in approved_pairs:
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
                assertion_context=str(rec.get("assertion_context") or "indication"),
                number_of_cases=int(rec["distinct_cases"]) + int(rec["anon_rows"]),
                case_ids=sorted_pipe([*case_ids, *anon_tokens]),
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
    stats(logger, "shape_faers_applied_to_treat", indications=len(indications), stoplist_drops=stoplist_drops, assertions=len(rows))
    return rows


transform = ObservedUsesShaper().transform

__all__ = ["ObservedUsesShaper", "build_observed_use_rows", "is_non_disease_indication", "transform"]
