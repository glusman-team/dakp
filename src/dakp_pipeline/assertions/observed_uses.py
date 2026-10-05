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

The cross-reference itself is :class:`ApprovedTreatsIndex`, and a row is approved when ANY of
three keys hits (see that class for the detail and the measured effect):

1. **FDA application identity.** The row's expanded application display forms (``NDA021343``,
   the values already written to its ``FDA_regulatory_approvals`` cell) name the exact product,
   and the approved table says which objects each application approves. This is the key that
   survives the brand/ingredient spelling gap: FAERS reports ``ELIGARD`` while the approved
   row's subject is the DailyMed ingredient ``LEUPROLIDE``, and no alias table can be expected
   to carry every brand.
2. **Object granularity, within one application.** An approved object occurring whole-word
   inside the observed object approves it, because the label naming the general condition
   (``prostate cancer``) covers the more specific report (``prostate cancer stage iv``). Same
   direction :func:`~dakp_pipeline.assertions.approved_treats._section_mentions_condition`
   already accepts when corroborating a candidate against a label; the reverse never approves.
   Scoped to one application on purpose: across drugs it would let any general approval cover
   any other drug's specific report.
3. **Normalized text pair.** ``(subject, object)`` equality, the legacy rule and the only one
   that can answer for a report carrying no application number (64% of production off-label
   rows).

Rules 1 and 2 only ever ADD approvals, so the change is monotone: no pair that read as
``approved_for_condition`` can read as ``off_label_use``.

Every text key on both sides runs through the same normalization chain (:func:`_pair_key`: the
textnorm chain then gazetteer normalization), so dosage/form junk, brand aliases
(:data:`~dakp_pipeline.textnorm.BRAND_ALIASES`), punctuation, and casing cannot make ONE pair
answer asymmetrically: every spelling variant of one real pair derives ONE
``clinical_approval_status``, and Tablassert's ``uuid_on_collision`` first-wins merge never sees
a conflicting scalar (v1.13.0's foldreport recorded 3,283 such conflicts, driven by pre-textnorm
drugname junk).

Residual limits, all inherited from the inputs rather than introduced here:

- A report with NO application number and a brand spelling the alias table lacks still reads as
  ``off_label_use`` (rule 3 is all that is left). ``ELIGARD`` is one such brand.
- An approved object can be MORE general than the label's own indication, because approved-treats
  objects come from FAERS wordings corroborated against the label: NDA021343 is indicated for
  *advanced* prostate cancer, its approved object is ``prostate cancer``, so rule 2 also promotes
  ``Prostate cancer stage I`` (2 cases in v1.16.0). Rule 3 already asserted the general pair.
- An approved object that is not a condition at all (``blood pressure``, ``surgery``,
  ``magnetic resonance imaging``; these are FAERS ``indi_pt`` values the non-disease stoplist does not
  catch) propagates to its specific variants under rule 2. Measured over v1.16.0, 1,837 of the
  26,307 rule-2 promotions (100,827 of 1,686,572 cases) carry such a residual, and on inspection
  most are still clinically right (amlodipine for ``blood pressure abnormal``, atorvastatin for
  ``low density lipoprotein increased``, everolimus for ``astrocytoma, low grade``). A
  residual-token denylist was measured and rejected: it also rejects correct specificity
  wordings such as ``astrocytoma, low grade``. The root fix is the stoplist, not this rule.
- A reporter-supplied application number that does not belong to the reported drug would approve
  on the wrong product; that mismatch is already published in the edge's own
  ``FDA_regulatory_approvals``, so it is visible rather than hidden. All-digit keys (the
  unknown-number fallback) are never indexed, so a shared reporter typo cannot cross-approve two
  products.

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

import functools
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import polars as pl

from dakp_pipeline.assertions import (
    AT_MANUAL,
    INFORES_CANADA_VIGILANCE,
    INFORES_DAILYMED,
    INFORES_DAKP,
    INFORES_FAERS,
    join_pipe,
    match_diseases,
    row_for,
)
from dakp_pipeline.assertions.contexts import assertion_context
from dakp_pipeline.assertions.evidence import (
    FDAApprovalIndex,
    build_fda_approval_index,
    faers_quarter_urls,
    faers_record_url,
    find_faers_cases,
    find_table,
    merge_unique,
    normalize_nda,
    sorted_pipe,
    write_assertion_table,
)
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.ner.dictionary import normalize_text
from dakp_pipeline.sources.canada_vigilance import CANADA_VIGILANCE_EXTRACTS_URL as CANADA_VIGILANCE_SOURCE_RECORD_URL
from dakp_pipeline.textnorm import defaers_text, defaersify

_TABLE = "faers_applied_to_treat_assertions"
_PREDICATE = "biolink:applied_to_treat"
#: Interim Canada Vigilance observation table (extract.canada_vigilance), joined into the SAME
#: applied-to-treat table with source-partitioned provenance (a CV row and a FAERS row for the
#: same (subject, object, context) stay separate rows; the upstream chains differ).
_CV_INDICATIONS_FILENAME = "cv_indications.parquet"
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
            # Per-phase timer (US-007): the FAERS case projection is the fixed cost here.
            with step(logger, "shape_faers_applied_to_treat.inputs"):
                faers_cases = find_faers_cases(
                    inputs, columns=("drugname", "indication", "primaryid", "nda", "nda_raw", "quarter", "source_record_id")
                )
                approved = find_table(inputs, "approved_treats_assertions.tsv")
                approved_index = ApprovedTreatsIndex.from_frame(approved) if approved is not None else None
                approvals = build_fda_approval_index(inputs)
                cv_indications = find_table(inputs, _CV_INDICATIONS_FILENAME)
            rows = build_observed_use_rows(
                faers_cases,
                disease_map,
                approved_index,
                approvals=approvals,
                faers_quarter_urls=faers_quarter_urls(inputs),
                cv_indications=cv_indications,
            )
            return write_assertion_table(_TABLE, rows, inputs, ctx, operation="shape_faers_applied_to_treat")


@functools.lru_cache(maxsize=1 << 20)
def _proper_spans(object_key: str) -> tuple[str, ...]:
    """Every contiguous token span of a normalized object key, excluding the key itself.

    :func:`~dakp_pipeline.ner.dictionary.normalize_text` folds each non-alphanumeric run into a
    single space, so a normalized key is ``[a-z0-9 ]`` only and a whole-word occurrence of one
    key inside another is EXACTLY a contiguous token span (``prostate cancer`` inside
    ``hormone refractory prostate cancer``). Enumerating the spans turns "does any approved
    object occur inside this object?" into hash lookups instead of a substring scan per
    candidate, which is what keeps the rule affordable over the 1.6M-row production table.
    """
    tokens = object_key.split(" ")
    return tuple(
        " ".join(tokens[start:end]) for start in range(len(tokens)) for end in range(start + 1, len(tokens) + 1) if end - start < len(tokens)
    )


def _split_pipe(value: object) -> list[str]:
    """The non-empty members of a pipe-joined assertion-table cell."""
    return [member for member in str(value or "").split("|") if member]


def _no_objects() -> Mapping[str, frozenset[str]]:
    """The empty read-only application index (the dataclass default)."""
    return MappingProxyType({})


@dataclass(frozen=True)
class ApprovedTreatsIndex:
    """Cross-reference index over ``approved_treats_assertions.tsv`` for the status rule.

    Two independent keys answer "is this observed (drug, condition) use label-approved?", and a
    row is approved when EITHER hits:

    ``objects_by_approval``
        FDA application display form (``NDA021343``) -> the normalized object keys that
        application approves. The application number is the exact product identity BOTH tables
        already carry, so it bridges the drug spellings the text key cannot: a FAERS report of
        the brand ``ELIGARD`` and the approved row for the DailyMed ingredient ``LEUPROLIDE``
        share ``NDA021343``. On the v1.16.0 production tables this alone reclassifies 38,044
        rows (6,678,203 cases) that read as off-label, including ``ELIGARD | Prostate cancer``
        (35,977 cases) whose approved counterpart is ``LEUPROLIDE | Prostate cancer``.

        Within one application the object match also spans granularity: an approved object
        occurring whole-word INSIDE the observed object approves it, because the label naming
        the general condition covers the more specific report (``prostate cancer`` covers
        ``prostate cancer stage iv``). That is the same direction
        :func:`~dakp_pipeline.assertions.approved_treats._section_mentions_condition` already
        accepts when it corroborates a FAERS candidate against a label; the reverse (an
        approved object MORE specific than the observation) never approves, and cannot: a
        longer key is not a span of a shorter one.

    ``pairs``
        Normalized ``(subject_text, object_text)`` keys: the text rule, and the only rule that
        can answer for a FAERS report carrying no application number (64% of production
        off-label rows). Matching is normalized text on both sides because the two tables spell
        drugs differently (observed-uses subjects are raw FAERS drugnames; approved-treats
        subjects are DailyMed ingredient text).

    Both keys run through :func:`_pair_key`, the SAME chain the observed-uses lookup uses
    (:func:`~dakp_pipeline.textnorm.defaers_text` followed by the gazetteer
    :func:`~dakp_pipeline.ner.dictionary.normalize_text`), so dosage/form junk, brand aliases,
    and punctuation differences cannot make ONE pair asymmetric: every spelling variant of one
    real pair derives ONE status, and Tablassert's ``uuid_on_collision`` first-wins merge then
    never sees a conflicting ``clinical_approval_status`` scalar (v1.13.0 foldreport recorded
    3,283 such conflicts, driven by pre-textnorm drugname junk). Residual free-text spelling
    variants beyond the alias table (non-English INN spellings, typos) on a report with NO
    application number can still miss and read as ``off_label_use``.
    """

    #: Normalized ``(subject_key, object_key)`` text pairs of the approved-treats table.
    pairs: frozenset[tuple[str, str]] = frozenset()
    #: FDA application display form -> the normalized object keys that application approves.
    #: A read-only view: the index is built once and shared by every row of a 1.6M-row table.
    objects_by_approval: Mapping[str, frozenset[str]] = field(default_factory=_no_objects)

    @classmethod
    def from_frame(cls, approved: pl.DataFrame) -> ApprovedTreatsIndex:
        """Build both keys from the approved-treats table (one pass, no copies).

        Rows with an empty subject or object contribute nothing to ``pairs``; a row with an
        empty object contributes nothing to ``objects_by_approval`` either, because an
        application with no condition approves nothing to cross-reference. ``FDA_regulatory_approvals``
        cells hold the FDA display forms :class:`~dakp_pipeline.assertions.evidence.FDAApprovalIndex`
        produced for that table, and the observed-uses rows carry display forms from the SAME
        index, so the two sides are directly comparable strings.

        A key that is ALL DIGITS is skipped. :meth:`FDAApprovalIndex.expand` falls back to the
        bare ``<prefix><digits>`` of a number no register knows, and with an empty prefix that
        publishes reporter typos (``99``, ``999999``, ``02248240``) as if they were identities.
        Two unrelated products sharing a typo would then cross-approve. On the v1.16.0 tables no
        promotion rests on such a key, so this removes a vector rather than a behavior.
        """
        pairs: set[tuple[str, str]] = set()
        objects: dict[str, set[str]] = {}
        for rec in approved.iter_rows(named=True):
            subject = _pair_key(str(rec.get("subject_text") or ""))
            obj = _pair_key(str(rec.get("object_text") or ""))
            if subject and obj:
                pairs.add((subject, obj))
            if not obj:
                continue
            for approval in _split_pipe(rec.get("FDA_regulatory_approvals")):
                if approval.isdigit():
                    continue  # a bare number is the unknown-application fallback, not an identity
                objects.setdefault(approval, set()).add(obj)
        return cls(pairs=frozenset(pairs), objects_by_approval=MappingProxyType({key: frozenset(value) for key, value in objects.items()}))

    def is_approved(self, subject_key: str, object_key: str, approvals: Iterable[str]) -> bool:
        """True when the approved-treats table already asserts this (drug, condition) pair.

        ``approvals`` are the row's expanded FDA display forms (the same values written to its
        ``FDA_regulatory_approvals`` cell); an empty iterable leaves only the text rule. Every
        branch is an existence test over deterministic keys, so the verdict never depends on
        set iteration order.

        ``approvals`` is the UNION of every case merged into the observed row, exactly as the
        row's own ``FDA_regulatory_approvals`` cell is, so one application approving the pair
        approves the merged edge. That is the same aggregation the published provenance already
        uses: the edge claims every application its cases cited, and the status agrees with the
        claim rather than contradicting it.
        """
        if (subject_key, object_key) in self.pairs:
            return True
        spans: tuple[str, ...] | None = None
        for approval in approvals:
            approved_objects = self.objects_by_approval.get(approval)
            if not approved_objects:
                continue
            if object_key in approved_objects:
                return True
            if spans is None:
                spans = _proper_spans(object_key)
            if any(span in approved_objects for span in spans):
                return True  # the general approved object covers this more specific report
        return False


@functools.lru_cache(maxsize=1 << 20)
def _pair_key(text: str) -> str:
    """Canonical (drug, condition) lookup key: textnorm chain, then gazetteer normalization.

    Single source of truth for both the approved-pair index and the observed-uses status
    lookup; keeping it identical on both sides is what makes the derived status canonical
    across spelling variants. Pure and memoized: the 1.7M production rows repeat each drug
    and object text many times (US-007 profiling: ~17% of the FAERS shaping wall time).
    """
    return normalize_text(defaers_text(text))


def _indication_mapping(indications: list[str], disease_map: Mapping[str, Mapping[str, str]], source: str) -> tuple[pl.DataFrame, int]:
    """Resolve distinct indication strings to object attributes + assertion contexts.

    Shared by the FAERS and Canada Vigilance aggregation paths: stop-listed placeholders are
    dropped (counted), disease-map matches win, misses degrade to text-first ``Disease``
    objects, and canonical object attributes per resolved text keep dictionary-casing variants
    deterministic when merged. ``source`` feeds :func:`assertion_context` (``faers`` and
    ``canada_vigilance`` share the spontaneous-report semantics).
    """
    resolution: dict[str, dict[str, str]] = {}
    misses: list[str] = []
    stoplist_drops = 0
    for indication in indications:
        if is_non_disease_indication(indication):
            stoplist_drops += 1
            continue  # placeholder/usage-context indication, not a drug->condition observation
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
        .map_elements(lambda value: assertion_context(source, "indication", str(value or "")), return_dtype=pl.Utf8)
        .alias("assertion_context")
    )
    return mapping, stoplist_drops


def _canada_vigilance_rows(
    cv_indications: pl.DataFrame | None, disease_map: Mapping[str, Mapping[str, str]], approved_index: ApprovedTreatsIndex | None
) -> list[dict[str, str]]:
    """Aggregate Canada Vigilance drug-indication observations into applied-to-treat rows.

    Mirrors the FAERS aggregation with the source-appropriate inputs: the subject is the active
    ingredient (brand-name fallback for products the ingredients member does not cover), the
    case count is the number of DISTINCT report ids, and provenance rides the
    ``infores:canada-vigilance`` + ``infores:dailymed`` chain (the corroboration-derived
    ``clinical_approval_status`` is DailyMed-backed, exactly like FAERS rows). CV rows are built
    SEPARATELY from FAERS rows on purpose: same triple, different upstream chain, never merged.
    """
    if cv_indications is None or cv_indications.is_empty():
        return []

    def _text_column(name: str) -> pl.Expr:
        return pl.col(name).fill_null("").cast(pl.Utf8) if name in cv_indications.columns else pl.lit("")

    report_id = _text_column("report_id").str.strip_chars()
    cases = (
        cv_indications.lazy()
        .select(
            # Ingredient subject, brand fallback: ingredient text matches the approved-treats
            # subjects (DailyMed ingredient text) under the same pair normalization.
            pl.when(_text_column("ingredient").str.strip_chars() != "")
            .then(_text_column("ingredient").str.strip_chars())
            .otherwise(_text_column("drugname").str.strip_chars())
            .alias("subject"),
            _text_column("indication").str.strip_chars().alias("indication"),
            report_id.alias("report_id"),
            _text_column("source_record_id").alias("source_record_id"),
        )
        .filter((pl.col("subject") != "") & (pl.col("indication") != ""))
    )

    mapping, _stoplist_drops = _indication_mapping(
        sorted(set(cases.select("indication").unique().collect().get_column("indication").to_list()) - {""}), disease_map, "canada_vigilance"
    )
    _rid = pl.col("report_id")
    pairs = (
        cases.join(mapping.lazy(), on="indication", how="inner")
        .group_by("subject", "object_text", "assertion_context")
        .agg(
            pl.col("object_curie").first(),
            pl.col("object_name").first(),
            pl.col("object_category").first(),
            _rid.filter(_rid != "").n_unique().alias("distinct_cases"),
            _rid.filter(_rid != "").unique().alias("case_ids"),
            _rid.filter(_rid == "").len().alias("anon_rows"),
            pl.col("source_record_id").filter(pl.col("source_record_id") != "").unique().alias("source_records"),
            pl.col("source_record_id").filter((_rid == "") & (pl.col("source_record_id") != "")).unique().alias("anon_records"),
        )
        .collect()
        .sort("subject", "object_text")
    )

    rows: list[dict[str, str]] = []
    for rec in pairs.iter_rows(named=True):
        subject = str(rec["subject"])
        case_ids = {str(value) for value in rec.get("case_ids") or () if value}
        source_records = {str(value) for value in rec.get("source_records") or () if value}
        # Anonymous rows (no report id) contribute `anon:<source_record_id>` tokens, padded so
        # len(case_ids) == number_of_cases stays exact (the FAERS convention).
        anon_pad = int(rec["anon_rows"])
        anon_records = {str(value) for value in rec.get("anon_records") or () if value}
        anon_tokens = {f"anon:{record}" for record in anon_records}
        anon_tokens.update(f"anon:row:{subject}:{rec['object_text']}:{index}" for index in range(max(anon_pad, 0)))
        if approved_index is None:
            status = _STATUS_NOT_PROVIDED
        # CV reports carry no FDA application number, so only the normalized text rule can answer.
        elif approved_index.is_approved(_pair_key(subject), _pair_key(str(rec["object_text"])), ()):
            status = _STATUS_APPROVED
        else:
            status = _STATUS_OFF_LABEL
        rows.append(
            row_for(
                _TABLE,
                subject_text=subject,
                subject_curie="",
                subject_name=subject,
                subject_category="ChemicalEntity",
                predicate=_PREDICATE,
                object_text=str(rec["object_text"]),
                object_curie=str(rec["object_curie"]),
                object_name=str(rec["object_name"]),
                object_category=str(rec["object_category"]),
                assertion_context=str(rec.get("assertion_context") or "indication"),
                number_of_cases=int(rec["distinct_cases"]) + int(rec["anon_rows"]),
                case_ids=sorted_pipe([*case_ids, *anon_tokens]),
                FDA_regulatory_approvals="",
                edge_evidence="",
                supporting_faers_records=sorted_pipe(source_records),
                supporting_faers_urls=CANADA_VIGILANCE_SOURCE_RECORD_URL,
                clinical_approval_status=status,
                knowledge_level=_KNOWLEDGE_LEVEL,
                agent_type=AT_MANUAL,
                primary_knowledge_source=INFORES_DAKP,
                upstream_resource_ids=join_pipe(INFORES_CANADA_VIGILANCE, INFORES_DAILYMED),
            )
        )
    return rows


def build_observed_use_rows(
    faers_cases: pl.DataFrame | None,
    disease_map: Mapping[str, Mapping[str, str]],
    approved_pairs: ApprovedTreatsIndex | Iterable[tuple[str, str]] | None = None,
    *,
    approvals: FDAApprovalIndex | None = None,
    faers_quarter_urls: Mapping[str, str] | None = None,
    cv_indications: pl.DataFrame | None = None,
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

    ``approved_pairs`` is the approved-treats cross-reference
    (:class:`ApprovedTreatsIndex`, or a bare iterable of normalized ``(subject, object)`` text
    pairs when the caller only wants the text rule), or ``None`` when that table is unavailable,
    in which case every row degrades to ``clinical_approval_status = not_provided``.

    ``approvals`` expands the FAERS application numbers, which FAERS records with both the
    application-type prefix and the leading zeros stripped (``125514``), back to the FDA form
    every other source uses (``BLA125514``). A number no FDA register knows contributes NOTHING:
    ``nda_num`` is reporter free text, and its placeholders (``999999``, ``99``) and concatenated
    junk are not application numbers, so they must not ride ``regulatory_approvals``
    (:func:`~dakp_pipeline.assertions.evidence.is_fda_application_number`). The drops are reported
    as ``unresolved_application_numbers``, counted per ``(assertion row, distinct number)`` pair
    rather than as distinct numbers, so one placeholder reported across a thousand groups counts a
    thousand times; a missing register then reads as a collapse instead of silence.
    """
    if faers_cases is None:
        return []
    del ner, devices, cache
    approvals = approvals if approvals is not None else FDAApprovalIndex()
    approved_index = _coerce_approved_index(approved_pairs)

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
            _text_column("quarter").str.strip_chars().str.to_uppercase().alias("quarter"),
            _text_column("source_record_id").alias("source_record_id"),
        )
        .filter((pl.col("drugname") != "") & (pl.col("indication") != ""))
    )

    # Resolve each distinct stop-list-passing indication to its object BEFORE aggregation.
    mapping, stoplist_drops = _indication_mapping(
        sorted(set(cases.select("indication").unique().collect().get_column("indication").to_list()) - {""}), disease_map, "faers"
    )
    # Vectorized per-group set building (US-007): the previous form materialized EVERY source
    # row as a Python dict (49M structs on production) and folded it in a nested loop - ~2/3 of
    # the 5 h phase. The polars aggregations below compute the same distinct sets in one pass;
    # the Python loop only walks the (much smaller) per-group result lists.
    _pid = pl.col("primaryid")
    pairs = (
        cases.join(mapping.lazy(), on="indication", how="inner")  # stop-listed indications carry no mapping entry
        .group_by("drugname", "object_text", "assertion_context")
        .agg(
            pl.col("object_curie").first(),
            pl.col("object_name").first(),
            pl.col("object_category").first(),
            pl.col("indication").first().alias("context_indication"),
            _pid.filter(_pid != "").n_unique().alias("distinct_cases"),
            _pid.filter(_pid != "").unique().alias("case_ids"),
            _pid.filter(_pid == "").len().alias("anon_rows"),
            # Distinct quarters that carry a case id: the evidence-URL set is exactly the URLs of
            # these (a quarter maps to one URL), so the URL strings themselves are resolved in the
            # small second pass instead of per source row.
            pl.col("quarter").filter((_pid != "") & (pl.col("quarter") != "")).unique().alias("quarters"),
            pl.col("source_record_id").filter(pl.col("source_record_id") != "").unique().alias("source_records"),
            # Anonymous rows (no primaryid) contribute `anon:<source_record_id>` tokens.
            pl.col("source_record_id").filter((_pid == "") & (pl.col("source_record_id") != "")).unique().alias("anon_records"),
            pl.struct([pl.col("nda_raw"), pl.col("nda")]).unique().alias("nda_pairs"),
        )
        .collect()
        .sort("drugname", "object_text")
    )

    quarter_urls = dict(faers_quarter_urls or {})  # once, not per case row (was 49M dict copies)
    rows: list[dict[str, str]] = []
    unresolved_numbers = 0
    for rec in pairs.iter_rows(named=True):
        drug = str(rec["drugname"])
        obj = {
            "text": str(rec["object_text"]),
            "curie": str(rec["object_curie"]),
            "name": str(rec["object_name"]),
            "category": str(rec["object_category"]),
        }
        evidence_urls = {faers_record_url(quarter, quarter_urls) for quarter in rec.get("quarters") or ()}
        source_records = {str(value) for value in rec.get("source_records") or () if value}
        approval_values_by_norm: dict[str, set[str]] = {}
        for raw_pair in rec.get("nda_pairs") or ():
            pair = raw_pair or {}
            raw_nda = str(pair.get("nda_raw") or pair.get("nda") or "").strip()
            norm_nda = normalize_nda(raw_nda)
            if norm_nda:
                approval_values_by_norm.setdefault(norm_nda, set()).add(raw_nda)
        # One expansion per DISTINCT application number: the FAERS spellings of a number
        # (``125514``/``0125514``) all normalize to the same key, and the index answers with the
        # FDA display form(s) for that key. A number that expands to nothing is reporter junk, not
        # an application; the edge keeps its case-id and URL provenance either way.
        approval_values: list[str] = []
        for raw_values in approval_values_by_norm.values():
            if displays := approvals.expand(min(raw_values)):
                approval_values.extend(displays)
            else:
                unresolved_numbers += 1
        approval_values = merge_unique(approval_values)
        # ``anon_rows`` counts RAW primaryid-less rows while ``anon_records`` dedups their
        # source_record_ids (and an id-less row leaves no token at all), so pad with per-group
        # synthetic tokens — unique across rows that could merge downstream because the group
        # key is embedded — keeping len(case_ids) == number_of_cases exact.
        anon_records = {str(value) for value in rec.get("anon_records") or () if value}
        anon_tokens = {f"anon:{record}" for record in anon_records}
        pad = int(rec["anon_rows"]) - len(anon_tokens)
        anon_tokens.update(f"anon:row:{drug}:{obj['text']}:{index}" for index in range(max(pad, 0)))
        # Identified case ids: distinct non-empty primaryids of the group.
        case_ids = {str(value) for value in rec.get("case_ids") or () if value}
        if approved_index is None:
            status = _STATUS_NOT_PROVIDED
        elif approved_index.is_approved(_pair_key(drug), _pair_key(obj["text"]), approval_values):
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
    cv_rows = _canada_vigilance_rows(cv_indications, disease_map, approved_index)
    # One deterministic total order across both sources: (subject, object, context, upstream).
    # The tiebreakers matter now that two sources share the table: a FAERS row and a CV row
    # for the same triple must land in a stable, source-discernible order.
    rows = sorted(
        [*rows, *cv_rows], key=lambda row: (row["subject_text"], row["object_text"], row["assertion_context"], row["upstream_resource_ids"])
    )
    stats(
        logger,
        "shape_faers_applied_to_treat",
        stoplist_drops=stoplist_drops,
        assertions=len(rows),
        canada_vigilance_assertions=len(cv_rows),
        unresolved_application_numbers=unresolved_numbers,
    )
    return rows


def _coerce_approved_index(approved_pairs: ApprovedTreatsIndex | Iterable[tuple[str, str]] | None) -> ApprovedTreatsIndex | None:
    """Accept either the full index or a bare iterable of text pairs (``None`` stays ``None``).

    A bare pair iterable carries no application numbers, so it exercises the text rule alone:
    the pre-approval-identity contract, kept so callers and tests can pin that rule in isolation.
    Pairs with an empty member are dropped, matching :meth:`ApprovedTreatsIndex.from_frame`: an
    empty key could otherwise approve every row whose subject or object normalizes to nothing.
    """
    if approved_pairs is None or isinstance(approved_pairs, ApprovedTreatsIndex):
        return approved_pairs
    return ApprovedTreatsIndex(pairs=frozenset((subject, obj) for subject, obj in approved_pairs if subject and obj))


transform = ObservedUsesShaper().transform

__all__ = ["ApprovedTreatsIndex", "ObservedUsesShaper", "_canada_vigilance_rows", "build_observed_use_rows", "is_non_disease_indication", "transform"]
