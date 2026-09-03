"""Tablassert handoff: generate Graph/table configs and (optionally) run Tablassert.

Flattened from the former ``tablassert/configs.py`` + ``tablassert/run.py`` (US-004).

DAKP does everything up to the shape Tablassert consumes, then emits ONE Graph config
(``tables/graph.yaml``) plus one table config per assertion table (:func:`generate`).
Canonical entity resolution, KGX compilation, dedup, deterministic IDs, and RIG generation
are delegated to ``../Tablassert`` — DAKP ships **no** local fallback KGX compiler.

The configs match the ACTUAL current Tablassert schema (verified against
``../Tablassert/src/tablassert/models.py`` and its ``ingests.to_sections`` loader):

* a table config is a ``template:``-wrapped :class:`~tablassert.models.Section`
  (``source`` / ``statement`` / ``provenance`` / ``annotations``). The loader only reads
  top-level ``template`` / ``sections`` keys — a bare ``source:``/``statement:`` shape is
  silently dropped, so the ``template:`` wrapper is mandatory;
* ``source.kind: text`` with a tab ``delimiter`` and the uncompressed assertion ``.tsv``
  as ``source.local``. ``source.url`` is required by the model and records the table's REAL
  upstream dataset URL (:data:`_TABLE_SOURCE_URLS`) — never a placeholder. It serves the RIG
  only: edge provenance comes entirely from the explicit ``override.sources`` template
  (:data:`_TABLE_SOURCES`), which carries no dataset-level URLs (they are irrelevant
  per-edge). Tablassert
  8.2.1+ models ``source.url`` as a ``list`` of one or more URLs per section and DAKP assertion
  rows aggregate across quarters, releases, and applications, so no per-row URL is truthful at
  row granularity; the dataset-level URLs are the honest RIG record, and per-row precision
  stays on the edge via ``publications`` (SPL set links) and ``FDA_regulatory_approvals`` (FDA
  application numbers);
* column-encoded ``statement.subject`` / ``statement.object`` / ``statement.predicate``
  with drug / disease ``prioritize`` categories plus a HARD category allow-list each:
  ``avoid`` is emitted as the sorted complement of the side's allow-list over the installed
  Tablassert's full Biolink ``Categories`` enum, so no off-allow-list category can ever
  resolve into the graph (``prioritize`` alone is only a soft ranking boost);
* a ``provenance.override`` (:class:`~tablassert.models.ManualProvenance`) block carrying
  ``knowledge_level`` and ``agent_type``
  (the DAKP ``infores`` is graph-level only since Tablassert >= 8.0.1 forbids it in the override;
  no ``publication`` — the override replaces repo/publication provenance), plus the explicit
  ``sources`` template (:data:`_TABLE_SOURCES`) replicating the legacy DAKP edge-provenance
  shape — the DAKP wrapper entry carries the gestalt ``{edge_id}`` record-URL template
  (:data:`GESTALT_RECORD_URL_TEMPLATE`), resolved by Tablassert on the final edges;
* column-encoded ``statement.qualifiers`` where an assertion column carries the qualifier's entity
  (per-table :data:`_TABLE_QUALIFIERS`). A Tablassert qualifier is a node encoding resolved through
  the fullmap alongside subject/object; the contraindication context qualifier is ``nullable``, so
  a blank or unresolvable context omits only the qualifier rather than dropping the edge, and it
  carries its own Disease-only category allow-list (narrower than the object's);
* column-encoded evidence ``annotations`` (aggregated evidence columns carry ``split_by: "|"``
  so pipe-joined assertion cells emit as real JSON arrays, not joined scalars).

Column letters are DERIVED from the assertion-table column contracts in
:mod:`dakp_pipeline.io.schemas` (never hardcoded). YAML is emitted by a tiny stdlib
emitter (no ``pyyaml`` runtime dependency) that round-trips through ``yaml.safe_load``
(asserted in the unit tests).

The real runner (:class:`TablassertRunner`) shells out to the installed ``tablassert`` CLI
(a CORE dependency installed by the single ``uv sync``) and captures stdout / exit code into
a handoff report; the deferred runner (:class:`DeferredTablassertRunner`) writes a
deferred-handoff report without ever touching Tablassert (used when no fullmap triggers the
real handoff, and in tests). DAKP requires Tablassert >= 14.0: the graph config carries the
fullmap path (the ``build-kg --fullmap`` flag was removed in Tablassert 8.1), the 8.2
Biolink-valid KGX modeling (``sources[]`` retrieval provenance, first-class evidence slots)
is what the emitted configs target, 9.1's per-row ``split_by`` (made the ONLY multivalued
annotation encoding in 10.0) is what DAKP's pipe-joined evidence cells need, 11.0 makes
the graph config's ``rig:`` section mandatory while dropping the flat
``primary_knowledge_source`` edge column (nested ``sources[]`` provenance only — the edge
primary source now derives from ``rig.source_info.infores_id``), and 12.0 keeps a curated FDA-application-number
edge field TOP-LEVEL instead of folding it into ``supporting_text`` and stops
fabricating an empty supporting study for publication-less sections like DAKP's. 13.0 (biolink-model 4.4.4) renames the
``AffinityMeasurement`` config class to ``ProteinLigandAssayResult`` — the generated
``avoid:``/``prioritize:`` lists assume the new name — reworks the inlined supporting study
(``Study.id`` is the publication CURIE or config stem, no ``#`` composition), and adds the
stage-7 ``--qc`` fail-the-build assertions (empty-or-null-values, unnamed/unidentified
nodes, incomplete-edges) the production build now runs. 14.0 adds the explicit
``override.sources`` template (SkyeAv/Tablassert#116) with post-dedup ``{edge_id}``
resolution — what DAKP's legacy-shaped edge provenance requires, so 14.0 is the floor.
Fullmaps must
be ``tablassert.fullmap.v5`` redb files — the on-disk format since Tablassert 8.2, unchanged
in 13.0; older ones (v1-v4) are rejected on read.

The DEFAULT invocation runs the installed package — the venv ``tablassert`` binary when it is
on ``PATH``, otherwise ``uv run tablassert``. An OPTIONAL editable-checkout override (the
``tablassert_dir`` ctx param, the ``DAKP_TABLASERT_DIR`` env var, or the
``TablassertRunner.tablassert_dir`` field; conventionally ``DEFAULT_TABLASERT_DIR``) switches
to ``uv run --with-editable <dir> tablassert`` for dev against a local ``../Tablassert``
checkout. ``--qc`` is appended only when requested AND the QC audit runtime
(sentence-transformers, part of the required ``tablassert[qc]`` install) is importable;
``--release`` is a boolean flag.

The module-level :func:`run` is the entry point the stage harness and ``dags.dakp_build``
invoke as a MODULE ATTRIBUTE at call time (``tablassert.run(...)``), so
``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` replaces the callable they see.
It dispatches to the real runner when ``ctx.params["run_tablassert"]`` is truthy, else the
deferred runner. Tests monkeypatch the runner's subprocess hook (:func:`run_subprocess`) and
the availability probes (:func:`tablassert_available` / :func:`qc_runtime_available`) — no
real Tablassert required.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dakp_pipeline import __version__
from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock
from dakp_pipeline.logging_setup import logger, stats, step
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import dailymed as dailymed_source
from dakp_pipeline.sources import drugsfda as drugsfda_source
from dakp_pipeline.sources import faers as faers_source

# --- Translator provenance constants (match dakp_pipeline.assertions + ../DINGO) ----

INFORES_DAKP = "infores:multiomics-drugapprovals"
AGENT_TYPE = "manual_validation_of_automated_agent"

GRAPH_NAME = "DRUG_APPROVALS_KP"
#: Fallback ``fullmap`` written into ``graph.yaml`` when no real fullmap is configured (deferred
#: runs, which never invoke ``build-kg``). Tablassert's ``Graph`` model REQUIRES a ``fullmap``
#: field, and Tablassert reads the fullmap path FROM this field on a graph build (the
#: ``build-kg --fullmap`` flag was removed in Tablassert 8.1) — so :func:`generate`
#: writes the real ``ctx.params["fullmap"]`` here for real runs. DAKP never downloads a fullmap.
#: Tablassert reads only ``tablassert.fullmap.v5`` redb files (the on-disk format since
#: Tablassert 8.2) — a fullmap built by 8.1 or older is rejected on read ("fullmap DB is
#: outdated"); rebuild it with the installed ``tablassert build-fullmap``.
FULLMAP_DEFAULT = ".fullmap"

#: Edge identity fields declared as the Graph config's ``uuid_fields`` (Tablassert >= 16.0,
#: SkyeAv/Tablassert#122): only these feed the derived edge ``id``, so an attribute-only
#: change (``supporting_text``, ``sources``) no longer mints a new edge.
#: Identity is the SEMANTIC STATEMENT ONLY — the resolved triple, the nullable
#: ``disease_context_qualifier``, and the pre-resolution mentions:
#: ``original_subject`` / ``original_object`` are REQUIRED discriminators, not extras: two
#: distinct source mentions can resolve to the same canonical CURIE (observed in production:
#: two objects resolving to UMLS:C4721779 collided as ``uuid-fields-not-a-key``), and the
#: pre-resolution value is what keeps those records distinct. A declared field ABSENT from an
#: edge record contributes nothing at all to the hash, so the nullable
#: ``disease_context_qualifier`` — emitted only on the contraindication edges that
#: carry one — still discriminates those edges without forcing the key onto the other tables.
#: Evidence fields (``publications``, ``FDA_regulatory_approvals``, ``number_of_cases``) are
#: deliberately NOT identity: rows agreeing on the six fields are ONE edge whose evidence is
#: the merged union (deduplicated, sorted, pipe-joined — the
#: :func:`~dakp_pipeline.assertions.evidence.sorted_pipe` convention). That merging happens in
#: the assertion shapers BEFORE Tablassert ever sees the rows, because Tablassert's deduper
#: aborts the build (``uuid-fields-not-a-key``) when two differing records derive one id —
#: observed in production: two FAERS indication wordings resolving to the same object text
#: (CHEBI:62088 applied_to_treat HP:0012531), which the observed-uses shaper now folds into a
#: single row with an exact distinct-case count (see
#: :func:`~dakp_pipeline.assertions.observed_uses.build_observed_use_rows`).
#: Declaring ``uuid_fields`` also moves the UUID namespace onto the graph's infores
#: (:data:`INFORES_DAKP`) instead of the historic ``TABLASSERT`` constant.
UUID_FIELDS = ["subject", "predicate", "object", "disease_context_qualifier", "original_subject", "original_object"]

#: Real upstream dataset URL recorded as each table's ``source.url`` — the constants the
#: acquisition layer itself uses, so provenance can never drift from what was downloaded.
#: ``source.url`` is the section's RIG/audit record ONLY: edges get their provenance from the
#: explicit ``override.sources`` template (:data:`_TABLE_SOURCES`), which carries no
#: dataset-level URLs. Approved-treats and contraindication
#: rows are extracted from DailyMed SPL releases (the DailyMed full-release index); FAERS
#: observed-use rows from the FAERS quarterly ASCII extracts (the FDA quarterly-data listing).
_TABLE_SOURCE_URLS: dict[str, str] = {
    "approved_treats_assertions": dailymed_source.FULL_RELEASE_INDEX_URL,
    "faers_applied_to_treat_assertions": faers_source.FDA_FAERS_INDEX_URL,
    "contraindication_assertions": dailymed_source.FULL_RELEASE_INDEX_URL,
}
GRAPH_DESCRIPTION = (
    "Drug Approvals Knowledge Provider: FDA-approved treatment relationships, "
    "FAERS-observed applied-to-treat uses, and contraindications text-mined from "
    "DailyMed, modeled from DailyMed, Drugs@FDA, and FAERS. "
    "Every edge carries the evidence identifiers backing it; approved-treats edges also carry "
    "clinical approval status and FDA application numbers, FAERS-observed use edges add case "
    "counts, and contraindication edges carry application numbers where available."
)

# --- RIG (Resource Ingest Guide) graph-config section -------------------------------
#
# Tablassert >= 11 REQUIRES a ``rig:`` section on every graph config and rejects the legacy
# top-level ``description`` / ``infores`` keys (``rig-legacy-keys``). The nested shape mirrors
# the released resource-ingest-guide-schema; Tablassert composes the generated-artifact
# ``relevant_files`` / ``included_content`` entries and the observed target summaries itself,
# so only human-authored facts live here. The edge primary knowledge source derives from
# ``rig.source_info.infores_id`` — it is no longer a graph-level key.
#
# Content is ADAPTED from the DINGO-reviewed upstream DAKP RIG — ``NCATSTranslator/
# translator-ingests`` ``src/translator_ingest/ingests/dakp/dakp_rig.yaml`` (review issue
# #416; both linked in :data:`RIG_PROVENANCE_ARTIFACTS`) — keeping the reviewed facts and
# rejecting four upstream parts, each grounded in this repository:
# * the ``infores:medi`` source entry: MEDI belongs to the LEGACY pipeline; this rebuild has
#   no MEDI source module, so adopting it would be fabricated provenance (details on
#   :data:`RIG_SUPPORTING_DATA_SOURCES`);
# * the ``source_info`` CC BY 4.0 data-license claim: this repository carries NO evidence of
#   a CC BY 4.0 data license (the code license is Apache-2.0 and the upstream feeds are US
#   government public-domain data), so ``terms_of_use_info`` records the NLM/FDA terms instead;
# * the legacy KGX-file ``relevant_files``: legacy-pipeline OUTPUT artifacts no table section
#   sources; listing them would trip Tablassert's RIG audit (``rig-validation-failed``, see
#   ``_rig_config``);
# * the ``target_info`` node/edge type summaries: Tablassert GENERATES those from the observed
#   build, so hand-authored ones would fail validation and drift (details on
#   :data:`RIG_TARGET_FUTURE_CONSIDERATIONS`).

#: Public URL prefix the generated KGX artifacts are published under; Tablassert appends each
#: ``.nodes.ndjson`` / ``.edges.ndjson`` name to build RIG file locations. The GitHub repo URL
#: stands in until a dedicated public artifact location exists.
RIG_ARTIFACT_BASE_URL = "https://github.com/glusman-team/dakp"
#: Workdir-relative directory ``build-kg`` writes the KGX + RIG artifacts into (the runner's cwd
#: is the workdir root, so outputs stay in ``./data`` as before).
RIG_ARTIFACT_BASE_PATH = "data"
#: Full human-readable RIG source name — the bare acronym is ambiguous outside this repository,
#: and downstream ingest maintainers index sources by this name.
RIG_SOURCE_NAME = "Drug Approvals Knowledge Provider (DAKP)"
#: RIG citations: the DAKP method preprint with its resolvable PMC landing page, so ingest
#: maintainers can trace how the graph is produced. ``RIGSourceInfo.citations`` takes free-text,
#: URL-bearing strings.
RIG_CITATIONS = (
    "Generating Biomedical Knowledge Graphs from Knowledge Bases, Registries, and Multiomic Data "
    "(preprint): https://pmc.ncbi.nlm.nih.gov/articles/PMC11601480/",
)
#: RIG versioning statement. Interpolates the live package version so the prose can never drift
#: from the version every build embeds in the graph config via ``graph_config(version=...)``.
#: Freshness gates (see README "acquire"): DailyMed and Drugs@FDA re-downloads use the
#: acquire-stage 7-day download-cache window; FAERS is cache-first content-addressed, no age gate.
RIG_DATA_VERSIONING_AND_RELEASES = (
    f"DAKP versions follow the Python package version (pyproject.toml, currently {__version__}); "
    "every build embeds it in the graph config via graph_config(version=...). Re-ingests track "
    "the upstream cadence: FAERS quarterly ASCII extracts and DailyMed SPL releases. DailyMed "
    "and Drugs@FDA re-downloads are freshness-gated to a 7-day cache window; FAERS downloads "
    "are content-addressed and cache-first, with no age gate."
)
#: RIG ``supporting_data_source_info``: the upstream data sources a DAKP graph derives its
#: knowledge from. Exactly the TWO edge-backed upstreams, adapted from the DINGO-reviewed
#: ``supporting_data_source_info`` of the upstream ``NCATSTranslator/translator-ingests`` DAKP
#: RIG (names, descriptions, public-domain terms assessments), with each ``relevant_files[0]``
#: location swapped for the URL constant DAKP's acquisition layer actually downloads, so the
#: documented provenance can never drift from the download source. Unlike ``ingest_info``'s
#: ``relevant_files`` (filtered per graph to the tables present and audit-cross-checked against
#: table section sources), this section is free-form and always complete.
#: NO ``infores:medi`` entry: this rebuild has no MEDI source module (``src/dakp_pipeline/sources/``
#: is dailymed, drugsfda, faers only); contraindications are mined from DailyMed SPL. MEDI
#: belonged to the legacy pipeline — listing it HERE would be fabricated provenance (the
#: edge-level ``sources[]`` template still carries its legacy ``infores:medi`` entry for
#: shape parity; see :data:`_TABLE_SOURCES`).
#: NO Drugs@FDA entry either: only the two EDGE-BACKED upstreams are listed; Drugs@FDA enriches
#: assertions at build time (application joins) but backs no edge as a supporting source.
RIG_SUPPORTING_DATA_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "infores_id": "infores:dailymed",
        "name": "DailyMed",
        "description": (
            "DailyMed provides trustworthy information about marketed drugs in the United States, "
            "based on FDA Structured Product Labeling (SPL) documents. DAKP uses DailyMed to "
            "identify FDA-approved drug-indication and drug-contraindication relationships."
        ),
        "terms_of_use_info": {
            "terms_of_use_url": "https://dailymed.nlm.nih.gov/dailymed/",
            "terms_of_use_description": "DailyMed data are freely available and in the public domain.",
        },
        "relevant_files": [
            {
                "file_name": "DailyMed Structured Product Labeling",
                "location": dailymed_source.FULL_RELEASE_INDEX_URL,
                "description": "Structured product labeling (SPL) documents for FDA-approved drugs",
            }
        ],
    },
    {
        "infores_id": "infores:faers",
        "name": "FDA Adverse Event Reporting System (FAERS)",
        "description": (
            "FAERS contains adverse event reports, medication error reports, and product quality "
            "complaints submitted to the FDA. DAKP uses FAERS to derive drug-disease usage "
            "relationships and case counts, including on-label and off-label use."
        ),
        "terms_of_use_info": {
            "terms_of_use_url": (
                "https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/"
                "fda-adverse-event-reporting-system-faers-quarterly-data-extract-files"
            ),
            "terms_of_use_description": "FAERS data files are in the public domain and freely available for download.",
        },
        "relevant_files": [
            {
                "file_name": "FAERS Quarterly Data Files",
                "location": faers_source.FDA_FAERS_INDEX_URL,
                "description": "Quarterly data files containing adverse event reports",
            }
        ],
    },
)
#: RIG ``ingest_info.included_content``: the upstream record types DAKP pulls into the graph.
#: ``fields_used`` names exactly what the assertion tables consume from SPL — the four mined
#: section kinds' text (indications_and_usage, contraindications, boxed warnings,
#: warnings/precautions), SPL set identifiers feeding ``publications``, and FDA application
#: numbers feeding ``FDA_regulatory_approvals``. Unlike ``relevant_files``, ``included_content`` is NOT
#: audit-cross-checked by Tablassert, so these entries document intent rather than gate the build.
RIG_INCLUDED_CONTENT: tuple[dict[str, str], ...] = (
    {
        "file_name": "DailyMed SPL sections",
        "included_records": (
            "indications_and_usage (LOINC 34067-9), contraindications (LOINC 34070-3), "
            "boxed warnings (LOINC 34066-1), and warnings/precautions (LOINC 43685-7, legacy "
            "34071-1/42232-9) sections; FDA application numbers are carried as provenance where available"
        ),
        "fields_used": (
            "indications_and_usage, contraindications, boxed-warning, and warnings/precautions section text, "
            "SPL set identifiers, FDA application numbers"
        ),
    },
    {"file_name": "FAERS quarterly ASCII zips", "included_records": "drug/indication case pairs; case counts"},
)
#: RIG ``ingest_info.filtered_content``: what DAKP deliberately drops from the upstream feeds,
#: and why. Both entries restate the pipeline's actual scope gates: approved-treats assertions
#: require an approval-backed indication (``assertions/approved_treats.py`` rule 2), and the
#: assertion modules mine only the four section kinds listed in :data:`RIG_INCLUDED_CONTENT`
#: (``assertions/contraindications.py`` passes 1-3).
RIG_FILTERED_CONTENT: tuple[dict[str, str], ...] = (
    {
        "file_name": "DailyMed SPL indication sections",
        "filtered_records": "indication sections on SPL sets whose NDA lacks a DailyMed SPL approval",
        "rationale": (
            "approved-treats assertions require an FDA approval backing the indication; observed-use and "
            "text-mined contraindication assertions are deliberately not approval-gated"
        ),
    },
    {
        "file_name": "DailyMed SPL sections",
        "filtered_records": (
            "all SPL sections other than indications_and_usage (LOINC 34067-9), contraindications (LOINC 34070-3), "
            "boxed warnings (LOINC 34066-1), and warnings/precautions (LOINC 43685-7, legacy 34071-1/42232-9)"
        ),
        "rationale": "the remaining sections carry no treatment or contraindication evidence",
    },
)
#: RIG ingest-level ``future_considerations``: content DAKP deliberately defers, each grounded in
#: code — the medication-context note is the ``_TABLE_QUALIFIERS`` disease-only policy (the
#: qualifier is emitted; only medications are excluded from it), and the
#: status-coercion note is the ``ClinicalApprovalStatusEnum`` membership rule documented in
#: ``assertions/observed_uses.py``.
RIG_INGEST_FUTURE_CONSIDERATIONS: tuple[dict[str, str], ...] = (
    {
        "category": "edge_content",
        "consideration": (
            "A drug-drug chemical-entity interaction assertion for medication context (currently "
            "the disease_context_qualifier is intentionally disease-only; medications belong in a "
            "future interaction assertion)"
        ),
        "relevant_files": "DailyMed SPL contraindication sections",
    },
    {
        "category": "edge_property_content",
        "consideration": (
            "clinical_approval_status is a first-class Biolink ClinicalApprovalStatusEnum field, "
            "so values outside the enum cannot be preserved: the legacy FAERS observed_use status "
            "is coerced to not_provided (degraded mode). Revisit if Biolink adds a dedicated "
            "observation status."
        ),
    },
)


#: RIG ``provenance_info.contributions``: the PEOPLE first, then the pipeline/tooling
#: statements. The first three are VERBATIM from the DINGO-reviewed upstream
#: ``NCATSTranslator/translator-ingests`` DAKP RIG. Skye Lane Goetz is added because she
#: authored the upstream Tablassert features this pipeline's provenance override requires
#: (``ManualProvenance.upstream_source_record_urls``, SkyeAv/Tablassert#104, and the explicit
#: ``override.sources`` template, SkyeAv/Tablassert#116) and co-authored
#: the cited DAKP method preprint (PMC11601480).
RIG_CONTRIBUTIONS: tuple[str, ...] = (
    "Gwenlyn Glusman - code author, domain expertise, data modeling",
    "Matthew Brush - data modeling",
    "Sierra Moxon - code, data modeling",
    "Skye Lane Goetz - code author, pipeline engineering, Tablassert integration",
    "DAKP pipeline (https://github.com/glusman-team/dakp): source acquisition, assertion modeling",
    "Tablassert: KGX and RIG generation",
)
#: RIG ``provenance_info.artifacts``: external provenance artifacts — this pipeline's
#: repository, the upstream DINGO-reviewed DAKP RIG this one descends from, and that RIG's
#: review ticket.
RIG_PROVENANCE_ARTIFACTS: tuple[str, ...] = (
    "DAKP pipeline repository: https://github.com/glusman-team/dakp",
    "Upstream DINGO-reviewed DAKP RIG: https://github.com/NCATSTranslator/translator-ingests/blob/main/src/translator_ingest/ingests/dakp/dakp_rig.yaml",
    "RIG review issue: https://github.com/NCATSTranslator/translator-ingests/issues/416",
)
#: RIG ``target_info`` modeling future considerations. The upstream DAKP RIG's
#: ``target_info.edge_type_info`` / ``node_type_info`` are REJECTED wholesale: Tablassert's
#: ``RIGTargetInfoExtras`` accepts ONLY ``future_considerations`` + ``additional_notes``
#: because node/edge type summaries are GENERATED by Tablassert from the observed build —
#: hand-written summaries would fail validation and drift from the graph. Categories are
#: exact ``tablassert.enums.ModelingCategories`` members.
RIG_TARGET_FUTURE_CONSIDERATIONS: tuple[dict[str, str], ...] = (
    {
        "category": "qualifiers",
        "consideration": (
            "disease_context_qualifier is emitted on contraindication edges via Tablassert's CLASS_FIELD_OVERRIDES grant "
            "(SkyeAv/Tablassert#120), which is deliberately ahead of the pinned Biolink model: Biolink declares the slot only on "
            "ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation, while these edges pin EntityToDisease/EntityToPhenotypicFeature "
            "for FDA_regulatory_approvals. Drop reliance on the grant once upstream Biolink widens disease_context_qualifier to the "
            "entity-to-disease classes"
        ),
    },
    {
        "category": "edge_properties",
        "consideration": (
            "FDA application numbers are emitted on the Biolink FDA_regulatory_approvals slot, which only "
            "EntityToDiseaseAssociation and EntityToPhenotypicFeatureAssociation declare, so every edge pins one of those classes by object "
            "category; monitor whether the slot is widened to the chemical-to-disease association classes"
        ),
    },
)
#: RIG ``target_info.additional_notes``: records that the type summaries are machine-generated
#: per build, so nobody re-authors them here.
RIG_TARGET_ADDITIONAL_NOTES: tuple[str, ...] = (
    "Node and edge type summaries are generated by Tablassert from the observed build, not authored in this configuration.",
)


def _rig_config(tables: list[str]) -> dict[str, Any]:
    """The required ``rig:`` graph-config section (Tablassert >= 11), all constants.

    Content is ADAPTED from the DINGO-reviewed upstream DAKP RIG (``NCATSTranslator/
    translator-ingests`` ``src/translator_ingest/ingests/dakp/dakp_rig.yaml``, review issue
    #416); the full adoption/rejection rationale lives in the section-header comment above
    the RIG constants.

    Shape verified against ``tablassert.models.RIGConfig`` (pinned directly by
    ``test_rig_section_validates_directly_against_tablassert_rig_config``):
    ``supporting_data_source_info`` (the two edge-backed upstreams,
    :data:`RIG_SUPPORTING_DATA_SOURCES`), ``source_info``
    (infores id, full source name, description + citation, non-empty terms-of-use assessment,
    URL-bearing data access locations, versioning story, source status),
    ``ingest_info`` (explicit ingest category, authored utility/scope, per-table upstream
    relevant files + included/filtered content + future considerations),
    ``target_info`` (modeling future considerations + generated-summary note; upstream type
    summaries rejected), ``provenance_info`` (named contributors + pipeline/tooling
    contributions, provenance artifact links), and the artifact base URL/path pair.

    ``relevant_files`` is filtered to the upstream URLs of the tables actually IN this graph:
    Tablassert's RIG audit fails the build when a configured relevant-file matches no table
    source (``rig-validation-failed``), so a single-table graph (tests, partial builds) must
    not list the other tables' upstreams.
    """
    included_urls = {_TABLE_SOURCE_URLS[table] for table in _TABLE_ORDER if any(Path(t).name == f"{_TABLE_SPECS[table][0]}.yaml" for t in tables)}
    relevant_files = [
        {
            "file_name": "DailyMed full-release SPL zips",
            "location": dailymed_source.FULL_RELEASE_INDEX_URL,
            "description": "Structured Product Labeling XML releases; indications/contraindications sections and approval numbers.",
        },
        {
            "file_name": "FAERS quarterly ASCII zips",
            "location": faers_source.FDA_FAERS_INDEX_URL,
            "description": "FDA Adverse Event Reporting System quarterly extracts; drug/indication case pairs.",
        },
        # No Drugs@FDA entry: no assertion table declares it as a section source (it is joined
        # in upstream, at assertion-build time), so the RIG audit would reject it.
        # REJECTED from the upstream ``NCATSTranslator/translator-ingests`` DAKP RIG: its legacy
        # KGX-file ``relevant_files`` (``drug_approvals_kg_{edges,nodes}.jsonl.gz`` at
        # db.systemsbiology.net) are legacy pipeline OUTPUT artifacts, not inputs; no table
        # section sources them here, so Tablassert's RIG audit would fail the build
        # (``rig-validation-failed``).
    ]
    return {
        # Deep copies: the returned config dict must never alias the module-level constants.
        "supporting_data_source_info": [copy.deepcopy(entry) for entry in RIG_SUPPORTING_DATA_SOURCES],
        "source_info": {
            "infores_id": INFORES_DAKP,
            "name": RIG_SOURCE_NAME,
            "description": GRAPH_DESCRIPTION,
            "citations": list(RIG_CITATIONS),
            "terms_of_use_info": {
                "terms_of_use_url": "https://www.nlm.nih.gov/terms.html",
                "terms_of_use_description": (
                    "DAKP is derived from DailyMed (NLM), Drugs@FDA, and FAERS (FDA) — US government "
                    "public-domain data; the NLM and FDA terms of use apply."
                ),
            },
            "data_access_locations": [
                f"DailyMed SPL releases - {dailymed_source.FULL_RELEASE_INDEX_URL}",
                f"FAERS quarterly ASCII extracts - {faers_source.FDA_FAERS_INDEX_URL}",
                f"Drugs@FDA data files - {drugsfda_source.DRUGSFDA_DATA_FILES_URL}",
            ],
            "data_provision_mechanisms": ["file_download"],
            "data_formats": ["kgx"],
            "data_versioning_and_releases": RIG_DATA_VERSIONING_AND_RELEASES,
            "source_status": "maintained_regular_updates",
        },
        "ingest_info": {
            # Explicit even though it equals the ``RIGIngestInfo`` default: DAKP CREATES
            # knowledge (the assertion tables this pipeline builds) rather than passing a source
            # through, and the upstream DAKP RIG declares the same category.
            "ingest_categories": ["translator_knowledge_creator"],
            "utility": (
                "Provides FDA-approved drug-disease treatment relationships, FAERS-observed "
                "applied-to-treat uses, and SPL-mined contraindications for Translator querying."
            ),
            "scope": (
                "Approved-treats edges (DailyMed SPL indications joined to Drugs@FDA applications and "
                "FAERS cases), FAERS observed-use edges, and contraindication edges text-mined from "
                "DailyMed SPL sections; all other content of the upstream feeds is out of scope."
            ),
            "relevant_files": [entry for entry in relevant_files if entry["location"] in included_urls],
            "included_content": [copy.deepcopy(entry) for entry in RIG_INCLUDED_CONTENT],
            "filtered_content": [copy.deepcopy(entry) for entry in RIG_FILTERED_CONTENT],
            "future_considerations": [copy.deepcopy(entry) for entry in RIG_INGEST_FUTURE_CONSIDERATIONS],
        },
        "target_info": {
            "future_considerations": [copy.deepcopy(entry) for entry in RIG_TARGET_FUTURE_CONSIDERATIONS],
            "additional_notes": list(RIG_TARGET_ADDITIONAL_NOTES),
        },
        "provenance_info": {"contributions": list(RIG_CONTRIBUTIONS), "artifacts": list(RIG_PROVENANCE_ARTIFACTS)},
        "artifact_base_url": RIG_ARTIFACT_BASE_URL,
        "artifact_base_path": RIG_ARTIFACT_BASE_PATH,
    }


# Canonical emission order for the three assertion tables.
_TABLE_ORDER = ("approved_treats_assertions", "faers_applied_to_treat_assertions", "contraindication_assertions")

# assertion table -> subject_text row exclusions, emitted as ``source.reindex`` ``ne`` filters so
# the rows are dropped at Tablassert LOAD time and can never reach the final KGX output. The FAERS
# off-label table aggregates methanol poison-exposure reports where the chemical is the EXPOSURE,
# not a treatment — an ``applied_to_treat`` edge for it is semantically wrong ("METHYL ALCHOL" is
# FAERS's own spelling of the ingredient). "GENERIC DRUG"/"GENERIC DRUGS" are FAERS verbatim
# drug-name placeholders that name no real substance, and "PLACEBO" is the control arm: by
# definition it treats nothing. Entries are written in their canonical uppercase form and matched
# case-insensitively (see :func:`_casing_variants`). Reindex conditions AND together: a row
# survives only when its subject_text matches no denylist entry in any emitted casing.
_TABLE_SUBJECT_DENYLIST: dict[str, tuple[str, ...]] = {
    "faers_applied_to_treat_assertions": ("METHYL ALCHOL", "METHANOL", "GENERIC DRUG", "GENERIC DRUGS", "PLACEBO")
}

# assertion table -> (config basename, predicate, knowledge_level, agent_type). Knowledge levels
# match the DINGO translator-ingest provenance contract
# (../DINGO/tests/unit/ingests/dakp/test_dakp.py): treats = knowledge_assertion;
# applied_to_treat = observation; contraindicated_in = knowledge_assertion mined from DailyMed.
# Every family uses ``AGENT_TYPE`` — the agent type the legacy DAKP KG shipped on all three
# predicates (``manual_validation_of_automated_agent``).
_TABLE_SPECS: dict[str, tuple[str, str, str, str]] = {
    "approved_treats_assertions": ("approved_treats", "treats", "knowledge_assertion", AGENT_TYPE),
    "faers_applied_to_treat_assertions": ("faers_applied_to_treat", "applied_to_treat", "observation", AGENT_TYPE),
    "contraindication_assertions": ("contraindications", "contraindicated_in", "knowledge_assertion", AGENT_TYPE),
}

#: Per-edge record URL template carried by the ``infores:multiomics-drugapprovals`` sources
#: entry — the gestalt viewer deep-links each edge by its own id. ``{edge_id}`` is resolved by
#: Tablassert in a post-dedup sweep of the final edges ndjson (``override.sources``,
#: SkyeAv/Tablassert#116).
GESTALT_RECORD_URL_TEMPLATE = "https://db.systemsbiology.net/gestalt/cgi-pub/KGinfo.pl?id={edge_id}"

# assertion table -> the explicit ``provenance.override.sources`` template, recovered from the
# shipped legacy ``drug_approvals_kg_edges.jsonl``: (resource_id, role, upstream ids) in legacy
# entry order. The DAKP entry always carries the gestalt record URL; the remaining entries carry
# no ``source_record_urls`` (dataset-level URLs are irrelevant per-edge). Contraindication edges
# keep the legacy ``infores:medi`` primary entry even though this rebuild has no MEDI source
# module — edge-shape parity only; the RIG deliberately stays MEDI-free (see the
# ``RIG_SUPPORTING_DATA_SOURCES`` comment). Requires Tablassert >= 14.0 (SkyeAv/Tablassert#116).
_TABLE_SOURCES: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "approved_treats_assertions": (
        (INFORES_DAKP, "primary_knowledge_source", ("infores:dailymed", "infores:faers")),
        ("infores:faers", "supporting_data_source", ()),
        ("infores:dailymed", "supporting_data_source", ()),
    ),
    "faers_applied_to_treat_assertions": (
        (INFORES_DAKP, "aggregator_knowledge_source", ("infores:dailymed", "infores:faers")),
        ("infores:faers", "primary_knowledge_source", ()),
        ("infores:dailymed", "supporting_data_source", ()),
    ),
    "contraindication_assertions": (
        (INFORES_DAKP, "aggregator_knowledge_source", ("infores:dailymed", "infores:medi")),
        ("infores:medi", "primary_knowledge_source", ("infores:dailymed",)),
        ("infores:dailymed", "supporting_data_source", ()),
    ),
}


def _sources_template(table: str) -> list[dict[str, Any]]:
    """The table's explicit ``override.sources`` entries (the DAKP entry gets the gestalt URL)."""
    entries: list[dict[str, Any]] = []
    for resource_id, role, upstream in _TABLE_SOURCES[table]:
        entry: dict[str, Any] = {"resource_id": resource_id, "resource_role": role}
        if upstream:
            entry["upstream_resource_ids"] = list(upstream)
        if resource_id == INFORES_DAKP:
            entry["source_record_urls"] = [GESTALT_RECORD_URL_TEMPLATE]
        entries.append(entry)
    return entries


# assertion column -> (annotation name, multivalued separator), per table. Every DAKP edge resolves
# to ``biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation`` (Tablassert derives the
# category from the (subject role, object role) pair, then keeps it for all three DAKP predicates),
# so an annotation name is only useful if THAT class declares the slot. Anything else is relocated:
# Tablassert's ``prune_to_class`` nulls what the class refuses and hands it to the inlined
# ``has_supporting_studies`` as a ``"name=value"`` string in the StudyResult ``description`` — a
# junk drawer, and nothing in ``NCATSTranslator/translator-ingests`` models evidence that way (there
# a ``Study`` is a real cohort/dataset/trial with TYPED ``StudyResult`` slots). So each name below is
# a slot the class actually holds:
# * ``publications`` (``list[str]``, declared by the root ``Association`` class so every class
#   :data:`OBJECT_CATEGORY_OVERRIDE` pins holds it) carries the sorted identifier union from the
#   single ``edge_evidence`` column: ``dailymed:<spl_set_id>`` CURIEs for the backing DailyMed
#   labels. This is the slot the deployed translator-ingests dakp transform lands DAKP evidence
#   in anyway (it appends the legacy ``has_evidence`` values onto ``publications``, "store as
#   publications for now"), so the edge ships the final slot directly and the ingest's stopgap
#   re-homing disappears. One column,
#   not two, because ANNOTATION NAMES MUST BE UNIQUE PER TABLE: Tablassert applies annotations
#   as ``with_columns(pl.col(src).alias(name))`` in declaration order with no duplicate check,
#   so a second ``publications`` entry would SILENTLY overwrite the first. Human-readable URLs
#   remain in the unannotated source-specific debug columns and source-record provenance. (The
#   deprecated Biolink ``supporting_documents`` slot — an alias of ``publications`` — is
#   attached to no association class and was what previously landed in the study description.)
#   The provenance-override path for the same slot is inert here: DAKP's ``ManualProvenance``
#   carries no ``publications`` of its own (the model demands ``PMCID:``-prefixed CURIEs
#   there), so Tablassert's provenance-phase ``publications`` op never runs to overwrite the
#   column-encoded values.
# * DAKP aggregates each cell into ONE pipe-joined string, so ``split_by: "|"`` is what makes it a
#   real JSON array. Without the split Tablassert wraps the joined scalar into a USELESS
#   one-element list (``["url1|url2"]``) that still passes Biolink validation — a silent-corruption
#   guard, not a hard-failure one. Tablassert 8.2.1 removed the old ``delimiter`` spelling, and 10.0
#   removed its ``method: list`` successor (a config-time literal — the same array on every row)
#   entirely, so ``split_by`` is the ONLY encoding that can express a per-row array;
# * ``clinical_approval_status`` is a first-class enum-typed field on the association class, so its
#   values must be ``ClinicalApprovalStatusEnum`` members (see the enum-membership note in
#   ``assertions/observed_uses.py``);
# * ``number_of_cases`` (integer) takes the FAERS case count — the literal Biolink slot ("cases
#   carrying the phenotype/disease"), declared by the classes :data:`OBJECT_CATEGORY_OVERRIDE`
#   pins. Until 15.1 the NAME was unusable: Tablassert's study-size classifier claimed it
#   (``coerce.study_size_target("number_of_cases") == "study_size"``), so the clean phase renamed
#   it and the count landed on the inlined supporting study as ``Study.study_size``. 15.1's
#   ``STUDY_SIZE_EXEMPT_PATTERN`` (SkyeAv/Tablassert#119) exempts the exact slot, so the column
#   reaches the edge as ``number_of_cases`` — DAKP used the ``evidence_count`` alias before that;
# * ``FDA_regulatory_approvals`` (FDA application numbers) IS a Biolink slot — "numbers that
#   identify specific drug applications", multivalued, declared by ``EntityToDiseaseAssociation``
#   and ``EntityToPhenotypicFeatureAssociation``, the classes
#   :data:`OBJECT_CATEGORY_OVERRIDE` pins. DAKP annotates it with ``split_by: "|"`` so the
#   pipe-joined cell reaches the final KGX edge as its own top-level JSON ARRAY (the legacy
#   ``approvals`` list shape) instead of a joined scalar. Its mixed case survives because
#   Tablassert >= 15.0 canonicalizes a declared annotation name onto the allow-listed slot
#   spelling instead of lowercasing it (SkyeAv/Tablassert#117); before that it reached the fold
#   sweep as ``fda_regulatory_approvals``, matched no slot, and folded into ``supporting_text``.
#   ``source_score`` still has no reachable
#   slot and no carve-out, so it folds into ``supporting_text`` as a ``"name: value"`` string —
#   visible provenance, deliberately kept. ``has_confidence_score``
#   would be mechanically available for ``source_score``, but that column is the max NER SPAN score
#   — confidence that a mention was recognized, not that the statement is true — so promoting it
#   would mislead any consumer that weights edges by confidence.
_TABLE_ANNOTATIONS: dict[str, tuple[tuple[str, str, str | None], ...]] = {
    "approved_treats_assertions": (
        ("FDA_regulatory_approvals", "FDA_regulatory_approvals", "|"),
        ("edge_evidence", "publications", "|"),
        ("clinical_approval_status", "clinical_approval_status", None),
    ),
    "faers_applied_to_treat_assertions": (
        ("case_count", "number_of_cases", None),
        ("FDA_regulatory_approvals", "FDA_regulatory_approvals", "|"),
        ("edge_evidence", "publications", "|"),
        ("clinical_approval_status", "clinical_approval_status", None),
    ),
    "contraindication_assertions": (
        ("FDA_regulatory_approvals", "FDA_regulatory_approvals", "|"),
        ("edge_evidence", "publications", "|"),
        # ``evidence_text`` (the SPL contraindication prose) is deliberately NOT annotated: mapped
        # to ``supporting_text`` it buried every edge under full sentences, making the KGX output
        # unreadable. The column stays in the assertion TSV as provenance; only the edge drops it.
        ("source_score", "source_score", None),
    ),
}

SUBJECT_COLUMN = "subject_text"
OBJECT_COLUMN = "object_text"
# Per-side category allow-lists. Tablassert ``prioritize`` is a soft ranking boost only, so each
# side ALSO emits a hard ``avoid`` guard — the sorted complement of its allow-list over the
# installed Tablassert's full Biolink ``Categories`` enum (see :func:`category_avoid_list`) —
# keeping wacky fullmap categories (genes, taxa, publications, devices, ...) out of the graph.
# The allow-list IS the ``prioritize`` tuple on each side; widen both together.
SUBJECT_PRIORITIZE = ("Drug", "SmallMolecule", "ChemicalEntity")
OBJECT_PRIORITIZE = ("Disease", "PhenotypicFeature")
#: ``statement.category_override`` (Tablassert >= 15.0): the association class each object
#: category pins, in place of the derived ``(subject role, object role)`` pair lookup. Tablassert
#: derives ``ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation`` for every DAKP pair, and that
#: class declares NEITHER ``FDA_regulatory_approvals`` nor ``number_of_cases`` — Biolink attaches
#: both to ``EntityToDiseaseAssociation`` / ``EntityToPhenotypicFeatureAssociation`` instead — so
#: ``prune_to_class`` nulled them off the edge and rescued them into the pruned column. Pinning the
#: two classes per object category is the split the legacy KG made for the same reason
#: (``ref/legacy/bin/dakp-postprocess2jsonlBL.py``: ``biolink:EntityToDiseaseAssociation`` for
#: Disease objects, ``biolink:EntityToPhenotypicFeatureAssociation`` for PhenotypicFeature ones,
#: with the chemical-to-disease class commented out beside it). The keys are exactly
#: :data:`OBJECT_PRIORITIZE`: every object category DAKP allows is pinned, so no row falls back to
#: the derived pair.
OBJECT_CATEGORY_OVERRIDE: dict[str, str] = {"Disease": "EntityToDiseaseAssociation", "PhenotypicFeature": "EntityToPhenotypicFeatureAssociation"}

# Per-table biolink statement qualifiers: (qualifier slot, backing assertion column). Emitted as
# ``statement.qualifiers`` entries ONLY where a column actually carries the qualifier's entity.
# Validity is two-layered: the slot must be a member of the installed Tablassert's Biolink
# ``Qualifiers`` enum and the association class must support it. The second layer is met for the
# contraindication context qualifier by Tablassert 15.1's ``CLASS_FIELD_OVERRIDES``
# (SkyeAv/Tablassert#120): Biolink declares ``disease_context_qualifier`` on
# ``ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation`` ONLY, while
# :data:`OBJECT_CATEGORY_OVERRIDE` pins ``EntityToDiseaseAssociation`` /
# ``EntityToPhenotypicFeatureAssociation`` — the only classes declaring
# ``FDA_regulatory_approvals`` — so without the grant a contraindication edge could carry the
# approvals or the qualifier, never both (``prune_to_class`` would null the qualifier into the
# pruned column). The grant is deliberately ahead of the pinned Biolink model; drop this note
# once upstream Biolink widens the slot to the entity-to-disease classes.
_TABLE_QUALIFIERS: dict[str, tuple[tuple[str, str], ...]] = {
    # ``clinical_approval_status`` ("approved_for_condition") is the Biolink ClinicalApprovalStatusEnum
    # ASSOCIATION slot, not a qualifier slot — no ``Qualifiers`` member expresses approval status, so
    # it stays an annotation; FDA application numbers / SPL ids are provenance strings, not entities.
    "approved_treats_assertions": (),
    # The indication (object) is the only disease on a FAERS row, so a ``disease_context_qualifier``
    # encoded from the object column just restates the object — biolink defines it as the condition
    # a relationship "took place" in, which only informs when it DIFFERS from the object. The
    # ``applied_to_treat`` predicate + ``knowledge_level: observation`` already carry the semantics.
    # The adverse event itself (FAERS ``effects``) is aggregated away and not part of the assertion
    # contract; it is an adverse reaction rather than a disease context, so it is not a substitute.
    "faers_applied_to_treat_assertions": (),
    # ``disease_context_text`` is a distinct disease from the contraindicated object only when the
    # extractor's explicit template classifier populated it; blank cells are valid and the qualifier
    # is emitted ``nullable``, so an unconditional contraindication keeps its edge minus only the
    # qualifier. Disease-only by design: medications belong in a future chemical-entity interaction
    # assertion, not this slot.
    "contraindication_assertions": (("disease_context_qualifier", "disease_context_text"),),
}

_GENERATE_OPERATION = "generate_tablassert_configs"


def _biolink_categories() -> tuple[str, ...]:
    """All entity category names of the installed Tablassert's Biolink model, sorted.

    Tablassert builds its ``Categories`` enum dynamically from the Biolink Model it ships, so
    deriving the universe from the installed package (instead of freezing a copy here) keeps the
    emitted ``avoid`` lists exactly consistent with the enum that validates them at config load —
    across Tablassert/Biolink upgrades and fullmap rebuilds alike.
    """
    from tablassert.biolink import Categories  # lazy: keep this module's own import light

    return tuple(sorted(category.value for category in Categories))


def category_avoid_list(allowed: Sequence[str]) -> list[str]:
    """Hard category allow-list for one node encoding, expressed as Tablassert's ``avoid`` list.

    Tablassert has no allow-list knob: ``NodeEncoding.avoid`` is the only hard category filter
    (``prioritize`` merely re-ranks), so the allow-list is emitted as the SORTED complement of
    ``allowed`` over every category in the installed Biolink model. Candidates carrying an avoided
    category are dropped during entity resolution; a mention whose only candidates are avoided
    stays unresolved (its row emits no edge, logged by Tablassert) — recall is deliberately traded
    for the guarantee that no off-allow-list category reaches the graph.

    Raises ``ValueError`` when an ``allowed`` entry is not a category the installed Tablassert
    knows (it would fail Tablassert config validation anyway — fail loudly at generation time).
    """
    universe = _biolink_categories()
    unknown = sorted(set(allowed) - set(universe))
    if unknown:
        msg = f"allowed categories not in the installed Tablassert Biolink model: {unknown}"
        raise ValueError(msg)
    allowed_set = set(allowed)
    return [category for category in universe if category not in allowed_set]


# --- Excel-style column letters ---------------------------------------------------


def excel_column(index: int) -> str:
    """0-based column index -> Excel-style letters (``0->A``, ``25->Z``, ``26->AA``).

    Tablassert reads source files headerless and addresses columns by Excel-style letters
    (``EncodingMethods.COLUMN``); this maps an assertion-table column position to its letter.
    """
    if index < 0:
        msg = f"column index must be >= 0, got {index}"
        raise ValueError(msg)
    letters = ""
    n = index + 1
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _casing_variants(name: str) -> tuple[str, ...]:
    """Casing renderings of a :data:`_TABLE_SUBJECT_DENYLIST` entry, canonical uppercase first.

    Tablassert's ``ne`` reindex filter is polars ``!=`` — an exact, case-SENSITIVE string compare
    (``tablassert.lib.reindex``, ``cast=False`` for ``eq``/``ne``) — and the ``Comparisons`` enum
    has no case-insensitive member, so case-insensitive denial has to be spelled out as one ``ne``
    per rendering. FAERS ``drugname`` is reporter-entered free text carried through verbatim as
    ``subject_text``: uppercase dominates, with title ("Generic Drug"), sentence ("Generic drug")
    and lower renderings all occurring in practice, so those four cover the realistic space.
    Deduplicated (single-word entries collapse title into sentence case) and order-stable so the
    committed configs stay byte-reproducible. Known gap: a mid-word oddity ("PLaCEBO") still slips
    through — closing it needs a case-insensitive comparison in Tablassert itself.
    """
    return tuple(dict.fromkeys((name.upper(), name.lower(), name.title(), name.capitalize())))


def column_letter(table: str, column: str) -> str:
    """Excel-style letter for ``column`` in ``table``'s ordered contract (KeyError if absent)."""
    columns = schemas.columns_for(table)
    if column not in columns:
        msg = f"column {column!r} not in {table} contract: {columns}"
        raise KeyError(msg)
    return excel_column(columns.index(column))


# --- config dict builders (single source of truth) --------------------------------


def table_config(table: str) -> dict[str, Any]:
    """Build the Tablassert ``Section`` body (the ``template:`` value) for an assertion table.

    Shape verified against ``tablassert.models.Section``: ``source`` (text), ``statement``
    (column-encoded subject/predicate/object plus the table's ``qualifiers`` when a column backs
    them), ``provenance.override`` (ManualProvenance), and column-encoded ``annotations`` for the
    table's evidence columns. Subject/object carry ``prioritize`` (soft ranking) plus ``avoid`` —
    the hard allow-list guard computed by :func:`category_avoid_list` from the side's ``prioritize``
    tuple. Qualifier values use their own category guard: contraindication context is constrained
    to ``Disease`` (not the object's broader Disease/PhenotypicFeature list), and is nullable so
    absent or unresolved context omits only the qualifier rather than deleting the edge. A table
    with a :data:`_TABLE_SUBJECT_DENYLIST` entry additionally carries ``source.reindex`` ``ne``
    filters — one per :func:`_casing_variants` rendering of each entry — that drop those
    subject_text rows before entity resolution.
    """
    _basename, predicate, knowledge_level, agent_type = _TABLE_SPECS[table]  # KeyError for unknown tables
    annotations: list[dict[str, Any]] = []
    for column, annotation, split_by in _TABLE_ANNOTATIONS[table]:
        entry: dict[str, Any] = {"annotation": annotation, "method": "column", "encoding": column_letter(table, column)}
        if split_by is not None:  # multivalued Biolink slot: split the pipe-joined cell into a JSON array
            entry["split_by"] = split_by
        annotations.append(entry)
    qualifiers = [
        {
            "qualifier": qualifier,
            "method": "column",
            "encoding": column_letter(table, column),
            "nullable": table == "contraindication_assertions",
            "prioritize": ["Disease"],
            "avoid": category_avoid_list(("Disease",)),
        }
        for qualifier, column in _TABLE_QUALIFIERS[table]
    ]
    statement: dict[str, Any] = {
        "subject": {
            "method": "column",
            "encoding": column_letter(table, SUBJECT_COLUMN),
            "prioritize": list(SUBJECT_PRIORITIZE),
            "avoid": category_avoid_list(SUBJECT_PRIORITIZE),
        },
        "predicate": predicate,
        "object": {
            "method": "column",
            "encoding": column_letter(table, OBJECT_COLUMN),
            "prioritize": list(OBJECT_PRIORITIZE),
            "avoid": category_avoid_list(OBJECT_PRIORITIZE),
        },
        "category_override": dict(OBJECT_CATEGORY_OVERRIDE),
    }
    if qualifiers:  # no backing column => no ``qualifiers`` key (Tablassert treats absent and empty alike; keep configs minimal)
        statement["qualifiers"] = qualifiers
    source: dict[str, Any] = {"kind": "text", "local": f"data/tabular/{table}.tsv", "url": [_TABLE_SOURCE_URLS[table]], "delimiter": "\t"}
    denylist = _TABLE_SUBJECT_DENYLIST.get(table)
    if denylist:
        subject_letter = column_letter(table, SUBJECT_COLUMN)
        source["reindex"] = [
            {"column": subject_letter, "comparison": "ne", "comparator": variant} for name in denylist for variant in _casing_variants(name)
        ]
    return {
        "source": source,
        "statement": statement,
        "provenance": {
            "override": {
                # Explicit ``sources`` list replicating the legacy DAKP edge-provenance shape
                # exactly (DAKP wrapper entry with the gestalt record-URL template, then the
                # per-resource entries); Tablassert >= 14.0 uses it verbatim on every edge and
                # resolves the ``{edge_id}`` placeholder post-build. No dataset-level record
                # URLs anywhere: they are irrelevant per-edge, and the section ``source.url``
                # stays RIG-only.
                "sources": _sources_template(table),
                "knowledge_level": knowledge_level,
                "agent_type": agent_type,
            }
        },
        "annotations": annotations,
    }


def graph_config(tables: list[str] | None = None, version: str | None = None, fullmap: str = FULLMAP_DEFAULT) -> dict[str, Any]:
    """Build the Tablassert ``Graph`` config dict (verified against ``tablassert.models.Graph``).

    ``tables`` defaults to the three committed table configs (``tables/<basename>.yaml``);
    ``version`` defaults to the DAKP package version. ``fullmap`` is the fullmap redb path the
    ``build-kg`` resolve step reads from the Graph config (Tablassert >= 8.1 has no ``--fullmap``
    flag); it defaults to :data:`FULLMAP_DEFAULT` for deferred runs that never invoke ``build-kg``.
    The mandatory ``rig:`` section (Tablassert >= 11) is constant — see :func:`_rig_config`.
    ``uuid_fields`` (Tablassert >= 16.0) pins edge identity to :data:`UUID_FIELDS`.
    """
    if tables is None:
        tables = [f"tables/{_TABLE_SPECS[table][0]}.yaml" for table in _TABLE_ORDER]
    return {
        "name": GRAPH_NAME,
        "version": version if version is not None else __version__,
        "fullmap": fullmap,
        "rig": _rig_config(tables),
        "uuid_fields": list(UUID_FIELDS),
        "tables": list(tables),
    }


# --- minimal stdlib YAML emitter (no pyyaml runtime dep; round-trips) -------------

_PLAIN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")
_RESERVED = {"", "~", "true", "false", "null", "yes", "no", "on", "off", "True", "False", "Null", "None"}
_FOLD_WIDTH = 70
_FOLD_THRESHOLD = 60


def _is_safe_plain(text: str) -> bool:
    """True when ``text`` may be emitted as an unquoted YAML plain scalar.

    Excludes YAML reserved words and anything with characters outside a conservative
    identifier-ish set (so infores CURIEs, URLs, and the tab delimiter are always quoted).
    A leading letter/underscore is required, which also keeps bare numbers quoted.
    """
    return text not in _RESERVED and bool(_PLAIN_RE.match(text))


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if _is_safe_plain(text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t").replace("\n", "\\n")
    return f'"{escaped}"'


def _should_fold(text: str) -> bool:
    """Long, space-bearing strings are emitted as ``>-`` folded blocks (readable committed configs)."""
    return len(text) > _FOLD_THRESHOLD and " " in text


def _wrap(text: str, width: int = _FOLD_WIDTH) -> list[str]:
    """Greedy word-wrap; ``>-`` folding joins the lines back with single spaces (lossless)."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


def _indent(level: int) -> str:
    return "  " * level


def _emit_mapping(mapping: dict[str, Any], level: int, lines: list[str]) -> None:
    pad = _indent(level)
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            _emit_mapping(value, level + 1, lines)
        elif isinstance(value, list):
            lines.append(f"{pad}{key}:")
            _emit_list(value, level + 1, lines)
        elif isinstance(value, str) and _should_fold(value):
            lines.append(f"{pad}{key}: >-")
            lines.extend(f"{_indent(level + 1)}{folded}" for folded in _wrap(value))
        else:
            lines.append(f"{pad}{key}: {_yaml_scalar(value)}")


def _emit_list(items: list[Any], level: int, lines: list[str]) -> None:
    pad = _indent(level)
    for item in items:
        if isinstance(item, dict):
            # Render as a mapping one level deeper, then rewrite the first line's indent into
            # a "- " dash (same width: one indent level == two spaces == "- ").
            sub: list[str] = []
            _emit_mapping(item, level + 1, sub)
            prefix = _indent(level + 1)
            sub[0] = f"{pad}- {sub[0][len(prefix) :]}"
            lines.extend(sub)
        else:
            lines.append(f"{pad}- {_yaml_scalar(item)}")


def _dump_yaml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _emit_mapping(data, 0, lines)
    return "\n".join(lines) + "\n"


# --- serialized config strings ----------------------------------------------------


def table_yaml(table: str) -> str:
    """Serialized table config YAML: the ``Section`` body wrapped in a top-level ``template:``."""
    return _dump_yaml({"template": table_config(table)})


def graph_yaml(tables: list[str] | None = None, version: str | None = None, fullmap: str = FULLMAP_DEFAULT) -> str:
    """Serialized Graph config YAML."""
    return _dump_yaml(graph_config(tables=tables, version=version, fullmap=fullmap))


# --- runtime generation into the workdir ------------------------------------------


def generate(assertion_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Write ``tables/graph.yaml`` plus one table config per assertion table into the workdir.

    Configs land in ``<workdir>/tables/`` so their workdir-relative references
    (``tables/<name>.yaml``, ``data/tabular/<table>.tsv``) resolve when Tablassert runs from the
    workdir root. ``graph.yaml`` carries the fullmap redb path the ``build-kg`` resolve step reads
    (Tablassert reads it from the Graph config; the ``--fullmap`` flag is gone in >= 8.1):
    the real ``ctx.params["fullmap"]`` when present, else the :data:`FULLMAP_DEFAULT`
    placeholder (deferred runs never invoke ``build-kg``). Returns ``[graph_ref, *table_refs]`` in
    the canonical table order; assertion refs are linked as input provenance by table stem.
    """
    workdir = Workdir(ctx.workdir)
    store = ArtifactStore(workdir)
    tables_dir = workdir.root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    input_ids = {ref.uri.stem: ref.blake3 for ref in assertion_refs}
    operation = OperationBlock(name=_GENERATE_OPERATION)

    event = "generate_tablassert_configs"
    stats(logger, event, configs_dir=str(tables_dir), assertion_inputs=len(assertion_refs))
    table_refs: list[ArtifactRef] = []
    table_paths: list[str] = []
    for table in _TABLE_ORDER:
        basename = _TABLE_SPECS[table][0]
        config_path = tables_dir / f"{basename}.yaml"
        config_path.write_text(table_yaml(table), encoding="utf-8")
        table_paths.append(f"tables/{basename}.yaml")
        inputs = [input_ids[table]] if table in input_ids else []
        table_refs.append(store.register(config_path, media_type="application/yaml", inputs=inputs, operation=operation))

    graph_path = tables_dir / "graph.yaml"
    fullmap = str(ctx.params["fullmap"]) if ctx.params.get("fullmap") else FULLMAP_DEFAULT
    graph_path.write_text(graph_yaml(table_paths, fullmap=fullmap), encoding="utf-8")
    graph_ref = store.register(graph_path, media_type="application/yaml", inputs=[ref.blake3 for ref in table_refs], operation=operation)

    for ref in table_refs:
        stats(logger, event, table_config=str(ref.uri), blake3=ref.blake3)
    stats(logger, event, graph_config=str(graph_ref.uri), blake3=graph_ref.blake3, fullmap=fullmap)
    return [graph_ref, *table_refs]


# --- runner ------------------------------------------------------------------------


class TablassertError(RuntimeError):
    """Raised when the ``tablassert`` subprocess exits non-zero.

    The handoff report (with ``status: failed`` and full stdout/stderr) is written to disk
    *before* this exception is raised, so it remains available for post-mortem debugging.
    """


DEFAULT_TABLASERT_DIR = "../Tablassert"
TABLASERT_DIR_ENV = "DAKP_TABLASERT_DIR"
REPORT_NAME = "tablassert_handoff.json"
_REPORT_SCHEMA = "dakp.tablassert_handoff.v1"
_RUN_OPERATION = "run_tablassert"


def run_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Execute ``command`` capturing stdout/stderr, never raising on non-zero exit.

    This is the monkeypatch point for tests: patch the ``run_subprocess`` attribute on THIS
    module (``dakp_pipeline.tablassert``) and no real process is spawned.
    """
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


#: Terminal control sequences (colors / cursor moves) stripped from streamed subprocess lines.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean_stream_line(raw: str) -> str | None:
    """Reduce one raw subprocess line to a loggable message, or ``None`` when it is noise.

    Progress-bar redraws collapse to their LAST non-empty ``\r`` segment (the visible state
    the writer left on the line); ANSI escapes are stripped; empty leftovers drop out.
    Tablassert renders its rich progress on stderr, and under a pipe the bar emits its final
    state rather than every frame — this filter keeps the rare stray redraw out of the task
    log either way.
    """
    segments = [_ANSI_ESCAPE_RE.sub("", segment).strip() for segment in raw.split("\r")]
    return next((segment for segment in reversed(segments) if segment), None)


def stream_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``command`` streaming stdout/stderr LIVE into the task log, never raising on exit.

    Each surviving line (see :func:`_clean_stream_line`) is logged as
    ``tablassert[stdout]: <line>`` / ``tablassert[stderr]: <line>`` while the process runs —
    a Tablassert build takes 20+ minutes and its stage transitions should be visible as they
    happen, not only after exit. Consecutive duplicate lines collapse (redraw spam). The full
    unfiltered output is still accumulated and returned for the handoff report.

    Monkeypatch point for tests: patch ``stream_subprocess`` on this module; fakes keep the
    same ``(command, cwd=None) -> CompletedProcess[str]`` shape as :func:`run_subprocess`.
    """
    proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    collected: dict[str, list[str]] = {"stdout": [], "stderr": []}
    last_line: dict[str, str | None] = {"stdout": None, "stderr": None}

    def _pump(stream: Any, tag: str) -> None:
        for raw in stream:
            collected[tag].append(raw)
            cleaned = _clean_stream_line(raw)
            if cleaned is None or cleaned == last_line[tag]:
                continue
            last_line[tag] = cleaned
            logger.info("tablassert[{}]: {}", tag, cleaned)

    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()
    returncode = proc.wait()
    for thread in threads:
        thread.join()
    return subprocess.CompletedProcess(args=command, returncode=returncode, stdout="".join(collected["stdout"]), stderr="".join(collected["stderr"]))


def tablassert_available() -> bool:
    """True when the ``tablassert`` package (a core DAKP dependency) is importable here."""
    return importlib.util.find_spec("tablassert") is not None


def qc_runtime_available() -> bool:
    """True when the QC audit runtime (sentence-transformers, via ``tablassert[qc]``) is importable."""
    return importlib.util.find_spec("sentence_transformers") is not None


def _command_prefix(tablassert_dir: str | None) -> list[str]:
    """argv prefix that launches the ``tablassert`` CLI.

    Editable override (dev against a local checkout): ``uv run --with-editable <dir> tablassert``;
    installed package: the venv ``tablassert`` binary when it is on ``PATH``, otherwise
    ``uv run tablassert`` (uv materializes the console script for the rare importable-but-no-PATH
    case; the availability guard in ``TablassertRunner.run`` has already confirmed ``tablassert``
    is importable before this fallback is reachable).
    """
    if tablassert_dir:
        return ["uv", "run", "--with-editable", tablassert_dir, "tablassert"]
    binary = shutil.which("tablassert")
    if binary is not None:
        return [binary]
    return ["uv", "run", "tablassert"]


def _resolve_tablassert_dir(runner_dir: str | None, params_dir: str | None) -> str | None:
    """Editable-checkout override precedence: ctx param > ``DAKP_TABLASERT_DIR`` env > runner default.

    ``None`` (the default everywhere) means "use the installed PyPI package".
    """
    return params_dir or os.environ.get(TABLASERT_DIR_ENV) or runner_dir


def _base_report(mode: str, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef]) -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "stage": "tablassert_handoff",
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "assertion_inputs": [{"table": ref.uri.stem, "artifact_id": ref.blake3, "rows": ref.rows} for ref in assertion_refs],
        "config_inputs": [str(ref.uri) for ref in config_refs],
    }


def _write_report(report: dict[str, Any], assertion_refs: list[ArtifactRef], ctx: TaskContext) -> ArtifactRef:
    workdir = Workdir(ctx.workdir)
    path = workdir.reports / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    store = ArtifactStore(workdir)
    return store.register(
        path, media_type="application/json", inputs=[ref.blake3 for ref in assertion_refs], operation=OperationBlock(name=_RUN_OPERATION)
    )


def _find_graph(config_refs: list[ArtifactRef], ctx: TaskContext) -> Path:
    """Locate ``graph.yaml`` among the generated config refs (fall back to the conventional path)."""
    for ref in config_refs:
        if ref.uri.name == "graph.yaml":
            return ref.uri
    return Workdir(ctx.workdir).root / "tables" / "graph.yaml"


@dataclass(frozen=True)
class TablassertRunner:
    """Run the INSTALLED ``tablassert`` CLI (a core DAKP dependency) as a subprocess.

    Builds ``tablassert build-kg <graph.yaml> [--qc] [--release]`` (the graph config carries the
    fullmap path — Tablassert 8.1 removed the ``build-kg --fullmap`` flag), streams the
    subprocess output live into the task log (:func:`stream_subprocess`), and records the full
    stdout / stderr / exit code in the handoff report. A non-zero exit is captured as
    ``status: failed`` in the report (written to disk before raising) and then raises
    :class:`TablassertError` so the calling task (Airflow or stage harness) fails correctly.
    Raises ``RuntimeError`` when ``tablassert`` is unavailable and no editable-checkout
    override is configured (reinstall with ``uv sync``).
    """

    tablassert_dir: str | None = None

    def build_command(
        self, graph_yaml: Path, *, tablassert_dir: str | None = None, qc: bool = False, release: bool = False, threads: int | None = None
    ) -> list[str]:
        """The exact Tablassert invocation (pure; testable without spawning a process)."""
        command = [*_command_prefix(tablassert_dir), "build-kg", str(graph_yaml)]
        if qc:
            command.append("--qc")
        if release:
            command.append("--release")
        if threads is not None:
            command.extend(["--threads", str(threads)])
        return command

    def run(self, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        stats(logger, "run_tablassert", mode="real")
        graph_yaml = _find_graph(config_refs, ctx)
        fullmap_value = ctx.params.get("fullmap")
        if not fullmap_value:
            msg = (
                "a fullmap redb path is required for a real Tablassert handoff but none was provided: pass "
                "`--fullmap <path>` to `dakp up` (DAKP no longer downloads a fullmap; build one with "
                "`tablassert build-fullmap`). Tablassert reads only `tablassert.fullmap.v5` redb "
                "files (the format since Tablassert 8.2) — a fullmap built by 8.1 or older must be "
                "rebuilt"
            )
            raise RuntimeError(msg)
        fullmap = str(fullmap_value)
        tablassert_dir = _resolve_tablassert_dir(self.tablassert_dir, ctx.params.get("tablassert_dir"))
        if tablassert_dir is None and not tablassert_available():
            msg = (
                "tablassert is not available: it is a core DAKP dependency, so reinstall with `uv sync`, or point at a "
                f"local editable checkout via the tablassert_dir param / {TABLASERT_DIR_ENV} env var"
            )
            raise RuntimeError(msg)

        event = "run_tablassert"
        qc_requested = bool(ctx.params.get("qc"))
        qc = qc_requested and qc_runtime_available()
        if qc_requested and not qc:
            logger.warning("{}: --qc requested but the QC audit runtime (sentence-transformers) is not importable; running without --qc", event)
        release = bool(ctx.params.get("release"))
        # Worker count for the parallel fullmap reads behind entity resolution; absent => Tablassert auto.
        threads_value = ctx.params.get("tablassert_threads")
        threads = int(str(threads_value)) if threads_value is not None else None

        command = self.build_command(graph_yaml, tablassert_dir=tablassert_dir, qc=qc, release=release, threads=threads)
        cwd = Workdir(ctx.workdir).root

        with step(logger, event):
            stats(
                logger,
                event,
                graph_config=str(graph_yaml),
                fullmap=fullmap,
                qc=qc,
                release=release,
                threads=threads,
                tablassert_dir=tablassert_dir or "-",
            )
            stats(logger, event, command=" ".join(command))
            completed = stream_subprocess(command, cwd=cwd)
        status = "ok" if completed.returncode == 0 else "failed"
        stats(logger, event, status=status, exit_code=completed.returncode)
        if completed.returncode != 0:
            logger.error("{}: exited {} — stderr = {}", event, completed.returncode, (completed.stderr or "").strip())

        report = _base_report("real", assertion_refs, config_refs)
        report.update(
            {
                "status": status,
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "graph_config": str(graph_yaml),
                "fullmap": fullmap,
                "tablassert_dir": tablassert_dir,
                "qc": qc,
                "release": release,
                "threads": threads,
            }
        )
        refs = [_write_report(report, assertion_refs, ctx)]
        if completed.returncode != 0:
            raise TablassertError(
                f"Tablassert exited {completed.returncode}; see handoff report: {refs[0].uri}\n{(completed.stderr or '').strip()[:500]}"
            )
        return refs


@dataclass(frozen=True)
class DeferredTablassertRunner:
    """Write a deferred-handoff report; never touch Tablassert (no fullmap trigger + tests)."""

    def run(self, assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        event = "run_tablassert"
        stats(logger, event, mode="deferred")
        stats(
            logger,
            event,
            reason="no fullmap provided (run_tablassert disabled); canonical resolution + KGX compilation delegated to the installed tablassert CLI",
        )
        report = _base_report("deferred", assertion_refs, config_refs)
        report.update(
            {
                "status": "deferred",
                "reason": "no fullmap provided (run_tablassert disabled); canonical resolution + KGX compilation delegated to the installed tablassert CLI",
            }
        )
        return [_write_report(report, assertion_refs, ctx)]


def run(assertion_refs: list[ArtifactRef], config_refs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
    """Module entry point (stage harness / ``dags.dakp_build``): dispatch to a runner.

    Callers invoke this as a module attribute at call time (``tablassert.run(...)``) so
    ``monkeypatch.setattr("dakp_pipeline.tablassert.run", ...)`` replaces the callable they see.
    Real execution requires ``run_tablassert`` truthy in ``ctx.params`` (derived from a fullmap
    path); otherwise the deferred runner writes a deferred-handoff report. Returns a list with
    one ArtifactRef to the handoff report.
    """
    run_real = bool(ctx.params.get("run_tablassert"))
    runner: TablassertRunner | DeferredTablassertRunner = TablassertRunner() if run_real else DeferredTablassertRunner()
    return runner.run(assertion_refs, config_refs, ctx)


__all__ = [
    "AGENT_TYPE",
    "DEFAULT_TABLASERT_DIR",
    "FULLMAP_DEFAULT",
    "GESTALT_RECORD_URL_TEMPLATE",
    "GRAPH_DESCRIPTION",
    "GRAPH_NAME",
    "INFORES_DAKP",
    "REPORT_NAME",
    "TABLASERT_DIR_ENV",
    "DeferredTablassertRunner",
    "TablassertError",
    "TablassertRunner",
    "category_avoid_list",
    "column_letter",
    "excel_column",
    "generate",
    "graph_config",
    "graph_yaml",
    "qc_runtime_available",
    "run",
    "run_subprocess",
    "tablassert_available",
    "table_config",
    "table_yaml",
]
