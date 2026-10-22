"""ICD-10-CM Table of Drugs and Chemicals extraction.

Turns the drug table inside the acquired release ZIP (``icd10cm-drug-<FY>.xml``) into one interim
table, ``data/interim/icd10cm/icd10cm_subject_tiers.tsv``, that classifies every indexed substance
by what the Table itself asserts about its therapeutic use:

===========================  ======================================================================
tier                         rule (read off the entry's own code cells)
===========================  ======================================================================
``therapeutic``              the **Adverse effect** cell carries a code
``toxic_effect``             no Adverse effect code AND an accidental code in ``T51``-``T65``
``no_accepted_medical_use``  no Adverse effect code AND an accidental code in ``T36``-``T50``
``other``                    no Adverse effect code and no drug-class accidental code (V/X-only,
                             non-drug toxic effects such as ``T78`` adverse reactions, gases of
                             undetermined class) -- screened, never dropped
===========================  ======================================================================

The rule is the FY Official Guidelines' own semantics (section I.C.19.e): an Adverse effect code
"is used when the drug was correctly prescribed or properly administered", so an entry with NO
Adverse effect code is the classification system asserting the substance has no proper-administration
use. ``T51``-``T65`` is the "toxic effects" range, which the guidelines reserve for substances "not
being used as drugs". Those two tiers are what the observed-use screen drops; the shaper, not this
module, decides the drop, so this table stays a faithful rendering of the source.

Structure notes (verified against the FY2027 release, ``5184`` ``mainTerm`` entries):

* Only top-level ``mainTerm`` entries are read. The nested ``term`` rows are indented *qualifiers*
  of their parent heading (``beverage``, ``liquid``, ``acid``, ``ester``, ``vapor``), which are not
  standalone substance names, and they inherit the parent's codes; harvesting them would add keys
  like ``liquid`` to a deny set.
* A ``title`` element's leading text node is the substance name; the parenthetical ``nemod``
  children are qualifiers (``(vapor)``, ``(salt)``, ``(compound)``), so ``findtext("title")``
  already excludes them. Trailing comma synonyms ("Arsenic, arsenicals") keep the first name only:
  the comma tail is a class label, not a distinct substance.
* ``--`` means "no code in this column"; the cell text is split on ``;`` for multi-coded entries.

Pure-Python ``xml.etree`` over one 2 MB member; no polars parse of the XML, no new dependency.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.io.manifests import OperationBlock, TableBlock
from dakp_pipeline.logging_setup import bind, stats
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources import icd10cm as icd10cm_source

#: Name of the interim table this extractor writes (the screen's only input).
ICD10CM_SUBJECT_TIERS_FILENAME = "icd10cm_subject_tiers.tsv"
#: Tier column values. The two non-therapeutic drug tiers are what the screen consumes.
TIER_THERAPEUTIC = "therapeutic"
TIER_TOXIC_EFFECT = "toxic_effect"
TIER_NO_ACCEPTED_MEDICAL_USE = "no_accepted_medical_use"
TIER_OTHER = "other"
#: The tiers the subject screen treats as non-therapeutic. Exported so the denylist generator
#: (tests/eval/icd10cm_denylist.py) never re-states the literals and a tier rename cannot silently
#: narrow or widen the shipped list.
NON_THERAPEUTIC_TIERS: frozenset[str] = frozenset({TIER_TOXIC_EFFECT, TIER_NO_ACCEPTED_MEDICAL_USE})

_COLUMNS = ("name", "icd10cm_code", "tier")
_EVENT = "extract_icd10cm"
_NO_CODE = "--"
#: Accidental-poisoning ranges that carry the Table's drug-vs-noxious-substance judgment.
_TOXIC_EFFECT_PREFIXES = tuple(f"T{code}" for code in range(51, 66))
_DRUG_RANGE_PREFIXES = tuple(f"T{code}" for code in (*range(36, 51), 50))


@dataclass(frozen=True)
class SubjectTier:
    """One classified Table entry: the substance name, its tier, and the code that decided it."""

    name: str
    icd10cm_code: str
    tier: str


def _codes(cells: dict[str, str], column: str) -> list[str]:
    """The coded values of one column, or ``[]`` for the ``--`` "no code" marker."""
    raw = cells.get(column, _NO_CODE).strip()
    if not raw or raw == _NO_CODE:
        return []
    return [member.strip() for member in raw.split(";") if member.strip() and member.strip() != _NO_CODE]


def _tier_for(adverse_effect: list[str], accidental: list[str]) -> tuple[str, str]:
    """``(tier, deciding code)`` for one entry's Adverse effect / Accidental poisoning cells."""
    if adverse_effect:
        return TIER_THERAPEUTIC, adverse_effect[0]
    if not accidental:
        return TIER_OTHER, ""
    code = accidental[0]
    if code.startswith(_TOXIC_EFFECT_PREFIXES):
        return TIER_TOXIC_EFFECT, code
    if code.startswith(_DRUG_RANGE_PREFIXES):
        return TIER_NO_ACCEPTED_MEDICAL_USE, code
    return TIER_OTHER, code


def _subject_name(title: str) -> str:
    """The substance name of a ``mainTerm`` title, without qualifiers or comma-tail synonyms."""
    return title.split(",")[0].strip()


def classify_drug_table(xml: str) -> list[SubjectTier]:
    """Classify every top-level drug-table entry into a :class:`SubjectTier`.

    Deterministic: entries are returned in document order, and each entry takes its own cells (the
    first code of each column decides, since the Table lists the accidental code first).
    """
    root = ET.fromstring(xml)
    rows: list[SubjectTier] = []
    for main_term in root.iter("mainTerm"):
        title = (main_term.findtext("title") or "").strip()
        name = _subject_name(title)
        if not name:
            continue
        cells = {str(cell.get("col") or ""): (cell.text or "") for cell in main_term.findall("cell")}
        tier, code = _tier_for(_codes(cells, "6"), _codes(cells, "2"))
        rows.append(SubjectTier(name=name, icd10cm_code=code, tier=tier))
    return rows


def _drug_xml_member(zf: zipfile.ZipFile) -> str:
    """The release ZIP member holding the drug table, matched by prefix so a fiscal-year rename works.

    Raises on zero or multiple matches: a silently empty screen is the failure this pipeline must
    not be able to produce, and a rename is exactly the way a version-pinned fetcher goes stale.
    """
    members = [name for name in zf.namelist() if name.rsplit("/", 1)[-1].startswith(icd10cm_source.ICD10CM_DRUG_XML_PREFIX) and name.endswith(".xml")]
    if not members:
        msg = f"no {icd10cm_source.ICD10CM_DRUG_XML_PREFIX}*.xml member in the acquired ICD-10-CM release ZIP"
        raise ValueError(msg)
    if len(members) > 1:
        msg = f"ambiguous ICD-10-CM drug table: {sorted(members)}"
        raise ValueError(msg)
    return members[0]


def parse_release_zip(path: Path) -> list[SubjectTier]:
    """Read the drug table out of one release ZIP and classify it."""
    with zipfile.ZipFile(path) as zf:
        xml = zf.read(_drug_xml_member(zf)).decode("utf-8-sig", errors="replace")
    return classify_drug_table(xml)


class ICD10CMExtractor:
    """Extract the substance-tier table from the acquired ICD-10-CM release ZIP."""

    def extract(self, inputs: list[ArtifactRef], ctx: TaskContext) -> list[ArtifactRef]:
        wd = Workdir(ctx.workdir)
        store = ArtifactStore(wd)
        log = bind(task_id=_EVENT)

        zip_ref = next((ref for ref in inputs if ref.uri.suffix.lower() == ".zip"), None)
        if zip_ref is None:
            msg = "no ICD-10-CM Table of Drugs and Chemicals release ZIP among the inputs"
            raise ValueError(msg)

        rows = parse_release_zip(zip_ref.uri)
        tiers = {row.tier for row in rows}
        if not tiers:
            msg = f"the ICD-10-CM drug table classified nothing ({zip_ref.uri})"
            raise ValueError(msg)

        frame = pl.DataFrame(
            {"name": [row.name for row in rows], "icd10cm_code": [row.icd10cm_code for row in rows], "tier": [row.tier for row in rows]},
            schema={"name": pl.Utf8, "icd10cm_code": pl.Utf8, "tier": pl.Utf8},
        )
        out = wd.interim / "icd10cm" / ICD10CM_SUBJECT_TIERS_FILENAME
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(out, separator="\t")

        fingerprint = schemas.schema_fingerprint(list(_COLUMNS))
        ref = store.register(
            out,
            media_type="text/tab-separated-values",
            rows=frame.height,
            schema_fingerprint=fingerprint,
            inputs=[zip_ref.blake3],
            operation=OperationBlock(name=_EVENT),
            table=TableBlock(rows=frame.height, schema_fingerprint=fingerprint),
        )
        stats(
            log,
            _EVENT,
            entries=frame.height,
            therapeutic=sum(1 for row in rows if row.tier == TIER_THERAPEUTIC),
            screened=sum(1 for row in rows if row.tier in NON_THERAPEUTIC_TIERS),
            other=sum(1 for row in rows if row.tier == TIER_OTHER),
            blake3=ref.blake3,
        )
        return [ref]


extract = ICD10CMExtractor().extract

__all__ = [
    "ICD10CM_SUBJECT_TIERS_FILENAME",
    "NON_THERAPEUTIC_TIERS",
    "TIER_NO_ACCEPTED_MEDICAL_USE",
    "TIER_OTHER",
    "TIER_THERAPEUTIC",
    "TIER_TOXIC_EFFECT",
    "ICD10CMExtractor",
    "SubjectTier",
    "classify_drug_table",
    "extract",
    "parse_release_zip",
]
