"""Tests for the ICD-10-CM Table of Drugs and Chemicals extractor.

Pins the tier rules the subject screen rests on, using a hand-written drug-table XML that
mirrors the FY2027 member's structure (``mainTerm`` / ``title`` + ``nemod`` + ``cell col=2..7``).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest

from dakp_pipeline.extract import icd10cm
from dakp_pipeline.io.contracts import TaskContext
from dakp_pipeline.io.xcom import refs_from_xcom, refs_to_xcom
from dakp_pipeline.paths import Workdir

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ICD10CM.index>
  <version>2027</version>
  <letter a="A" z="Z">
    <head>Poisoning by, adverse effects of and underdosing of drugs, medicaments and biological substances</head>
    <mainTerm>
      <title>Aspirin<nemod> (salicylic acid)</nemod></title>
      <cell col="2">T39.011</cell>
      <cell col="3">T39.012</cell>
      <cell col="4">T39.013</cell>
      <cell col="5">T39.014</cell>
      <cell col="6">T39.015</cell>
      <cell col="7">T39.016</cell>
    </mainTerm>
    <mainTerm>
      <title>Chlorothalonil</title>
      <cell col="2">T60.4X1</cell>
      <cell col="3">T60.4X2</cell>
      <cell col="4">T60.4X3</cell>
      <cell col="5">T60.4X4</cell>
      <cell col="6">--</cell>
      <cell col="7">--</cell>
    </mainTerm>
    <mainTerm>
      <title>Heroin<nemod> (diacetylmorphine)</nemod></title>
      <cell col="2">T40.1X1</cell>
      <cell col="6">--</cell>
    </mainTerm>
    <mainTerm>
      <title>Arsenic, arsenicals (compounds)<nemod> (arsine)</nemod></title>
      <cell col="2">T56.0X1; T56.8X1</cell>
      <cell col="6">--</cell>
      <term level="1">
        <title>beverage</title>
        <cell col="2">T51.0X1</cell>
        <cell col="6">--</cell>
      </term>
    </mainTerm>
    <mainTerm>
      <title>Adverse reaction to nuts</title>
      <cell col="2">T78.01XA</cell>
      <cell col="6">--</cell>
    </mainTerm>
    <mainTerm>
      <title>Voltaren</title>
      <see>see Diclofenac</see>
    </mainTerm>
    <mainTerm>
      <title>Lead</title>
      <cell col="2">T56.911</cell>
      <cell col="6">--</cell>
    </mainTerm>
  </letter>
</ICD10CM.index>
"""


def _tiers(xml: str = _XML) -> dict[str, tuple[str, str]]:
    """``name -> (tier, code)`` from the extracted table."""
    return {row.name: (row.tier, row.icd10cm_code) for row in icd10cm.classify_drug_table(xml)}


def test_adverse_effect_code_makes_an_entry_therapeutic() -> None:
    assert _tiers()["Aspirin"] == (icd10cm.TIER_THERAPEUTIC, "T39.015")


def test_empty_adverse_effect_in_the_toxic_range_is_a_toxic_effect() -> None:
    assert _tiers()["Chlorothalonil"] == (icd10cm.TIER_TOXIC_EFFECT, "T60.4X1")


def test_empty_adverse_effect_in_the_drug_range_is_no_accepted_medical_use() -> None:
    """Heroin carries accidental codes in T40 with no Adverse-effect code at all."""
    assert _tiers()["Heroin"] == (icd10cm.TIER_NO_ACCEPTED_MEDICAL_USE, "T40.1X1")


def test_non_drug_code_families_are_screened_not_dropped() -> None:
    """T78 (adverse reactions) is neither a drug range nor a toxic-effect range."""
    assert _tiers()["Adverse reaction to nuts"] == (icd10cm.TIER_OTHER, "T78.01XA")


def test_title_qualifiers_and_comma_synonyms_are_stripped() -> None:
    """``nemod`` children are parenthetical qualifiers; a comma tail is a synonym, not a substance."""
    tiers = _tiers()
    assert "Arsenic" in tiers
    assert tiers["Arsenic"][0] == icd10cm.TIER_TOXIC_EFFECT
    assert tiers["Arsenic"][1] == "T56.0X1"  # the first accidental code decides


def test_indented_subterms_are_not_harvested_as_substances() -> None:
    """``term`` rows qualify their parent heading (``beverage`` under ``Alcohol``); they are not
    standalone substance names, and harvesting them would put fragments in a deny set."""
    assert "beverage" not in _tiers()


def test_brand_cross_references_without_codes_are_not_screened() -> None:
    """A ``see``-only entry classifies as ``other``: it is harvested, but the screen ignores it.

    The title survives as an audit row rather than being dropped at parse time (the parser states a
    tier for every mainTerm it sees), and only the two non-therapeutic drug/toxic tiers feed the
    denylist, so a codeless brand pointer can never deny a subject.
    """
    assert _tiers()["Voltaren"] == (icd10cm.TIER_OTHER, "")
    assert "Voltaren" not in {row.name for row in icd10cm.classify_drug_table(_XML) if row.tier in icd10cm.NON_THERAPEUTIC_TIERS}


def test_short_entries_are_classified_for_the_audit() -> None:
    """``Lead`` classifies here; the 4-character guard lives in the denylist generator, not the parse."""
    assert _tiers()["Lead"] == (icd10cm.TIER_TOXIC_EFFECT, "T56.911")


def test_only_non_therapeutic_tiers_are_the_screen_input() -> None:
    screen = {row for row in icd10cm.classify_drug_table(_XML) if row.tier in icd10cm.NON_THERAPEUTIC_TIERS}
    assert {row.name for row in screen} == {"Chlorothalonil", "Heroin", "Arsenic", "Lead"}


def test_member_is_resolved_by_prefix_and_rejects_ambiguity(tmp_path: Path) -> None:
    def build(names: list[str]) -> Path:
        path = tmp_path / f"release-{len(names)}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, _XML)
            archive.writestr("release/readme.txt", "not the table")
        return path

    single = build(["release/icd10cm-drug-2027.xml"])
    assert [row.name for row in icd10cm.parse_release_zip(single)][:1] == ["Aspirin"]
    with pytest.raises(ValueError, match="ambiguous"):
        icd10cm.parse_release_zip(build(["r/icd10cm-drug-2027.xml", "r/icd10cm-drug-2028.xml"]))
    with pytest.raises(ValueError, match="no icd10cm-drug"):
        icd10cm.parse_release_zip(build(["r/icd10cm-code-descriptions-2027.xml"]))


def test_extractor_writes_the_interim_table(monkeypatch, tmp_path: Path) -> None:
    from dakp_pipeline.io.artifact_store import ArtifactStore

    zip_path = tmp_path / "icd10cm-table-and-index-2027.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("icd10cm-table-and-index-2027/icd10cm-drug-2027.xml", _XML)
    Workdir(tmp_path / "work").create()
    store = ArtifactStore(Workdir(tmp_path / "work"))
    raw_ref, _ = store.ingest(zip_path, media_type="application/zip", alias="icd10cm/release.zip")
    ctx = TaskContext(workdir=tmp_path / "work", fixture_root=None, params={})

    refs = icd10cm.extract([raw_ref], ctx)
    assert len(refs) == 1
    out = refs[0].uri
    assert out.name == icd10cm.ICD10CM_SUBJECT_TIERS_FILENAME
    assert out.parent.name == "icd10cm"
    frame = pl.read_csv(out, separator="\t")
    assert frame.columns == list(icd10cm._COLUMNS)
    assert frame.height == 7

    assert set(frame.get_column("tier")) == {
        icd10cm.TIER_THERAPEUTIC,
        icd10cm.TIER_TOXIC_EFFECT,
        icd10cm.TIER_NO_ACCEPTED_MEDICAL_USE,
        icd10cm.TIER_OTHER,
    }
    # The artifact manifest records the ZIP it was derived from.
    manifest = store.read_manifest(refs[0].blake3)
    assert manifest is not None
    assert manifest.inputs == [raw_ref.blake3]
    assert refs_from_xcom(refs_to_xcom(refs))[0].blake3 == refs[0].blake3


def test_extractor_requires_a_zip_input() -> None:
    from dakp_pipeline.io.contracts import ArtifactRef

    with pytest.raises(ValueError, match="no ICD-10-CM"):
        icd10cm.extract(
            [ArtifactRef(uri=Path("/tmp/not-a-zip.tsv"), blake3="b3:x", media_type="text/plain")],
            TaskContext(workdir=Path("."), fixture_root=None, params={}),
        )
