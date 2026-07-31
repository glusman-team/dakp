"""Edge-case tests for the streaming DailyMed SPL XML extractor (100% branch coverage drive).

The happy-path suite only exercises the namespace-free *mock* shape. This drives the real
**HL7 v3** (``urn:hl7-org:v3``) parse path end-to-end plus the mock/defensive branches the
fixture never reaches:

* HL7 v3 document parse — ``setId@root`` (lowercased), approvals (NDA-OID id + application-type
  code, dedup, wrong-root skip), ingredients (activeMoiety/activeIngredientSubstance active,
  inactiveIngredientSubstance + ``ingredient@classCode=IACT`` inactive, duplicate-key and
  name-less skips), and LOINC-coded sections nested under ``component/section``.
* ``_collect_sections`` code loop — ``<code>`` descendants, a code without a ``code`` attr
  (skipped), LOINC-shape preference, ``<title>`` element vs title-fallback-to-name.
* mock document missing ``setId`` (warning + empty set rows), non-SPL artifact skip, empty
  table emission, and the small ``_looks_loinc`` / ``_looks_like_spl`` / ``_to_frame`` helpers.
* gzip-aware streaming of the HL7 v3 shape.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from dakp_pipeline.extract import spl_xml
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"

# A rich HL7 v3 SPL batch: one fully-populated document + one document missing its setId root.
_HL7V3_XML = """<?xml version="1.0" encoding="UTF-8"?>
<splBatch xmlns="urn:hl7-org:v3">
  <document>
    <setId/>
    <setId root="SET-ABC-123"/>
    <subjectOf>
      <approval>
        <id root="2.16.840.1.113883.3.150" extension="012345"/>
        <code code="WRONG" codeSystem="1.2.3.4"/>
        <code code="NDA" codeSystem="2.16.840.1.113883.3.26.1.1"/>
      </approval>
      <approval>
        <id root="2.16.840.1.113883.3.150" extension="012345"/>
      </approval>
      <approval>
        <id root="9.9.9.9" extension="SKIPME"/>
      </approval>
    </subjectOf>
    <activeMoiety>
      <activeMoiety>
        <name>Examplestatin</name>
        <code code="QFX8B1R4QF"/>
      </activeMoiety>
    </activeMoiety>
    <activeMoiety>
      <activeIngredientSubstance>
        <name>Ibuprofen</name>
        <code code="WK2XYI10QM"/>
      </activeIngredientSubstance>
    </activeMoiety>
    <activeMoiety>
      <activeMoiety>
        <name>Examplestatin</name>
        <code code="QFX8B1R4QF"/>
      </activeMoiety>
    </activeMoiety>
    <inactiveIngredientSubstance>
      <name>Lactose</name>
      <code code="J2B2A4N98G"/>
    </inactiveIngredientSubstance>
    <inactiveIngredientSubstance>
      <code code="NONAME000"/>
    </inactiveIngredientSubstance>
    <ingredient classCode="IACT">
      <ingredientSubstance>
        <name>Water</name>
        <code code="059QF0KO0R"/>
      </ingredientSubstance>
    </ingredient>
    <ingredient classCode="OTHER">
      <ingredientSubstance>
        <name>Ignored</name>
        <code code="ZZZ"/>
      </ingredientSubstance>
    </ingredient>
    <component><section>
      <code code="34067-9" codeSystem="2.16.840.1.113883.6.1"/>
      <title>INDICATIONS AND USAGE</title>
      <text>Examplestatin is indicated for   hypercholesterolemia.</text>
    </section></component>
    <component><section>
      <code codeSystem="2.16.840.1.113883.6.1"/>
      <code code="34070-3"/>
      <title>CONTRAINDICATIONS</title>
      <text>Contraindicated in active liver disease.</text>
    </section></component>
    <component><section>
      <code code="ABC"/>
      <code code="34066-1"/>
      <text>Prefer a LOINC code over an earlier non-LOINC one.</text>
    </section></component>
    <component><section>
      <code code="ABC-9"/>
      <text>Prefix-not-digit code.</text>
    </section></component>
    <component><section>
      <code code="123-"/>
      <text>Empty-suffix code.</text>
    </section></component>
  </document>
  <document>
    <component><section>
      <code code="34067-9"/>
      <text>Document without a setId root.</text>
    </section></component>
  </document>
</splBatch>"""

# A namespace-free mock document missing setId, with no approvals/ingredients (one section).
_MOCK_NO_SETID_XML = """<?xml version="1.0" encoding="UTF-8"?>
<splBatch>
  <document>
    <section loinc="34067-9" name="INDICATIONS AND USAGE">Some indication text here.</section>
  </document>
</splBatch>"""


def _ctx(tmp_path: Path) -> TaskContext:
    wd = tmp_path / "work"
    Workdir(wd).create()
    return TaskContext(workdir=wd, fixture_root=_FIXTURE_ROOT, params={})


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3=hash_file(path), media_type="application/octet-stream")


def _write_xml(tmp_path: Path, name: str, content: str) -> ArtifactRef:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return _ref(p)


def _by_name(refs: list[ArtifactRef]) -> dict[str, pl.DataFrame]:
    return {r.uri.name: pl.read_parquet(r.uri) for r in refs if r.uri.suffix == ".parquet"}


# --- HL7 v3 end-to-end ---------------------------------------------------------


def test_hl7v3_extract_emits_all_tables(tmp_path: Path) -> None:
    ref = _write_xml(tmp_path, "hl7.xml", _HL7V3_XML)
    refs = spl_xml.extract([ref], _ctx(tmp_path))
    tables = _by_name(refs)
    # 5 sections in doc1 + 1 in doc2 -> 6 section/document rows; 1 set; 1 approval; 4 ingredients.
    assert tables["spl_sections.parquet"].height == 6
    assert tables["spl_documents.parquet"].height == 6
    assert tables["spl_sets.parquet"].height == 1
    assert tables["spl_approvals.parquet"].height == 1
    assert tables["spl_ingredients.parquet"].height == 4


def test_hl7v3_set_id_lowercased_from_root(tmp_path: Path) -> None:
    ref = _write_xml(tmp_path, "hl7.xml", _HL7V3_XML)
    tables = _by_name(spl_xml.extract([ref], _ctx(tmp_path)))
    assert tables["spl_sets.parquet"]["spl_set_id"].to_list() == ["set-abc-123"]


def test_hl7v3_approvals_dedup_and_wrong_root_skip(tmp_path: Path) -> None:
    ref = _write_xml(tmp_path, "hl7.xml", _HL7V3_XML)
    tables = _by_name(spl_xml.extract([ref], _ctx(tmp_path)))
    approvals = tables["spl_approvals.parquet"]
    # The two 012345 approvals collapse to one; the wrong-root approval is skipped. The final
    # type reflects the last code seen for the id (the code-less second approval -> "").
    assert approvals["approval_id"].to_list() == ["012345"]
    assert approvals["approval_type"].to_list() == [""]


def test_hl7v3_ingredients_roles_dedup_and_skips(tmp_path: Path) -> None:
    ref = _write_xml(tmp_path, "hl7.xml", _HL7V3_XML)
    tables = _by_name(spl_xml.extract([ref], _ctx(tmp_path)))
    ing = tables["spl_ingredients.parquet"]
    rows = {(r["ingredient_name"], r["ingredient_unii"], r["role"]) for r in ing.iter_rows(named=True)}
    assert rows == {
        ("Examplestatin", "UNII:QFX8B1R4QF", "active"),  # activeMoiety/activeMoiety (deduped)
        ("Ibuprofen", "UNII:WK2XYI10QM", "active"),  # activeMoiety/activeIngredientSubstance
        ("Lactose", "UNII:J2B2A4N98G", "inactive"),  # inactiveIngredientSubstance
        ("Water", "UNII:059QF0KO0R", "inactive"),  # ingredient@classCode=IACT
    }
    # The name-less substance and the non-IACT ingredient are excluded.
    assert "Ignored" not in ing["ingredient_name"].to_list()
    assert all(n for n in ing["ingredient_name"].to_list())


def test_hl7v3_sections_loinc_title_and_fallbacks(tmp_path: Path) -> None:
    ref = _write_xml(tmp_path, "hl7.xml", _HL7V3_XML)
    tables = _by_name(spl_xml.extract([ref], _ctx(tmp_path)))
    sec = tables["spl_sections.parquet"]
    # doc1 (set-abc-123) has unique LOINCs; doc2 ("") reuses 34067-9, so scope to doc1.
    doc1 = sec.filter(pl.col("spl_set_id") == "set-abc-123")
    by_loinc = {r["loinc_code"]: r for r in doc1.iter_rows(named=True)}
    # LOINC from a <code> descendant; name mapped from the section-code registry.
    assert by_loinc["34067-9"]["section_name"] == "indications_and_usage"
    # A code without a `code` attr is skipped; the following 34070-3 code wins.
    assert by_loinc["34070-3"]["section_name"] == "contraindications"
    # A LOINC code is preferred over an earlier non-LOINC code in the same section.
    assert by_loinc["34066-1"]["section_name"] == "boxed_warning"
    # <title> element text becomes the section title.
    assert by_loinc["34067-9"]["section_title"] == "INDICATIONS AND USAGE"
    # Non-LOINC codes are preserved verbatim; title falls back to the name.
    assert by_loinc["ABC-9"]["section_title"] == "ABC-9"
    assert by_loinc["123-"]["loinc_code"] == "123-"
    # clean_text is whitespace-collapsed (the raw had a triple space).
    assert "  " not in by_loinc["34067-9"]["clean_text"]
    # doc2's section has no <title> -> title falls back to the mapped name.
    doc2 = sec.filter(pl.col("spl_set_id") == "")
    assert doc2.get_column("section_title").to_list() == ["indications_and_usage"]


def test_hl7v3_missing_setid_root_warns(tmp_path: Path) -> None:
    ref = _write_xml(tmp_path, "hl7.xml", _HL7V3_XML)
    refs = spl_xml.extract([ref], _ctx(tmp_path))
    sections_ref = next(r for r in refs if r.uri.name == "spl_sections.parquet")
    from dakp_pipeline.io.manifests import read_manifest

    assert sections_ref.manifest is not None
    manifest = read_manifest(sections_ref.manifest)
    assert manifest.table.warnings is not None
    assert manifest.table.warnings >= 1


def test_hl7v3_gzip_input_matches_plain(tmp_path: Path) -> None:
    plain = _write_xml(tmp_path, "hl7.xml", _HL7V3_XML)
    gz_path = tmp_path / "hl7.xml.gz"
    gz_path.write_bytes(gzip.compress(_HL7V3_XML.encode("utf-8")))
    gz = _ref(gz_path)

    plain_tables = _by_name(spl_xml.extract([plain], _ctx(tmp_path / "p")))
    gz_tables = _by_name(spl_xml.extract([gz], _ctx(tmp_path / "g")))
    assert plain_tables["spl_sections.parquet"]["loinc_code"].sort().to_list() == gz_tables["spl_sections.parquet"]["loinc_code"].sort().to_list()
    assert plain_tables["spl_ingredients.parquet"].height == gz_tables["spl_ingredients.parquet"].height


# --- mock shape edges ----------------------------------------------------------


def test_mock_missing_setid_warns_and_emits_empty_tables(tmp_path: Path) -> None:
    ref = _write_xml(tmp_path, "mock.xml", _MOCK_NO_SETID_XML)
    refs = spl_xml.extract([ref], _ctx(tmp_path))
    tables = _by_name(refs)
    # No setId -> no set row; no approvals/ingredients -> empty tables (empty-frame path).
    assert tables["spl_sets.parquet"].is_empty()
    assert tables["spl_approvals.parquet"].is_empty()
    assert tables["spl_ingredients.parquet"].is_empty()
    # The single section still produces a document + section row.
    assert tables["spl_documents.parquet"].height == 1
    assert tables["spl_sections.parquet"].height == 1
    # Missing-setId warning recorded on the section table manifest.
    from dakp_pipeline.io.manifests import read_manifest

    sections_ref = next(r for r in refs if r.uri.name == "spl_sections.parquet")
    assert sections_ref.manifest is not None
    manifest = read_manifest(sections_ref.manifest)
    assert manifest.table.warnings is not None
    assert manifest.table.warnings >= 1


def test_extract_skips_non_spl_artifacts(tmp_path: Path) -> None:
    # A .txt and a .zip are not SPL -> skipped (only the .xml is processed).
    txt = tmp_path / "notes.txt"
    txt.write_text("not spl", encoding="utf-8")
    xml = _write_xml(tmp_path, "mock.xml", _MOCK_NO_SETID_XML)
    refs = spl_xml.extract([_ref(txt), xml], _ctx(tmp_path))
    tables = _by_name(refs)
    assert tables["spl_sections.parquet"].height == 1  # only the XML contributed


# --- small helper units --------------------------------------------------------


def test_to_frame_empty_rows_yields_empty_schema() -> None:
    frame = spl_xml._to_frame([], ["a", "b"])
    assert frame.is_empty()
    assert frame.columns == ["a", "b"]


def test_to_frame_drops_extra_and_fills_missing_keys() -> None:
    frame = spl_xml._to_frame([{"a": "1", "extra": "x"}], ["a", "b"])
    assert frame.columns == ["a", "b"]
    assert frame.row(0) == ("1", "")  # extra dropped, missing -> ""


def test_looks_loinc_shapes() -> None:
    assert spl_xml._looks_loinc("34067-9") is True
    assert spl_xml._looks_loinc("ABC") is False  # no dash
    assert spl_xml._looks_loinc("-9") is False  # leading dash (dash <= 0)
    assert spl_xml._looks_loinc("ABC-9") is False  # prefix not digits
    assert spl_xml._looks_loinc("123-") is False  # empty suffix


def test_looks_like_spl_suffixes() -> None:
    assert spl_xml._looks_like_spl(Path("a.xml")) is True
    assert spl_xml._looks_like_spl(Path("a.xml.gz")) is True
    assert spl_xml._looks_like_spl(Path("a.XML")) is True
    assert spl_xml._looks_like_spl(Path("a.txt")) is False
    assert spl_xml._looks_like_spl(Path("a.zip")) is False
