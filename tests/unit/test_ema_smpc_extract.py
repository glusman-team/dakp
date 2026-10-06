"""Tests for the EMA product-information (SmPC) section extractor.

The extractor decides WHICH text becomes EU contraindication/indication evidence, so a wrong cut is a
wrong edge. These tests pin: the real-PDF happy path (a trimmed live EMA document), the Annex I
boundary (the package leaflet must never be mined), the tolerant QRD heading match (case, dotted
numbers, wrapped titles, cross-references that are not headings), the section end rule (a missing
section cannot leak into the next), newest-wins dedupe for a product with two documents, and every
structured warning path (unreadable PDF, no text layer, unknown document, missing section).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from dakp_pipeline.extract import ema_smpc
from dakp_pipeline.io import schemas
from dakp_pipeline.io.artifact_store import ArtifactStore
from dakp_pipeline.io.contracts import ArtifactRef, TaskContext
from dakp_pipeline.paths import Workdir
from dakp_pipeline.sources.ema_documents import EparDocument, load_documents, product_information_documents

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_MANIFEST = _FIXTURE_ROOT / "ema" / "epar_documents_en.json"
_CEPLENE_PDF = _FIXTURE_ROOT / "ema" / "smpc" / "ceplene-epar-product-information_en.pdf"

#: Real QRD wording, upper-case heading form (older documents), with a leaflet that repeats
#: "Do not use" contraindication-like text under Annex III. Mining the leaflet would duplicate
#: (and paraphrase) the regulatory section, so the cut must stop at ANNEX II.
_UPPERCASE_DOC = """ANNEX I
SUMMARY OF PRODUCT CHARACTERISTICS
1. NAME OF THE MEDICINAL PRODUCT
Examplomab 100 mg powder
4. CLINICAL PARTICULARS
4.1 THERAPEUTIC INDICATIONS
Examplomab is indicated for the treatment of adult patients with moderate to severe
plaque psoriasis who are candidates for systemic therapy.
4.2 POSOLOGY AND METHOD OF ADMINISTRATION
The recommended dose is 100 mg. See section 4.3 for patients who must not receive it.
4.3. CONTRAINDICATIONS
Hypersensitivity to the active substance or to any of the excipients listed in section 6.1.
Active tuberculosis.
4.4 Special warnings and special precautions for use, including a title long enough
that the text layer wraps it
Serious infections have been reported.
4.5 Interaction with other medicinal products
None known.
ANNEX II
MANUFACTURER RESPONSIBLE FOR BATCH RELEASE
ANNEX III
LABELLING AND PACKAGE LEAFLET
2. What you need to know before you use Examplomab
Do not use Examplomab if you have active hepatitis B.
"""


def _doc(stem: str, *, number: str = "EMEA/H/C/009999", name: str = "Examplomab", updated: str = "2025-01-01T00:00:00Z") -> EparDocument:
    return EparDocument(
        id=stem,
        type="product-information",
        medicine_name=name,
        ema_product_number=number,
        document_url=f"https://www.ema.europa.eu/en/documents/product-information/{stem}.pdf",
        last_updated_date=updated,
    )


def _ref(path: Path) -> ArtifactRef:
    return ArtifactRef(uri=path, blake3="b3:" + "0" * 64, media_type="application/pdf")


def _manifest_ref() -> ArtifactRef:
    return ArtifactRef(uri=_MANIFEST, blake3="b3:" + "1" * 64, media_type="application/json")


def _ceplene_documents() -> dict[str, EparDocument]:
    return {doc.stem: doc for doc in product_information_documents(load_documents(_MANIFEST))}


@pytest.fixture(autouse=True)
def _cap_parse_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the default parse-pool width so this file costs the same on any machine.

    ``_workers`` falls back to ``min(os.cpu_count(), 32)``, so on the 80-core build box every
    ``extract()`` call spawned 32 interpreters while CI's 4-vCPU runner spawns 4: the same test
    file cost 10x more there than in CI, which also made any duration-based CI shard split
    machine-specific. Two workers keep the spawn shim, input-order reassembly, and cross-process
    cache behaviour under test at a fixed small cost; tests that need a specific width pass
    ``workers=`` explicitly.
    """
    monkeypatch.setattr(os, "cpu_count", lambda: 2)


@pytest.fixture
def _untraced_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a plain ``subprocess`` child out of coverage's interpreter-start tracer.

    ``tests/conftest.py`` exports ``COVERAGE_PROCESS_START`` so spawned NER workers are measured.
    For the Airflow-shim test below that variable is pure cost: the child re-imports the whole
    ``dakp_pipeline`` chain under a line tracer and measured 47 s (1.5 s without it), while the
    test asserts only spawn SEMANTICS (``__spec__ is None``, the CLI is not re-executed) and every
    line it runs is also executed in-process by the serial path.

    This does NOT stop tracing in ``ProcessPoolExecutor`` children: ``[tool.coverage.run]`` sets
    ``concurrency = ["multiprocessing", ...]``, whose ``ProcessWithCoverage`` bootstrap starts a
    tracer unconditionally (that is what makes the child-only NER worker lines measurable). So
    ``test_parallel_parse_is_identical_to_serial`` still pays ~23 s for its three pool children
    under coverage; the shard-balancing data in ``.github/ci/unit-test-durations.json`` accounts
    for it rather than hiding it.
    """
    monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)


# --- the real PDF ------------------------------------------------------------------


def test_real_smpc_pdf_yields_all_three_mined_sections() -> None:
    """A trimmed LIVE EMA document: the cut must recover 4.1, 4.3 and 4.4 verbatim."""
    rows, warnings = ema_smpc.parse_documents([_ref(_CEPLENE_PDF)], _ceplene_documents())

    by_kind = {row["section_kind"]: row for row in rows}
    assert set(by_kind) == {"indications", "contraindications", "warnings"}
    assert "acute myeloid leukaemia" in by_kind["indications"]["section_text"]
    contra = by_kind["contraindications"]["section_text"]
    assert contra.startswith("Hypersensitivity to the active substance")
    assert "NYHA Class III/IV" in contra
    # Ends at 4.4: the next section's heading text never rides into 4.3.
    assert "Special warnings" not in contra
    assert all(row["ema_product_number"] == "EMEA/H/C/000796" for row in rows)
    assert all(row["medicine_name"] == "Ceplene" for row in rows)
    assert all(row["source_record_id"] == f"EMEA/H/C/000796#{row['section_kind']}" for row in rows)
    assert not [w for w in warnings if w["code"] != "missing_section"]


def test_real_smpc_text_is_whitespace_collapsed() -> None:
    """Layout newlines, NBSP, Symbol-font bullets (U+F0B7) and footer page numbers are layout, not
    text: left in they split NER spans and ride into the assertion TSV evidence cells."""
    rows, _ = ema_smpc.parse_documents([_ref(_CEPLENE_PDF)], _ceplene_documents())
    for row in rows:
        assert "\n" not in row["section_text"]
        assert "\xa0" not in row["section_text"]
        assert "\uf0b7" not in row["section_text"]
        assert "  " not in row["section_text"]
    contra = next(row["section_text"] for row in rows if row["section_kind"] == "contraindications")
    assert contra.endswith("During breast feeding.")  # the trailing page-number footer is gone


# --- cut_sections ------------------------------------------------------------------


def test_uppercase_and_dotted_headings_are_recognized() -> None:
    sections, warnings = ema_smpc.cut_sections(_UPPERCASE_DOC)
    assert set(sections) == {"indications", "contraindications", "warnings"}
    assert sections["contraindications"][1] == (
        "Hypersensitivity to the active substance or to any of the excipients listed in section 6.1. Active tuberculosis."
    )
    assert warnings == []


def test_the_package_leaflet_is_never_mined() -> None:
    """Annex III repeats 'Do not use' wording; it must not become contraindication evidence."""
    sections, _ = ema_smpc.cut_sections(_UPPERCASE_DOC)
    assert all("hepatitis B" not in body for _, body in sections.values())


def test_a_cross_reference_is_not_a_heading() -> None:
    """'See section 4.3' inside 4.2 must not open a contraindication section early."""
    sections, _ = ema_smpc.cut_sections(_UPPERCASE_DOC)
    assert "recommended dose" not in sections["contraindications"][1]


def test_a_wrapped_long_title_does_not_swallow_the_body() -> None:
    sections, _ = ema_smpc.cut_sections(_UPPERCASE_DOC)
    title, body = sections["warnings"]
    assert title.lower().startswith("special warnings")
    assert "Serious infections have been reported." in body


def test_a_missing_section_cannot_leak_into_the_next() -> None:
    """With no 4.2 heading, 4.1 must still stop at 4.3 (the next numbered heading of ANY kind)."""
    text = "ANNEX I\n4.1 Therapeutic indications\nTreats gout.\n4.3 Contraindications\nRenal failure.\n4.4 Special warnings\nNone.\nANNEX II\n"
    sections, warnings = ema_smpc.cut_sections(text)
    assert sections["indications"][1] == "Treats gout."
    assert sections["contraindications"][1] == "Renal failure."
    assert warnings == []


def test_missing_sections_are_reported_not_invented() -> None:
    text = "ANNEX I\n4.1 Therapeutic indications\nTreats gout.\n4.2 Posology\nOnce daily.\nANNEX II\n"
    sections, warnings = ema_smpc.cut_sections(text)
    assert set(sections) == {"indications"}
    assert ("missing_section", "section 4.3 (contraindications) was not found in Annex I") in warnings
    assert ("missing_section", "section 4.4 (warnings) was not found in Annex I") in warnings


def test_a_heading_with_the_wrong_title_is_not_a_mined_section() -> None:
    """A '4.3' line whose title is not 'Contraindications' (a table row, a renumbered template)."""
    text = "ANNEX I\n4.1 Therapeutic indications\nTreats gout.\n4.3 Table of doses\n10 mg\nANNEX II\n"
    sections, _ = ema_smpc.cut_sections(text)
    assert "contraindications" not in sections
    assert sections["indications"][1] == "Treats gout."


def test_an_empty_section_is_reported() -> None:
    text = "ANNEX I\n4.1 Therapeutic indications\nTreats gout.\n4.3 Contraindications\n4.4 Special warnings\nNone.\nANNEX II\n"
    sections, warnings = ema_smpc.cut_sections(text)
    assert "contraindications" not in sections
    assert ("empty_section", "section 4.3 (contraindications) has no text") in warnings


def test_a_duplicate_section_keeps_the_first_and_says_so() -> None:
    text = "ANNEX I\n4.3 Contraindications\nFirst.\n4.3 Contraindications\nSecond.\nANNEX II\n"
    sections, warnings = ema_smpc.cut_sections(text)
    assert sections["contraindications"][1] == "First."
    assert any(code == "duplicate_section" for code, _ in warnings)


def test_no_annex_marker_falls_back_but_still_stops_at_the_leaflet() -> None:
    text = "4.3 Contraindications\nPregnancy.\nANNEX III\n2. What you need to know\nDo not use if pregnant.\n"
    sections, warnings = ema_smpc.cut_sections(text)
    assert sections["contraindications"][1] == "Pregnancy."
    assert warnings[0][0] == "no_annex_i"


def test_no_annex_marker_and_no_later_annex_uses_the_whole_text() -> None:
    sections, warnings = ema_smpc.cut_sections("4.3 Contraindications\nPregnancy.\n")
    assert sections["contraindications"][1] == "Pregnancy."
    assert warnings[0][0] == "no_annex_i"


def test_no_headings_at_all_is_a_warning() -> None:
    sections, warnings = ema_smpc.cut_sections("ANNEX I\nfree prose with no numbering\nANNEX II\n")
    assert sections == {}
    assert ("no_headings", "no numbered QRD headings found in Annex I") in warnings


# --- parse_documents: attribution, dedupe, failures --------------------------------


def _pdf_with_text(tmp_path: Path, stem: str, text: str) -> Path:
    """Write a minimal one-page PDF whose text layer is ``text`` (pypdf, no system tools)."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")}
    )
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    lines = text.splitlines()
    ops = ["BT", "/F1 10 Tf", "12 TL", "40 760 Td"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({escaped}) Tj T*")
    ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    path = tmp_path / f"{stem}.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_two_documents_for_one_product_keep_the_newest(tmp_path: Path) -> None:
    """Observed live: the pre-endorsement 2021 AstraZeneca document beside the 2024 Vaxzevria one."""
    old = _pdf_with_text(tmp_path, "old-doc_en", "ANNEX I\n4.3 Contraindications\nOld wording.\nANNEX II")
    new = _pdf_with_text(tmp_path, "new-doc_en", "ANNEX I\n4.3 Contraindications\nNew wording.\nANNEX II")
    documents = {"old-doc_en": _doc("old-doc_en", updated="2021-01-29T16:30:00Z"), "new-doc_en": _doc("new-doc_en", updated="2024-05-07T12:41:00Z")}

    rows, warnings = ema_smpc.parse_documents([_ref(new), _ref(old)], documents)

    contra = [row for row in rows if row["section_kind"] == "contraindications"]
    assert len(contra) == 1
    assert contra[0]["section_text"] == "New wording."
    superseded = [w for w in warnings if w["code"] == "superseded_document"]
    assert len(superseded) == 1
    assert "old-doc_en.pdf" in superseded[0]["message"]


def test_dedupe_is_input_order_independent(tmp_path: Path) -> None:
    old = _pdf_with_text(tmp_path, "old-doc_en", "ANNEX I\n4.3 Contraindications\nOld wording.\nANNEX II")
    new = _pdf_with_text(tmp_path, "new-doc_en", "ANNEX I\n4.3 Contraindications\nNew wording.\nANNEX II")
    documents = {"old-doc_en": _doc("old-doc_en", updated="2021-01-29T16:30:00Z"), "new-doc_en": _doc("new-doc_en", updated="2024-05-07T12:41:00Z")}
    forward = ema_smpc.parse_documents([_ref(old), _ref(new)], documents)
    backward = ema_smpc.parse_documents([_ref(new), _ref(old)], documents)
    assert forward == backward


def test_a_pdf_outside_the_manifest_is_a_warning_not_a_row(tmp_path: Path) -> None:
    stray = _pdf_with_text(tmp_path, "stray_en", "ANNEX I\n4.3 Contraindications\nX.\nANNEX II")
    rows, warnings = ema_smpc.parse_documents([_ref(stray)], {})
    assert rows == []
    assert warnings == [
        {"ema_product_number": "", "code": "unknown_document", "message": "PDF stray_en.pdf is not in the documents report selection", "count": "1"}
    ]


def test_an_unreadable_pdf_is_a_warning(tmp_path: Path) -> None:
    broken = tmp_path / "broken_en.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a real pdf body")
    rows, warnings = ema_smpc.parse_documents([_ref(broken)], {"broken_en": _doc("broken_en")})
    assert rows == []
    assert [w["code"] for w in warnings] == ["pdf_unreadable"]


def test_a_pdf_without_a_text_layer_is_a_warning(tmp_path: Path) -> None:
    """A scanned (image-only) PDF yields no text; that must be reported, not silently empty."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    path = tmp_path / "scanned_en.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    rows, warnings = ema_smpc.parse_documents([_ref(path)], {"scanned_en": _doc("scanned_en")})
    assert rows == []
    assert [w["code"] for w in warnings] == ["no_text"]


def _multi_page_pdf(tmp_path: Path, stem: str, pages: list[str]) -> Path:
    """One PDF page per entry of ``pages`` (each a newline-separated text layer)."""
    writer = PdfWriter()
    for text in pages:
        single = PdfReader(str(_pdf_with_text(tmp_path, f"{stem}-page", text)))
        writer.add_page(single.pages[0])
    path = tmp_path / f"{stem}.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_annex_i_read_stops_at_the_page_that_ends_annex_i(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Labelling/leaflet pages after the Annex II marker are never decoded (the 89-minute fix)."""
    path = _multi_page_pdf(
        tmp_path,
        "stop_en",
        [
            "ANNEX I\n4.1 Therapeutic indications\nPsoriasis.",
            "4.3 Contraindications\nActive tuberculosis.\nANNEX II\nMANUFACTURER",
            "ANNEX III\nDo not use if you have hepatitis B.",
            "leaflet page that must never be decoded",
        ],
    )
    decoded: list[str] = []
    original = PageObject.extract_text

    def counting_extract_text(self: PageObject, *args: object, **kwargs: object) -> str:
        text = original(self, *args, **kwargs)  # type: ignore[arg-type]
        decoded.append(text)
        return text

    monkeypatch.setattr(PageObject, "extract_text", counting_extract_text)
    text = ema_smpc.read_annex_i_text(path)
    monkeypatch.undo()

    assert len(decoded) == 2
    assert "hepatitis B" not in text
    # Byte-identical cut to the full-document read: the trailing pages never reach a section.
    assert ema_smpc.cut_sections(text) == ema_smpc.cut_sections(ema_smpc.read_pdf_text(path))


def test_annex_i_read_ignores_an_annex_ii_marker_before_annex_i(tmp_path: Path) -> None:
    """A table of contents naming ANNEX II BEFORE the ANNEX I line must not end the read early."""
    path = _multi_page_pdf(
        tmp_path,
        "toc_en",
        [
            "CONTENTS\nANNEX II\nconditions",
            "ANNEX I\n4.3 Contraindications\nActive tuberculosis.",
            "4.4 Special warnings\nInfections.\nANNEX II",
            "ANNEX III\nleaflet",
        ],
    )
    text = ema_smpc.read_annex_i_text(path)
    assert "Infections." in text
    assert "leaflet" not in text
    assert ema_smpc.cut_sections(text) == ema_smpc.cut_sections(ema_smpc.read_pdf_text(path))


def test_annex_i_read_without_an_annex_i_marker_reads_everything(tmp_path: Path) -> None:
    """No ANNEX I line => the whole document, because the fallback cut needs it unchanged."""
    path = _multi_page_pdf(tmp_path, "nomarker_en", ["4.3 Contraindications\nX.", "ANNEX II\nY", "tail"])
    assert ema_smpc.read_annex_i_text(path) == ema_smpc.read_pdf_text(path)


def test_parallel_parse_is_identical_to_serial(tmp_path: Path) -> None:
    """The spawn pool reassembles in input order: rows and warnings match a serial run exactly."""
    pdfs = [
        _pdf_with_text(tmp_path, "a-doc_en", "ANNEX I\n4.3 Contraindications\nA wording.\nANNEX II"),
        _pdf_with_text(tmp_path, "b-doc_en", "ANNEX I\n4.1 Therapeutic indications\nB indication.\nANNEX II"),
        _pdf_with_text(tmp_path, "stray_en", "ANNEX I\n4.3 Contraindications\nZ.\nANNEX II"),
    ]
    broken = tmp_path / "broken-doc_en.pdf"
    broken.write_bytes(b"%PDF-1.4\nnot a pdf body")
    refs = [_ref(path) for path in [*pdfs, broken]]
    documents = {
        "a-doc_en": _doc("a-doc_en", number="EMEA/H/C/000001"),
        "b-doc_en": _doc("b-doc_en", number="EMEA/H/C/000002"),
        "broken-doc_en": _doc("broken-doc_en", number="EMEA/H/C/000003"),
    }
    serial = ema_smpc.parse_documents(refs, documents, workers=1)
    parallel = ema_smpc.parse_documents(refs, documents, workers=3)
    assert parallel == serial
    assert {w["code"] for w in serial[1]} >= {"unknown_document", "pdf_unreadable"}


@pytest.mark.usefixtures("_untraced_subprocess")
def test_parallel_pdf_read_does_not_reexecute_the_airflow_main_script(tmp_path: Path) -> None:
    """Airflow's main has no spec and is not safe to reexecute in a spawn child."""
    script = tmp_path / "unsafe_main.py"
    script.write_text(
        "if __name__ != '__main__':\n"
        "    raise RuntimeError('CLI was reexecuted in the child')\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from dakp_pipeline.extract.ema_smpc import _read_texts\n"
        "assert __spec__ is None\n"
        "result = _read_texts([Path(sys.argv[1]), Path(sys.argv[1])], 2)\n"
        "assert all(text and error is None for text, error in result)\n"
        "assert __spec__ is None\n",
        encoding="utf-8",
    )
    run = subprocess.run([sys.executable, str(script), str(_CEPLENE_PDF)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize(("threads", "expected"), [(12, 12), (80, 32), (0, None), (True, None), ("8", None), (None, None)])
def test_parse_workers_follow_the_threads_param(tmp_path: Path, threads: object, expected: int | None) -> None:
    params = {} if threads is None else {"threads": threads}
    ctx = TaskContext(workdir=tmp_path, fixture_root=_FIXTURE_ROOT, params=params)
    assert ema_smpc._workers(ctx) == (expected if expected is not None else min(os.cpu_count() or 1, 32))


def test_read_pdf_text_decrypts_an_empty_password(tmp_path: Path) -> None:
    """EMA serves some documents with an encryption dictionary but no user password."""
    writer = PdfWriter(clone_from=PdfReader(str(_CEPLENE_PDF)))
    writer.encrypt(user_password="", owner_password="owner-secret")
    path = tmp_path / "encrypted_en.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    assert "4.3 Contraindications" in ema_smpc.read_pdf_text(path)
    assert "4.3 Contraindications" in ema_smpc.read_annex_i_text(path)


# --- the extractor (artifact contract) ----------------------------------------------


def _ctx(workdir: Path) -> TaskContext:
    Workdir(workdir).create()
    return TaskContext(workdir=workdir, fixture_root=_FIXTURE_ROOT, params={})


def test_extract_writes_the_sections_and_warnings_tables(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    ctx = _ctx(workdir)
    pdf = tmp_path / "ceplene-epar-product-information_en.pdf"
    shutil.copyfile(_CEPLENE_PDF, pdf)

    refs = ema_smpc.extract([_manifest_ref(), _ref(pdf)], ctx)

    names = sorted(ref.uri.name for ref in refs)
    assert names == ["smpc_sections.parquet", "smpc_warnings.parquet"]
    sections = pl.read_parquet(next(ref.uri for ref in refs if ref.uri.name == "smpc_sections.parquet"))
    assert sections.columns == ema_smpc.SMPC_SECTIONS_COLUMNS
    assert sections.height == 3
    assert sections["section_kind"].to_list() == ["indications", "contraindications", "warnings"]
    sections_ref = next(ref for ref in refs if ref.uri.name == "smpc_sections.parquet")
    assert sections_ref.schema_fingerprint == schemas.schema_fingerprint(ema_smpc.SMPC_SECTIONS_COLUMNS)
    store = ArtifactStore(Workdir(workdir))
    manifest = store.read_manifest(sections_ref.blake3)
    assert manifest is not None
    assert manifest.operation is not None
    assert manifest.operation.name == "extract_ema_smpc"


def test_extract_is_byte_deterministic(tmp_path: Path) -> None:
    """Same inputs => same parquet bytes => same BLAKE3, so the shaping cache keys stay stable."""
    pdf = tmp_path / "ceplene-epar-product-information_en.pdf"
    shutil.copyfile(_CEPLENE_PDF, pdf)
    manifest = _manifest_ref()
    first = ema_smpc.extract([manifest, _ref(pdf)], _ctx(tmp_path / "a"))
    second = ema_smpc.extract([manifest, _ref(pdf)], _ctx(tmp_path / "b"))
    assert [ref.blake3 for ref in first] == [ref.blake3 for ref in second]


def test_extract_reuses_unchanged_outputs_and_force_bypasses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path / "work")
    inputs = [_manifest_ref(), _ref(_CEPLENE_PDF)]
    original = ema_smpc.parse_documents
    calls: list[int] = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ema_smpc, "parse_documents", counted)
    first = ema_smpc.extract(inputs, ctx)
    second = ema_smpc.extract(inputs, ctx)
    assert [r.blake3 for r in first] == [r.blake3 for r in second]
    assert len(calls) == 1
    forced = TaskContext(workdir=ctx.workdir, fixture_root=ctx.fixture_root, params={"force": True})
    ema_smpc.extract(inputs, forced)
    assert len(calls) == 2
    first[0].uri.unlink()
    ema_smpc.extract(inputs, ctx)
    assert len(calls) == 3
    assert first[0].uri.exists()


def test_extract_cache_invalidates_on_parser_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path / "work")
    inputs = [_manifest_ref(), _ref(_CEPLENE_PDF)]
    ema_smpc.extract(inputs, ctx)
    monkeypatch.setattr(ema_smpc, "_PARSE_CACHE_VERSION", "new-parser")
    original = ema_smpc.parse_documents
    calls: list[int] = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ema_smpc, "parse_documents", counted)
    ema_smpc.extract(inputs, ctx)
    assert len(calls) == 1


def test_extract_writes_an_empty_typed_table_when_nothing_is_minable(tmp_path: Path) -> None:
    """Zero rows is a valid (warned) outcome; the typed empty table keeps the contract stable."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    pdf = tmp_path / "ceplene-epar-product-information_en.pdf"
    with pdf.open("wb") as handle:
        writer.write(handle)
    manifest = _manifest_ref()

    refs = ema_smpc.extract([manifest, _ref(pdf)], _ctx(tmp_path / "work"))

    sections = pl.read_parquet(next(ref.uri for ref in refs if ref.uri.name == "smpc_sections.parquet"))
    assert sections.height == 0
    assert sections.columns == ema_smpc.SMPC_SECTIONS_COLUMNS
    assert all(dtype == pl.Utf8 for dtype in sections.dtypes)
    warnings = pl.read_parquet(next(ref.uri for ref in refs if ref.uri.name == "smpc_warnings.parquet"))
    assert warnings["code"].to_list() == ["no_text"]


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        ([], "no EMA documents report"),
        ([ArtifactRef(uri=_MANIFEST, blake3="b3:" + "1" * 64, media_type="application/json")], "no EMA product-information PDFs"),
    ],
)
def test_extract_fails_loudly_without_its_inputs(tmp_path: Path, inputs: list[ArtifactRef], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ema_smpc.extract(inputs, _ctx(tmp_path / "work"))
