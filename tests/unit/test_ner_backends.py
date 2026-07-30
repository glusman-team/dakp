"""Unit tests for the pluggable NER backend layer (``dakp_pipeline.ner.backends``).

ALL of these pass with the ``[ner]`` extra NOT installed: the mock + dictionary backends are
deterministic and dep-free; the real GLiNER/SciSpacy backends are asserted to be *lazy*
(importing ``ner.backends`` imports no heavy deps) and to raise a clear "install the [ner]
extra" error when used without their dep (or skip, if the dep happens to be present).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dakp_pipeline.ner.backends import (
    CONTRAINDICATION_DISEASE_TYPES,
    DictionaryNERBackend,
    GLiNERBackend,
    MockNERBackend,
    NERBackend,
    NERDependencyError,
    SciSpacyBackend,
    canonical_type,
    extract_contraindication_diseases,
    get_backend,
)
from dakp_pipeline.ner.dictionary import DictionaryEntry, DictionaryIndex
from dakp_pipeline.ner.model_cache import SCHEMA_VERSION, default_model_cache_dir, ensure_model, manifest_path, model_root, read_manifest

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"
_ONTOLOGY_TSV = _FIXTURE_ROOT / "ontology" / "disease_map.tsv"

GLINER_AVAILABLE = importlib.util.find_spec("gliner") is not None
SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None


def _entry(normalized: str, curie: str, category: str = "Disease", source: str = "MONDO") -> DictionaryEntry:
    return DictionaryEntry(normalized, curie, normalized, category, source, normalized)


# --- canonical types + EntitySpan ----------------------------------------------


def test_canonical_type_folds_labels() -> None:
    assert canonical_type("DISEASE") == "disease"
    assert canonical_type("PhenotypicFeature") == "phenotype"
    assert canonical_type("  chemical ") == "chemical"
    assert canonical_type("SmallMolecule") == "drug"
    # Unknown labels fall back to their lowercased form.
    assert canonical_type("SomethingElse") == "somethingelse"


def test_contraindication_types_are_disease_and_phenotype() -> None:
    assert CONTRAINDICATION_DISEASE_TYPES == ("disease", "phenotype")


# --- MockNERBackend ------------------------------------------------------------


def test_mock_backend_true_offsets_and_types() -> None:
    backend = MockNERBackend({"hepatitis B": "disease", "fever": "phenotype"})
    text = "Contraindicated in patients with Hepatitis B and fever."
    spans = backend.extract(text, ["disease", "phenotype"])
    by_text = {span.text: span for span in spans}
    assert set(by_text) == {"Hepatitis B", "fever"}
    for span in spans:
        assert text[span.start : span.end] == span.text
    assert by_text["Hepatitis B"].type == "disease"
    assert by_text["fever"].type == "phenotype"


def test_mock_backend_is_deterministic() -> None:
    backend = MockNERBackend({"pain": "disease", "headache": "phenotype"})
    text = "pain and headache and pain"
    first = [(s.start, s.end, s.type, s.text) for s in backend.extract(text, ["disease", "phenotype"])]
    for _ in range(5):
        again = [(s.start, s.end, s.type, s.text) for s in backend.extract(text, ["disease", "phenotype"])]
        assert again == first
    # "pain" occurs twice -> both non-overlapping occurrences are returned.
    assert [s.text for s in backend.extract(text, ["disease"])] == ["pain", "pain"]


def test_mock_backend_filters_by_type() -> None:
    backend = MockNERBackend({"pain": "disease", "aspirin": "drug"})
    text = "pain relieved by aspirin"
    assert [s.text for s in backend.extract(text, ["disease"])] == ["pain"]
    assert [s.text for s in backend.extract(text, ["drug"])] == ["aspirin"]
    assert backend.extract(text, ["phenotype"]) == []


def test_mock_backend_word_boundaries() -> None:
    backend = MockNERBackend({"pain": "disease"})
    # "pain" must not match inside "painting".
    assert backend.extract("painting and repainting", ["disease"]) == []
    assert [s.text for s in backend.extract("pain", ["disease"])] == ["pain"]


def test_mock_backend_empty_and_blank_text() -> None:
    backend = MockNERBackend({"pain": "disease"})
    assert backend.extract("", ["disease"]) == []
    assert backend.extract("   ", ["disease"]) == []


def test_mock_backend_from_tsv(tmp_path: Path) -> None:
    tsv = tmp_path / "ents.tsv"
    tsv.write_text("text\ttype\npain\tdisease\nfever\tphenotype\n", encoding="utf-8")
    backend = MockNERBackend.from_tsv(tsv)
    spans = backend.extract("pain and fever", ["disease", "phenotype"])
    assert sorted(s.text for s in spans) == ["fever", "pain"]


# --- DictionaryNERBackend ------------------------------------------------------


def test_dictionary_backend_from_fixture_offsets_and_types() -> None:
    backend = DictionaryNERBackend.from_tsv(_ONTOLOGY_TSV)
    text = "History of peptic ulcer disease and headache."
    spans = backend.extract(text, ["disease", "phenotype"])
    by_text = {span.text: span for span in spans}
    assert set(by_text) == {"peptic ulcer disease", "headache"}
    for span in spans:
        assert text[span.start : span.end] == span.text
    # headache is a PhenotypicFeature -> phenotype; peptic ulcer disease -> disease.
    assert by_text["headache"].type == "phenotype"
    assert by_text["peptic ulcer disease"].type == "disease"


def test_dictionary_backend_is_deterministic() -> None:
    backend = DictionaryNERBackend.from_tsv(_ONTOLOGY_TSV)
    text = "asthma and pain and hypercholesterolemia"
    first = [(s.start, s.end, s.type, s.text) for s in backend.extract(text, ["disease", "phenotype"])]
    for _ in range(5):
        assert [(s.start, s.end, s.type, s.text) for s in backend.extract(text, ["disease", "phenotype"])] == first


def test_dictionary_backend_filters_by_type() -> None:
    index = DictionaryIndex.from_entries(
        [_entry("asthma", "MONDO:1"), DictionaryEntry("aspirin", "DRUGBANK:DB1", "aspirin", "Drug", "DRUGBANK", "aspirin")]
    )
    backend = DictionaryNERBackend(index)
    text = "asthma treated with aspirin"
    assert [s.text for s in backend.extract(text, ["disease"])] == ["asthma"]
    assert [s.text for s in backend.extract(text, ["drug"])] == ["aspirin"]
    # Empty types = no filter -> both.
    assert sorted(s.text for s in backend.extract(text, [])) == ["aspirin", "asthma"]


# --- protocol conformance ------------------------------------------------------


def test_backends_satisfy_protocol() -> None:
    assert isinstance(MockNERBackend({}), NERBackend)
    assert isinstance(DictionaryNERBackend(DictionaryIndex.from_entries([])), NERBackend)
    assert isinstance(GLiNERBackend(), NERBackend)
    assert isinstance(SciSpacyBackend(), NERBackend)


# --- factory -------------------------------------------------------------------


def test_get_backend_builds_each_kind() -> None:
    assert isinstance(get_backend("mock"), MockNERBackend)
    index = DictionaryIndex.from_entries([_entry("pain", "MONDO:1")])
    assert isinstance(get_backend("dictionary", dictionary=index), DictionaryNERBackend)
    assert isinstance(get_backend("gliner"), GLiNERBackend)
    assert isinstance(get_backend("scispacy"), SciSpacyBackend)


def test_get_backend_is_case_insensitive() -> None:
    assert isinstance(get_backend("  MOCK "), MockNERBackend)


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown ner_backend"):
        get_backend("nope")


# --- laziness: importing ner.backends imports no heavy deps --------------------


def test_importing_backends_does_not_import_heavy_deps() -> None:
    # The module is already imported by the top-level `from ...backends import ...` above;
    # that import path must not have pulled in any heavy optional dep.
    assert "dakp_pipeline.ner.backends" in sys.modules
    for module in ("gliner", "spacy", "scispacy", "huggingface_hub"):
        assert module not in sys.modules, f"importing ner.backends must not import {module}"


def test_constructing_real_backends_does_not_import_heavy_deps() -> None:
    get_backend("gliner")
    get_backend("scispacy")
    for module in ("gliner", "spacy", "scispacy"):
        assert module not in sys.modules, f"constructing a real backend must not import {module}"


# --- real backends raise a clear install error without the extra ---------------


def test_gliner_backend_raises_clear_error_without_extra() -> None:
    if GLINER_AVAILABLE:
        pytest.skip("gliner is installed; the missing-dep error path is not exercised")
    backend = get_backend("gliner")
    with pytest.raises(NERDependencyError, match=r"uv sync --extra ner"):
        backend.extract("contraindicated in severe hepatic impairment", ["disease"])


def test_scispacy_backend_raises_clear_error_without_extra() -> None:
    if SPACY_AVAILABLE:
        pytest.skip("spacy is installed; the missing-dep error path is not exercised")
    backend = get_backend("scispacy")
    with pytest.raises(NERDependencyError, match=r"uv sync --extra ner"):
        backend.extract("contraindicated in severe hepatic impairment", ["disease"])


# --- high-level helper ---------------------------------------------------------


def test_extract_contraindication_diseases_uses_disease_phenotype_types() -> None:
    backend = MockNERBackend({"liver disease": "disease", "rash": "phenotype", "aspirin": "drug"})
    text = "Contraindicated in liver disease, rash, or aspirin use."
    spans = extract_contraindication_diseases(text, backend)
    assert sorted(s.text for s in spans) == ["liver disease", "rash"]  # drug excluded
    assert all(s.type in {"disease", "phenotype"} for s in spans)


# --- model cache ---------------------------------------------------------------


def _fake_downloader(calls: list[str], payload: bytes = b"weights"):
    def download(model_id: str, dest: Path) -> None:
        calls.append(model_id)
        (dest / "weights.bin").write_bytes(payload)

    return download


def test_default_cache_dir_uses_workdir(tmp_path: Path) -> None:
    assert default_model_cache_dir(tmp_path) == tmp_path / "models"


def test_default_cache_dir_honors_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_model_cache_dir() == tmp_path / "dakp" / "models"


def test_default_cache_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert default_model_cache_dir() == Path.home() / ".cache" / "dakp" / "models"


def test_ensure_model_is_idempotent(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert ref1.path.exists()
    assert ref1.manifest.exists()
    assert ref1.b3.startswith("b3:")
    assert calls == ["acme/tiny-ner"]

    ref2 = ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert calls == ["acme/tiny-ner"]  # cache hit: not re-downloaded
    assert ref2.b3 == ref1.b3
    assert ref2.path == ref1.path


def test_ensure_model_writes_manifest(tmp_path: Path) -> None:
    calls: list[str] = []
    ensure_model("acme/tiny-ner", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    data = read_manifest(manifest_path(model_root(tmp_path, "acme/tiny-ner")))
    assert data is not None
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["model_id"] == "acme/tiny-ner"
    assert data["source"] == "huggingface"
    assert data["b3"].startswith("b3:")


def test_ensure_model_force_redownloads(tmp_path: Path) -> None:
    calls: list[str] = []
    ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls), force=True)
    assert len(calls) == 2


def test_ensure_model_verify_detects_drift(tmp_path: Path) -> None:
    calls: list[str] = []
    ref1 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"original"))
    (ref1.path / "weights.bin").write_bytes(b"tampered")  # simulate cache corruption
    ref2 = ensure_model("m/x", cache_dir=tmp_path, downloader=_fake_downloader(calls, payload=b"restored"))
    assert calls == ["m/x", "m/x"]  # drifted content triggered a re-download
    assert ref2.b3 != ref1.b3


def test_ensure_model_sanitizes_model_id_in_path(tmp_path: Path) -> None:
    calls: list[str] = []
    ref = ensure_model("urchade/gliner_small-v2.1", cache_dir=tmp_path, downloader=_fake_downloader(calls))
    assert "urchade--gliner_small-v2.1" in ref.path.as_posix()
