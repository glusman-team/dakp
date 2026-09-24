"""Unit tests for the shared FAERS text normalization helpers (`dakp_pipeline.textnorm`).

The legacy FAERS ASCII extracts replace every non-ASCII byte with `?`; these tests pin the
restoration contract (`PFIZER?BIONTECH COVID?19 VACCINE` -> `PFIZER-BIONTECH COVID-19
VACCINE`), idempotence on clean text, and the polars-expression wrapper.
"""

from __future__ import annotations

import polars as pl

from dakp_pipeline.textnorm import defaers_text, defaersify


def test_defaers_text_restores_mangled_hyphens() -> None:
    assert defaers_text("PFIZER?BIONTECH COVID?19 VACCINE") == "PFIZER-BIONTECH COVID-19 VACCINE"
    assert defaers_text("NATURE?THROID") == "NATURE-THROID"
    assert defaers_text("MODERNA COVID?19 VACCINE") == "MODERNA COVID-19 VACCINE"


def test_defaers_text_collapses_runs_and_trims_edges() -> None:
    assert defaers_text("A??B") == "A-B"
    assert defaers_text("?X") == "X"
    assert defaers_text("X?") == "X"
    assert defaers_text("-X-") == "X"
    assert defaers_text("?") == ""


def test_defaers_text_is_idempotent_on_clean_text() -> None:
    for clean in ("Advil", "Ibuprofen", "Acetylsalicylic acid", "PFIZER-BIONTECH COVID-19 VACCINE"):
        assert defaers_text(clean) == clean


def test_defaersify_expr_matches_str_twin() -> None:
    out = pl.DataFrame({"t": ["PFIZER?BIONTECH COVID?19 VACCINE", "Advil", None]}).select(defaersify(pl.col("t")).alias("t"))["t"].to_list()
    assert out == ["PFIZER-BIONTECH COVID-19 VACCINE", "Advil", None]


def test_defaers_text_strips_dosage_tail() -> None:
    # v1.13.0 audit: 45,929 distinct unresolved FAERS drugnames carry a dosage/form
    # tail; truncation at the first dosage token rescued 10/12 of the sampled texts.
    assert defaers_text("# CIPROFLOXACIN CIPROFLOXACIN HCL 500MG TAB)") == "CIPROFLOXACIN CIPROFLOXACIN HCL"
    assert defaers_text("(ALEMTUZUMAB) - UNKNOWN - 30 MG") == "(ALEMTUZUMAB) - UNKNOWN"
    assert defaers_text("TESSALON PERLES 100 mg") == "TESSALON PERLES"
    assert defaers_text("CIPROFLOXACIN INJ. USP 10MG/ ML - BEDFORD LAB") == "CIPROFLOXACIN INJ. USP"
    assert defaers_text("AMOXICILLINE/ACIDE CLAVULANIQUE TEVA 1G/125MG ADULTES") == "AMOXICILLINE/ACIDE CLAVULANIQUE TEVA"
    assert defaers_text("#5 TIMOLOL 0.5% EYE DROPS (ALC)") == "5 TIMOLOL"


def test_defaers_text_dosage_truncation_never_empties_the_name() -> None:
    # An all-dosage string has no non-empty prefix before the first dosage token,
    # so it is left untouched (empty truncation would invent an empty name).
    assert defaers_text("0.9% NACL INJECTION USP, B. BRAUN MEDICAL INC.") == "0.9% NACL INJECTION USP, B. BRAUN MEDICAL INC"


def test_defaers_text_strips_trailing_period_junk_and_greek_tokens() -> None:
    # v1.13.0 audit: `PREDNISONE.` alone was 2,219 dropped FAERS rows.
    assert defaers_text("PREDNISONE.") == "PREDNISONE"
    assert defaers_text("()-1 FENTOS (FENTANYL CITRATE)") == "1 FENTOS (FENTANYL CITRATE)"
    # Legacy NCR-style Greek (DailyMed + FAERS): normalization preserves semantics.
    assert defaers_text(".ALPHA.-TOCOPHEROL") == "alpha-TOCOPHEROL"
    assert defaers_text(".BETA.-CITRONELLOL, (R)-") == "beta-CITRONELLOL, (R)"


def test_defaers_text_leaves_clean_names_alone() -> None:
    # Guardrails from the audit: unit letters that are part of a name never truncate.
    assert defaers_text("HEPATITIS G") == "HEPATITIS G"
    assert defaers_text("ALBUMIN (HUMAN)") == "ALBUMIN (HUMAN)"
    assert defaers_text("PYRIDOXINE HYDROCHLORIDE 5'-PHOSPHATE SODIUM") == "PYRIDOXINE HYDROCHLORIDE 5'-PHOSPHATE SODIUM"


def test_brand_aliases_normalize_to_generic_ingredient_text() -> None:
    # v1.13.0 audit: tablassert QC rejected TRUE brand matches on weak fuzz/SapBERT
    # scores (XEFO->Lornoxicam fuzz 33, BETOLVEX->Cyanocobalamin fuzz 27,
    # rADAMTS13->Apadamtase alfa fuzz 70/sapbert 0.47). Normalizing the mention TEXT to
    # the generic ingredient name gives the fullmap a strong match so QC never fires on
    # a true match. TEXT-level only: DAKP never resolves CURIEs (tablassert's job).
    assert defaers_text("XEFO") == "Lornoxicam"
    assert defaers_text("Xefo") == "Lornoxicam"
    assert defaers_text("XEFOCAM 8MG") == "Lornoxicam"
    assert defaers_text("BETOLVEX") == "Cyanocobalamin"
    assert defaers_text("Betolvex") == "Cyanocobalamin"
    assert defaers_text("rADAMTS13") == "apadamtase alfa"
    assert defaers_text("R-ADAMTS-13") == "apadamtase alfa"
    assert defaers_text("recombinant ADAMTS13") == "apadamtase alfa"


def test_brand_alias_requires_the_recombinant_marker() -> None:
    # Bare ADAMTS13 is the endogenous enzyme, not the drug -- aliasing it would fabricate
    # drug edges from enzyme mentions.
    assert defaers_text("ADAMTS13") == "ADAMTS13"
    assert defaers_text("ADAMTS-13 DEFICIENCY") == "ADAMTS-13 DEFICIENCY"


def test_brand_alias_composes_with_dosage_and_hash_rules() -> None:
    # Real FAERS-shaped junk: the alias must survive the dosage-tail and line-label rules.
    assert defaers_text("XEFO 90 MG TABLET") == "Lornoxicam"
    assert defaers_text("# BETOLVEX 1MG/ML INJ") == "Cyanocobalamin"


def test_brand_aliases_never_touch_clean_or_deliberate_text() -> None:
    # Generic names, unrelated strings, and the deliberate non-alias (the v1.13.0 QC
    # rejection of PFIZER-BIONTECH to tozinameran product concepts was a CORRECT
    # garbage-catch) must pass through unchanged.
    for clean in ("Lornoxicam", "lornoxicam", "Cyanocobalamin", "apadamtase alfa", "Advil", "PFIZER-BIONTECH COVID-19 VACCINE"):
        assert defaers_text(clean) == clean


def test_brand_aliases_are_idempotent() -> None:
    for value in ("XEFO 90 MG TABLET", "rADAMTS13 INJ", "Betolvex"):
        once = defaers_text(value)
        assert defaers_text(once) == once


def test_defaersify_expr_matches_str_twin_on_brand_aliases() -> None:
    cases = ["XEFO", "Betolvex 1MG/ML INJ", "rADAMTS13", "ADAMTS13", "Advil"]
    out = pl.DataFrame({"t": cases}).select(defaersify(pl.col("t")).alias("t"))["t"].to_list()
    assert out == [defaers_text(case) for case in cases]


def test_defaers_text_junk_rules_are_idempotent() -> None:
    for value in ("PREDNISONE.", "# CIPROFLOXACIN CIPROFLOXACIN HCL 500MG TAB)", ".ALPHA.-TOCOPHEROL", "(ALEMTUZUMAB) - UNKNOWN - 30 MG"):
        once = defaers_text(value)
        assert defaers_text(once) == once


def test_defaersify_expr_matches_str_twin_on_audit_junk() -> None:
    cases = [
        "PREDNISONE.",
        "# CIPROFLOXACIN CIPROFLOXACIN HCL 500MG TAB)",
        ".ALPHA.-TOCOPHEROL ACETATE\\ASCORBIC ACID\\BIOTIN",
        "()-1 FENTOS (FENTANYL CITRATE)",
        "ALBUMIN (HUMAN)",
    ]
    out = pl.DataFrame({"t": cases}).select(defaersify(pl.col("t")).alias("t"))["t"].to_list()
    assert out == [defaers_text(case) for case in cases]
