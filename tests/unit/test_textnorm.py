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


def test_defaers_text_strips_unclosed_paren_junk() -> None:
    # v1.13.1 audit: 3,723 distinct FAERS subjects carry an unterminated ``( ...`` tail from
    # ASCII truncation. The tail is noise around a longer name: 'ACETYLSALICYLIC ACID (}'
    # fuzzy-matched onto Clofedanol (wrong drug) in the v1.13.1 full build.
    assert defaers_text("ACETYLSALICYLIC ACID (}") == "ACETYLSALICYLIC ACID"
    assert defaers_text("NAPROXEN SODIUM ({") == "NAPROXEN SODIUM"
    assert defaers_text("BARICITINIB (baricitinib") == "BARICITINIB"
    assert defaers_text("(CARBIDOPA") == "CARBIDOPA"
    assert defaers_text("EFRACEA (DOXYCYCLINE) (40 MG") == "EFRACEA (DOXYCYCLINE)"
    assert defaers_text("A (B (C") == "A"
    # Balanced parens are never touched, and a truncation that would empty the name keeps it.
    assert defaers_text("bosentan (as monohydrate)") == "bosentan (as monohydrate)"
    assert defaers_text("((") == "(("


def test_defaersify_expr_matches_str_twin_on_unclosed_parens() -> None:
    cases = ["ACETYLSALICYLIC ACID (}", "(CARBIDOPA", "A (B (C", "bosentan (as monohydrate)", "((", None]
    out = pl.DataFrame({"t": cases}).select(defaersify(pl.col("t")).alias("t"))["t"].to_list()
    assert out == [defaers_text(case) if case is not None else None for case in cases]


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
    # v1.14.0 growth: the top unresolved FAERS brand subjects by case mass (HUMIRA alone was
    # 632,251 dropped cases in v1.13.1 -- brand texts miss the fullmap, the generic resolves;
    # targets verified against the v1.13.1 node names / treats-side CURIEs).
    assert defaers_text("HUMIRA") == "Adalimumab"
    assert defaers_text("DUPIXENT") == "Dupilumab"
    assert defaers_text("ENBREL") == "Etanercept"
    assert defaers_text("Enbrel") == "Etanercept"
    assert defaers_text("TYSABRI") == "Natalizumab"
    assert defaers_text("REMICADE") == "Infliximab"
    assert defaers_text("INFLECTRA") == "Infliximab"
    assert defaers_text("ZANTAC") == "Ranitidine"
    assert defaers_text("COSENTYX") == "Secukinumab"
    assert defaers_text("REPATHA") == "Evolocumab"
    assert defaers_text("PROLIA") == "Denosumab"
    assert defaers_text("LANTUS SOLOSTAR") == "Insulin glargine"
    assert defaers_text("PROVENTIL HFA") == "Albuterol HFA"
    assert defaers_text("AVONEX") == "Interferon beta-1a"
    assert defaers_text("REBIF") == "Interferon beta-1a"
    assert defaers_text("CABOZANTINIB S-MALATE") == "Cabozantinib"
    # Indication-text canonicalization: v1.13.1 resolved 'CUSHING'S SYNDROME' into the
    # hyperaldosteronism wording channel while the treats side carries Cushing syndrome.
    assert defaers_text("CUSHING'S SYNDROME") == "Cushing syndrome"
    assert defaers_text("CUSHINGS SYNDROME") == "Cushing syndrome"


def test_brand_alias_growth_never_captures_lookalikes() -> None:
    # FORTECORTIN (dexamethasone brand) contains FORTEO as a prefix; the word boundary must
    # not fire, and the deliberate non-aliases stay put.
    assert defaers_text("FORTECORTIN") == "FORTECORTIN"
    assert defaers_text("FORTECORTIN DEXAMETHASONE") == "FORTECORTIN DEXAMETHASONE"
    assert defaers_text("ADAMTS13") == "ADAMTS13"


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


def test_combo_products_canonicalize_to_slash_joined_mixture() -> None:
    # US-004 decision: mixture-level subject. Fullmap quick-map on wenceslaus resolved 0/20
    # raw backslash-joined combos and 4/20 canonical ' / '-joined forms (incl. real
    # MolecularMixture concepts), so the product the patient took stays ONE subject.
    assert (
        defaers_text(".ALPHA.-TOCOPHEROL ACETATE\\ASCORBIC ACID\\BIOTIN\\CHOLECALCIFEROL")
        == "alpha-TOCOPHEROL ACETATE / ASCORBIC ACID / BIOTIN / CHOLECALCIFEROL"
    )
    assert defaers_text("IBUPROFEN 400MG TABLET\\CAFFEINE 50MG CAP") == "IBUPROFEN / CAFFEINE"
    assert (
        defaers_text("(6S)-5-METHYLTETRAHYDROFOLATE GLUCOSAMINE;BIOTIN;CALCIUM PANTOTHENATE")
        == "(6S)-5-METHYLTETRAHYDROFOLATE GLUCOSAMINE / BIOTIN / CALCIUM PANTOTHENATE"
    )


def test_combo_single_survivor_collapses_and_junk_never_empties() -> None:
    # One surviving component collapses to that ingredient; all-junk combos fall back to the
    # stripped original so the value never empties (both invariants from specs/combo-products).
    assert defaers_text("()\\;PREDNISONE.") == "PREDNISONE"
    assert defaers_text("()\\;") != ""


def test_combo_rule_never_splits_salt_pair_slash_notation() -> None:
    # Bare '/' names one conceptual product (a salt pair), NOT an ingredient list; only the
    # list separators '\\' and ';' trigger mixture canonicalization.
    once = defaers_text("AMOXICILLIN/CLAVULANATE POTASSIUM")
    assert "/" in once
    assert " / " not in once


def test_combo_canonicalization_is_idempotent() -> None:
    for value in (".ALPHA.-TOCOPHEROL ACETATE\\ASCORBIC ACID\\BIOTIN", "CARBOPLATIN;PACLITAXEL", "()\\;PREDNISONE."):
        once = defaers_text(value)
        assert defaers_text(once) == once


def test_defaersify_expr_matches_str_twin_on_combos() -> None:
    cases = [".ALPHA.-TOCOPHEROL ACETATE\\ASCORBIC ACID\\BIOTIN", "CARBOPLATIN;PACLITAXEL", "()\\;PREDNISONE.", "Advil", None]
    out = pl.DataFrame({"t": cases}).select(defaersify(pl.col("t")).alias("t"))["t"].to_list()
    non_null = [case for case in cases if case is not None]
    assert [value for value in out if value is not None] == [defaers_text(case) for case in non_null]
    assert out[4] is None


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
