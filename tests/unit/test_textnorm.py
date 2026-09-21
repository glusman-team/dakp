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
