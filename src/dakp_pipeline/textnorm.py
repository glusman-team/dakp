"""Text normalization shared by FAERS ingestion and assertion shaping.

The legacy FAERS ASCII extracts replace every non-ASCII byte with ``?``, so
hyphenated trade names arrive as ``PFIZER?BIONTECH COVID?19 VACCINE`` or
``NATURE?THROID`` and lexical QC matching against the curated vocabulary
collapses to ~0 fuzz. Restoring the separator hyphen — the overwhelmingly
common mangled character in drug names — recovers the match without inventing
data: any character that was mangled to ``?`` already broke exact matching, so
this transform can only raise lexical similarity, never flip a match away from
a string it used to hit.
"""

from __future__ import annotations

import re

import polars as pl

_COLLAPSED_HYPHENS = re.compile(r"-{2,}")


def defaers_text(value: str) -> str:
    """Restore ``?``-mangled separators in one FAERS text value (str-level twin of :func:`defaersify`)."""
    return _COLLAPSED_HYPHENS.sub("-", value.replace("?", "-")).strip("- ").strip()


def defaersify(expr: pl.Expr) -> pl.Expr:
    """Normalize a FAERS-mangled text column: ``?`` → ``-``, collapsed runs, trimmed edges.

    Idempotent on already-clean text, so it is safe to apply both at extraction
    (``extract.faers_ascii`` row shaping) and at assertion shaping (the
    approved-treats / observed-uses consumers).
    """
    return expr.cast(pl.Utf8).str.replace_all("?", "-", literal=True).str.replace_all(r"-{2,}", "-").str.strip_chars("- ")


__all__ = ["defaers_text", "defaersify"]
