"""Text normalization shared by FAERS ingestion and assertion shaping.

Two families of mangling reach the lexical/fullmap resolvers through these helpers:

1. **Legacy ASCII `?` bytes** — the FAERS ASCII extracts replace every non-ASCII byte
   with ``?``, so hyphenated trade names arrive as ``PFIZER?BIONTECH COVID?19 VACCINE``
   or ``NATURE?THROID`` and lexical QC matching against the curated vocabulary collapses
   to ~0 fuzz. Restoring the separator hyphen — the overwhelmingly common mangled
   character in drug names — recovers the match without inventing data.

2. **FAERS ``drugname`` field junk** — the free-text drugname field mixes product label
   text with dosage/form suffixes (``CIPROFLOXACIN CIPROFLOXACIN HCL 500MG TAB)``,
   ``(ALEMTUZUMAB) - UNKNOWN - 30 MG``), line-label ``#`` prefixes, empty ``()`` pairs,
   trailing sentence periods (``PREDNISONE.`` — 2,219 rows in v1.13.0), and legacy
   NCR-style Greek tokens (``.ALPHA.-TOCOPHEROL``). None of that reaches the resolver
   as a clean ingredient name: the v1.13.0 audit measured 45,929 distinct unresolved
   texts carrying a dosage tail (10/12 sample rescue rate once truncated), 3,541
   trailing-dot texts (6/12 rescue), plus hundreds of ``()``/``#`` rows. Every rule
   here only REMOVES suffix/punctuation noise from around a longer name, so it can
   only raise lexical similarity — never flip a match away from a string it used to hit
   (the same conservation argument as the ``?`` restoration). Dosage tails truncate
   from the FIRST dosage token only when a non-empty name remains before it, so a
   string that is all dosage (``0.9% NACL ...``) is left untouched.
"""

from __future__ import annotations

import re

import polars as pl

_COLLAPSED_HYPHENS = re.compile(r"-{2,}")
_LEADING_HASH = re.compile(r"^#+\s*")
_EMPTY_PARENS = re.compile(r"\(\)")
# Truncate at the first dosage token (``500MG TAB)``, ``0.5% EYE DROPS``, ``30 MG``);
# lazy ``(.+?)`` + required leading ``\s`` keep a non-empty name prefix mandatory.
_DOSAGE_TAIL = re.compile(
    r"^(.+?)\s+\d+(?:[.,]\d+)?\s*(?:(?:MCGS?|GMS?|MGS?|MLS?|UNITS?|IUS?|G)\b|%).*$",
    re.IGNORECASE | re.DOTALL,
)
_WRAPPED_PARENS = re.compile(r"^\((.+)\)$")
_TRAILING_PERIOD = re.compile(r"([A-Za-z])\.$")
_MULTI_SPACE = re.compile(r"\s{2,}")
# DailyMed/FAERS legacy non-ASCII Greek spellings, e.g. ``.ALPHA.-TOCOPHEROL``.
_GREEK_TOKENS = tuple(
    f".{word}."
    for word in (
        "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA",
        "IOTA", "KAPPA", "LAMBDA", "MU", "NU", "XI", "OMICRON", "PI", "RHO",
        "SIGMA", "TAU", "UPSILON", "PHI", "CHI", "PSI", "OMEGA",
    )
)
_POLARS_DOSAGE_TAIL = (
    r"(?is)^(.+?)\s+\d+(?:[.,]\d+)?\s*(?:(?:MCGS?|GMS?|MGS?|MLS?|UNITS?|IUS?|G)\b|%).*$"
)
_POLARS_TRAILING_PERIOD = r"(?i)([A-Za-z])\.$"
_POLARS_WRAPPED_PARENS = r"^\((.+)\)$"
_POLARS_LEADING_HASH = r"^#+\s*"
_POLARS_MULTI_SPACE = r"\s{2,}"

_EDGE_JUNK = " -,;"


def defaers_text(value: str) -> str:
    """Normalize one FAERS text value (str-level twin of :func:`defaersify`).

    Restores ``?``-mangled separators, strips dosage/form tails, ``#`` line labels,
    empty parens, trailing periods, and legacy ``.GREEK.`` tokens, unwraps a
    fully-wrapped parenthesized name, and trims edge junk. Idempotent on already-clean
    text.
    """
    value = value.replace("?", "-")
    value = _COLLAPSED_HYPHENS.sub("-", value)
    value = _LEADING_HASH.sub("", value)
    value = _EMPTY_PARENS.sub("", value)
    tail = _DOSAGE_TAIL.match(value)
    if tail and tail.group(1).strip(_EDGE_JUNK):
        value = tail.group(1)
    value = _WRAPPED_PARENS.sub(r"\1", value)
    value = _TRAILING_PERIOD.sub(r"\1", value)
    for token in _GREEK_TOKENS:
        if token in value:
            value = value.replace(token, token[1:-1].lower())
    return _MULTI_SPACE.sub(" ", value).strip(_EDGE_JUNK)


def defaersify(expr: pl.Expr) -> pl.Expr:
    """Normalize a FAERS-mangled text column (polars twin of :func:`defaers_text`).

    Idempotent on already-clean text, so it is safe to apply both at extraction
    (``extract.faers_ascii`` row shaping) and at assertion shaping (the
    approved-treats / observed-uses consumers).
    """
    expr = expr.cast(pl.Utf8).str.replace_all("?", "-", literal=True)
    expr = expr.str.replace_all(r"-{2,}", "-")
    expr = expr.str.replace_all(_POLARS_LEADING_HASH, "")
    expr = expr.str.replace_all("()", "", literal=True)
    expr = expr.str.replace_all(_POLARS_DOSAGE_TAIL, "${1}")
    expr = expr.str.replace_all(_POLARS_WRAPPED_PARENS, "${1}")
    expr = expr.str.replace_all(_POLARS_TRAILING_PERIOD, "${1}")
    for token in _GREEK_TOKENS:
        expr = expr.str.replace_all(token, token[1:-1].lower(), literal=True)
    return expr.str.replace_all(_POLARS_MULTI_SPACE, " ").str.strip_chars(_EDGE_JUNK)


__all__ = ["defaers_text", "defaersify"]
