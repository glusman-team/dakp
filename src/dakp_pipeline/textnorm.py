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

3. **Brand/alias spellings** — brand drug names whose fullmap candidates are weak-score
   (``XEFO``, ``BETOLVEX``, ``rADAMTS13``) got QC-rejected in v1.13.0 despite being TRUE
   matches. The :data:`BRAND_ALIASES` table normalizes the mention text to the generic
   ingredient name, which the fullmap resolves strongly — text normalization only, DAKP
   still never resolves CURIEs (tablassert owns ontology mapping).
"""

from __future__ import annotations

import re

import polars as pl

_COLLAPSED_HYPHENS = re.compile(r"-{2,}")
_LEADING_HASH = re.compile(r"^#+\s*")
_EMPTY_PARENS = re.compile(r"\(\)")
# Truncate at the first dosage token (``500MG TAB)``, ``0.5% EYE DROPS``, ``30 MG``);
# lazy ``(.+?)`` + required leading ``\s`` keep a non-empty name prefix mandatory.
_DOSAGE_TAIL = re.compile(r"^(.+?)\s+\d+(?:[.,]\d+)?\s*(?:(?:MCGS?|GMS?|MGS?|MLS?|UNITS?|IUS?|G)\b|%).*$", re.IGNORECASE | re.DOTALL)
_WRAPPED_PARENS = re.compile(r"^\((.+)\)$")
_TRAILING_PERIOD = re.compile(r"([A-Za-z])\.$")
_MULTI_SPACE = re.compile(r"\s{2,}")
# DailyMed/FAERS legacy non-ASCII Greek spellings, e.g. ``.ALPHA.-TOCOPHEROL``.
_GREEK_TOKENS = tuple(
    f".{word}."
    for word in (
        "ALPHA",
        "BETA",
        "GAMMA",
        "DELTA",
        "EPSILON",
        "ZETA",
        "ETA",
        "THETA",
        "IOTA",
        "KAPPA",
        "LAMBDA",
        "MU",
        "NU",
        "XI",
        "OMICRON",
        "PI",
        "RHO",
        "SIGMA",
        "TAU",
        "UPSILON",
        "PHI",
        "CHI",
        "PSI",
        "OMEGA",
    )
)
# Brand/alias mention TEXT normalization (v1.13.0 data audit, plans/v1.13.0-data-audit-findings.md):
# tablassert's QC rejected these brand spellings on weak fuzz/SapBERT scores even though the
# matches were TRUE (tmp/.tablassert/log/tablassert.log on wenceslaus): XEFO -> Lornoxicam
# (fuzz 33), BETOLVEX -> Cyanocobalamin (fuzz 27), rADAMTS13 -> Apadamtase alfa (fuzz 70 but
# sapbert 0.47). Normalizing the mention text to the generic ingredient name gives the fullmap
# a strong exact match, so QC never fires on a true match. This is TEXT-level only -- DAKP
# performs no ontology resolution; tablassert still resolves the generic name to its CURIE.
# Deliberately absent: PFIZER-BIONTECH COVID-19 VACCINE (its QC rejection to the tozinameran
# product concepts was a CORRECT garbage-catch -- the brand maps to a product concept, not the
# administered vaccine, and no generic single-ingredient name exists).
BRAND_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Xefo / Xefocam are lornoxicam brand names (Nycomed/Takeda).
    (re.compile(r"(?i)\bXEFO(?:CAM)?\b"), "Lornoxicam"),
    # Betolvex is a cyanocobalamin (vitamin B12) depot brand.
    (re.compile(r"(?i)\bBETOLVEX\b"), "Cyanocobalamin"),
    # rADAMTS13 (also written R-ADAMTS-13 / recombinant ADAMTS13) is the drug apadamtase alfa;
    # bare ADAMTS13 is the endogenous enzyme and must NOT alias.
    (re.compile(r"(?i)\bR[- ]?ADAMTS-?13\b|\bRECOMBINANT\s+ADAMTS-?13\b"), "apadamtase alfa"),
)
_POLARS_BRAND_ALIASES: tuple[tuple[str, str], ...] = tuple((pattern.pattern, replacement) for pattern, replacement in BRAND_ALIASES)
_POLARS_DOSAGE_TAIL = r"(?is)^(.+?)\s+\d+(?:[.,]\d+)?\s*(?:(?:MCGS?|GMS?|MGS?|MLS?|UNITS?|IUS?|G)\b|%).*$"
_POLARS_TRAILING_PERIOD = r"(?i)([A-Za-z])\.$"
_POLARS_WRAPPED_PARENS = r"^\((.+)\)$"
_POLARS_LEADING_HASH = r"^#+\s*"
_POLARS_MULTI_SPACE = r"\s{2,}"

_EDGE_JUNK = " -,;"


def defaers_text(value: str) -> str:
    """Normalize one FAERS text value (str-level twin of :func:`defaersify`).

    Restores ``?``-mangled separators, canonicalizes multi-ingredient combinations
    (see :func:`_combo_canonical`), strips dosage/form tails, ``#`` line labels,
    empty parens, trailing periods, and legacy ``.GREEK.`` tokens, unwraps a
    fully-wrapped parenthesized name, applies brand aliases, and trims edge junk.
    Idempotent on already-clean text.
    """
    if "\\" in value or ";" in value:
        return _combo_canonical(value)
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
    for pattern, generic in BRAND_ALIASES:
        if pattern.search(value):
            value = pattern.sub(generic, value)
    return _MULTI_SPACE.sub(" ", value).strip(_EDGE_JUNK)


def _combo_canonical(value: str) -> str:
    """Canonicalize a multi-ingredient combination to its mixture-level subject text.

    FAERS ``drugname`` and DailyMed packaging strings join ingredients with ``\\`` or
    ``;`` (v1.13.0 audit: 3,398 distinct backslash combos + 4,499 semicolon combos among
    unresolved FAERS subjects; fullmap quick-map on wenceslaus resolved 0/20 raw
    ``\\``-joined forms and 4/20 canonical `` / ``-joined forms, including real mixture
    concepts, e.g. ``CARBOPLATIN / PACLITAXEL`` -> PUBCHEM.COMPOUND MolecularMixture).

    Representation decision (US-004): MIXTURE-LEVEL subject -- the product the patient
    took IS the subject, joined with `` / `` (the fullmap's mixture-wording convention).
    Each component is normalized by the full :func:`defaers_text` chain (dosage tails,
    greek tokens, brand aliases), empty components drop, and exactly one surviving
    component collapses to that ingredient. One FAERS case contributes to ONE product
    edge: components are NEVER split into per-ingredient edges, because case attribution
    is at product level and splitting would fabricate per-ingredient evidence.
    Combinations whose components all clean away fall back to the stripped original so the
    value never empties. Bare ``/`` (salt-pair notation like ``AMOXICILLIN/CLAVULANATE``)
    is deliberately NOT a separator: it names one conceptual product, not an ingredient
    list. Unresolved mixtures stay unresolved (their rows drop) rather than fabricate an
    ingredient -- a curated alias table for top mixtures is backlog, not silent inference.
    """
    parts = [defaers_text(part) for part in re.split(r"[\\;]", value)]
    parts = [part for part in parts if part]
    if len(parts) >= 2:
        return " / ".join(parts)
    if len(parts) == 1:
        return parts[0]
    return value.strip(_EDGE_JUNK) or value.strip()


def defaersify(expr: pl.Expr) -> pl.Expr:
    """Normalize a FAERS-mangled text column (polars twin of :func:`defaers_text`).

    Idempotent on already-clean text, so it is safe to apply both at extraction
    (``extract.faers_ascii`` row shaping) and at assertion shaping (the
    approved-treats / observed-uses consumers).
    """
    has_combo = expr.cast(pl.Utf8).str.contains(r"[\\;]")
    expr = pl.when(has_combo).then(expr.cast(pl.Utf8).map_elements(_combo_canonical, return_dtype=pl.Utf8)).otherwise(expr.cast(pl.Utf8))
    expr = expr.str.replace_all("?", "-", literal=True)
    expr = expr.str.replace_all(r"-{2,}", "-")
    expr = expr.str.replace_all(_POLARS_LEADING_HASH, "")
    expr = expr.str.replace_all("()", "", literal=True)
    expr = expr.str.replace_all(_POLARS_DOSAGE_TAIL, "${1}")
    expr = expr.str.replace_all(_POLARS_WRAPPED_PARENS, "${1}")
    expr = expr.str.replace_all(_POLARS_TRAILING_PERIOD, "${1}")
    for token in _GREEK_TOKENS:
        expr = expr.str.replace_all(token, token[1:-1].lower(), literal=True)
    for pattern, generic in _POLARS_BRAND_ALIASES:
        expr = expr.str.replace_all(pattern, generic)
    return expr.str.replace_all(_POLARS_MULTI_SPACE, " ").str.strip_chars(_EDGE_JUNK)


__all__ = ["BRAND_ALIASES", "defaers_text", "defaersify"]
