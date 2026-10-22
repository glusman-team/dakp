"""Audit the curated species lexicon against a fullmap (evaluation artifact, not pytest).

Every CURIE in :data:`dakp_pipeline.assertions.species.SPECIES_LEXICON` plus
:data:`~dakp_pipeline.assertions.species.HUMAN_TAXON` must resolve, through the exact build
path (``tablassert.fullmap.quick_map``), back to itself as an ``OrganismTaxon``. The
``species_context_qualifier`` column carries these CURIEs verbatim, so a CURIE that stops
resolving would silently drop (strict qualifier) every edge carrying it.

Run on wenceslaus against the production fullmap (never on the laptop):

    uv run python tests/eval/species_lexicon_audit.py /local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap

Exits non-zero listing every CURIE that fails to round-trip.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tablassert.fullmap import quick_map

from dakp_pipeline.assertions.species import HUMAN_TAXON, SPECIES_LEXICON


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: species_lexicon_audit.py <fullmap redb path>", file=sys.stderr)
        return 2
    curies = sorted({HUMAN_TAXON, *SPECIES_LEXICON.values()})
    results = quick_map(curies, Path(argv[1]), taxon=None)
    failures: list[str] = []
    for curie in curies:
        frame = results.get(curie)
        rows = [] if frame is None else frame.to_dicts()
        if not any(row["CURIE"] == curie and row["CATEGORY_NAME"] == "OrganismTaxon" for row in rows):
            failures.append(f"{curie}: resolved to {sorted({str(row['CURIE']) for row in rows}) or 'nothing'}")
        else:
            name = next(row["PREFERRED_NAME"] for row in rows if row["CURIE"] == curie)
            print(f"ok   {curie}  {name}")
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"{len(curies) - len(failures)}/{len(curies)} lexicon CURIEs round-trip as OrganismTaxon")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
