"""Derive and audit the ICD-10-CM non-therapeutic subject denylist that ships in the table configs.

``src/dakp_pipeline/icd10cm_denylist.py`` is a GENERATED snapshot and this tool is its only writer.
It is ruff-format normalized before it is written or compared, so the committed snapshot stays
``make fmt-check``-clean and ``--check`` compares like for like.
It drives the shipped classifier (``dakp_pipeline.extract.icd10cm.parse_release_zip``) and the shipped
observed-use pair key, so a green run proves the rule rather than a copy of it.

Why a snapshot and not a live join
----------------------------------
The screen lands in ``tables/faers_applied_to_treat.yaml`` as ONE Tablassert ``source.reindex``
``antijoin`` membership filter on column ``A`` (the ``join``/``antijoin`` comparison added in
Tablassert 22.2.0): a polars anti hash join against the whole deny set, sub-second on a 1.7M-row
column, where the one-``ne``-per-casing shape it replaces measured 1.6 s at 1,200 literal
compares and 13.9 s at 2,700, while the shape stage they would otherwise force runs for hours.
Deriving the list from build data inside
``table_yaml()`` instead would break the committed-config byte-equality test, so the derivation is
an explicit, per-release act.

Exemption rule
--------------
A Table entry is NOT denied when the same name occurs as a contiguous whole-word run inside a
substance DAKP's own regulator corpora treat as a medicine:

* an ``approved_treats_assertions`` subject (DailyMed and EMA label ingredient text), or
* a Drugs@FDA approved ``ActiveIngredient`` name (split on the combination ``;`` separator), or
* a :data:`MEDICINAL_EXCEPTIONS` entry below.

Without it the screen drops ``LITHIUM`` (label ``LITHIUM CARBONATE``), ``ZINC``, ``ARSENIC``
(``ARSENIC TRIOXIDE``), ``PERMETHRIN`` and ``ALCOHOL``: the Table's empty Adverse-effect column is a
US *coding* statement about a substance-as-poison, and poisons and medicines share its rows. Deriving
the exemption from regulator data rather than a hand-kept medicine list lets it follow the corpora.

Run from the repo root on the machine holding the production tables (wenceslaus)::

    uv run python tests/eval/icd10cm_denylist.py \
        --release-zip /path/icd10cm-table-and-index-2027.zip \
        --approved-treats /path/approved_treats_assertions.tsv \
        --drugsfda-zip /path/drugsfda.zip \
        --edges /path/DRUG_APPROVALS_KP_1.24.1.edges.ndjson [--write]

Exit is non-zero when the derivation cannot run (missing release member, empty label vocabulary) or
when ``--check`` finds the shipped snapshot stale. ``--write`` rewrites the snapshot; regenerate
``tables/*.yaml`` afterwards so the configs embed it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import polars as pl

from dakp_pipeline.assertions.observed_uses import _pair_key
from dakp_pipeline.extract.icd10cm import NON_THERAPEUTIC_TIERS, SubjectTier, parse_release_zip
from dakp_pipeline.io.content_hash import hash_file
from dakp_pipeline.sources.icd10cm import ICD10CM_DRUG_TABLE_ZIP_URL, ICD10CM_RELEASE

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO_ROOT / "src" / "dakp_pipeline" / "icd10cm_denylist.py"

#: Substances the corpora-derived exemption cannot see: human medicines licensed in DAKP's
#: jurisdictions but absent from the mined label tables (US-only DailyMed/Drugs@FDA/EMA-narrative
#: coverage). Keyed by normalized name; the rationale ships into the snapshot so a reviewer can see
#: why a Tier-1 "no Adverse effect code" entry survived.
MEDICINAL_EXCEPTIONS: dict[str, str] = {
    "dihydromorphine": "morphine metabolite analgesic licensed in Germany/Austria; DAKP edges are pain indications",
    "tilidine": "opioid analgesic licensed in Germany/Belgium/Israel; DAKP edges are neuralgia",
    "piritramide": "opioid antitussive licensed in Germany/Austria/Switzerland",
    "diamorphine": "pharmaceutical heroin, licensed for pain in the UK/Switzerland; distinct from the street entry 'heroin'",
    "paraffin": "liquid/soft paraffin is a licensed laxative and emollient in UK/EU products; DAKP edges are dry skin and constipation",
    "chromium": "essential trace element dosed in parenteral nutrition and supplements, like the exempted zinc/copper/selenium",
    "phosphorus": "elemental name under which phosphate salts are dosed in supplementation (the dropped edge treats hypophosphataemia, i.e. replacement therapy)",
}


def name_spans(key: str) -> set[str]:
    """Every contiguous whole-word run of a normalized key, including the key itself."""
    words = key.split(" ")
    return {" ".join(words[start:end]) for start in range(len(words)) for end in range(start + 1, len(words) + 1)}


def label_vocabulary(approved_treats: Path, drugsfda_zip: Path) -> dict[str, str]:
    """``exempt term -> example evidence`` from the corpora DAKP treats as label material."""
    exempt: dict[str, str] = {}

    subjects = (
        pl.scan_csv(approved_treats, separator="\t")
        .select("subject_text")
        .unique()
        .sort("subject_text")  # sorted order: which subject supplies a shared span's evidence must not depend on polars' parallel unique()
        .collect()
        .get_column("subject_text")
        .to_list()
    )
    for subject in subjects:
        for run in name_spans(_pair_key(str(subject))):
            exempt.setdefault(run, f"approved-treats subject: {subject}")

    with zipfile.ZipFile(drugsfda_zip) as zf:
        member = next(name for name in zf.namelist() if name.endswith("Products.txt"))
        text = io.TextIOWrapper(zf.open(member), encoding="utf-8-sig", errors="replace").read()
    cells = {(row.get("ActiveIngredient") or "").strip() for row in csv.DictReader(io.StringIO(text), delimiter="\t")}
    for cell in sorted(cells - {""}):
        for ingredient in cell.split(";"):
            for run in name_spans(_pair_key(ingredient)):
                exempt.setdefault(run, f"Drugs@FDA active ingredient: {cell}")

    for name, rationale in MEDICINAL_EXCEPTIONS.items():
        exempt.setdefault(name, f"medicinal exception: {rationale}")
    return exempt


def screen(rows: list[SubjectTier], exempt: dict[str, str]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Split classified Table rows into ``(denylist entries, exempt entries)``."""
    denied: dict[str, tuple[str, str]] = {}
    spared: dict[str, str] = {}
    for row in rows:
        key = _pair_key(row.name)
        if len(key) < 4:
            continue  # a class label or fragment (GAS, DUST, OIL), not a screenable substance name
        if row.tier not in NON_THERAPEUTIC_TIERS:
            continue
        if key in exempt:
            spared.setdefault(row.name, exempt[key])
            continue
        previous = denied.get(key)
        # toxic_effect is the stronger claim (a noxious substance rather than merely a drug with no
        # US adverse-effect code), so it wins when one normalized key arrives on several rows.
        if previous is None or (previous[0] == "no_accepted_medical_use" and row.tier == "toxic_effect"):
            denied[key] = (row.tier, row.name)
    entries = sorted((name, tier, code) for name, (tier, code) in ((k, v) for k, v in denied.items()))
    return entries, sorted(spared.items())


def _subject_text(edge: dict) -> str:
    raw = edge.get("original_subject")
    return " ".join(map(str, raw)) if isinstance(raw, list) else str(raw or "")


def edge_impact(edges: Path, names: tuple[str, ...]) -> tuple[int, list[tuple[int, str]], int, list[tuple[int, str]]]:
    """Effect of the screen on a released KGX edge file, measured two ways.

    Returns ``(exact_edges, exact_rows, ceiling_edges, ceiling_rows)``. ``exact`` is what the emitted
    config really drops: a row survives the ``antijoin`` unless its subject text equals one comparator
    *literally*, in whichever casing the list spells (the membership join compares raw cell values,
    so the shipped list is four casings of every name). ``ceiling`` is the normalized-key match,
    i.e. everything a semantic screen would remove; the gap is reporter spelling no literal compare
    can reach.
    """
    casings = {casing for name in names for casing in (name.upper(), name.lower(), name.title(), name.capitalize())}
    keys = {_pair_key(name) for name in names}
    exact: dict[str, int] = {}
    ceiling: dict[str, int] = {}
    with edges.open(encoding="utf-8") as handle:
        for line in handle:
            edge = json.loads(line)
            if edge.get("predicate") != "biolink:applied_to_treat":
                continue
            text = _subject_text(edge)
            if text in casings:
                exact[text] = exact.get(text, 0) + 1
            key = _pair_key(text)
            if key in keys:
                ceiling[key] = ceiling.get(key, 0) + 1

    def rank(counts: dict[str, int]) -> list[tuple[int, str]]:
        return sorted(((n, k) for k, n in counts.items()), reverse=True)

    return sum(exact.values()), rank(exact), sum(ceiling.values()), rank(ceiling)


MODULE_HEADER = '''"""Generated ICD-10-CM non-therapeutic subject denylist.

GENERATED FILE, do not edit by hand. Regenerate with::

    uv run python tests/eval/icd10cm_denylist.py --release-zip <zip> \\
        --approved-treats <tsv> --drugsfda-zip <zip> --write

then regenerate ``tables/*.yaml``: ``tablassert._TABLE_SUBJECT_DENYLIST`` embeds these names as
one ``source.reindex`` ``antijoin`` membership filter in ``tables/faers_applied_to_treat.yaml``,
so observed-use edges about a known poison never reach the KG, and an edit costs a config
regeneration, not a re-shape.

Source: the CDC/NCHS ICD-10-CM {release} Table of Drugs and Chemicals, ``{url}``. Its drug table
lists each substance's poisoning, adverse-effect and underdosing codes; an entry with an EMPTY
Adverse-effect column asserts the substance has no use under proper administration (Official
Guidelines section I.C.19.e: adverse-effect coding requires proper administration, so a substance
that never gets an adverse-effect code is one the classification never treats as properly
administered). Such names are screened out of the observed-use (``biolink:applied_to_treat``) edges.

Only the two drug/toxic-effect tiers are screened (``toxic_effect`` = T51-T65, ``no_accepted_medical_use``
= T36-T50, both with no adverse-effect code). Any entry carrying an adverse-effect code is a
therapeutic use and never lands here. Names that also appear inside regulator label text are exempt
and recorded in ``LABEL_BACKED_EXCEPTIONS`` for audit; see tests/eval/icd10cm_denylist.py.
"""

from __future__ import annotations

#: The Table release this snapshot was derived from.
ICD10CM_RELEASE = "{release}"
#: The exact artifact ``sources/icd10cm.py`` acquires, pinned by fiscal year.
ICD10CM_SOURCE_URL = "{url}"
#: blake3 of that release ZIP, so the snapshot is tied to one artifact.
ICD10CM_SOURCE_BLAKE3 = "{blake3}"

#: ``(screened key, tier, published Table spelling)`` for every substance the release screens as
#: non-therapeutic, sorted by key. The key is the name as DAKP normalizes a mention (lowercased,
#: punctuation folded); the third field keeps the Table's own spelling for review.
SCREENED_SUBJECTS: tuple[tuple[str, str, str], ...] = (
{{screened}}
)

#: The names the generated table config denies: both spellings of every screened substance, because a
#: Tablassert ``ne`` filter compares whole cells literally. The key catches reporter text differing
#: only in case ("HEROIN"); the published spelling catches punctuation the normalizer folds away
#: ("P-DICHLOROBENZENE" against the key "p dichlorobenzene").
SUBJECT_DENYLIST: tuple[str, ...] = tuple(dict.fromkeys(name for row in SCREENED_SUBJECTS for name in (row[0], row[2])))

#: ``(subject, evidence)`` for screened-looking substances that are EXEMPT because regulator label
#: text backs them as medicines: an approved-treats subject, a Drugs@FDA active ingredient, or a
#: documented MEDICINAL_EXCEPTIONS rationale.
LABEL_BACKED_EXCEPTIONS: tuple[tuple[str, str], ...] = (
{{exempt}}
)
'''


def render_module(*, blake3: str, entries: list[tuple[str, str, str]], exempt: list[tuple[str, str]]) -> str:
    screened = "\n".join(f'    ("{subject}", "{tier}", "{code}"),' for subject, tier, code in entries) or "    # (nothing screened)"
    spared = "\n".join(f'    ("{subject}", "{evidence}"),' for subject, evidence in exempt) or "    # (nothing exempted)"
    return (
        MODULE_HEADER.format(release=ICD10CM_RELEASE, url=ICD10CM_DRUG_TABLE_ZIP_URL, blake3=blake3)
        .replace("{screened}", screened)
        .replace("{exempt}", spared)
    )


def ruff_normalize(text: str) -> str:
    """Run ruff-format over ``text`` so the snapshot matches the repo's ``fmt-check`` gate.

    A missing ruff (the eval script run outside the project env) leaves the text untouched.
    """
    ruff = shutil.which("ruff")
    if ruff is None:
        return text
    proc = subprocess.run([ruff, "format", "--stdin-filename", SNAPSHOT_PATH.name, "-"], input=text, capture_output=True, text=True, check=True)
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="derive/audit the ICD-10-CM subject denylist")
    parser.add_argument("--release-zip", type=Path, required=True, help="acquired ICD-10-CM Table and Index release ZIP")
    parser.add_argument("--approved-treats", type=Path, required=True, help="production approved_treats_assertions.tsv")
    parser.add_argument("--drugsfda-zip", type=Path, required=True, help="Drugs@FDA data-files ZIP (Products.txt)")
    parser.add_argument("--edges", type=Path, help="released KGX edges NDJSON, to report the screen's edge impact")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite the generated snapshot module")
    mode.add_argument("--check", action="store_true", help="fail if the shipped snapshot differs from the derivation")
    args = parser.parse_args(argv)

    if not args.release_zip.is_file():
        print(f"release ZIP not found: {args.release_zip}", file=sys.stderr)
        return 1
    rows = parse_release_zip(args.release_zip)
    if not rows:
        print("the release ZIP yielded no classified Table entries", file=sys.stderr)
        return 1
    exempt_terms = label_vocabulary(args.approved_treats, args.drugsfda_zip)
    if len(exempt_terms) < 1000:
        print(f"label vocabulary looks truncated ({len(exempt_terms)} terms); check the inputs", file=sys.stderr)
        return 1

    entries, spared = screen(rows, exempt_terms)
    text = ruff_normalize(render_module(blake3=hash_file(args.release_zip), entries=entries, exempt=spared))
    tiers: dict[str, int] = {}
    for _s, tier, _c in entries:
        tiers[tier] = tiers.get(tier, 0) + 1
    print(f"classified Table entries: {len(rows)}")
    print(f"label vocabulary terms : {len(exempt_terms)}")
    print(f"screened subjects      : {len(entries)} {tiers}")
    print(f"label-backed exemptions: {len(spared)}")
    if args.edges is not None:
        names = tuple(name for row in entries for name in (row[0], row[2]))
        exact, exact_rows, ceiling, ceiling_rows = edge_impact(args.edges, names)
        print(f"applied_to_treat edges removed by the emitted filters: {exact} (normalized-key ceiling {ceiling})")
        for count, spelling in exact_rows[:20]:
            print(f"    removed {count:>4}  {spelling}")
        for count, key in ceiling_rows[:20]:
            print(f"    ceiling {count:>4}  {key}")

    if args.check:
        shipped = SNAPSHOT_PATH.read_text(encoding="utf-8")
        if shipped != text:
            print(f"STALE: {SNAPSHOT_PATH} differs from the derivation; rerun with --write", file=sys.stderr)
            return 1
        print("snapshot is current")
        return 0
    if args.write:
        SNAPSHOT_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {SNAPSHOT_PATH}")
        return 0
    print("(dry run: pass --write to store the snapshot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
