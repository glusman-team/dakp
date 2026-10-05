"""Approval-status audit: re-derive clinical_approval_status with the SHIPPED rule.

Compares the ``clinical_approval_status`` column that a build actually shipped in
``faers_applied_to_treat_assertions.tsv`` against the value the current code derives, over the
REAL production tables. Because the audit drives ``ApprovedTreatsIndex`` itself (not a
reimplementation), a green audit proves the rule, not a copy of it.

Why it exists: the shipped status used to come from normalized text-pair equality only, which
misread real FDA-approved uses as off-label whenever the FAERS brand spelling differed from the
DailyMed ingredient spelling (the production edge ``32c1b660-8fcf-3330-8576-21ec0ac0fab3``:
``ELIGARD applied_to_treat Prostate cancer stage IV`` shipped as ``off_label_use`` although the
FDA label for NDA021343 is indicated "for the treatment of advanced prostate cancer"). The rule
now also matches on the FDA application display form and bridges object granularity within one
application; this tool measures how much status mass those keys move and gates on monotonicity.

The audit is a REGRESSION GATE: it exits non-zero when any row is DEMOTED
(``approved_for_condition`` -> ``off_label_use``). The rule only ever adds approvals, so a
demotion means a change broke a previously-correct status.

Run from the repo root on the machine that holds the production tables (wenceslaus)::

    PYTHONUTF8=1 PYTHONPATH=$PWD/src python tests/eval/approval_status_audit.py /path/to/tmp/tabular
    PYTHONUTF8=1 PYTHONPATH=$PWD/src python tests/eval/approval_status_audit.py /path/to/tmp/tabular \
        --subject ELIGARD --object "Prostate cancer stage IV"   # inspect one pair

This is an evaluation artifact: NOT collected by pytest (filename is not ``test_*.py``) and not
part of the coverage-gated package (same convention as ``benchmark_ner.py`` / ``ab_ner_diff.py``).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import polars as pl

from dakp_pipeline.assertions.observed_uses import ApprovedTreatsIndex, _pair_key

_APPROVED = "approved_for_condition"
_OFF_LABEL = "off_label_use"


def _derive(index: ApprovedTreatsIndex, subject: str, obj: str, approvals: list[str]) -> str:
    """The status the shipped rule assigns to one table row (empty approvals -> text rule only)."""
    return _APPROVED if index.is_approved(_pair_key(subject), _pair_key(obj), approvals) else _OFF_LABEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tabular", help="directory holding the real *_assertions.tsv tables")
    parser.add_argument("--subject", help="only audit rows with this exact subject_text")
    parser.add_argument("--object", help="only audit rows with this exact object_text")
    parser.add_argument("--top", type=int, default=25, metavar="N", help="how many changed rows to print (default 25)")
    args = parser.parse_args(argv)

    approved = pl.read_csv(f"{args.tabular}/approved_treats_assertions.tsv", separator="\t", infer_schema_length=0)
    index = ApprovedTreatsIndex.from_frame(approved)
    print(f"approved table rows          : {approved.height:,}")
    print(f"text pairs in the index      : {len(index.pairs):,}")
    print(f"application keys in the index: {len(index.objects_by_approval):,}")

    query = pl.col("clinical_approval_status").is_in([_APPROVED, _OFF_LABEL])  # not_provided rows had no approved table
    if args.subject:
        query &= pl.col("subject_text") == args.subject
    if args.object:
        query &= pl.col("object_text") == args.object
    frame = pl.read_csv(f"{args.tabular}/faers_applied_to_treat_assertions.tsv", separator="\t", infer_schema_length=0).filter(query)
    print(f"audited rows                 : {frame.height:,}")

    rows: Counter[str] = Counter()
    cases: Counter[str] = Counter()
    changed: dict[str, list[tuple[int, str, str, str]]] = {"promoted": [], "demoted": []}

    for rec in frame.iter_rows(named=True):
        shipped = str(rec.get("clinical_approval_status") or "")
        subject = str(rec.get("subject_text") or "")
        obj = str(rec.get("object_text") or "")
        n = int(rec.get("number_of_cases") or 0)
        approvals = [a for a in (rec.get("FDA_regulatory_approvals") or "").split("|") if a]
        derived = _derive(index, subject, obj, approvals)
        if derived == shipped:
            rows["unchanged"] += 1
            continue
        direction = "promoted" if derived == _APPROVED else "demoted"
        rows[direction] += 1
        cases[direction] += n
        changed[direction].append((n, subject, obj, "|".join(approvals[:4])))

    print()
    for key in ("unchanged", "promoted", "demoted"):
        print(f"{key:10s} rows={rows[key]:>10,}  cases={cases[key]:>12,}")
    for direction, label in (("promoted", f"{_OFF_LABEL} -> {_APPROVED}"), ("demoted", f"{_APPROVED} -> {_OFF_LABEL}")):
        if not changed[direction]:
            continue
        print(f"\n{direction} ({label}), top {args.top} by case count:")
        for n, subject, obj, appr in sorted(changed[direction], key=lambda t: -t[0])[: args.top]:
            print(f"   {n:>9,}  {subject} | {obj} | [{appr}]")

    print(f"\nverdict: {rows['demoted']} demotion(s)")
    return 1 if rows["demoted"] else 0


if __name__ == "__main__":
    sys.exit(main())
