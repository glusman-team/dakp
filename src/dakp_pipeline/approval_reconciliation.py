"""Preserve the legacy approval rule after ontology resolution and edge folding.

Text/application matching happens in assertion shapers. Only the resolved KG can answer
whether two source spellings name the same drug and condition. This pass promotes observed
uses with an exact resolved treats counterpart; it never demotes a source-derived approval.
It is not a KG compiler: Tablassert owns resolution, identities and evidence folding.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

TREATS = "biolink:treats"
OBSERVED = "biolink:applied_to_treat"
APPROVED = "approved_for_condition"


def resolved_pair(edge: Mapping[str, Any]) -> tuple[str, str] | None:
    """Only nonempty scalar string endpoints identify a pair; validation reports defects."""
    subject, obj = edge.get("subject"), edge.get("object")
    return (subject, obj) if isinstance(subject, str) and subject and isinstance(obj, str) and obj else None


def _record(line: str) -> Mapping[str, Any]:
    record = json.loads(line)
    if not isinstance(record, dict):
        msg = f"KGX edge record must be a JSON object, got {type(record).__name__}"
        raise ValueError(msg)
    return record


def approved_resolved_pairs(edges: Iterable[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """Collect exact resolved pairs, without ancestor inference or qualifier changes."""
    return {pair for edge in edges if edge.get("predicate") == TREATS and (pair := resolved_pair(edge)) is not None}


def reconcile_approval_status(path: Path) -> int:
    """Atomically promote observed uses with resolved treats counterparts in NDJSON.

    Two streaming passes retain only the approved pair set, not the whole graph. Input order,
    unchanged record bytes, identities, qualifiers and all evidence stay intact. JSON or I/O
    failures leave the original untouched. The output is idempotent. A qualified treats edge
    is still evidence for its exact drug/condition pair; its qualifier and ID remain distinct.
    """
    with path.open(encoding="utf-8") as handle:
        approved = approved_resolved_pairs(_record(line) for line in handle if line.strip())
    promoted = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
            temporary = Path(out.name)
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        out.write(line)
                        continue
                    edge = dict(_record(line))
                    if edge.get("predicate") == OBSERVED and resolved_pair(edge) in approved and edge.get("clinical_approval_status") != APPROVED:
                        edge["clinical_approval_status"] = APPROVED
                        out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                        promoted += 1
                    else:
                        out.write(line)
            out.flush()
            os.fsync(out.fileno())
        if promoted:
            os.chmod(temporary, path.stat().st_mode)
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return promoted
