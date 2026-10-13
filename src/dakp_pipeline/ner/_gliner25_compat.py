"""Compatibility shim for the gliner2 2.1.0 boundary-architecture CUDA decode bug.

The shipped NER checkpoint (``fastino/gliner2.5-base-v1``) requires the gliner2.5
``BoundaryExtractor`` architecture (gliner2 >= 2.1). gliner2 2.1.0 ships a device bug in
``BoundaryExtractor._attach_entity_attributes``: the attribute logits are deliberately moved to
CPU (``(logits / ...).float().cpu()``) but the row-index tensor is built on the accelerator
(``torch.tensor(rows, dtype=torch.long, device=device)`` with ``device`` taken from the live
query states). Indexing a CPU tensor with CUDA indices raises
``RuntimeError: indices should be either on cpu or on the same device as the indexed tensor``
on every CUDA call whose schema carries entity attribute groups — which DAKP's always does
(the ``assertion_context`` group is a pipeline feature, so the buggy path cannot be avoided by
schema construction). CPU-only inference is unaffected, but CPU is not a viable production
device for NER mining.

This module replaces that one method with a byte-faithful copy of the 2.1.0 implementation
whose only change is indexing on ``logits.device`` instead of the accelerator device. The
replacement is applied only when the live method's source shows the exact buggy pairing, so a
future gliner2 release that fixes the bug (same signature absent) is detected and left alone.
The vendored copy tracks gliner2 2.1.0; re-derive it from a newer release only if that release
reintroduces or reshapes the bug.

Remove this module once the project's minimum gliner2 version ships the fix upstream (track
the upstream issue; bump the floor in ``pyproject.toml`` and delete the ``_load_model`` call).
"""

from __future__ import annotations

import importlib.metadata
import inspect
from collections import OrderedDict
from typing import Any

from dakp_pipeline.logging_setup import logger

#: Source signatures identifying the buggy pairing (CUDA row index into CPU-resident logits).
_BUGGY_INDEX = "torch.tensor(rows, dtype=torch.long, device=device),"
_LOGITS_TO_CPU = ".float().cpu()"
_FIXED_INDEX = "torch.tensor(rows, dtype=torch.long, device=logits.device),"


def _fixed_attach_entity_attributes(
    self: Any,
    sample_index: int,
    core: dict[str, Any],
    specs: list[dict[str, Any]],
    metadata: dict[str, Any],
    decoded: OrderedDict[str, list[dict[str, Any]]],
    token_coordinates: dict[int, tuple[int, int]],
) -> None:
    """gliner2 2.1.0 ``BoundaryExtractor._attach_entity_attributes`` with one device fix.

    Byte-faithful to the 2.1.0 implementation except the row-index tensor below is built on
    ``logits.device`` (CPU) instead of the accelerator: the 2.1.0 code moves ``logits`` to CPU
    and then indexes it with accelerator-resident indices, which raises. Everything else —
    early returns, ordering, scoring, thresholds — is the upstream logic, unmodified.
    """
    # `torch` is bound by apply_boundary_attribute_device_fix before this method can run; the
    # shim module itself must stay import-light (ner.py imports no heavy deps at module load).
    groups = metadata.get("entity_attribute_groups", {})
    if not groups:
        return
    prompt_labels = metadata.get("entity_attribute_prompt_labels", {})
    query_by_name = {spec["field_name"]: query_id for query_id, spec in enumerate(specs) if spec["task_type"] == "entities"}
    attribute_rows: list[tuple[str, int]] = []
    for group in groups.values():
        for label in group.labels:
            prompt = prompt_labels.get(label, label)
            query_id = query_by_name.get(prompt)
            if query_id is not None:
                attribute_rows.append((label, query_id))
    # Labels are unique across groups by Schema.entity_attributes validation.
    attribute_rows = list(dict.fromkeys(attribute_rows))
    if not attribute_rows:
        return

    unique_pairs = list(dict.fromkeys(token_coordinates[id(item)] for items in decoded.values() for item in items))
    if not unique_pairs:
        return
    device = core["query_states"].device
    query_ids = torch.tensor([query_id for _, query_id in attribute_rows], dtype=torch.long, device=device)
    pair_tensor = torch.tensor(unique_pairs, dtype=torch.long, device=device)
    indices = pair_tensor.view(1, 1, len(unique_pairs), 2).expand(1, len(attribute_rows), len(unique_pairs), 2)
    query_states = core["query_states"][sample_index : sample_index + 1].index_select(1, query_ids)
    query_mask = core["query_mask"][sample_index : sample_index + 1].index_select(1, query_ids)
    logits = self.boundary_head.score_explicit_spans(
        core["text_states"][sample_index : sample_index + 1], core["text_mask"][sample_index : sample_index + 1], query_states, query_mask, indices
    )[0]
    logits = (logits / self.boundary_settings.pair_temperature).float().cpu()
    row_by_label = {label: row for row, (label, _) in enumerate(attribute_rows)}
    pair_index = {pair: index for index, pair in enumerate(unique_pairs)}

    for entity_name, items in decoded.items():
        for item in items:
            column = pair_index[token_coordinates[id(item)]]
            for group_name, group in groups.items():
                if group.applies_to is not None and entity_name not in group.applies_to:
                    continue
                present = [(label, row_by_label[label]) for label in group.labels if label in row_by_label]
                if not present:
                    continue
                labels, rows = zip(*present, strict=False)
                values = logits[torch.tensor(rows, dtype=torch.long, device=logits.device), column]
                if group.multi_label:
                    probabilities = torch.sigmoid(values)
                    item[group_name] = [
                        {"label": label, "confidence": float(probabilities[index])}
                        for index, label in enumerate(labels)
                        if float(probabilities[index]) >= group.threshold
                    ]
                else:
                    probabilities = torch.softmax(values, dim=-1)
                    best = int(probabilities.argmax())
                    item[group_name] = {"label": labels[best], "confidence": float(probabilities[best])}


def apply_boundary_attribute_device_fix() -> bool:
    """Patch ``BoundaryExtractor._attach_entity_attributes`` when the 2.1.0 bug is live.

    Returns True when the vendored fix was applied. No-op (False, silent) when the boundary
    engine is absent (e.g. gliner2 < 2.1 or test fakes) or already fixed upstream.
    """
    global torch  # the vendored method resolves `torch` from this module's globals
    try:
        import torch  # lazy: the shim module stays import-light; `global torch` above makes this bind the module global
        from gliner2.models.boundary import engine as boundary_engine
    except ImportError:
        return False

    method = getattr(getattr(boundary_engine, "BoundaryExtractor", None), "_attach_entity_attributes", None)
    if method is None:
        return False
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return False
    if _FIXED_INDEX in source or not (_BUGGY_INDEX in source and _LOGITS_TO_CPU in source):
        return False
    boundary_engine.BoundaryExtractor._attach_entity_attributes = _fixed_attach_entity_attributes
    logger.info("ner_gliner25_compat: boundary attribute device fix applied (gliner2 %s)", importlib.metadata.version("gliner2"))
    return True
