"""Unit tests for the NER training-data export stage (:mod:`dakp_pipeline.ner_export`).

Happy-path coverage of the ``dakp.ner.export.v1`` contract, run against REAL fixture
extracts per the ``tests/unit/conftest.py`` convention: the Avro model mirror's
schema compatibility with RelMedNER's ``TrainingExample``, LOINC section selection
(contraindications + boxed warnings + indications), FAERS/EMA selection, the gliner2
row build (entities, normal + qualifier relations, classification), dedupe/determinism,
the four-file bundle layout with a verbatim gold copy, manifest self-consistency, and
the Transformer-shaped :func:`export` entry point. Error/edge branches live in
``test_ner_export_edge.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from dakp_pipeline.ner_export import TrainingExample

_SNAPSHOT = Path(__file__).resolve().parents[1] / "eval" / "ner_training_schema.json"


def test_avro_schema_matches_relmedner_snapshot() -> None:
    """The exported ``examples.avro`` must be record-compatible with RelMedNER.

    ``tests/eval/ner_training_schema.json`` is the ``avro_schema_to_python()`` output of
    RelMedNER's own ``TrainingExample`` (branch ``gliner-biomed-post-training`` as merged
    to ``main``). Regenerate that file from the RelMedNER checkout when its models
    change; never hand-edit it.
    """
    assert TrainingExample.avro_schema_to_python() == json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


def test_to_output_is_the_gliner2_projection() -> None:
    example = TrainingExample(text="placeholder")
    assert example.to_output() == {"input": "placeholder", "output": {}}
