"""Translator-readiness layer (Milestone 8).

KGX/Translator contract validation, DAKP RIG generation, and legacy-informed regression
guardrails. These modules are import-safe and monkeypatchable:

* :mod:`dakp_pipeline.translator.contract` — assertion-table contract (:func:`~.contract.validate`)
  and full KGX validation (:func:`~.contract.validate_kgx`).
* :mod:`dakp_pipeline.translator.rig` — RIG content + dependency-free YAML
  (:func:`~.rig.generate_rig`, :func:`~.rig.write_rig`).
* :mod:`dakp_pipeline.translator.regression` — family/provenance/label guardrails
  (:func:`~.regression.check_rows`, :func:`~.regression.check_assertion_tables`).
"""

from __future__ import annotations

from dakp_pipeline.translator.contract import ContractProblem, ContractReport, validate, validate_kgx
from dakp_pipeline.translator.regression import RegressionReport, RegressionViolation, check_assertion_tables, check_rows
from dakp_pipeline.translator.rig import generate_rig, rig_text, rig_yaml, write_rig

__all__ = [
    "ContractProblem",
    "ContractReport",
    "RegressionReport",
    "RegressionViolation",
    "check_assertion_tables",
    "check_rows",
    "generate_rig",
    "rig_text",
    "rig_yaml",
    "validate",
    "validate_kgx",
    "write_rig",
]
