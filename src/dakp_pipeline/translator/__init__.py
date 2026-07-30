"""Translator-readiness layer.

KGX/Translator contract validation and legacy-informed regression
guardrails. These modules are import-safe and monkeypatchable:

* :mod:`dakp_pipeline.translator.contract` — assertion-table contract (:func:`~.contract.validate`)
  and full KGX validation (:func:`~.contract.validate_kgx`).
* :mod:`dakp_pipeline.translator.regression` — family/provenance/label guardrails
  (:func:`~.regression.check_rows`, :func:`~.regression.check_assertion_tables`).
"""

from __future__ import annotations

from dakp_pipeline.translator.contract import ContractProblem, ContractReport, validate, validate_kgx
from dakp_pipeline.translator.regression import RegressionReport, RegressionViolation, check_assertion_tables, check_rows

__all__ = [
    "ContractProblem",
    "ContractReport",
    "RegressionReport",
    "RegressionViolation",
    "check_assertion_tables",
    "check_rows",
    "validate",
    "validate_kgx",
]
