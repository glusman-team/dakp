"""Go worker runner integration.

Heavy parsing/extraction workers are native Go CLIs; Airflow/CLI tasks remain thin Python
orchestrators that shell out through :mod:`dakp_pipeline.workers.go_runner`.
"""

from __future__ import annotations
