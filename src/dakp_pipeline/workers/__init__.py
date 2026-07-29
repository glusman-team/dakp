"""Go worker runner (stub).

Heavy parsing/extraction workers become native Go CLIs in later milestones; Airflow/CLI
tasks remain thin Python orchestrators that shell out via :func:`run_worker`. Milestone 1
ships only the stub — the ``go/`` tree and Go hooks land later.
"""

from __future__ import annotations
