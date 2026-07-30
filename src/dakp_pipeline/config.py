"""Pipeline profiles and configuration.

Profiles are defined in Python (not parsed from YAML) so the base install needs no
``pyyaml`` dependency. ``configs/pipeline.yaml`` and ``configs/local.example.yaml``
ship as human-readable declarations of the same knobs.

Three profiles (per ``PLAN.md`` "Resolved planning decisions" + performance tiers):

* ``mock``   — tiny fixtures, all external calls monkeypatchable, CI-friendly.
* ``sample`` — laptop-safe bounded sample, real acquisition but limited scope.
* ``prod``   — real full build on the 80-thread / 187 GiB workstation (the ``wenceslaus`` host).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PROFILES: frozenset[str] = frozenset({"mock", "sample", "prod"})


class Profile(BaseModel):
    """Execution profile: concurrency budgets and source-acquisition behavior."""

    name: str = Field(description="Profile name (mock | sample | prod).")
    threads: int = Field(ge=1, description="Worker threads/processes per task.")
    memory_budget_gb: int = Field(ge=1, description="Soft per-task memory budget in GiB.")
    quarter_limit: int | None = Field(default=None, description="Cap FAERS quarters processed (dev/sample); None = all available.")
    release_limit: int | None = Field(default=None, description="Cap DailyMed full releases processed (dev/sample/smoke); None = all available.")
    mock_sources: bool = Field(description="If True, fetchers load fixtures instead of the network (mock profile).")
    force: bool = Field(default=False, description="Ignore cached artifacts and rerun every stage.")
    run_tablassert: bool = Field(default=False, description="Invoke real ../Tablassert at the handoff stage (deferred in mock profile).")
    use_go_workers: bool = Field(
        default=False,
        description=(
            "Delegate the heavy extractors (DailyMed/FAERS/Drugs@FDA) to the compiled Go "
            "dakp-worker binary when one is available (Go worker optimization). Defaults to "
            "False so mock/sample/CI keep using the pure-Python extractors; enable for prod "
            "full builds via load_profile('prod', use_go_workers=True) once the binary is "
            "deployed. The extractors fall back to Python automatically when Go is unavailable "
            "(see workers/go_runner.go_available)."
        ),
    )


# use_go_workers stays False for every shipped profile so the Python extractors remain the
# default (and the test suite never shells out to Go); prod opts in explicitly via override.
_MOCK = Profile(
    name="mock", threads=1, memory_budget_gb=1, quarter_limit=1, mock_sources=True, force=False, run_tablassert=False, use_go_workers=False
)

_SAMPLE = Profile(
    name="sample", threads=4, memory_budget_gb=8, quarter_limit=1, mock_sources=False, force=False, run_tablassert=False, use_go_workers=False
)

_PROD = Profile(
    name="prod", threads=64, memory_budget_gb=128, quarter_limit=None, mock_sources=False, force=False, run_tablassert=True, use_go_workers=False
)

_PROFILE_TABLE: dict[str, Profile] = {"mock": _MOCK, "sample": _SAMPLE, "prod": _PROD}


def load_profile(name: str, **overrides: object) -> Profile:
    """Return the named profile with optional field overrides applied.

    Raises ``KeyError`` for an unknown profile name rather than silently defaulting,
    so typos fail loudly at startup.
    """
    if name not in _PROFILE_TABLE:
        known = ", ".join(sorted(PROFILES))
        msg = f"Unknown profile {name!r}; expected one of: {known}"
        raise KeyError(msg)
    base = _PROFILE_TABLE[name]
    if not overrides:
        return base
    data = base.model_dump()
    data.update(overrides)
    return Profile(**data)


__all__ = ["PROFILES", "Profile", "load_profile"]
