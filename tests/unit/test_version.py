"""Pin :data:`dakp_pipeline.__version__` to ``project.version`` in ``pyproject.toml``.

The two drifted once (1.3.2 bumped ``pyproject.toml`` only), and nothing caught it: every other
test spells the version symbolically as ``__version__``, so the whole suite stayed green while a
1.3.2 build published ``drug_approvals_kg_*_v1.3.1`` artifacts. This test is the one place that
compares the literal to the packaging metadata.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from dakp_pipeline import __version__

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    declared = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert __version__ == declared, f"dakp_pipeline.__version__ ({__version__}) != pyproject project.version ({declared})"
