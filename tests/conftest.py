"""pytest configuration for DAIL-SQL tests.

Ensures DAIL-SQL's own `runner` package takes precedence over
OpenSearch-SQL's same-named package when both test suites run in
the same pytest process. sys.path and the runner.* module cache are
saved before each test and restored after, so OpenSearch-SQL tests
are not affected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DAIL_SQL_ROOT = str(Path(__file__).resolve().parents[1])


@pytest.fixture(autouse=True)
def _isolate_dail_runner():
    """Pin DAIL-SQL's runner at the front of sys.path for this test only."""
    saved_path = sys.path[:]
    saved_modules = {k: v for k, v in sys.modules.items()
                     if k == "runner" or k.startswith("runner.")}

    # Evict any runner.* that came from a different location
    for key in list(sys.modules.keys()):
        if key == "runner" or key.startswith("runner."):
            del sys.modules[key]

    # DAIL-SQL root first
    if _DAIL_SQL_ROOT in sys.path:
        sys.path.remove(_DAIL_SQL_ROOT)
    sys.path.insert(0, _DAIL_SQL_ROOT)

    yield

    # Restore everything
    sys.path[:] = saved_path
    for key in list(sys.modules.keys()):
        if key == "runner" or key.startswith("runner."):
            del sys.modules[key]
    sys.modules.update(saved_modules)
