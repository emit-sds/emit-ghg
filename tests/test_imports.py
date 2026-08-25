"""
Import smoke-tests for emit_ghg modules.

Only intent is for basic syntax breaks.  DAAC delivery not includes,
as there are external dependencies only for the SDS.
"""

import importlib

import pytest

MODULES = [
    "apply_glt",
    "cli",
    "diffmf",
    "files",
    "scale",
    "target_generation",
    "utils",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    """Each module imports without syntax or import errors."""
    importlib.import_module(f"emit_ghg.{mod}")
