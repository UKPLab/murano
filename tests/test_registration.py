"""Registration guards: the parts of AGENTS.md section 8 a new method most often forgets.

Every Step must be exported from ``murano.steps``, every public module must be
listed in the API reference, and every extra in ``pyproject.toml`` must be
registered with ``require_optional``. These lists are hand-maintained, so they
drift silently unless a test pins them.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from importlib.metadata import metadata
from pathlib import Path

import murano
import murano.plotting
import murano.steps
from murano._optional import EXTRA_IMPORTS
from murano.steps.base import Step

DOCS_SCRIPTS = Path(__file__).parent.parent / "docs" / "scripts"


def _public_modules(package) -> list[str]:
    return [
        f"{package.__name__}.{info.name}"
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_") and not info.ispkg
    ]


def test_every_step_is_exported_from_murano_steps():
    defined = {
        cls.__name__
        for module_name in _public_modules(murano.steps)
        for _, cls in inspect.getmembers(
            importlib.import_module(module_name), inspect.isclass
        )
        if issubclass(cls, Step)
        and cls.__module__ == module_name
        and not cls.__name__.startswith("_")
    }
    assert defined - set(murano.steps.__all__) == set()


def test_every_public_module_is_in_the_api_reference():
    sys.path.insert(0, str(DOCS_SCRIPTS))
    try:
        gen_api_docs = importlib.import_module("gen_api_docs")
    finally:
        sys.path.remove(str(DOCS_SCRIPTS))
    documented = {module for module, _ in gen_api_docs.MODULES}
    public = (
        set(_public_modules(murano))
        | set(_public_modules(murano.steps))
        | set(_public_modules(murano.plotting))
    )
    assert public - documented == set()


def test_every_extra_is_registered_with_require_optional():
    extras = set(metadata("murano-interp").get_all("Provides-Extra") or []) - {"all"}
    assert extras == set(EXTRA_IMPORTS)
