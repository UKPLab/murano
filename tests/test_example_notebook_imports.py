"""Check that every example script and notebook resolves its imports.

CI runners have no GPU and no model weights, so the example scripts and the
notebooks cannot run end to end. What a refactor actually breaks is cheaper to
catch: a renamed, moved, or removed public symbol that leaves an example or
notebook importing something that no longer exists.

Each file is parsed and every ``murano`` import in it is resolved against the
installed package. No model, network, or GPU is touched, so the check runs in
the ordinary pytest matrix. Discovery is glob-based, so a new example or
notebook is covered the moment it lands, without editing this file.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _notebook_code_units(path: Path) -> list[str]:
    """Return each code cell of a notebook as a parseable source string.

    Jupyter cell magics (``%%``) and line magics / shell escapes (``%``, ``!``)
    are not valid Python, so they are dropped. Cells are kept separate because
    Jupyter executes them independently.
    """
    notebook = json.loads(path.read_text())
    units: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith("%%"):
            continue
        lines = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        unit = "\n".join(lines).strip()
        if unit:
            units.append(unit)
    return units


def _discover_artifacts() -> list[tuple[str, list[str]]]:
    artifacts: list[tuple[str, list[str]]] = []
    for path in sorted((_ROOT / "examples").glob("*.py")):
        artifacts.append((path.name, [path.read_text()]))
    for pattern in ("notebooks/**/*.ipynb", "tutorials/**/*.ipynb"):
        for path in sorted(_ROOT.glob(pattern)):
            artifacts.append((str(path.relative_to(_ROOT)), _notebook_code_units(path)))
    return artifacts


def _murano_imports(tree: ast.Module):
    """Yield ``(module, name)`` for every ``murano`` import in a parsed unit.

    ``name`` is ``None`` for a plain ``import murano`` / ``import murano.sub``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "murano" or module.startswith("murano."):
                for alias in node.names:
                    yield module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "murano" or alias.name.startswith("murano."):
                    yield alias.name, None


def _resolve(module: str, name: str | None) -> None:
    imported = importlib.import_module(module)
    if name is None or name == "*":
        return
    if hasattr(imported, name):
        return
    # A submodule is importable without being an attribute of its parent until
    # first imported (e.g. ``from murano.plotting import attention``).
    importlib.import_module(f"{module}.{name}")


_ARTIFACTS = _discover_artifacts()


@pytest.mark.parametrize(
    "name, units", _ARTIFACTS, ids=[name for name, _ in _ARTIFACTS]
)
def test_example_and_notebook_imports_resolve(name: str, units: list[str]) -> None:
    for unit in units:
        try:
            tree = ast.parse(unit, filename=name)
        except SyntaxError as exc:
            pytest.fail(f"{name}: syntax error ({exc})")
        for module, symbol in _murano_imports(tree):
            try:
                _resolve(module, symbol)
            except (ImportError, AttributeError) as exc:
                target = module if symbol is None else f"{module}.{symbol}"
                pytest.fail(f"{name}: unresolved import `{target}` ({exc})")
