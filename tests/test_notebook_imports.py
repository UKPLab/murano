"""Check that every notebook resolves its imports.

CI runners have no GPU and no model weights, so the notebooks cannot run end to
end. What a refactor actually breaks is cheaper to catch: a renamed, moved, or
removed public symbol that leaves a notebook importing something that no longer
exists.

Each notebook is parsed and every ``murano`` import in it is resolved against the
installed package. No model, network, or GPU is touched, so the check runs in the
ordinary pytest matrix. Discovery is glob-based, so a new notebook is covered the
moment it lands, without editing this file.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
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


def _discover_notebooks() -> list[tuple[str, list[str]]]:
    return [
        (str(path.relative_to(_ROOT)), _notebook_code_units(path))
        for path in sorted(_ROOT.glob("notebooks/**/*.ipynb"))
    ]


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


_NOTEBOOKS = _discover_notebooks()


def test_discovery_finds_the_notebooks() -> None:
    """Guard the glob: an empty parametrize list would pass vacuously."""
    assert _NOTEBOOKS, f"no notebooks discovered under {_ROOT / 'notebooks'}"


@pytest.mark.parametrize("readme", ["README.md", "notebooks/README.md"])
def test_readme_notebook_links_resolve(readme: str) -> None:
    """A renamed notebook must fail here rather than 404 on GitHub.

    Notebook discovery is glob-based, so removing a notebook silently drops its
    parametrized case instead of failing. The READMEs name notebooks by path, and
    nothing else checks those paths still exist.
    """
    path = _ROOT / readme
    targets = re.findall(r"\]\((?!https?://|#)([^)]+)\)", path.read_text())
    linked = [t for t in targets if t.endswith(".ipynb") or t.endswith("/")]
    assert linked, f"{readme} links no notebooks; did the link format change?"
    for target in linked:
        assert (path.parent / target).exists(), f"{readme} links missing {target}"


@pytest.mark.parametrize(
    "name, units", _NOTEBOOKS, ids=[name for name, _ in _NOTEBOOKS]
)
def test_notebook_imports_resolve(name: str, units: list[str]) -> None:
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
