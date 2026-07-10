"""Enforce one shape for the tutorial notebooks.

Fifteen notebooks written by hand drift: one grows a section the outline never
mentions, another claims an extra it does not import, a third re-declares a
dataset the library already ships. Each check below caught a real defect, and
keeps catching it.

The reproduction notebooks predate this template and are exempt.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NOTEBOOKS = [_ROOT / "notebooks" / "getting_started.ipynb"] + sorted(
    (_ROOT / "notebooks" / "applications").glob("*.ipynb")
)
_IDS = [p.name for p in _NOTEBOOKS]

_REQUIRED_HEADINGS = [
    "**Key questions**",
    "**Structure**",
    "**Model and data.**",
    "**Requirements.**",
]

# What in a notebook implies which extra. A step can need an extra without the
# notebook importing the third-party package itself: the Plot step pulls in
# plotly, and the Probe step pulls in scikit-learn. Anything not listed here is
# satisfied by the core install.
_EXTRA_IMPLIED_BY = {
    "plot": [r"\bmurano\.plotting\b", r"\bPlot\("],
    "probe": [r"\bsklearn\b", r"\bProbe\("],
    "sae": [r"\bsae_lens\b", r"\bSAEEncode\(", r"\bSAETopActivations\("],
    "data": [r"\bdatasets\b", r"\.from_hub\("],
}

# Steps whose construction inside a loop means the notebook hand-rolled a sweep.
# `Sweep` exists precisely so a component study is one step rather than a closure
# over the model, the task, and a baseline Results.
_CAUSAL_STEPS = {
    "Ablate",
    "AblateAttention",
    "Intervene",
    "LogitAttribution",
    "LogitLens",
    "Logits",
    "PathPatch",
    "Patch",
    "Probe",
    "Record",
    "RecordAttention",
    "SAEEncode",
    "SteeringVector",
    "WeightAblation",
    "sae_steer",  # builds an Intervene
}


def _cells(path: Path, kind: str) -> list[str]:
    notebook = json.loads(path.read_text())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == kind]


def _code(path: Path) -> str:
    return "\n".join(_cells(path, "code"))


def _outline(header: str) -> list[str]:
    block = header.split("**Structure**", 1)[1].split("**Model and data.**", 1)[0]
    return re.findall(r"^\d+\.\s+(.*?)\s*$", block, re.M)


def _sections(markdown: list[str]) -> list[tuple[int, str]]:
    found = []
    for cell in markdown:
        found += [
            (int(n), title.strip())
            for n, title in re.findall(r"^## (\d+)\.\s+(.*?)\s*$", cell, re.M)
        ]
    return found


def _stated_extras(header: str) -> set[str]:
    match = re.search(r"murano-interp\[([^\]]*)\]", header)
    return set(match.group(1).split(",")) if match else set()


def _trees(path: Path) -> list[ast.Module]:
    """Parse each code cell; skip any that does not stand alone (checked elsewhere)."""
    trees = []
    for cell in _cells(path, "code"):
        try:
            trees.append(ast.parse(cell))
        except SyntaxError:
            continue
    return trees


def _normalized_definitions(path: Path) -> dict[str, str]:
    """Return ``{normalized body: name}`` for every function and class defined.

    The name is stripped and docstrings are blanked before dumping the AST, so two
    definitions that differ only in what they are called collide. Comparing names
    alone let ``patch_head`` and ``recovered_fraction``, the same function twice,
    live in two notebooks unnoticed.
    """
    definitions = {}
    for tree in _trees(path):
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            clone = ast.parse(ast.unparse(node)).body[0]
            clone.name = "_"  # type: ignore[attr-defined]
            for inner in ast.walk(clone):
                if (
                    isinstance(inner, ast.Expr)
                    and isinstance(inner.value, ast.Constant)
                    and isinstance(inner.value.value, str)
                ):
                    inner.value.value = ""
            definitions[ast.dump(clone)] = node.name
    return definitions


def test_every_notebook_is_discovered():
    """Guard the glob: an empty parametrize list would pass vacuously."""
    assert len(_NOTEBOOKS) >= 15, _IDS


# ── Template ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_header_carries_every_required_field(path):
    header = _cells(path, "markdown")[0]

    assert header.startswith("# "), "the first cell must open with the title"
    for heading in _REQUIRED_HEADINGS:
        assert heading in header, f"missing {heading}"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_sections_match_the_outline(path):
    """The header promises an outline; the sections must deliver exactly it."""
    markdown = _cells(path, "markdown")
    outline = _outline(markdown[0])
    sections = _sections(markdown)

    assert outline, "the Structure block lists no sections"
    assert [n for n, _ in sections] == list(range(1, len(outline) + 1)), (
        "sections are missing, duplicated, or out of order"
    )
    assert [title for _, title in sections] == outline


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_closes_with_what_next(path):
    assert _cells(path, "markdown")[-1].startswith("## What next")


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_what_next_links_resolve(path):
    last = _cells(path, "markdown")[-1]
    targets = re.findall(r"\]\((?!https?://)([^)]+\.ipynb)\)", last)

    assert targets, "What next links to no notebook"
    for target in targets:
        assert (path.parent / target).exists(), f"broken link to {target}"


# ── Claims match reality ──────────────────────────────────────────────


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_declared_extras_match_the_imports(path):
    """A notebook must name every extra it needs, and none that it does not."""
    header = _cells(path, "markdown")[0]
    code = _code(path)

    needed = {
        extra
        for extra, patterns in _EXTRA_IMPLIED_BY.items()
        if any(re.search(pattern, code) for pattern in patterns)
    }
    assert _stated_extras(header) == needed, (
        f"header states {_stated_extras(header) or 'no extras'}, the notebook "
        f"needs {needed or 'no extras'}"
    )


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_every_code_cell_parses(path):
    """Jupyter executes cells independently, so each must be valid on its own."""
    for index, cell in enumerate(_cells(path, "code")):
        try:
            ast.parse(cell)
        except SyntaxError as exc:
            pytest.fail(f"{path.name} code cell {index}: {exc}")


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_writes_to_its_own_output_directory(path):
    code = _code(path)
    match = re.search(r'^OUTPUT_DIR = "murano_outputs/([^"]+)"', code, re.M)

    assert match, "no OUTPUT_DIR declared"
    assert match.group(1) == path.stem, "OUTPUT_DIR must be named after the notebook"


# ── No duplicated code across notebooks ───────────────────────────────


def test_no_function_or_class_is_defined_in_two_notebooks():
    """Shared helpers belong in the library, not copy-pasted between tutorials."""
    owners: dict[str, list[str]] = {}
    for path in _NOTEBOOKS:
        for name in re.findall(r"^(?:def|class)\s+(\w+)", _code(path), re.M):
            owners.setdefault(name, []).append(path.name)

    duplicated = {name: files for name, files in owners.items() if len(files) > 1}
    assert not duplicated, f"define these once, in murano: {duplicated}"


def test_no_two_notebooks_define_the_same_body_under_different_names():
    """Renaming a copy-paste does not make it not a copy-paste."""
    owners: dict[str, list[str]] = {}
    for path in _NOTEBOOKS:
        for body, name in _normalized_definitions(path).items():
            owners.setdefault(body, []).append(f"{path.name}:{name}")

    duplicated = [names for names in owners.values() if len(names) > 1]
    assert not duplicated, f"these are the same definition twice: {duplicated}"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_no_causal_step_is_constructed_inside_a_loop(path):
    """A step built in a `for` body is a hand-rolled sweep; use the Sweep step."""
    offenders = set()
    for tree in _trees(path):
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id in _CAUSAL_STEPS
                ):
                    offenders.add(inner.func.id)

    assert not offenders, (
        f"{sorted(offenders)} constructed inside a for-loop; sweep with `Sweep`, "
        f"whose steps= callback builds the chain per item"
    )


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_IDS)
def test_no_pipeline_is_built_inside_a_function(path):
    """Pipelines are read at cell level; only Sweep's steps= callback builds steps."""
    offenders = set()
    for tree in _trees(path):
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "Pipeline"
                ):
                    offenders.add(node.name)

    assert not offenders, (
        f"{sorted(offenders)} hide a Pipeline; build it at cell level so a reader "
        f"sees the steps, and sweep with `Sweep`"
    )


def test_notebooks_do_not_re_declare_what_murano_tasks_provides():
    """The IOI prompts and the sentiment vocabulary have exactly one home."""
    forbidden = {
        "When {a} and {b}": "murano.tasks.IOI_TEMPLATE",
        '("Mary", "John")': "murano.tasks.ioi",
        "POSITIVE_WORDS = {": "murano.tasks.POSITIVE_WORDS",
    }
    for path in _NOTEBOOKS:
        code = _code(path)
        for snippet, home in forbidden.items():
            assert snippet not in code, f"{path.name} re-declares {home}"


def test_output_directories_are_unique():
    """Two notebooks writing to one directory would clobber each other's figures."""
    dirs = [
        re.search(r'^OUTPUT_DIR = "([^"]+)"', _code(path), re.M).group(1)
        for path in _NOTEBOOKS
    ]
    assert len(set(dirs)) == len(dirs), dirs


def test_the_docs_generator_lists_every_notebook():
    """A new notebook missing from ORDER would only fail at docs-deploy time."""
    order = re.findall(
        r'^    "([^"]+\.ipynb)",',
        (_ROOT / "docs" / "scripts" / "gen_notebook_docs.py").read_text(),
        re.M,
    )
    on_disk = [path.relative_to(_ROOT / "notebooks").as_posix() for path in _NOTEBOOKS]

    assert sorted(order) == sorted(on_disk), (
        "docs/scripts/gen_notebook_docs.py ORDER is out of sync with notebooks/"
    )
