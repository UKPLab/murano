"""Generate API reference Markdown pages from Python docstrings using griffe.

Run from the repo root:
    python docs/scripts/gen_api_docs.py

Outputs Markdown files to docs/src/content/docs/reference/.
"""

from __future__ import annotations

from pathlib import Path

import griffe

REPO_ROOT = Path(__file__).parent.parent.parent
SRC = REPO_ROOT / "src"
OUT = REPO_ROOT / "docs" / "src" / "content" / "docs" / "reference"

# Modules to document: (import path, output slug)
MODULES = [
    ("murano.model", "model"),
    ("murano.pipeline", "pipeline"),
    ("murano.dataset", "dataset"),
    ("murano.artifacts", "artifacts"),
    ("murano.results", "results"),
    ("murano.steps.record", "steps/record"),
    ("murano.steps.intervene", "steps/intervene"),
    ("murano.steps.train", "steps/train"),
    ("murano.steps.probe", "steps/probe"),
    ("murano.steps.evaluate", "steps/evaluate"),
    ("murano.steps.load", "steps/load"),
    ("murano.steps.save", "steps/save"),
]


def _docstring(obj: griffe.Object) -> str:
    if obj.docstring is None:
        return ""
    return obj.docstring.value.strip()


def _signature(func: griffe.Function) -> str:
    params = []
    for p in func.parameters:
        part = p.name
        if p.annotation:
            part += f": {p.annotation}"
        if p.default:
            part += f" = {p.default}"
        params.append(part)
    ret = f" -> {func.returns}" if func.returns else ""
    return f"({', '.join(params)}){ret}"


def render_function(func: griffe.Function, heading: int = 3) -> str:
    h = "#" * heading
    doc = _docstring(func)
    sig = _signature(func)
    lines = [f"{h} `{func.name}{sig}`", ""]
    if doc:
        lines += [doc, ""]
    return "\n".join(lines)


def render_class(cls: griffe.Class) -> str:
    doc = _docstring(cls)
    lines = [f"## `{cls.name}`", ""]
    if doc:
        lines += [doc, ""]

    for name, member in cls.members.items():
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(member, griffe.Function):
            lines.append(render_function(member, heading=3))

    return "\n".join(lines)


def render_module(module_path: str) -> str:
    loader = griffe.GriffeLoader(search_paths=[str(SRC)])
    module = loader.load(module_path)

    lines = []
    doc = _docstring(module)
    if doc:
        lines += [doc, ""]

    for name, member in module.members.items():
        if name.startswith("_"):
            continue
        if isinstance(member, griffe.Class):
            lines.append(render_class(member))
        elif isinstance(member, griffe.Function):
            lines.append(render_function(member, heading=2))

    return "\n".join(lines)


def slug_to_title(slug: str) -> str:
    name = slug.split("/")[-1].replace("_", " ").title()
    prefix = {"Steps/": "Steps — "}.get(slug.rsplit("/", 1)[0].title() + "/", "")
    return prefix + name


def generate():
    OUT.mkdir(parents=True, exist_ok=True)

    for module_path, slug in MODULES:
        out_file = OUT / f"{slug}.md"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        title = slug_to_title(slug)
        try:
            body = render_module(module_path)
        except Exception as e:
            body = f":::caution[Generation error]\n{e}\n:::"

        content = f"---\ntitle: {title}\ndescription: API reference for {module_path}\n---\n\n{body}\n"
        out_file.write_text(content)
        print(f"  wrote {out_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    generate()