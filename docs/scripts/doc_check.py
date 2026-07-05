"""Analyze docstring quality across the murano codebase using griffe.

Run from the repo root:
    python docs/scripts/doc_check.py

Outputs a report to stdout and optionally to docs/doc_report.md.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import griffe

REPO_ROOT = Path(__file__).parent.parent.parent
SRC = REPO_ROOT / "src"
REPORT_PATH = REPO_ROOT / "docs" / "doc_report.md"

# All public modules to check
MODULES = [
    "murano.model",
    "murano.pipeline",
    "murano.dataset",
    "murano.artifacts",
    "murano.results",
    "murano.evaluation",
    "murano.io",
    "murano.steps.base",
    "murano.steps.record",
    "murano.steps.intervene",
    "murano.steps.train",
    "murano.steps.probe",
    "murano.steps.evaluate",
    "murano.steps.load",
    "murano.steps.save",
    "murano.steps.paired",
    "murano.steps.logits",
    "murano.steps.ablate",
    "murano.steps.patch",
    "murano.steps.path_patch",
    "murano.steps.logit_attribution",
]

# Skip these names everywhere
SKIP_NAMES = {
    "__repr__",
    "__str__",
    "__len__",
    "__contains__",
    "__getitem__",
    "__setitem__",
    "__hash__",
    "__eq__",
}


@dataclass
class Issue:
    module: str
    object_name: str
    kind: str  # "error" | "warning" | "info"
    message: str


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)

    def add(self, module: str, name: str, kind: str, msg: str):
        self.issues.append(Issue(module, name, kind, msg))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.kind == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.kind == "warning"]


SECTION_HEADERS = (
    "Args:",
    "Arguments:",
    "Parameters:",
    "Returns:",
    "Return:",
    "Yields:",
    "Raises:",
    "Attributes:",
    "Example:",
    "Examples:",
    "Note:",
    "Warning:",
)


def _has_section(docstring: str, header: str) -> bool:
    for line in docstring.splitlines():
        if line.strip().lower().startswith(header.lower()):
            return True
    return False


def _has_any_section(docstring: str) -> bool:
    """Return True if the docstring uses any standard Napoleon section header."""
    return any(_has_section(docstring, h) for h in SECTION_HEADERS)


def _non_self_params(func: griffe.Function) -> list[griffe.Parameter]:
    return [p for p in func.parameters if p.name not in ("self", "cls")]


def check_function(
    func: griffe.Function, module: str, report: Report, parent_cls: str | None = None
):
    full_name = f"{parent_cls}.{func.name}" if parent_cls else func.name

    if func.name.startswith("_") and func.name != "__init__":
        return
    if func.name in SKIP_NAMES:
        return

    doc = func.docstring.value.strip() if func.docstring else ""

    if not doc:
        if func.name == "__init__":
            # __init__ can rely on class docstring for Args
            return
        report.add(module, full_name, "error", "Missing docstring")
        return

    # Summary-only docstrings (no Napoleon sections) are typical for override
    # one-liners and trivial getters; trust the author and skip Args/Returns
    # warnings. Once the author writes any section, they're being thorough,
    # so missing Args/Returns is genuinely a gap.
    if not _has_any_section(doc):
        return

    params = _non_self_params(func)
    if len(params) >= 2 and not _has_section(doc, "Args:"):
        report.add(
            module,
            full_name,
            "warning",
            f"Has {len(params)} parameters but no Args section",
        )

    if func.returns and str(func.returns) not in ("None", ""):
        if not _has_section(doc, "Returns:") and not _has_section(doc, "Return:"):
            report.add(
                module, full_name, "warning", "Has return type but no Returns section"
            )


def check_class(cls: griffe.Class, module: str, report: Report):
    if cls.name.startswith("_"):
        return

    doc = cls.docstring.value.strip() if cls.docstring else ""

    if not doc:
        report.add(module, cls.name, "error", "Missing class docstring")
        return

    # Check if __init__ has params that should be documented somewhere
    init = cls.members.get("__init__")
    if init and isinstance(init, griffe.Function):
        params = _non_self_params(init)
        if len(params) >= 2:
            has_args = (
                _has_section(doc, "Args:")
                or _has_section(doc, "Parameters:")
                or _has_section(doc, "Attributes:")
            )
            init_doc = init.docstring.value.strip() if init.docstring else ""
            has_init_args = _has_section(init_doc, "Args:") if init_doc else False
            if not has_args and not has_init_args:
                report.add(
                    module,
                    cls.name,
                    "warning",
                    f"__init__ has {len(params)} parameters but neither class nor __init__ docstring documents them",
                )

    # Check public methods
    for name, member in cls.members.items():
        if isinstance(member, griffe.Function):
            check_function(member, module, report, parent_cls=cls.name)


def check_module(module_path: str, report: Report):
    loader = griffe.GriffeLoader(search_paths=[str(SRC)])
    try:
        module = loader.load(module_path)
    except Exception as e:
        report.add(module_path, module_path, "error", f"Failed to load: {e}")
        return

    doc = module.docstring.value.strip() if module.docstring else ""
    if not doc:
        report.add(module_path, module_path, "info", "Missing module docstring")

    for name, member in module.members.items():
        if name.startswith("_"):
            continue
        if isinstance(member, griffe.Class):
            check_class(member, module_path, report)
        elif isinstance(member, griffe.Function):
            check_function(member, module_path, report)


def format_report(report: Report) -> str:
    lines = ["# Docstring Quality Report", ""]

    errors = report.errors
    warnings = report.warnings

    lines.append(
        f"**{len(errors)} errors, {len(warnings)} warnings** across {len(MODULES)} modules"
    )
    lines.append("")

    if errors:
        lines.append("## Errors (missing docstrings)")
        lines.append("")
        lines.append("| Module | Name | Issue |")
        lines.append("|--------|------|-------|")
        for i in errors:
            lines.append(f"| `{i.module}` | `{i.object_name}` | {i.message} |")
        lines.append("")

    if warnings:
        lines.append("## Warnings (incomplete docstrings)")
        lines.append("")
        lines.append("| Module | Name | Issue |")
        lines.append("|--------|------|-------|")
        for i in warnings:
            lines.append(f"| `{i.module}` | `{i.object_name}` | {i.message} |")
        lines.append("")

    if not errors and not warnings:
        lines.append("All public APIs are documented!")

    return "\n".join(lines)


def run() -> Report:
    report = Report()
    for module_path in MODULES:
        check_module(module_path, report)
    return report


def main():
    report = run()
    text = format_report(report)
    print(text)

    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH.relative_to(REPO_ROOT)}")

    if report.errors:
        print(f"\n{len(report.errors)} error(s) found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
