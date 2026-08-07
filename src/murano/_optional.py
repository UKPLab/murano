"""Guard helper for optional, use-case-specific dependencies.

Murano's core install carries only what every workflow needs. Feature-specific
libraries (scikit-learn for probing, datasets for benchmark loading, plotting
backends, sae-lens) ship as ``pip install murano-interp[<extra>]`` extras and
are imported lazily at their point of use.

:func:`require_optional` is the single gate every such import passes through, so
a missing dependency always raises the same actionable error naming the extra to
install. Adding a new application means adding one entry to
:data:`EXTRA_IMPORTS` (and the matching ``[project.optional-dependencies]`` group
in ``pyproject.toml``); no new guard code is needed.
"""

from __future__ import annotations

from importlib.util import find_spec

# Maps each extra to the top-level import names it provides. Keys must match the
# extra names in pyproject.toml; values are import names (not pip package names,
# which can differ, e.g. scikit-learn -> sklearn) so they can be probed directly.
EXTRA_IMPORTS: dict[str, tuple[str, ...]] = {
    "probe": ("sklearn",),
    "data": ("datasets",),
    "plot": ("plotly",),
    "sae": ("sae_lens",),
    "notebook": ("ipykernel", "nltk"),
}


def require_optional(extra: str, *modules: str) -> None:
    """Ensure the optional dependencies for a use case are importable.

    Call this before lazily importing a feature-specific library so that an
    absent dependency surfaces as a clear instruction rather than a raw
    ``ImportError`` deep in the call stack.

    Args:
        extra: Name of the extra to check, e.g. ``"probe"``. Must be a key of
            :data:`EXTRA_IMPORTS`.
        *modules: Specific import names to require. Defaults to every import
            name registered for ``extra``. Pass a subset when a function needs
            only part of an extra (e.g. ``require_optional("plot", "plotly")``
            for a Plotly-only helper).

    Raises:
        ValueError: If ``extra`` is not a registered extra.
        RuntimeError: If any required module is not installed. The message names
            the missing modules and the ``pip install`` command that provides
            them.
    """
    if modules:
        names: tuple[str, ...] = modules
    elif extra in EXTRA_IMPORTS:
        names = EXTRA_IMPORTS[extra]
    else:
        raise ValueError(
            f"Unknown extra {extra!r}; expected one of {sorted(EXTRA_IMPORTS)}."
        )
    missing = [name for name in names if find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"The '{extra}' feature needs missing optional "
            f"{'dependency' if len(missing) == 1 else 'dependencies'}: "
            f"{', '.join(missing)}. "
            f"Install with: pip install murano-interp[{extra}]"
        )
