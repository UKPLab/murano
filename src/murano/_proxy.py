"""Resolve nnsight trace handles to their underlying tensors."""

from __future__ import annotations

from torch import Tensor  # pyright: ignore[reportPrivateImportUsage]


def unwrap_traced(saved) -> Tensor:
    """Resolve an nnsight ``.save()`` handle to its underlying tensor.

    Reads ``.value`` when present, then unwraps any tuple nesting (decoder
    layers return ``(hidden_states, ...)``; module inputs may be ``(args, ...)``)
    down to the first tensor.

    Raises:
        TypeError: If the handle does not resolve to a tensor; the module's
            output/input layout is then unsupported.
    """
    val = saved.value if hasattr(saved, "value") else saved
    while isinstance(val, tuple):
        val = val[0]
    if not isinstance(val, Tensor):
        raise TypeError(
            f"Expected a tensor from the traced handle, got {type(val).__name__}; "
            "the module's output/input layout is not supported."
        )
    return val
