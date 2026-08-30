"""FEGA reconstruction, ablation, and effect-normalization helpers."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any

import torch
from torch import Tensor, nn


def normalize_delta_rows(
    delta_rows: Tensor,
    gram: Tensor,
    tau_zero: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize raw `ablated - baseline` rows in the Gram metric.

    Args:
        delta_rows: Residual deltas with shape `(n_rows, d_model)`.
        gram: Symmetric Gram matrix with shape `(d_model, d_model)`.
        tau_zero: Maximum Gram magnitude treated as numerically zero.

    Returns:
        A tuple `(directions, magnitudes, mask)`. `directions` contains retained
        Gram-unit rows in original order, `magnitudes` contains their Gram
        magnitudes, and `mask` selects those rows from `delta_rows`.

    Raises:
        ValueError: If tensor shapes are incompatible or `tau_zero` is negative.
        FloatingPointError: If retrying a negative quadratic form in float64 still
            yields a material negative value.
    """
    # Validate the compact raw-cloud boundary before any metric arithmetic.
    if delta_rows.ndim != 2:
        raise ValueError("delta_rows must be rank two")
    if gram.shape != (delta_rows.shape[1], delta_rows.shape[1]):
        raise ValueError("gram must be square with one row per effect dimension")
    if tau_zero < 0:
        raise ValueError("tau_zero must be nonnegative")

    # Match FEGA's compute-dtype path, retrying suspicious negatives in float64.
    effects32 = delta_rows.to(torch.float32)
    compute_dtype = torch.float64 if gram.dtype == torch.float64 else torch.float32
    gram_compute = gram.to(device=gram.device, dtype=compute_dtype)
    delta_compute = effects32.to(device=gram.device, dtype=compute_dtype)
    squared_magnitudes = torch.sum(
        (delta_compute @ gram_compute) * delta_compute, dim=1
    )
    finite_rows = torch.isfinite(effects32).all(dim=1) & torch.isfinite(
        squared_magnitudes
    )
    negative_rows = finite_rows & (squared_magnitudes < 0)
    if bool(negative_rows.any().item()):
        negative_indices = torch.nonzero(negative_rows, as_tuple=False).flatten()
        delta64 = effects32[negative_indices].to(
            device=gram.device, dtype=torch.float64
        )
        gram64 = gram.to(device=gram.device, dtype=torch.float64)
        retried = torch.sum((delta64 @ gram64) * delta64, dim=1)
        if not torch.isfinite(retried).all() or bool((retried < 0).any().item()):
            failing = negative_indices[
                (~torch.isfinite(retried)) | (retried < 0)
            ].tolist()
            raise FloatingPointError(
                f"persistent negative Gram quadratic form for rows {failing}"
            )
        squared_magnitudes = squared_magnitudes.to(torch.float64)
        squared_magnitudes[negative_indices] = retried.to(squared_magnitudes.dtype)

    # Filter only true zero or invalid rows, leaving retained metadata aligned.
    mask = finite_rows & (squared_magnitudes > float(tau_zero) ** 2)
    retained = torch.nonzero(mask, as_tuple=False).flatten()
    if retained.numel() == 0:
        width = int(delta_rows.shape[1])
        return (
            torch.empty((0, width), dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            mask.cpu(),
        )
    retained_magnitudes = torch.sqrt(squared_magnitudes[retained]).to(torch.float32)
    directions = effects32[retained] / retained_magnitudes[:, None]
    finite_directions = torch.isfinite(directions).all(dim=1)
    if not bool(finite_directions.all().item()):
        failing = retained[~finite_directions]
        mask[failing] = False
        directions = directions[finite_directions]
        retained_magnitudes = retained_magnitudes[finite_directions]
    return directions.to(torch.float32), retained_magnitudes, mask.cpu()


def normalize_effect_rows(
    baseline: Tensor,
    ablated: Tensor,
    gram: Tensor,
    tau_zero: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize finite nonzero ``ablated - baseline`` rows in the Gram metric.

    Args:
        baseline: Baseline rows with shape ``(n_rows, d_model)``.
        ablated: Ablated rows with the same shape as ``baseline``.
        gram: Symmetric Gram matrix with shape ``(d_model, d_model)``.
        tau_zero: Maximum Gram magnitude treated as numerically zero.

    Returns:
        A tuple ``(directions, magnitudes, mask)``. ``directions`` contains
        retained Gram-unit rows in original order, ``magnitudes`` contains their
        Gram magnitudes, and ``mask`` selects those rows from the inputs. The
        floating-point outputs are always ``float32``.

    Raises:
        ValueError: If tensor shapes are incompatible or ``tau_zero`` is
            negative.
    """
    # Validate the row and metric shapes before computing ablation effects.
    if baseline.ndim != 2 or ablated.shape != baseline.shape:
        raise ValueError("baseline and ablated must have the same rank-2 shape")
    if gram.shape != (baseline.shape[1], baseline.shape[1]):
        raise ValueError("gram must be square with one row per effect dimension")
    if tau_zero < 0:
        raise ValueError("tau_zero must be nonnegative")

    # Use the same normalization for live effects and saved effect matrices.
    deltas = ablated.to(torch.float32) - baseline.to(torch.float32)
    return normalize_delta_rows(deltas, gram, tau_zero)


def _hidden_tensor(output: Any) -> Tensor:
    """Return the hidden-state tensor carried by a layer output.

    Args:
        output: A tensor, tuple-like output, or mapping-like model output.

    Returns:
        The first hidden-state tensor in the supported output container.

    Raises:
        TypeError: If no supported hidden-state tensor is present.
    """
    # Follow the common tensor, tuple, then ModelOutput/mapping conventions.
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    if isinstance(output, Mapping):
        for value in output.values():
            if isinstance(value, Tensor):
                return value
    raise TypeError("layer output does not contain a supported hidden-state tensor")


def _replace_hidden_tensor(output: Any, hidden: Tensor) -> Any:
    """Replace the hidden-state tensor while preserving its output container.

    Args:
        output: The original tensor, tuple-like output, or mapping-like output.
        hidden: Replacement hidden-state tensor.

    Returns:
        An output with the same supported container convention as ``output``.

    Raises:
        TypeError: If ``output`` has no replaceable hidden-state tensor.
    """
    # Preserve plain and tuple outputs without introducing a wrapper type.
    if isinstance(output, Tensor):
        return hidden
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        values = (hidden, *output[1:])
        return type(output)(*values) if hasattr(output, "_fields") else values

    # Copy mapping/ModelOutput containers and replace their first tensor field.
    if isinstance(output, Mapping):
        replaced: Any = copy(output)
        for key, value in output.items():
            if isinstance(value, Tensor):
                replaced[key] = hidden
                if hasattr(replaced, key):
                    setattr(replaced, key, hidden)
                return replaced
    raise TypeError("layer output does not contain a replaceable hidden-state tensor")


def _position_tensor(
    target_positions: Tensor | Sequence[int],
    batch: int,
    device: torch.device,
) -> Tensor:
    """Convert one target position per batch row to a device-local index tensor.

    Args:
        target_positions: Sequence containing one sequence index per batch row.
        batch: Expected batch size.
        device: Device on which hidden states are stored.

    Returns:
        A rank-one ``torch.long`` tensor of length ``batch``.

    Raises:
        ValueError: If the number of positions does not match the batch size.
    """
    # Materialize positions once so row-wise gather and patch use identical indices.
    positions = torch.as_tensor(target_positions, device=device, dtype=torch.long)
    if positions.shape != (batch,):
        raise ValueError("target_positions must contain one position per batch row")
    return positions


def run_reconstruction_readout_batch(
    raw_model: nn.Module,
    raw_layer: nn.Module,
    output_embedding: nn.Module,
    sae: Any,
    tokens: Mapping[str, Tensor],
    target_positions: Tensor | Sequence[int],
    feature_ids: Tensor | Sequence[int] | None = None,
    z_batch: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor] | Tensor:
    """Run a reconstruction baseline or one-feature-per-row FEGA ablation.

    The layer hook always patches the selected residual rows with ``sae.decode``.
    In baseline mode it first computes ``z = sae.encode(original)``. In ablation
    mode it clones the supplied ``z_batch`` and zeros the selected feature in
    each row. A pre-hook on ``output_embedding`` captures the exact rank-three
    tensor entering that module, before any transformations applied to returned
    logits.

    Args:
        raw_model: Model invoked as ``raw_model(**tokens)``.
        raw_layer: Layer whose selected output rows are reconstructed or ablated.
        output_embedding: Output-head module whose positional input is captured.
        sae: Object exposing callable ``encode`` and ``decode`` methods.
        tokens: Keyword arguments forwarded to ``raw_model``.
        target_positions: One sequence position per batch row.
        feature_ids: Optional one-dimensional feature index per batch row. When
            omitted, the function runs the full-reconstruction baseline.
        z_batch: Baseline SAE activations. Required when ``feature_ids`` is set.

    Returns:
        Baseline mode returns ``(original_rows, z, readout_input)``. Ablation
        mode returns only ``readout_input``. ``readout_input`` is the exact
        rank-three tensor observed by the output-embedding pre-hook.

    Raises:
        ValueError: If required ablation inputs are missing, shapes disagree,
            feature indices are invalid, or the output head is not captured
            exactly once with a rank-three input.
        TypeError: If the layer output or output-head input is unsupported.
    """
    # Reject incomplete ablation requests before installing model hooks.
    baseline_mode = feature_ids is None
    if not baseline_mode and z_batch is None:
        raise ValueError("z_batch is required when feature_ids are provided")

    original_rows: list[Tensor] = []
    encoded_rows: list[Tensor] = []
    readout_inputs: list[Tensor] = []
    layer_calls = 0

    def patch_layer(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        """Patch target residual rows with full or feature-ablated SAE decodes."""
        # Extract target rows and construct the mode-specific latent batch.
        nonlocal layer_calls
        layer_calls += 1
        hidden = _hidden_tensor(output)
        if hidden.ndim != 3:
            raise ValueError("raw_layer hidden states must be rank three")
        batch = hidden.shape[0]
        positions = _position_tensor(target_positions, batch, hidden.device)
        row_indices = torch.arange(batch, device=hidden.device)
        original = hidden[row_indices, positions]

        if baseline_mode:
            z = torch.stack([sae.encode(row) for row in original])
            original_rows.append(original)
            encoded_rows.append(z)
        else:
            assert z_batch is not None
            z = z_batch.to(device=hidden.device).clone()
            features = torch.as_tensor(
                feature_ids, device=hidden.device, dtype=torch.long
            )
            if z.ndim != 2 or z.shape[0] != batch or features.shape != (batch,):
                raise ValueError(
                    "z_batch and feature_ids must contain one row/item per batch row"
                )
            if bool(((features < 0) | (features >= z.shape[1])).any()):
                raise ValueError("feature_ids contains an out-of-range feature index")
            z[row_indices, features] = 0

        # Decode and patch only the selected positions, leaving sibling rows intact.
        reconstructed = (
            torch.stack([sae.decode(row) for row in z])
            if baseline_mode
            else sae.decode(z)
        )
        if reconstructed.shape != original.shape:
            raise ValueError("sae.decode output must match the selected residual rows")
        patched = hidden.clone()
        patched[row_indices, positions] = reconstructed
        return _replace_hidden_tensor(output, patched)

    def capture_readout(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        """Capture the positional tensor entering the output embedding."""
        # Store the actual pre-output tensor before returned-logit transformations.
        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("output_embedding must receive a positional tensor input")
        readout_inputs.append(inputs[0])

    # Install both hooks for one forward and always remove them after execution.
    layer_handle = raw_layer.register_forward_hook(patch_layer)
    readout_handle = output_embedding.register_forward_pre_hook(capture_readout)
    try:
        model_inputs: dict[str, Any] = dict(tokens)
        model_inputs.setdefault("use_cache", False)
        model_inputs.setdefault("output_hidden_states", False)
        raw_model(**model_inputs)
    finally:
        layer_handle.remove()
        readout_handle.remove()

    # Enforce one unambiguous rank-three readout capture and one layer invocation.
    if len(readout_inputs) != 1 or readout_inputs[0].ndim != 3:
        raise ValueError("expected exactly one rank-three output-embedding input")
    if layer_calls != 1:
        raise ValueError("expected raw_layer to run exactly once")
    if baseline_mode:
        return original_rows[0], encoded_rows[0], readout_inputs[0]
    return readout_inputs[0]
