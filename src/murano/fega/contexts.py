"""Prompt and token-position contracts for native FEGA runs."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import torch
from torch import Tensor

from murano.artifacts import PromptBatch


PositionSpec: TypeAlias = Literal["last"] | int | Sequence[int]
_OPTIONAL_FIELDS = ("attribute_label", "pair_role", "pair_index", "group_label")


@dataclass(frozen=True)
class FEGAContext:
    """One ordered FEGA prompt with an unpadded target-token position."""

    index: int
    prompt: str
    input_ids: tuple[int, ...]
    target_position: int
    attribute_label: str | None = None
    pair_role: str | None = None
    pair_index: int | None = None
    group_label: str | None = None


@dataclass(frozen=True)
class FEGAContextBatch:
    """Ordered contexts and their prompt source."""

    contexts: tuple[FEGAContext, ...]
    source: str


@dataclass(frozen=True)
class LeftPaddedBatch:
    """One model-ready left-padded FEGA batch and its physical target indices."""

    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    target_positions: tuple[int, ...]
    context_indices: tuple[int, ...]

    def as_tokens(self) -> dict[str, Tensor]:
        """Return the tokenizer-style mapping accepted by Murano model traces."""
        # Keep explicit position IDs so batching cannot change prompt-local positions.
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
        }


def prepare_contexts(
    prompts: PromptBatch,
    tokenizer: Any,
    position: PositionSpec = "last",
) -> FEGAContextBatch:
    """Tokenize prompts independently and attach aligned optional FEGA metadata.

    Independent tokenization freezes each unpadded row before batching. Later
    left padding therefore changes only its physical index, not its token IDs or
    logical target position.
    """
    # Validate the prompt-level metadata before any model-facing work begins.
    if not prompts.prompts:
        raise ValueError("FEGA requires at least one prompt")
    metadata = prompts.metadata.get("fega", {})
    if not isinstance(metadata, dict):
        raise ValueError("PromptBatch.metadata['fega'] must be a mapping")
    aligned = {
        name: _aligned_optional(metadata, name, len(prompts))
        for name in _OPTIONAL_FIELDS
    }

    # Tokenize without padding so target positions remain prompt-relative.
    encoded = tokenizer(
        prompts.prompts,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
    )
    token_rows = cast(list[list[int]], encoded["input_ids"])
    if len(token_rows) != len(prompts):
        raise ValueError("tokenizer returned a different number of prompt rows")
    positions = _resolve_positions(position, [len(row) for row in token_rows])

    # Materialize immutable rows in original prompt order for exact resume behavior.
    contexts = []
    for index, (prompt, token_ids, target) in enumerate(
        zip(prompts.prompts, token_rows, positions, strict=True)
    ):
        if not token_ids:
            raise ValueError(f"prompt {index} tokenized to an empty row")
        contexts.append(
            FEGAContext(
                index=index,
                prompt=prompt,
                input_ids=tuple(int(token) for token in token_ids),
                target_position=target,
                attribute_label=_optional_str(aligned["attribute_label"][index]),
                pair_role=_optional_str(aligned["pair_role"][index]),
                pair_index=_optional_int(aligned["pair_index"][index]),
                group_label=_optional_str(aligned["group_label"][index]),
            )
        )
    return FEGAContextBatch(contexts=tuple(contexts), source=prompts.source)


def left_pad_contexts(
    contexts: Sequence[FEGAContext],
    *,
    pad_token_id: int,
    device: torch.device | str = "cpu",
) -> LeftPaddedBatch:
    """Left-pad contexts and translate logical positions to physical indices."""
    # Build the exact source FEGA padding, mask, and prompt-local position IDs.
    rows = list(contexts)
    if not rows:
        raise ValueError("cannot build a FEGA batch from zero contexts")
    max_length = max(len(context.input_ids) for context in rows)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    targets: list[int] = []
    for context in rows:
        pad_length = max_length - len(context.input_ids)
        input_rows.append([int(pad_token_id)] * pad_length + list(context.input_ids))
        mask_rows.append([0] * pad_length + [1] * len(context.input_ids))
        targets.append(pad_length + context.target_position)
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    # Fail closed if a translated target ever points at padding.
    for row, target in enumerate(targets):
        if int(attention_mask[row, target].item()) != 1:
            raise RuntimeError("translated FEGA target position points at padding")
    return LeftPaddedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        target_positions=tuple(targets),
        context_indices=tuple(context.index for context in rows),
    )


def _resolve_positions(position: PositionSpec, lengths: list[int]) -> list[int]:
    """Normalize supported position forms against unpadded prompt lengths."""
    # Expand one shared selector or validate an aligned per-row selector.
    if position == "last":
        values = [length - 1 for length in lengths]
    elif isinstance(position, int) and not isinstance(position, bool):
        values = [int(position)] * len(lengths)
    elif isinstance(position, Sequence) and not isinstance(position, (str, bytes)):
        values = [int(value) for value in position]
        if len(values) != len(lengths):
            raise ValueError("per-row FEGA positions must match the prompt count")
    else:
        raise ValueError(
            "position must be 'last', a non-negative int, or one int per row"
        )
    for row, (value, length) in enumerate(zip(values, lengths, strict=True)):
        if value < 0 or value >= length:
            raise ValueError(
                f"FEGA position {value} for row {row} is outside token length {length}"
            )
    return values


def _aligned_optional(metadata: dict[str, Any], name: str, count: int) -> list[Any]:
    """Return one optional aligned metadata field with explicit missing values."""
    # Missing optional fields become aligned ``None`` entries; malformed fields fail.
    value = metadata.get(name)
    if value is None:
        return [None] * count
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"FEGA metadata {name!r} must be a sequence")
    values = list(value)
    if len(values) != count:
        raise ValueError(f"FEGA metadata {name!r} must match the prompt count")
    return values


def _optional_str(value: Any) -> str | None:
    """Normalize an optional label without inventing a missing value."""
    # Preserve user labels as strings at the durable context boundary.
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    """Normalize an optional pair index."""
    # Keep missing pair identities explicit while accepting integer-like inputs.
    return None if value is None else int(value)
