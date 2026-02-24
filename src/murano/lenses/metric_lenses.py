"""
Metric Computation Lenses for the Murano interpretability pipeline.

These are BaseComputationLenses that enrich the shared artifact dictionary
with quantitative evaluation metrics derived from model outputs.

Expected artifact keys (inputs / outputs) are documented per-class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any, Dict, Literal

from .base_lens import BaseComputationLens


# ---------------------------------------------------------------------------
# CrossEntropyLossLens
# ---------------------------------------------------------------------------


class CrossEntropyLossLens(BaseComputationLens):
    """
    Computes cross-entropy loss between the model's predicted logits and the
    ground-truth token IDs.

    Artifact inputs
    ---------------
    ``final_logits`` : torch.Tensor, shape ``[batch, seq_len, vocab_size]``
        Raw (un-softmaxed) logit scores from the language model head.
    ``target_ids`` : torch.Tensor, shape ``[batch, seq_len]``
        Ground-truth token IDs (typically ``input_ids`` shifted left by one).

    Artifact outputs
    ----------------
    ``loss`` : torch.Tensor
        * ``reduction='mean'`` → scalar (0-dim tensor).
        * ``reduction='none'`` → per-token tensor of shape ``[batch, seq_len]``.

    Parameters
    ----------
    reduction : {'mean', 'sum', 'none'}
        Passed directly to ``torch.nn.functional.cross_entropy``.
        Default is ``'mean'``.
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__(name="CrossEntropyLossLens")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. Choose from 'mean', 'sum', 'none'."
            )
        self.reduction = reduction

    def process(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute cross-entropy loss and store the result in the artifact.

        Args:
            artifact: Pipeline artifact dict. Must contain ``final_logits``
                      and ``target_ids``.

        Returns:
            The same artifact dict enriched with a ``loss`` key.
        """
        logits: torch.Tensor = artifact["final_logits"]  # [B, S, V]
        targets: torch.Tensor = artifact["target_ids"]  # [B, S]

        B, S, V = logits.shape

        # F.cross_entropy expects (N, C) logits and (N,) targets
        flat_logits = logits.reshape(B * S, V)
        flat_targets = targets.reshape(B * S)

        if self.reduction == "none":
            # Keep per-token losses and reshape back to [B, S]
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            loss = loss.reshape(B, S)
        else:
            loss = F.cross_entropy(flat_logits, flat_targets, reduction=self.reduction)

        # Immutability guarantee: we only ADD a new key
        artifact["loss"] = loss
        return artifact


# ---------------------------------------------------------------------------
# AccuracyLens
# ---------------------------------------------------------------------------


class AccuracyLens(BaseComputationLens):
    """
    Computes token-level prediction accuracy.

    Artifact inputs
    ---------------
    ``final_logits`` : torch.Tensor, shape ``[batch, seq_len, vocab_size]``
    ``target_ids``   : torch.Tensor, shape ``[batch, seq_len]``

    Artifact outputs
    ----------------
    ``accuracy`` : float
        Fraction of tokens for which ``argmax(logits) == target_id``.
    """

    def __init__(self) -> None:
        super().__init__(name="AccuracyLens")

    def process(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute token-level accuracy and store the result in the artifact.

        Args:
            artifact: Pipeline artifact dict. Must contain ``final_logits``
                      and ``target_ids``.

        Returns:
            The same artifact dict enriched with an ``accuracy`` key (float).
        """
        logits: torch.Tensor = artifact["final_logits"]  # [B, S, V]
        targets: torch.Tensor = artifact["target_ids"]  # [B, S]

        predicted_tokens = logits.argmax(dim=-1)  # [B, S]
        accuracy = (predicted_tokens == targets).float().mean().item()

        artifact["accuracy"] = accuracy
        return artifact


# ---------------------------------------------------------------------------
# ComparisonComputationLens
# ---------------------------------------------------------------------------


class ComparisonComputationLens(BaseComputationLens):
    """
    Compares two tensors stored in the artifact and stores the result.

    Useful for comparing activations from a clean vs. corrupted run, or for
    any pairwise tensor comparison needed in an interpretability experiment.

    Artifact inputs
    ---------------
    The keys supplied via ``key_a`` and ``key_b`` at construction time.
    Both values must be ``torch.Tensor`` instances of compatible shapes.

    Artifact outputs
    ----------------
    The tensor stored under ``output_key``.

    Parameters
    ----------
    key_a : str
        Artifact key for the first tensor (e.g. ``'clean_acts'``).
    key_b : str
        Artifact key for the second tensor (e.g. ``'corrupt_acts'``).
    output_key : str
        Artifact key under which the result is stored.
    comparison_type : {'difference', 'cosine_similarity'}
        * ``'difference'``         – element-wise subtraction ``A - B``.
        * ``'cosine_similarity'``  – row-wise cosine similarity, yielding
          a 1-D tensor of length ``A.shape[0]``.

    Raises
    ------
    ValueError
        If an unsupported ``comparison_type`` is supplied at process time.
    """

    SUPPORTED_TYPES = frozenset({"difference", "cosine_similarity"})

    def __init__(
        self,
        key_a: str,
        key_b: str,
        output_key: str,
        comparison_type: str = "difference",
    ) -> None:
        super().__init__(name="ComparisonComputationLens")
        self.key_a = key_a
        self.key_b = key_b
        self.output_key = output_key
        self.comparison_type = comparison_type

    def process(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve the two tensors, compute the requested comparison, and store
        the result in the artifact without overwriting the source tensors.

        Args:
            artifact: Pipeline artifact dict. Must contain ``key_a`` and ``key_b``.

        Returns:
            The same artifact dict enriched with ``output_key``.

        Raises:
            ValueError: On unsupported ``comparison_type``.
        """
        tensor_a: torch.Tensor = artifact[self.key_a]
        tensor_b: torch.Tensor = artifact[self.key_b]

        if self.comparison_type == "difference":
            result = tensor_a - tensor_b

        elif self.comparison_type == "cosine_similarity":
            # Row-wise: flatten to 2-D first for general support
            a_2d = tensor_a.reshape(tensor_a.shape[0], -1)
            b_2d = tensor_b.reshape(tensor_b.shape[0], -1)
            result = F.cosine_similarity(a_2d, b_2d, dim=1)

        else:
            raise ValueError(
                f"Unsupported comparison_type '{self.comparison_type}'. "
                f"Choose from: {sorted(self.SUPPORTED_TYPES)}"
            )

        # Immutability: only add the new key
        artifact[self.output_key] = result
        return artifact
