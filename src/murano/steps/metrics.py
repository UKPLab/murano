"""Metric computation steps for Murano pipelines.

Converts the legacy ``BaseComputationLens`` classes into ``Step`` subclasses
that operate on ``Results`` objects.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from murano import keys
from murano.results import Results
from murano.steps.base import Step


class CrossEntropyLossStep(Step):
    """Compute cross-entropy loss from logits and target IDs.

    Reads ``logits_key`` and ``targets_key`` from results, computes the loss,
    and writes the result under ``output_key``.

    Args:
        logits_key: Key in results containing the logits tensor ``[B, S, V]``.
        targets_key: Key in results containing the target IDs ``[B, S]``.
        output_key: Key under which to store the computed loss.
        reduction: Loss reduction — ``"mean"``, ``"sum"``, or ``"none"``.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        logits_key: str = keys.FINAL_LOGITS,
        targets_key: str = keys.TARGET_IDS,
        output_key: str = keys.LOSS,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        self.logits_key = logits_key
        self.targets_key = targets_key
        self.output_key = output_key
        self.reduction = reduction
        self.reads = [logits_key, targets_key]
        self.writes = [output_key]

    def __call__(self, results: Results) -> Results:
        logits: torch.Tensor = results[self.logits_key]
        targets: torch.Tensor = results[self.targets_key]

        # F.cross_entropy ignores the -100 targets that mark padded and
        # last-position slots, but a mean over an all-ignored batch divides by
        # zero and returns NaN; report a 0.0 loss for that empty case instead.
        if self.reduction == "mean" and not (targets != -100).any():
            results[self.output_key] = logits.new_tensor(0.0)
            return results

        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
            reduction=self.reduction,
        )

        # When reduction='none', reshape back to [B, S] so the shape
        # matches the original sequence dimension.
        if self.reduction == "none":
            loss = loss.reshape(B, S)

        results[self.output_key] = loss
        return results


class AccuracyStep(Step):
    """Compute token-level accuracy from logits and target IDs.

    Reads ``logits_key`` and ``targets_key`` from results, computes the
    fraction of argmax predictions that match the target, and writes the
    result (a plain Python float) under ``output_key``.

    Args:
        logits_key: Key in results containing the logits tensor ``[B, S, V]``.
        targets_key: Key in results containing the target IDs ``[B, S]``.
        output_key: Key under which to store the computed accuracy.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        logits_key: str = keys.FINAL_LOGITS,
        targets_key: str = keys.TARGET_IDS,
        output_key: str = keys.ACCURACY,
    ):
        self.logits_key = logits_key
        self.targets_key = targets_key
        self.output_key = output_key
        self.reads = [logits_key, targets_key]
        self.writes = [output_key]

    def __call__(self, results: Results) -> Results:
        logits: torch.Tensor = results[self.logits_key]
        targets: torch.Tensor = results[self.targets_key]

        predicted = logits.argmax(dim=-1)
        # Score only the real next-token positions; -100 marks padded and
        # last-position slots that have no target. An all-ignored batch has
        # nothing to score, so report 0.0 rather than averaging an empty tensor.
        valid = targets != -100
        accuracy = (
            (predicted[valid] == targets[valid]).float().mean().item()
            if valid.any()
            else 0.0
        )

        results[self.output_key] = accuracy
        return results


class ComparisonComputationStep(Step):
    """Compute element-wise difference or row-wise cosine similarity.

    Reads two tensors from results, computes the comparison, and writes
    the result under ``output_key``.

    Args:
        key_a: Key in results for the first tensor.
        key_b: Key in results for the second tensor.
        output_key: Key under which to store the result.
        comparison_type: ``"difference"`` or ``"cosine_similarity"``.
    """

    reads: list[str] = []
    writes: list[str] = []

    def __init__(
        self,
        key_a: str,
        key_b: str,
        output_key: str,
        comparison_type: str = "difference",
    ):
        self.key_a = key_a
        self.key_b = key_b
        self.output_key = output_key
        self.comparison_type = comparison_type
        self.reads = [key_a, key_b]
        self.writes = [output_key]

    def __call__(self, results: Results) -> Results:
        tensor_a: torch.Tensor = results[self.key_a]
        tensor_b: torch.Tensor = results[self.key_b]

        if self.comparison_type == "difference":
            result = tensor_a - tensor_b
        elif self.comparison_type == "cosine_similarity":
            a_norm = F.normalize(tensor_a.float(), dim=-1)
            b_norm = F.normalize(tensor_b.float(), dim=-1)
            result = (a_norm * b_norm).sum(dim=-1)
        else:
            raise ValueError(
                f"Unsupported comparison_type '{self.comparison_type}'. "
                f"Expected 'difference' or 'cosine_similarity'."
            )

        results[self.output_key] = result
        return results
