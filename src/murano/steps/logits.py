"""Logits step: expose the model's output logits and next-token LM targets.

Runs a single forward pass over the batched prompts and stores the model's raw
``[B, S, V]`` output logits, plus (by default) the left-shifted next-token
target IDs the cross-entropy and accuracy metric steps consume. The shift and
padding mask live only in the targets, never in the logits, so ``final_logits``
stays the honest model output for downstream analyses (logit-diff, DLA) while
the generic LM metrics stay correct without slicing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, full, long  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.backend import ModelBackend


class Logits(Step):
    """Run the model and write its output logits (plus next-token targets).

    Runs one forward pass over the batched prompts via the backend's
    ``forward_logits`` primitive and stores the raw ``[B, S, V]`` logits. When
    ``targets`` is set (the default), it also writes left-shifted next-token
    target IDs so ``CrossEntropyLossStep``/``AccuracyStep`` can run directly.

    Reads from results:
        results['prompts']: PromptBatch

    Writes to results:
        results['final_logits']: Tensor [B, S, vocab] of output logits.
        results['target_ids']: Tensor [B, S] of next-token targets (only when
            ``targets`` is set).

    Args:
        model: Model backend to run.
        logits_key: Results key to write the logits under.
        targets_key: Results key to write the next-token targets under.
        targets: Target-generation mode. ``"next_token"`` writes left-shifted
            next-token targets; ``None`` writes logits only, for callers that
            supply their own task targets (e.g. logit-diff).

    Raises:
        ValueError: If ``targets`` is neither ``"next_token"`` nor ``None``.
    """

    reads = [keys.PROMPTS]
    read_types = {keys.PROMPTS: PromptBatch}

    def __init__(
        self,
        model: ModelBackend,
        logits_key: str = keys.FINAL_LOGITS,
        targets_key: str = keys.TARGET_IDS,
        targets: str | None = "next_token",
    ):
        if targets not in ("next_token", None):
            raise ValueError(f"targets must be 'next_token' or None, got {targets!r}")
        self.model = model
        self.logits_key = logits_key
        self.targets_key = targets_key
        self.targets = targets
        # Only advertise target_ids when we will actually produce it, so the
        # step never claims a key it does not write.
        self.writes = [logits_key] + ([targets_key] if targets else [])
        self.write_types = {
            logits_key: Tensor,
            **({targets_key: Tensor} if targets else {}),
        }

    def __call__(self, results: Results) -> Results:
        prompt_batch: PromptBatch = results[keys.PROMPTS]
        prompts = prompt_batch.prompts
        logger.info("Logits: %d prompts", len(prompts))

        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        results[self.logits_key] = self.model.forward_logits(tokens)
        if self.targets:
            results[self.targets_key] = self._next_token_targets(
                cast(Tensor, tokens["input_ids"]),
                cast(Tensor, tokens["attention_mask"]),
            )
        return results

    @staticmethod
    def _next_token_targets(input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Build left-shifted next-token targets with ``-100`` for ignored slots.

        The target at position ``s`` is the token at ``s + 1`` — the token a
        causal LM predicts from position ``s``. The last real position of each
        row, and any position whose source or next token is padding, have no
        next token and are set to ``-100`` so the metric steps skip them.
        Validity is read from the attention mask, which is correct for either
        padding side and yields an all-``-100`` row for length-1 inputs.

        Args:
            input_ids: Token IDs ``[B, S]``.
            attention_mask: Attention mask ``[B, S]`` (1 = real, 0 = padding).

        Returns:
            Next-token targets ``[B, S]`` (long), ``-100`` where ignored.
        """
        B, S = input_ids.shape
        targets = full((B, S), -100, dtype=long, device=input_ids.device)
        if S > 1:
            valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
            targets[:, :-1] = input_ids[:, 1:].to(long).masked_fill(~valid, -100)
        return targets
