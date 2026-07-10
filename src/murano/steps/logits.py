"""Logits step: expose the model's output logits and next-token LM targets.

Runs a single forward pass over the batched prompts and stores the model's raw
``[B, S, V]`` output logits, plus (by default) the left-shifted next-token
target IDs the cross-entropy and accuracy metric steps consume. The shift and
padding mask live only in the targets, never in the logits, so ``final_logits``
stays the honest model output for downstream analyses (logit-diff, DLA) while
the generic LM metrics stay correct without slicing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

from torch import Tensor, full, long  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import Node
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

    With ``fn`` given, the pass is intervened: ``fn`` rewrites each hooked
    module's activation before the logits are read. This is the forward-pass
    analogue of :class:`~murano.steps.intervene.Intervene`, which applies the same
    edit during generation. Reach for it to *measure* an intervention with a
    metric rather than read it off a completion: a twelve-layer steering sweep
    costs twelve forward passes instead of twelve decodes.

    Reads from results:
        results[prompts_key]: PromptBatch (default ``prompts``). Set this to run
            the step on a different prompt set, e.g. the corrupt side of a
            paired run.

    Writes to results:
        results['final_logits']: Tensor [B, S, vocab] of output logits.
        results['attention_mask']: Tensor [B, S] marking real (1) vs padding
            (0) positions, so downstream metric steps can locate the answer
            position without re-tokenizing.
        results['target_ids']: Tensor [B, S] of next-token targets (only when
            ``targets`` is set).

    Args:
        model: Model backend to run.
        prompts_key: Results key to read the prompts from. When a second Logits
            runs in the same pipeline (e.g. the corrupt side of a paired run),
            also override logits_key and mask_key (and set targets accordingly)
            so it does not overwrite the first run's outputs.
        logits_key: Results key to write the logits under.
        targets_key: Results key to write the next-token targets under.
        mask_key: Results key to write the attention mask under.
        targets: Target-generation mode. ``"next_token"`` writes left-shifted
            next-token targets; ``None`` writes logits only, for callers that
            supply their own task targets (e.g. logit-diff).
        fn: ``(activation, node) -> activation``, applied to each hooked module's
            activation before the logits are read; for example
            ``steer_direction(direction, alpha=4)`` or ``ablate_direction(...)``.
            ``None`` (the default) runs the plain forward pass, in which case
            ``layers``, ``modules``, and ``per_head`` are ignored.
        layers: Layer indices to hook, or ``"all"``.
        modules: Module name(s) to hook at each layer.
        per_head: Hook the attention output projection's input and hand ``fn`` the
            per-head activation, so it can rewrite one head's slice.

    Raises:
        ValueError: If ``targets`` is neither ``"next_token"`` nor ``None``.
    """

    reads = [keys.PROMPTS]
    read_types = {keys.PROMPTS: PromptBatch}

    def __init__(
        self,
        model: ModelBackend,
        prompts_key: str = keys.PROMPTS,
        logits_key: str = keys.FINAL_LOGITS,
        targets_key: str = keys.TARGET_IDS,
        mask_key: str = keys.ATTENTION_MASK,
        targets: str | None = "next_token",
        fn: Callable[[Tensor, Node], Tensor] | None = None,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        per_head: bool = False,
    ):
        if targets not in ("next_token", None):
            raise ValueError(f"targets must be 'next_token' or None, got {targets!r}")
        self.model = model
        self.prompts_key = prompts_key
        self.logits_key = logits_key
        self.targets_key = targets_key
        self.mask_key = mask_key
        self.targets = targets
        self.fn = fn
        self.layers = layers
        self.modules = modules
        self.per_head = per_head
        self.reads = [prompts_key]
        self.read_types = {prompts_key: PromptBatch}
        # The mask is independent of targets; only advertise target_ids when we
        # will actually produce it, so the step never claims a key it does not
        # write.
        self.writes = [logits_key, mask_key] + ([targets_key] if targets else [])
        self.write_types = {
            logits_key: Tensor,
            mask_key: Tensor,
            **({targets_key: Tensor} if targets else {}),
        }

    def __call__(self, results: Results) -> Results:
        prompt_batch: PromptBatch = results[self.prompts_key]
        prompts = prompt_batch.prompts
        logger.info(
            "Logits: %d prompts%s", len(prompts), " (intervened)" if self.fn else ""
        )

        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        attention_mask = cast(Tensor, tokens["attention_mask"])
        results[self.logits_key] = self.model.forward_logits(
            tokens,
            fn=self.fn,
            layers=self.layers,
            modules=self.modules,
            per_head=self.per_head,
        )
        results[self.mask_key] = attention_mask.detach().cpu()
        if self.targets:
            results[self.targets_key] = self._next_token_targets(
                cast(Tensor, tokens["input_ids"]),
                attention_mask,
            )
        return results

    @staticmethod
    def _next_token_targets(input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Build left-shifted next-token targets with ``-100`` for ignored slots.

        The target at position ``s`` is the token at ``s + 1``, the token a
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
