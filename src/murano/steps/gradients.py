"""Gradient recording: teacher-forced backward passes over saved rollouts.

Forward capture (:class:`~murano.steps.record.Record`) answers "what did the
model represent"; this module answers "what would training move". It records,
for a frozen checkpoint, the gradient of the completion log-likelihood with
respect to the residual stream, one vector per completion token. That gradient
is the direction factor of the first-epoch policy-gradient update (the
advantage enters only as a per-rollout scalar), so recording it on saved
rollouts reads the update a trainer *would* apply without ever taking an
optimizer step.

Two design decisions differ from the rest of the capture code, on purpose:

* Every other capture path detaches at the earliest opportunity; a gradient
  pass cannot, so this module owns its forward end to end through
  :meth:`~murano.model.MuranoModel.grad_forward` (the graph-preserving
  counterpart of ``forward_logits``) and detaches only at the store boundary.
* Rollouts are processed one at a time rather than in padded batches. The
  rollouts of one corpus differ in length, and padding would put the completion
  window at a different sequence offset per row; per-rollout passes keep the
  prompt/completion boundary exact and keep memory proportional to one
  sequence.

The loss is the plain summed log-likelihood of the realized completion tokens
over a fixed window: no advantage weighting, no normalization. Rewards and
GRPO group structure travel with the rollouts (:class:`RolloutBatch`) so that
downstream analyses can weight per-rollout results by the standardized
advantage, but the recorded gradient itself is reward-free.

Typical flow::

    rollouts = RolloutBatch(input_ids=ids, prompt_lengths=lengths)
    pipeline = Pipeline([
        LoadRollouts(rollouts),
        RecordGradients(model, layer=30, window=512),
    ])
    results = pipeline.run()
    store = results["gradient_record"]
    store.gradients[0]  # [T, d_model] gradient at layer 30, first rollout
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, as_tensor, empty, float32, isfinite, log_softmax  # pyright: ignore[reportPrivateImportUsage]
from torch import long, tensor, zeros_like  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.backend import ModelBackend


@dataclass
class RolloutBatch:
    """A corpus of saved rollouts: full token ids plus the completion boundary.

    A rollout is a prompt and the completion some model sampled for it. For
    gradient recording only the fixed token ids matter: every checkpoint is
    teacher-forced through the same ids, so results differ only in weights,
    never in text. Rewards and group ids are optional passengers for
    advantage-weighted analyses.

    Attributes:
        input_ids: One 1-D long tensor of token ids per rollout, prompt and
            completion concatenated.
        prompt_lengths: Number of leading prompt tokens per rollout. Loss terms
            and gradient reads start at this offset. Must be at least 1 (the
            first completion token is predicted from the last prompt position)
            and leave at least one completion token.
        rewards: Optional per-rollout scalar verifier rewards.
        groups: Optional per-rollout group ids (e.g. the GRPO task group), used
            by :meth:`advantages` to standardize rewards within each group.
    """

    input_ids: list[Tensor]
    prompt_lengths: list[int]
    rewards: list[float] | None = None
    groups: list[int] | None = None

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.prompt_lengths):
            raise ValueError(
                f"input_ids and prompt_lengths must align: got "
                f"{len(self.input_ids)} rollouts but "
                f"{len(self.prompt_lengths)} prompt lengths."
            )
        coerced: list[Tensor] = []
        for n, ids in enumerate(self.input_ids):
            ids = as_tensor(ids, dtype=long)
            if ids.dim() != 1:
                raise ValueError(
                    f"Rollout {n}: input_ids must be 1-D token ids, got shape "
                    f"{tuple(ids.shape)}."
                )
            prompt_len = self.prompt_lengths[n]
            if not 1 <= prompt_len < ids.shape[0]:
                raise ValueError(
                    f"Rollout {n}: prompt_length must be in [1, {ids.shape[0]}) "
                    f"so at least one completion token remains, got {prompt_len}."
                )
            coerced.append(ids)
        self.input_ids = coerced
        for name, extra in (
            ("rewards", self.rewards),
            ("groups", self.groups),
        ):
            if extra is not None and len(extra) != len(self.input_ids):
                raise ValueError(
                    f"{name} must have one entry per rollout: got {len(extra)} "
                    f"for {len(self.input_ids)} rollouts."
                )

    def __len__(self) -> int:
        return len(self.input_ids)

    def advantages(self, eps: float = 1e-4) -> Tensor:
        """Standardized within-group advantages ``(r - mean) / (std + eps)``.

        Groups with a single outcome (all rollouts passed, or all failed) carry
        no within-group signal and get advantage 0 rather than a division
        artifact.

        Args:
            eps: Stabilizer added to the within-group standard deviation.

        Returns:
            Float tensor ``[n_rollouts]`` of advantages.

        Raises:
            ValueError: If ``rewards`` or ``groups`` was not provided.
        """
        if self.rewards is None or self.groups is None:
            raise ValueError(
                "advantages() needs both rewards and groups on the RolloutBatch."
            )
        rewards = tensor(self.rewards, dtype=float32)
        out = zeros_like(rewards)
        for group in set(self.groups):
            mask = tensor([g == group for g in self.groups])
            r = rewards[mask]
            std = r.std(unbiased=False)
            if std.item() == 0.0:
                continue
            out[mask] = (r - r.mean()) / (std + eps)
        return out


class LoadRollouts(Step):
    """Load a rollout corpus into the pipeline results.

    Writes to results:
        results['rollouts']: RolloutBatch

    Args:
        rollouts: The rollout corpus to make available to downstream steps.
    """

    reads = []
    writes = [keys.ROLLOUTS]
    write_types = {keys.ROLLOUTS: RolloutBatch}

    def __init__(self, rollouts: RolloutBatch):
        if not isinstance(rollouts, RolloutBatch):
            raise TypeError(
                f"LoadRollouts expects a RolloutBatch, got {type(rollouts).__name__}."
            )
        self.rollouts = rollouts

    def __call__(self, results: Results) -> Results:
        results[keys.ROLLOUTS] = self.rollouts
        return results


@dataclass
class GradientStore:
    """Per-rollout residual-stream gradients from a teacher-forced pass.

    Attributes:
        gradients: One ``[T, d_model]`` float32 CPU tensor per rollout: the
            gradient of the completion log-likelihood with respect to the
            residual stream after block ``layer``, at each completion position
            inside the window. ``T`` is the per-rollout window length.
        nll: ``[n_rollouts]`` mean negative log-likelihood per completion token
            over the same window, the functional-change scale used to match
            finetuning runs.
        finite: ``[n_rollouts]`` bool; False where the backward produced
            non-finite values (near-random checkpoints can), in which case that
            rollout's gradient tensor is empty and its nll is NaN.
        layer: Residual-stream layer the gradients were read at.
        window: Maximum completion tokens read per rollout.
    """

    gradients: list[Tensor]
    nll: Tensor
    finite: Tensor
    layer: int
    window: int

    def __post_init__(self) -> None:
        n = len(self.gradients)
        if self.nll.shape[0] != n or self.finite.shape[0] != n:
            raise ValueError(
                f"nll and finite must have one entry per rollout: got "
                f"{self.nll.shape[0]} and {self.finite.shape[0]} for {n} "
                f"gradient tensors."
            )

    def __len__(self) -> int:
        return len(self.gradients)


def _completion_window(ids: Tensor, prompt_length: int, window: int) -> tuple[int, int]:
    """Return the ``[start, end)`` sequence positions of the read window."""
    return prompt_length, min(int(ids.shape[0]), prompt_length + window)


def _completion_loglik(
    model: ModelBackend, hidden_last: Tensor, ids: Tensor, start: int, end: int
) -> Tensor:
    """Summed log-likelihood of completion tokens ``ids[start:end]``.

    The token at position ``t`` is predicted from the logits at ``t - 1``, so
    the loss reads logits at ``[start-1, end-1)``. Computed in float32 for a
    stable softmax regardless of the model dtype.
    """
    logits = model.project_on_vocab(hidden_last[:, start - 1 : end - 1])
    logprobs = log_softmax(logits.float(), dim=-1)
    targets = ids[start:end].to(logprobs.device)
    return logprobs[0].gather(1, targets.unsqueeze(1)).sum()


class RecordGradients(Step):
    """Record residual-stream gradients of the completion log-likelihood.

    For each rollout, run the frozen model teacher-forced over the saved
    tokens, form the summed log-likelihood of the completion tokens inside the
    window, and record its gradient with respect to the residual stream after
    ``layer``, at every completion position in the window. One forward and one
    backward per rollout; the weights are never touched.

    Reads from results:
        results['rollouts']: RolloutBatch

    Writes to results:
        results['gradient_record']: GradientStore

    Args:
        model: Wrapped model to read gradients from.
        layer: Residual-stream layer (block output, 0-based) to read at.
        window: Maximum completion tokens contributing loss terms and gradient
            reads per rollout. The window bounds memory; shorter rollouts use
            all their completion tokens.

    Raises:
        ValueError: If ``layer`` is out of range or ``window`` is not positive.

    Note:
        Gradients are stored per rollout as ``[T, d_model]`` float32 on CPU:
        memory is ``n_rollouts * window * d_model`` floats, the gradient-side
        analogue of full-position recording. Load large models in float32 for
        this step where feasible; bfloat16 gradients are noticeably noisier.
    """

    reads = [keys.ROLLOUTS]
    writes = [keys.GRADIENT_RECORD]
    read_types = {keys.ROLLOUTS: RolloutBatch}
    write_types = {keys.GRADIENT_RECORD: GradientStore}

    def __init__(self, model: ModelBackend, layer: int, window: int = 512):
        if not 0 <= layer < model.n_layers:
            raise ValueError(f"layer must be in [0, {model.n_layers}), got {layer}.")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        self.model = model
        self.layer = layer
        self.window = window

    def __call__(self, results: Results) -> Results:
        rollouts: RolloutBatch = results[keys.ROLLOUTS]
        last = self.model.n_layers - 1
        layers = sorted({self.layer, last})

        gradients: list[Tensor] = []
        nll: list[float] = []
        finite: list[bool] = []
        logger.info(
            "Recording gradients: %d rollouts, layer %d, window %d",
            len(rollouts),
            self.layer,
            self.window,
        )
        for n, ids in enumerate(rollouts.input_ids):
            start, end = _completion_window(
                ids, rollouts.prompt_lengths[n], self.window
            )
            hidden = self.model.grad_forward(ids.unsqueeze(0), layers)
            loglik = _completion_loglik(self.model, hidden[last], ids, start, end)
            (grad,) = torch.autograd.grad(loglik, hidden[self.layer])
            grad_window = grad[0, start:end].detach().float().cpu()
            if bool(isfinite(grad_window).all()):
                gradients.append(grad_window)
                nll.append(-loglik.item() / (end - start))
                finite.append(True)
            else:
                logger.warning(
                    "Rollout %d produced non-finite gradients; storing empty.", n
                )
                gradients.append(empty(0, self.model.d_model))
                nll.append(float("nan"))
                finite.append(False)

        results[keys.GRADIENT_RECORD] = GradientStore(
            gradients=gradients,
            nll=tensor(nll, dtype=float32),
            finite=tensor(finite),
            layer=self.layer,
            window=self.window,
        )
        logger.info(
            "Gradient recording done: %d/%d rollouts finite.",
            sum(finite),
            len(finite),
        )
        return results
