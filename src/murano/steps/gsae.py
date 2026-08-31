"""Gradient Sparse Autoencoders (GSAE): a TopK dictionary over recorded gradients.

The forward-capture SAE path (:mod:`murano.steps.sae`) loads pre-trained
dictionaries; gradients have none, so the GSAE ships its own minimal trainer.
The object of study is the Gradient Feature Circuits feature census:
train one shared dictionary on the pooled gradient inputs of several
checkpoints (base and RL alike), then compare how often each feature fires per
checkpoint. A feature is *introduced* when it is nearly silent on the base
gradients but common after RL, and *turned up* (the paper also says
"recruited") when RL multiplies an existing firing rate. Because the
dictionary is shared and fit once, a firing-rate shift is a property of the
gradients, not of a per-checkpoint refit.

The encoder input follows the paper: per token position,
``x = sign(A) * g / RMS(g)``, where ``g`` is the recorded residual-stream
gradient and ``A`` the rollout's standardized advantage, so the code sees the
advantage's sign but not its size, and every input has unit RMS regardless of
how loud its position was.

The autoencoder is the TopK variant: ``z = TopK_k(W_enc (x - b_pre))`` keeps
the ``k`` largest coordinates and zeroes the rest, ``xhat = W_dec z + b_pre``,
decoder rows are renormalized to unit length after every step, and features
that never fired in an epoch are re-initialized to the residual of a
badly-reconstructed input so the dictionary stays alive.

Typical flow::

    inputs = normalized_gradient_inputs(store, advantages)   # per checkpoint
    gsae = GSAE.train(torch.cat(all_inputs), m=16384, k=64)  # fit ONCE, pooled
    base_rates = firing_rates(gsae, base_inputs)
    rl_rates = firing_rates(gsae, rl_inputs).amax(dim=0)     # max over RL ckpts
    census = classify_features(base_rates, rl_rates)
    census["introduced"]  # feature ids nearly silent before RL, common after
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import (
    Generator,  # pyright: ignore[reportPrivateImportUsage]
    Tensor,
    bool as torch_bool,  # pyright: ignore[reportPrivateImportUsage]
    cat,  # pyright: ignore[reportPrivateImportUsage]
    randn,
    randperm,  # pyright: ignore[reportPrivateImportUsage]
    zeros,  # pyright: ignore[reportPrivateImportUsage]
)

from murano.logging import logger
from murano.steps.gradients import GradientStore

if TYPE_CHECKING:
    from murano.backend import ModelBackend

CENSUS_CLASSES = ("introduced", "turned_up", "suppressed", "stable")


def normalized_gradient_inputs(
    store: GradientStore,
    advantages: Tensor | None = None,
    subsample: float | None = None,
    seed: int = 0,
) -> Tensor:
    """Pool a gradient store into GSAE inputs ``sign(A) * g / RMS(g)``.

    Every kept rollout's per-position gradients are RMS-normalized (each row
    ends at unit RMS, so loud and quiet positions weigh alike) and multiplied
    by the sign of that rollout's advantage. Rollouts with advantage 0 (an
    all-pass or all-fail group carries no within-group signal) are dropped,
    matching the paper's capture.

    Args:
        store: Recorded gradients, one ``[T, d_model]`` tensor per rollout.
        advantages: Per-rollout advantages; ``None`` keeps every finite
            rollout with sign +1.
        subsample: Keep this fraction of pooled positions, drawn without
            replacement (the paper keeps 5% at scale); ``None`` keeps all.
        seed: Seed for the subsample draw.

    Returns:
        Float32 tensor ``[n_positions, d_model]`` of encoder inputs.

    Raises:
        ValueError: If ``advantages`` misaligns with the store or
            ``subsample`` is not in (0, 1].
    """
    if advantages is not None and len(advantages) != len(store):
        raise ValueError(
            f"advantages must have one entry per rollout: got "
            f"{len(advantages)} for {len(store)} rollouts."
        )
    if subsample is not None and not 0.0 < subsample <= 1.0:
        raise ValueError(f"subsample must be in (0, 1], got {subsample}.")
    rows = []
    for n, grad in enumerate(store.gradients):
        if not bool(store.finite[n]) or grad.numel() == 0:
            continue
        sign = 1.0
        if advantages is not None:
            a = float(advantages[n])
            if a == 0.0:
                continue
            sign = 1.0 if a > 0 else -1.0
        rms = grad.float().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        rows.append(sign * grad.float() / rms)
    if not rows:
        raise ValueError("No usable rollouts: all dropped or zero-advantage.")
    pooled = cat(rows)
    if subsample is not None and subsample < 1.0:
        generator = Generator().manual_seed(seed)
        n_keep = max(1, int(pooled.shape[0] * subsample))
        keep = randperm(pooled.shape[0], generator=generator)[:n_keep]
        pooled = pooled[keep]
    return pooled


@dataclass
class GSAE:
    """A trained TopK gradient sparse autoencoder.

    Attributes:
        w_enc: Encoder weight ``[m, d_model]``.
        w_dec: Decoder weight ``[m, d_model]``, unit-norm rows; row ``j`` is
            feature ``j``'s direction in the residual stream.
        b_pre: Pre-encoder bias ``[d_model]``, subtracted before encoding and
            added back after decoding.
        k: Active features per input.
        fvu: Fraction of variance unexplained on the training inputs after the
            final epoch (0 = perfect reconstruction).
    """

    w_enc: Tensor
    w_dec: Tensor
    b_pre: Tensor
    k: int
    fvu: float = float("nan")

    @property
    def m(self) -> int:
        """Dictionary size (number of features)."""
        return int(self.w_enc.shape[0])

    def encode(self, inputs: Tensor) -> Tensor:
        """Sparse codes ``[n, m]``: TopK of the pre-activations, rest zeroed.

        Args:
            inputs: ``[n, d_model]`` encoder inputs.

        Returns:
            ``[n, m]`` codes with exactly ``min(k, m)`` nonzeros per row.
        """
        pre = (inputs.float() - self.b_pre) @ self.w_enc.T
        values, indices = pre.topk(min(self.k, self.m), dim=-1)
        codes = zeros(pre.shape, dtype=pre.dtype, device=pre.device)
        return codes.scatter_(-1, indices, values)

    def decode(self, codes: Tensor) -> Tensor:
        """Reconstruction ``[n, d_model]`` from sparse codes."""
        return codes @ self.w_dec + self.b_pre

    def feature_direction(self, feature: int) -> Tensor:
        """Feature ``feature``'s unit decoder direction ``[d_model]``."""
        return self.w_dec[feature]

    @classmethod
    def train(
        cls,
        inputs: Tensor,
        m: int,
        k: int,
        lr: float = 5e-4,
        batch_size: int = 4096,
        epochs: int = 3,
        seed: int = 0,
    ) -> "GSAE":
        """Fit a TopK autoencoder on pooled gradient inputs.

        Plain reconstruction training: Adam on the squared error, decoder rows
        renormalized to unit length after every step (so a feature's strength
        lives in its code, not its direction), and features that never fired
        in an epoch re-initialized between epochs to the residual of the
        worst-reconstructed input of a probe batch, so the dictionary stays
        alive instead of accumulating dead features.

        Args:
            inputs: ``[n, d_model]`` pooled encoder inputs (pool the
                checkpoints first: the census needs one shared dictionary).
            m: Dictionary size.
            k: Active features per input; at most ``m``.
            lr: Adam learning rate.
            batch_size: Minibatch size.
            epochs: Passes over the inputs.
            seed: Seed for initialization and batch order.

        Returns:
            The trained :class:`GSAE`, with ``fvu`` filled in.

        Raises:
            ValueError: If ``inputs`` is not 2-D or ``k``/``m`` are invalid.
        """
        if inputs.dim() != 2:
            raise ValueError(
                f"inputs must be [n, d_model], got shape {tuple(inputs.shape)}."
            )
        if not 0 < k <= m:
            raise ValueError(f"Need 0 < k <= m, got k={k}, m={m}.")
        inputs = inputs.float()
        n, d_model = inputs.shape
        generator = Generator().manual_seed(seed)
        w_dec = randn(m, d_model, generator=generator)
        w_dec = w_dec / w_dec.norm(dim=1, keepdim=True)
        gsae = cls(
            w_enc=w_dec.clone().requires_grad_(True),
            w_dec=w_dec.clone().requires_grad_(True),
            b_pre=zeros(d_model).requires_grad_(True),
            k=k,
        )
        optimizer = torch.optim.Adam([gsae.w_enc, gsae.w_dec, gsae.b_pre], lr=lr)
        variance = inputs.var(dim=0, unbiased=False).sum().clamp_min(1e-12)

        for epoch in range(epochs):
            order = randperm(n, generator=generator)
            fired = zeros(m, dtype=torch_bool)
            for begin in range(0, n, batch_size):
                batch = inputs[order[begin : begin + batch_size]]
                codes = gsae.encode(batch)
                reconstruction = gsae.decode(codes)
                loss = (reconstruction - batch).pow(2).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    gsae.w_dec /= gsae.w_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    fired |= codes.ne(0).any(dim=0)
            logger.info(
                "GSAE epoch %d/%d: %d/%d features alive",
                epoch + 1,
                epochs,
                int(fired.sum()),
                m,
            )
            if epoch < epochs - 1:
                _reinit_dead_features(gsae, inputs, fired, generator)

        gsae.w_enc = gsae.w_enc.detach()
        gsae.w_dec = gsae.w_dec.detach()
        gsae.b_pre = gsae.b_pre.detach()
        with torch.no_grad():
            error = 0.0
            for begin in range(0, n, batch_size):
                batch = inputs[begin : begin + batch_size]
                residual = batch - gsae.decode(gsae.encode(batch))
                error += float(residual.pow(2).sum())
        gsae.fvu = error / n / float(variance)
        logger.info("GSAE trained: FVU %.4f on the training inputs", gsae.fvu)
        return gsae


def _reinit_dead_features(
    gsae: GSAE, inputs: Tensor, fired: Tensor, generator: Generator
) -> None:
    """Point dead features at what the live ones reconstruct worst.

    For every feature that fired on nothing this epoch, pick a high-error
    input from a probe batch and set the feature's decoder row to that input's
    residual (renormalized), with the encoder row matched to it, so the next
    epoch can recruit the feature for exactly the structure the dictionary is
    missing.

    Each revived feature needs its own residual, so the probe grows with the
    dead count. When more features are dead than there are inputs to draw from,
    the remainder keep their current directions and get another chance next
    epoch rather than the pass failing.
    """
    dead = (~fired).nonzero(as_tuple=True)[0]
    if dead.numel() == 0:
        return
    with torch.no_grad():
        n_probe = min(max(4096, dead.numel()), inputs.shape[0])
        probe = inputs[randperm(inputs.shape[0], generator=generator)[:n_probe]]
        residual = probe - gsae.decode(gsae.encode(probe))
        errors = residual.pow(2).sum(dim=1)
        revived = dead[: probe.shape[0]]
        worst = errors.argsort(descending=True)[: revived.numel()]
        directions = residual[worst]
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        gsae.w_dec[revived] = directions
        gsae.w_enc[revived] = directions
    logger.info(
        "Re-initialized %d of %d dead features",
        int(revived.numel()),
        int(dead.numel()),
    )


def firing_rates(gsae: GSAE, inputs: Tensor, batch_size: int = 4096) -> Tensor:
    """Fraction of inputs on which each feature is among the selected k.

    Args:
        gsae: The trained autoencoder.
        inputs: ``[n, d_model]`` encoder inputs of one checkpoint.
        batch_size: Encode batch size.

    Returns:
        Float tensor ``[m]`` of per-feature firing rates in [0, 1].
    """
    counts = zeros(gsae.m)
    with torch.no_grad():
        for begin in range(0, inputs.shape[0], batch_size):
            codes = gsae.encode(inputs[begin : begin + batch_size])
            counts += codes.ne(0).sum(dim=0).float()
    return counts / inputs.shape[0]


def classify_features(
    base_rates: Tensor,
    rl_rates: Tensor,
    introduced: tuple[float, float] = (0.05, 0.15),
    turned_up: tuple[float, float] = (3.0, 0.005),
) -> dict[str, list[int]]:
    """The paper's firing-rate census, first matching class per feature.

    With base rate ``f0`` and RL rate ``fR`` (at scale, the max over the RL
    checkpoints):

    * introduced: ``f0 < introduced[0]`` and ``fR > introduced[1]``. The
      feature barely existed in the base gradients and is common after RL.
    * turned_up: else ``fR >= turned_up[0] * f0`` and
      ``fR - f0 > turned_up[1]``. An existing feature RL fires much more
      often (the paper also calls this "recruited").
    * suppressed: else ``f0 >= turned_up[0] * fR``.
    * stable: everything else.

    Args:
        base_rates: ``[m]`` firing rates on the base checkpoint's inputs.
        rl_rates: ``[m]`` firing rates after RL (max over RL checkpoints).
        introduced: ``(base_below, rl_above)`` thresholds.
        turned_up: ``(ratio, min_gain)`` thresholds, shared with suppressed.

    Returns:
        ``{class_name: sorted feature ids}`` covering every feature exactly
        once, keys ordered as :data:`CENSUS_CLASSES`.

    Raises:
        ValueError: If the rate vectors' shapes differ.
    """
    if base_rates.shape != rl_rates.shape:
        raise ValueError(
            f"Rate vectors must share a shape, got {tuple(base_rates.shape)} "
            f"vs {tuple(rl_rates.shape)}."
        )
    census: dict[str, list[int]] = {name: [] for name in CENSUS_CLASSES}
    # A feature silent everywhere (f0 = fR = 0) lands in "suppressed" because
    # 0 >= ratio * 0 holds; the criteria are kept exactly as the paper states
    # them (no extra guard), and the trained dictionaries keep every feature
    # alive via the dead-feature re-initialization, so the edge is theoretical.
    for j in range(base_rates.shape[0]):
        f0, fr = float(base_rates[j]), float(rl_rates[j])
        if f0 < introduced[0] and fr > introduced[1]:
            census["introduced"].append(j)
        elif fr >= turned_up[0] * f0 and (fr - f0) > turned_up[1]:
            census["turned_up"].append(j)
        elif f0 >= turned_up[0] * fr:
            census["suppressed"].append(j)
        else:
            census["stable"].append(j)
    return census


def promoted_tokens(
    gsae: GSAE, model: ModelBackend, feature: int, top_k: int = 5
) -> list[str]:
    """The tokens a feature's direction promotes through the unembedding.

    Projects the decoder direction through the final norm's gain and the
    unembedding, ``W_U (gamma * w_j)``, and returns the ``top_k`` most
    promoted token strings; the cheap readout the paper uses to say what a
    recruited gradient feature would push the model toward.

    Args:
        gsae: The trained autoencoder.
        model: Backend supplying ``final_norm``, ``unembed_weight``, and the
            tokenizer. Use one fixed checkpoint for every feature.
        feature: Feature id.
        top_k: Tokens to return.

    Returns:
        Token strings, most promoted first.

    Raises:
        ValueError: If ``feature`` is out of range or ``top_k`` is not
            positive.
    """
    if not 0 <= feature < gsae.m:
        raise ValueError(f"feature must be in [0, {gsae.m}), got {feature}.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")
    direction = gsae.feature_direction(feature)
    gain = model.final_norm.weight
    direction = direction.to(gain.device, gain.dtype) * gain
    unembed = model.unembed_weight
    logits = unembed.float() @ direction.to(unembed.device).float()
    ids = logits.topk(top_k).indices.tolist()
    return model.tokenizer.convert_ids_to_tokens(ids)
