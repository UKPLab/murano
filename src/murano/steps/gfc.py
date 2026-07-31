"""Gradient Feature Circuits (GFC): where the gradient travels between layers.

The method asks not what a checkpoint computes but how it *routes credit*:
which directions of the residual stream at a late source layer connect,
through the frozen network's own backward transport, to which coordinates at
an earlier target layer. Comparing that routing map across checkpoints (early
pretraining, base, RL steps, and finetuned or perturbed variants) separates
"training rewired the paths" from "training re-weighted reads along fixed
paths".

The construction, per rollout:

1. Read the completion log-likelihood gradient ``g_t`` at the source layer
   (the same quantity :class:`~murano.steps.gradients.RecordGradients` stores).
2. Project it onto ``k`` fixed random orthonormal directions ``d_i`` drawn
   once from a seed: read strengths ``a_it = <g_t, d_i>`` and seeds
   ``s_i(t) = a_it * d_i``. The basis is fixed before any checkpoint is seen,
   so any cross-checkpoint agreement belongs to the models, not the readout.
3. Transport every direction's seed field to the target layer with one
   batched vector-Jacobian product against the frozen forward map, evaluated
   at the same teacher-forced activations (chunkable via ``direction_chunk``).
   Signs are kept through the cross-position sum; the absolute value is taken
   per coordinate only afterwards.
4. Row ``i`` of the operator is the transported magnitude summed over target
   positions: ``R_i = sum_tau |(J^T s_i)(tau)|``, giving ``R`` of shape
   ``[k, d_model]``. Rollouts accumulate by summation.

Two switches carry the method's controls. ``gradient_off=True`` sets every
read strength to 1, so the operator reads the frozen transport alone: any
surviving overlap is carried by the network's paths, not by the gradient read
along them. ``per_position`` keeps the target position open instead of
summing, giving one operator per completion position for opening-versus-rest
comparisons.

Typical flow::

    rollouts = RolloutBatch(input_ids=ids, prompt_lengths=lengths)
    step = GFCOperator(model, source_layer=30, target_layer=25, k=128)
    results = Pipeline([LoadRollouts(rollouts), step]).run()
    base_map = results["gfc_operator"].operator      # [k, d_model]
    ...  # rerun with another checkpoint's model
    score = pairing_overlap(base_map, other_map)          # 1 = same routing
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Generator, Tensor, full, isfinite, ones, outer  # pyright: ignore[reportPrivateImportUsage]
from torch import float32, long, randperm, tensor, zeros, zeros_like  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step
from murano.steps.gradients import (
    RolloutBatch,
    _completion_loglik,
    _completion_window,
)

if TYPE_CHECKING:
    from murano.backend import ModelBackend


def read_directions(d_model: int, k: int, seed: int = 0) -> Tensor:
    """Draw ``k`` fixed random orthonormal read directions in ``d_model`` dims.

    A ``[d_model, k]`` matrix of iid standard Gaussians is drawn from ``seed``
    and orthonormalized by QR; the directions are the first ``k`` columns of
    the orthonormal factor. The same seed gives the same basis on every call,
    which is the point: the basis is fixed once, before any checkpoint is
    read.

    Args:
        d_model: Ambient dimension (the residual-stream width).
        k: Number of directions; must not exceed ``d_model``.
        seed: Seed for the Gaussian draw.

    Returns:
        Float32 tensor ``[k, d_model]`` with orthonormal rows.

    Raises:
        ValueError: If ``k`` exceeds ``d_model`` or either is not positive.
    """
    if not 0 < k <= d_model:
        raise ValueError(
            f"k must be in [1, d_model={d_model}], got {k}: more orthonormal "
            f"directions than dimensions do not exist."
        )
    generator = Generator().manual_seed(seed)
    gaussian = torch.randn(d_model, k, generator=generator, dtype=float32)
    q, _ = torch.linalg.qr(gaussian)
    return q[:, :k].T.contiguous()


def marginal(operator: Tensor) -> Tensor:
    """The rank-one background fixed by an operator's row and column totals.

    ``marg(R) = rowsum(R) * colsum(R)^T / total(R)`` is what the totals would
    produce if sources and targets were unlinked: it encodes how loud each
    source direction and each target coordinate is, but nothing about which
    source wires to which target. Two operators with identical marginals can
    route completely differently, so every routing comparison subtracts it.

    Args:
        operator: ``[k, d_model]`` routing operator.

    Returns:
        Rank-one tensor of the same shape. Zero when the operator sums to 0.
    """
    total = operator.sum()
    if total.abs().item() == 0.0:
        return zeros_like(operator)
    return outer(operator.sum(dim=1), operator.sum(dim=0)) / total


def background_subtract(operator: Tensor) -> Tensor:
    """Remove the rank-one marginal background: ``R - marg(R)``.

    The result has (numerically) zero row sums and zero column sums; what
    remains is the pairing structure, who wires to whom beyond loudness.
    """
    return operator - marginal(operator)


def pairing_operator(operator: Tensor, rank: int = 8) -> Tensor:
    """Keep the top singular pairs at unit strength: ``sum_i u_i v_i^T``.

    The SVD of a (background-subtracted) operator splits it into paired source
    and target directions with strengths. Replacing the top ``rank`` singular
    values by 1 keeps *which* sources pair with *which* targets and discards
    how hard each route is driven, so the comparison reads routing, not gain.
    The result always has Frobenius norm ``sqrt(rank)``.

    Args:
        operator: ``[k, d_model]`` operator, normally background-subtracted.
        rank: Number of singular pairs to keep.

    Returns:
        ``[k, d_model]`` unit-strength pairing operator.

    Raises:
        ValueError: If ``rank`` is not in ``[1, min(operator.shape)]``.
    """
    if not 0 < rank <= min(operator.shape):
        raise ValueError(
            f"rank must be in [1, {min(operator.shape)}] for an operator of "
            f"shape {tuple(operator.shape)}, got {rank}."
        )
    u, _, vh = torch.linalg.svd(operator.float(), full_matrices=False)
    return u[:, :rank] @ vh[:rank, :]


def pairing_overlap(
    a: Tensor, b: Tensor, rank: int = 8, subtract: bool = True
) -> float:
    """Pairing-only overlap between two routing operators.

    Background-subtracts both operators (unless ``subtract=False``), reduces
    each to its unit-strength top-``rank`` pairing, and returns their Frobenius
    cosine. 1 means the two checkpoints pair the same source directions with
    the same target coordinates (identical routing, up to consistent
    relabeling of the pairs); values near the :func:`permutation_floor` mean
    chance agreement.

    Args:
        a: ``[k, d_model]`` routing operator.
        b: ``[k, d_model]`` routing operator on the same basis and layer pair.
        rank: Singular pairs kept per operator.
        subtract: Background-subtract before pairing. Raw operators share most
            of their energy through the marginal alone, so leaving this on is
            what makes the floor sit near zero.

    Returns:
        The overlap as a float.

    Raises:
        ValueError: If the operators' shapes differ.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Operators must share a shape, got {tuple(a.shape)} vs "
            f"{tuple(b.shape)}; they are only comparable on the same basis "
            f"and layer pair."
        )
    if subtract:
        a = background_subtract(a)
        b = background_subtract(b)
    pa = pairing_operator(a, rank=rank)
    pb = pairing_operator(b, rank=rank)
    denom = pa.norm() * pb.norm()
    return ((pa * pb).sum() / denom).item()


def permutation_floor(
    a: Tensor,
    b: Tensor,
    rank: int = 8,
    subtract: bool = True,
    draws: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """Chance level of :func:`pairing_overlap`, by destroying the pairing.

    Randomly permutes the target columns of ``b`` (which preserves both
    operators' row and column totals but breaks who-wires-to-whom) and
    recomputes the overlap. On background-subtracted operators the floor sits
    near zero; on raw operators it is high, which is why subtraction is the
    default everywhere.

    Args:
        a: ``[k, d_model]`` routing operator.
        b: ``[k, d_model]`` routing operator to permute.
        rank: Singular pairs kept per operator.
        subtract: Background-subtract before pairing.
        draws: Number of random permutations.
        seed: Seed for the permutations.

    Returns:
        ``(mean, sd)`` of the overlap across draws.
    """
    generator = Generator().manual_seed(seed)
    scores = []
    for _ in range(draws):
        perm = randperm(b.shape[1], generator=generator)
        scores.append(pairing_overlap(a, b[:, perm], rank=rank, subtract=subtract))
    stacked = tensor(scores)
    return stacked.mean().item(), stacked.std().item()


def per_position_overlap(
    a: Tensor, b: Tensor, rank: int = 8, subtract: bool = True
) -> Tensor:
    """Position-resolved routing overlap ``o(tau)`` between two runs.

    Applies :func:`pairing_overlap` independently at every completion position
    of two per-position operator stacks (the ``per_position`` output of
    :class:`GFCOperator`). A dip over the first few positions with
    recovery afterwards localizes a routing difference to how rollouts open.

    Args:
        a: ``[positions, k, d_model]`` per-position operator stack.
        b: Stack of the same shape from another checkpoint.
        rank: Singular pairs kept per operator.
        subtract: Background-subtract each position's operator.

    Returns:
        Float tensor ``[positions]`` of overlaps.

    Raises:
        ValueError: If the stacks' shapes differ.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Per-position stacks must share a shape, got {tuple(a.shape)} vs "
            f"{tuple(b.shape)}."
        )
    return tensor(
        [
            pairing_overlap(a[p], b[p], rank=rank, subtract=subtract)
            for p in range(a.shape[0])
        ]
    )


@dataclass
class GFCOperatorResult:
    """A routing operator and its capture settings.

    Attributes:
        operator: ``[k, d_model]`` float32 operator, summed over target
            positions and rollouts.
        source_layer: Layer the gradient is read at.
        target_layer: Layer the seeds are transported to.
        k: Number of read directions.
        per_position: ``[positions, k, d_model]`` stack with the target
            position kept open, when requested; ``None`` otherwise. Position 0
            is the first completion token.
        position_counts: ``[positions]`` rollouts contributing at each
            position (rollouts shorter than the horizon stop contributing);
            ``None`` when ``per_position`` was not requested.
        nll: ``[n_rollouts]`` mean teacher-forced negative log-likelihood per
            completion token on the same window, the functional-change scale
            for matching runs; NaN for dropped rollouts.
        kept: ``[n_rollouts]`` bool; False where a rollout was dropped for
            non-finite gradients or transport.
        basis_seed: Seed of the direction draw, or ``None`` when an explicit
            ``directions`` basis was passed instead of a seeded draw.
        gradient_off: True when every read strength was held at 1 (the
            gradient-off control); False for the gradient-weighted operator.
        window: Maximum completion tokens read per rollout.
    """

    operator: Tensor
    source_layer: int
    target_layer: int
    k: int
    per_position: Tensor | None = None
    position_counts: Tensor | None = None
    nll: Tensor | None = None
    kept: Tensor | None = None
    basis_seed: int | None = 0
    gradient_off: bool = False
    window: int = 512


class GFCOperator(Step):
    """Build the GFC routing operator for one checkpoint over a rollout corpus.

    For each rollout: one teacher-forced forward preserving the graph, one
    backward for the source-layer gradient (skipped in gradient-off mode),
    then one batched vector-Jacobian product transporting every read
    direction's seed field from the source layer to the target layer at the
    frozen activations (chunkable via ``direction_chunk``). Transported
    magnitudes accumulate into ``[k, d_model]`` rows; rollouts accumulate by
    summation.

    Reads from results:
        results['rollouts']: RolloutBatch

    Writes to results:
        results['gfc_operator']: GFCOperatorResult

    Args:
        model: Wrapped model to read.
        source_layer: Residual-stream layer (block output) the gradient is
            read and seeds are placed at.
        target_layer: Earlier layer the seeds are transported to; must be
            strictly below ``source_layer``.
        k: Number of fixed random orthonormal read directions; at most
            ``d_model``.
        basis_seed: Seed of the direction draw. Use the same seed for every
            checkpoint being compared; redraw with several seeds to check
            basis robustness. Ignored when ``directions`` is given.
        window: Maximum completion tokens read per rollout (loss terms, seed
            positions, and target sum alike).
        gradient_off: If True, hold every read strength at 1 instead of
            reading it from the gradient, so the operator reads the frozen
            transport alone (the paper's gradient-off control).
        per_position: If given, additionally keep the target position open for
            the first ``per_position`` completion positions, producing the
            per-position operator stack for opening-versus-rest analyses.
        directions: A pre-drawn ``[k, d_model]`` basis with unit-norm rows,
            for reproducing an analysis whose read directions were drawn
            elsewhere. When omitted, the basis is drawn from ``basis_seed``
            via :func:`read_directions`.
        direction_chunk: Directions transported per vector-Jacobian product.
            ``None`` (the default) batches all ``k`` in one VJP via
            ``is_grads_batched``, the fast path; an integer transports that
            many at a time, trading speed for peak memory; ``1`` uses plain
            unbatched VJPs, the escape hatch for an architecture whose
            backward does not compose with ``torch.vmap``. Every setting
            produces the identical operator.

    Raises:
        ValueError: If the layer pair, ``k``, ``window``, ``per_position``,
            ``directions``, or ``direction_chunk`` is invalid.

    Note:
        Cost scales with ``k * n_rollouts`` VJP work; batching the directions
        (the default) amortizes the backward traversal so the wall clock is
        far below ``k`` sequential backwards. Peak memory adds one
        ``[chunk, seq, d_model]`` seed-field stack in the model dtype on top
        of one sequence's activations. Rollouts whose gradients are
        non-finite (near-random checkpoints can produce them) are dropped and
        counted in ``kept`` (non-finite transport likewise), matching
        :class:`~murano.steps.gradients.RecordGradients`.
    """

    reads = [keys.ROLLOUTS]
    writes = [keys.GFC_OPERATOR]
    read_types = {keys.ROLLOUTS: RolloutBatch}
    write_types = {keys.GFC_OPERATOR: GFCOperatorResult}

    def __init__(
        self,
        model: ModelBackend,
        source_layer: int,
        target_layer: int,
        k: int = 128,
        basis_seed: int = 0,
        window: int = 512,
        gradient_off: bool = False,
        per_position: int | None = None,
        directions: Tensor | None = None,
        direction_chunk: int | None = None,
    ):
        if not 0 <= target_layer < source_layer < model.n_layers:
            raise ValueError(
                f"Need 0 <= target_layer < source_layer < n_layers="
                f"{model.n_layers}; got source {source_layer}, target "
                f"{target_layer}. The operator transports backwards, from a "
                f"later layer to an earlier one."
            )
        if not 0 < k <= model.d_model:
            raise ValueError(f"k must be in [1, d_model={model.d_model}], got {k}.")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        if per_position is not None and per_position < 1:
            raise ValueError(
                f"per_position must be a positive horizon or None, got {per_position}."
            )
        if directions is not None:
            if directions.shape != (k, model.d_model):
                raise ValueError(
                    f"directions must be [k={k}, d_model={model.d_model}], got "
                    f"{tuple(directions.shape)}."
                )
            norms = directions.float().norm(dim=1)
            if not bool((norms - 1.0).abs().lt(1e-3).all()):
                raise ValueError(
                    "directions must have unit-norm rows; draw them with "
                    "read_directions or normalize before passing."
                )
        if direction_chunk is not None and direction_chunk < 1:
            raise ValueError(
                f"direction_chunk must be a positive count or None, got "
                f"{direction_chunk}."
            )
        self.model = model
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.k = k
        self.basis_seed = basis_seed
        self.window = window
        self.gradient_off = gradient_off
        self.per_position = per_position
        self.directions = directions
        self.direction_chunk = direction_chunk

    def __call__(self, results: Results) -> Results:
        rollouts: RolloutBatch = results[keys.ROLLOUTS]
        model = self.model
        last = model.n_layers - 1
        layers = sorted({self.target_layer, self.source_layer, last})
        directions = (
            self.directions.float().cpu()
            if self.directions is not None
            else read_directions(model.d_model, self.k, self.basis_seed)
        )

        operator = zeros(self.k, model.d_model)
        horizon = self.per_position
        per_position = (
            zeros(horizon, self.k, model.d_model) if horizon is not None else None
        )
        position_counts = zeros(horizon, dtype=long) if horizon is not None else None
        nll = full((len(rollouts),), float("nan"))
        kept = full((len(rollouts),), False)

        logger.info(
            "GFC operator: %d rollouts, layers %d->%d, k=%d, gradient_off=%s",
            len(rollouts),
            self.source_layer,
            self.target_layer,
            self.k,
            self.gradient_off,
        )
        for n, ids in enumerate(rollouts.input_ids):
            start, end = _completion_window(
                ids, rollouts.prompt_lengths[n], self.window
            )
            hidden = model.grad_forward(ids.unsqueeze(0), layers)
            source = hidden[self.source_layer]
            target = hidden[self.target_layer]

            strengths = self._read_strengths(
                model, hidden[last], source, ids, start, end, directions, nll, n
            )
            if strengths is None:
                continue

            contribution = zeros_like(operator)
            pos_contribution = (
                zeros(horizon, self.k, model.d_model) if horizon is not None else None
            )
            chunk = self.direction_chunk or self.k
            for begin in range(0, self.k, chunk):
                stop = min(begin + chunk, self.k)
                last_chunk = stop == self.k
                # Seed field per direction in the chunk: a_it * d_i at every
                # window position, zero elsewhere. [c, T, d_model] built in one
                # broadcast, then placed into the full-sequence field stack.
                seeds = strengths[:, begin:stop].T.unsqueeze(-1) * directions[
                    begin:stop
                ].unsqueeze(1)
                fields = zeros(
                    stop - begin,
                    *source.shape,
                    dtype=source.dtype,
                    device=source.device,
                )
                fields[:, 0, start:end] = seeds.to(source.device, source.dtype)
                if fields.shape[0] == 1:
                    # Plain unbatched VJP: the escape hatch for architectures
                    # whose backward does not compose with torch.vmap.
                    (transported,) = torch.autograd.grad(
                        source,
                        target,
                        grad_outputs=fields[0],
                        retain_graph=not last_chunk,
                    )
                    transported = transported.unsqueeze(0)
                else:
                    (transported,) = torch.autograd.grad(
                        source,
                        target,
                        grad_outputs=fields,
                        is_grads_batched=True,
                        retain_graph=not last_chunk,
                    )
                magnitudes = transported[:, 0, start:end].detach().float().cpu().abs()
                contribution[begin:stop] = magnitudes.sum(dim=1)
                if pos_contribution is not None:
                    p = min(magnitudes.shape[1], pos_contribution.shape[0])
                    pos_contribution[:p, begin:stop] = magnitudes[:, :p].transpose(0, 1)

            if not bool(isfinite(contribution).all()):
                logger.warning("Rollout %d produced non-finite transport; dropped.", n)
                nll[n] = float("nan")
                continue
            operator += contribution
            if horizon is not None:
                assert per_position is not None
                assert pos_contribution is not None
                assert position_counts is not None
                per_position += pos_contribution
                position_counts[: min(end - start, per_position.shape[0])] += 1
            kept[n] = True

        results[keys.GFC_OPERATOR] = GFCOperatorResult(
            operator=operator,
            per_position=per_position,
            position_counts=position_counts,
            nll=nll,
            kept=kept,
            source_layer=self.source_layer,
            target_layer=self.target_layer,
            k=self.k,
            basis_seed=None if self.directions is not None else self.basis_seed,
            gradient_off=self.gradient_off,
            window=self.window,
        )
        logger.info(
            "GFC operator done: kept %d/%d rollouts.",
            int(kept.sum()),
            len(rollouts),
        )
        return results

    def _read_strengths(
        self,
        model: ModelBackend,
        hidden_last: Tensor,
        source: Tensor,
        ids: Tensor,
        start: int,
        end: int,
        directions: Tensor,
        nll: Tensor,
        n: int,
    ) -> Tensor | None:
        """Per-position read strengths ``[T, k]``, or None to drop the rollout.

        Normally the strengths are the gradient's components along each
        direction and the graph is retained for the transport VJPs; with
        ``gradient_off`` every strength is 1 and only the (grad-free) NLL is
        computed. Fills ``nll[n]`` as a side effect.
        """
        context = torch.no_grad() if self.gradient_off else contextlib.nullcontext()
        with context:
            loglik = _completion_loglik(model, hidden_last, ids, start, end)
        nll[n] = -loglik.item() / (end - start)
        if self.gradient_off:
            return ones(end - start, self.k)
        (grad,) = torch.autograd.grad(loglik, source, retain_graph=True)
        grad_window = grad[0, start:end].detach().float().cpu()
        if not bool(isfinite(grad_window).all()):
            logger.warning("Rollout %d produced non-finite gradients; dropped.", n)
            nll[n] = float("nan")
            return None
        return grad_window @ directions.T
