"""RACE step: rank neurons by how consistently they drive their module's update.

A module's write to the residual stream is a sum of per-neuron writes: the MLP
update is ``W_down a`` and the attention update is ``W_O h``, so channel ``j`` of
the output projection's input writes ``a_j W[:, j]``. RACE scores each neuron on
every observed token by how much of that write lies along the module's own
normalised update (residual-direction alignment), then pools the scores over a
domain with a Normal-Inverse-Gamma posterior. The ranking score (CAM) is the
lower credible bound on the posterior mean, so a neuron that pushes hard on a
few tokens but idles or flips sign on the rest ranks below a steady one.

The step streams teacher-forced prompts once and keeps three numbers per neuron
(count, mean, centred sum of squares), so the prior and ``gamma`` can change
afterwards without another forward pass. Selection is per layer and module
(:meth:`RACEResult.select`), optionally skipping the top neurons of a reference
domain's result (reference-set filtering), and returns a
:class:`~murano.artifacts.NeuronSelection`. ``SelectComponents`` is not used
because it ranks whole components, not the ``10^5``-``10^6`` neurons of a model.

Attribution:
    Method: Wang, Liu, Zhang, Han et al. (2026), "RACE: Scalable Statistical
    Estimation of Functional Consistency in LLM Neurons", EMNLP 2026.
    https://arxiv.org/abs/2608.24758
    Reference code: Nexround/RACE@cf6fb23c22190c1c233f85256931e41441eda8a5 (MIT),
    adapted: ``compute_ffn_evidence_batch`` / ``compute_attn_evidence_batch`` and
    ``NIGPosterior`` (``packages/race/src/race/core/analyzer.py``) and the
    reference-set filter of ``create_reference_filtered_ablation_plan``
    (``packages/race-eval/src/race_eval/ablation/plan.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from torch import Tensor
from torch import float64, full, long, no_grad, sort  # pyright: ignore[reportPrivateImportUsage]
from torch import tensor, where, zeros, zeros_like  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import NeuronSelection, PromptBatch
from murano.logging import logger
from murano.nodes import MLP, SELF_ATTN, Neuron, Node, NodeDict, canonical_module
from murano.results import Results
from murano.steps.attention import _projection_weight
from murano.steps.base import Step
from murano.steps.record import _batched, _rank_positions

if TYPE_CHECKING:
    from murano.backend import ModelBackend


# Names of an MLP's down projection, tried in order (Llama/Qwen, GPT-2, OPT/Phi,
# GPT-NeoX/Falcon, original Llama ``feed_forward.w2``).
_DOWN_PROJ_NAMES = ("down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2")

_SCORES = ("cam", "neg_cam", "posterior_mean")


def _evidence(a: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    """Residual-direction alignment evidence (Eqs. 6-9), shape ``[N, C]``.

    Args:
        a: Output-projection inputs ``[N, C]`` (MLP intermediate activations or
            concatenated head outputs).
        weight: Projection weight ``[d_model, C]``, bias excluded.
        eps: Stabiliser added to the update norm.

    Returns:
        ``e[t, j] = a[t, j] * (W[:, j] . d_hat[t])`` with ``d_hat`` the unit
        module update. Rows sum to ``||W a|| ** 2 / (||W a|| + eps)``.
    """
    update = a @ weight.T
    d_hat = update / (update.norm(dim=-1, keepdim=True) + eps)
    return a * (d_hat @ weight)


def _merge_moments(
    n: int, mean: Tensor, ss: Tensor, batch: Tensor
) -> tuple[int, Tensor, Tensor]:
    """Fold a batch of evidence ``[m, C]`` into running float64 moments (Eq. 11)."""
    m = batch.shape[0]
    if m == 0:
        return n, mean, ss
    batch = batch.double()
    b_mean = batch.mean(dim=0)
    b_ss = (batch - b_mean).pow(2).sum(dim=0)
    total = n + m
    delta = b_mean - mean
    # Differs from Nexround/RACE@cf6fb23 packages/race/src/race/core/analyzer.py
    # NIGPosterior: merges centred float64 moments (Chan et al.) instead of float32
    # sum_e2 - n * e_bar**2, because the latter cancels catastrophically at large n.
    return (
        total,
        mean + delta * (m / total),
        ss + b_ss + delta.pow(2) * (n * m / total),
    )


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)`` by Lentz's continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        for num in (
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-16:
            break
    return math.exp(log_front) * h / a


def _t_cdf(t: float, df: float) -> float:
    """Student-t CDF with ``df`` degrees of freedom."""
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def _t_ppf(p: float, df: float) -> float:
    """Student-t quantile, found by bisection on :func:`_t_cdf`."""
    # Differs from Nexround/RACE@cf6fb23 packages/race/src/race/core/analyzer.py
    # cam_score: computes the quantile with the standard library instead of
    # scipy.stats.t.ppf, because scipy is not a Murano base dependency.
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    return 0.5 * (lo + hi)


def _ranking(scores: Tensor) -> list[int]:
    """Neuron indices by descending score, ties by ascending index."""
    return sort(-scores.double(), stable=True).indices.tolist()


def _budget(size: int, top_k: int | None, fraction: float) -> int:
    """How many neurons to take from one module of ``size`` neurons."""
    if top_k is not None:
        return min(top_k, size)
    # Differs from Nexround/RACE@cf6fb23 packages/race-eval/src/race_eval/ablation/
    # plan.py _resolve_llm_layer_k: rounds instead of ceil, as the same file's
    # select_top_indices does, because the file uses both and the issue's plan
    # settled on rounding.
    return min(max(int(round(size * fraction)), 1), size)


@dataclass
class RACEResult:
    """Streamed RACE statistics for every neuron of the scored modules.

    Neurons are the input channels of each module's output projection, keyed by
    ``Node(layer, "mlp")`` or ``Node(layer, "self_attn")``. Every module sees the
    same observations, so one count ``n`` covers all of them.

    Attributes:
        n: Number of observations (token positions) pooled.
        mean: ``{Node: [C] float64}`` empirical mean evidence per neuron.
        ss: ``{Node: [C] float64}`` centred sum of squared evidence per neuron.
        mu_0: Prior mean of the evidence.
        lambda_0: Prior pseudo-count for the mean.
        alpha_0: Prior shape of the evidence variance.
        beta_0: Prior scale of the evidence variance.
        gamma: One-sided credible level for :meth:`cam` (``0.05`` is a 95% bound).
        metadata: Provenance (prompt count, start positions).
    """

    n: int
    mean: dict[Node, Tensor]
    ss: dict[Node, Tensor]
    mu_0: float = 0.0
    lambda_0: float = 1.0
    alpha_0: float = 1.0
    beta_0: float = 1.0
    gamma: float = 0.05
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = NodeDict({k: v.detach().cpu() for k, v in self.mean.items()})
        self.ss = NodeDict({k: v.detach().cpu() for k, v in self.ss.items()})

    @property
    def lambda_n(self) -> float:
        """Posterior pseudo-count, Eq. 13."""
        return self.lambda_0 + self.n

    @property
    def alpha_n(self) -> float:
        """Posterior variance shape, Eq. 14."""
        return self.alpha_0 + self.n / 2.0

    @property
    def posterior_mean(self) -> dict[Node, Tensor]:
        """Signed posterior mean ``mu_n`` per neuron, Eq. 15."""
        return NodeDict(
            {
                node: (self.lambda_0 * self.mu_0 + self.n * mean) / self.lambda_n
                for node, mean in self.mean.items()
            }
        )

    @property
    def beta_n(self) -> dict[Node, Tensor]:
        """Posterior variance scale per neuron, Eq. 16."""
        return NodeDict(
            {
                node: self.beta_0
                + self.ss[node] / 2.0
                + self.lambda_0
                * self.n
                * (mean - self.mu_0).pow(2)
                / (2.0 * self.lambda_n)
                for node, mean in self.mean.items()
            }
        )

    def _bounds(self, gamma: float | None, sign: float) -> dict[Node, Tensor]:
        """CAM (``sign=1``) or negative CAM (``sign=-1``), Eq. 19."""
        gamma = self.gamma if gamma is None else gamma
        t_q = _t_ppf(1.0 - gamma, 2.0 * self.alpha_n)
        mu, beta = self.posterior_mean, self.beta_n
        out = NodeDict()
        for node, m in mu.items():
            sigma = (beta[node] / (self.alpha_n * self.lambda_n)).sqrt()
            m = sign * m
            out[node] = where(m > 0, (m - t_q * sigma).clamp(min=0.0), zeros_like(m))
        return out

    def cam(self, gamma: float | None = None) -> dict[Node, Tensor]:
        """Conservative alignment magnitude per neuron (the RACE score).

        Args:
            gamma: Credible level; defaults to :attr:`gamma`.

        Returns:
            ``{Node: [C]}`` non-negative scores, zero where ``mu_n <= 0``.
        """
        return self._bounds(gamma, 1.0)

    def neg_cam(self, gamma: float | None = None) -> dict[Node, Tensor]:
        """CAM of the negated evidence: steady opposition to the module update."""
        return self._bounds(gamma, -1.0)

    def scores(
        self, score: str = "cam", gamma: float | None = None
    ) -> dict[Node, Tensor]:
        """Return one of ``"cam"``, ``"neg_cam"`` or ``"posterior_mean"``."""
        if score not in _SCORES:
            raise ValueError(f"score must be one of {_SCORES}, got {score!r}.")
        if score == "posterior_mean":
            return self.posterior_mean
        return self.cam(gamma) if score == "cam" else self.neg_cam(gamma)

    def select(
        self,
        top_k: int | None = None,
        fraction: float = 0.01,
        *,
        score: str = "cam",
        gamma: float | None = None,
        exclude: RACEResult | None = None,
        exclude_top_k: int | None = None,
        exclude_fraction: float | None = None,
    ) -> NeuronSelection:
        """Pick the top neurons of every layer and module (§3.2, Eq. 20).

        Args:
            top_k: Neurons per module. Overrides ``fraction`` when given.
            fraction: Share of each module's neurons to keep, at least one.
            score: Ranking score, ``"cam"``, ``"neg_cam"`` or ``"posterior_mean"``.
            gamma: Credible level for the CAM scores; defaults to :attr:`gamma`.
            exclude: A reference-domain result. Its top neurons per module are
                skipped while walking this result's ranking (reference-set
                filtering), so the selection keeps target-specific neurons.
            exclude_top_k: Reference neurons to skip per module.
            exclude_fraction: Share of reference neurons to skip per module.
                With neither given, the reference budget equals the target one.

        Returns:
            A NeuronSelection ordered by module, then by descending score.

        Raises:
            ValueError: If a budget is not positive, both reference budgets are
                given, or ``exclude`` covers different modules or widths.
        """
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
        if exclude_top_k is not None and exclude_fraction is not None:
            raise ValueError("Pass exclude_top_k or exclude_fraction, not both.")
        if exclude_top_k is not None and exclude_top_k < 0:
            raise ValueError(f"exclude_top_k must be >= 0, got {exclude_top_k}.")
        if exclude_fraction is not None and not 0.0 <= exclude_fraction <= 1.0:
            raise ValueError(
                f"exclude_fraction must be in [0, 1], got {exclude_fraction}."
            )

        target = self.scores(score, gamma)
        reference = None if exclude is None else exclude.scores(score, gamma)
        if reference is not None and set(reference) != set(target):
            raise ValueError(
                "exclude must cover the same modules as this result; got "
                f"{sorted(map(str, reference))} vs {sorted(map(str, target))}."
            )

        neurons: list[Neuron] = []
        values: dict[Neuron, float] = {}
        for node in sorted(target):
            scores = target[node]
            size = scores.numel()
            k = _budget(size, top_k, fraction)
            ranking = _ranking(scores)
            if reference is not None:
                if reference[node].numel() != size:
                    raise ValueError(
                        f"{node} has {size} neurons here but "
                        f"{reference[node].numel()} in exclude."
                    )
                if exclude_top_k is not None:
                    k_ref = min(exclude_top_k, size)
                elif exclude_fraction is not None:
                    k_ref = (
                        0
                        if exclude_fraction == 0
                        else _budget(size, None, exclude_fraction)
                    )
                else:
                    k_ref = k
                skip = set(_ranking(reference[node])[:k_ref])
                ranking = [j for j in ranking if j not in skip]
                if len(ranking) < k:
                    logger.warning(
                        "RACE: %s keeps %d/%d neurons; only %d remain outside the "
                        "reference top %d.",
                        node,
                        len(ranking),
                        k,
                        len(ranking),
                        k_ref,
                    )
            for j in ranking[:k]:
                neuron = Neuron(node, j)
                neurons.append(neuron)
                values[neuron] = float(scores[j])
        return NeuronSelection(
            neurons=neurons,
            scores=values,
            metadata={
                "score": score,
                "gamma": self.gamma if gamma is None else gamma,
                "top_k": top_k,
                "fraction": None if top_k is not None else fraction,
                "filtered": exclude is not None,
            },
        )


class RACE(Step):
    """Score every neuron's functional consistency over a set of prompts.

    Runs the prompts teacher-forced in batches. A forward pre-hook on each
    scored module's output projection turns its input ``a`` into per-neuron
    evidence ``a_j * (W[:, j] . d_hat)``, with ``d_hat = W a / ||W a||`` the
    module's own update direction, and folds the evidence of every non-padding
    position into running per-neuron moments. Nothing larger than one batch's
    evidence is ever held.

    Reads from results:
        results[prompts_key]: PromptBatch.

    Writes to results:
        results[output_key]: RACEResult.

    Args:
        model: Model backend to run.
        modules: ``"mlp"``, ``"self_attn"`` or both. An MLP neuron is a channel
            of the down projection's input; an attention neuron is a channel of
            the concatenated head output entering ``o_proj``.
        layers: Layer indices to score, or ``"all"``.
        mu_0: Prior mean of the evidence.
        lambda_0: Prior pseudo-count for the mean (``> 0``).
        alpha_0: Prior shape of the evidence variance (``> 0``).
        beta_0: Prior scale of the evidence variance (``> 0``).
        gamma: One-sided credible level of the CAM bound, in ``(0, 1)``.
        eps: Stabiliser on the update norm.
        start_positions: Skip each prompt's first tokens: an int for every
            prompt, or one int per prompt, counted over the prompt's own
            tokens (padding excluded). Use it to score only a response that
            follows a prompt. ``None`` scores every token.
        batch_size: Prompts per forward pass.
        prompts_key: Results key to read the prompts from.
        output_key: Results key to write the result under.

    Raises:
        ValueError: If an argument is out of range.
        NotImplementedError: At run time, if a scored module has no known
            output projection (an MoE block or an unknown architecture).
    """

    def __init__(
        self,
        model: ModelBackend,
        modules: str | Sequence[str] = (MLP, SELF_ATTN),
        layers: list[int] | str = "all",
        *,
        mu_0: float = 0.0,
        lambda_0: float = 1.0,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        gamma: float = 0.05,
        eps: float = 1e-8,
        start_positions: int | Sequence[int] | None = None,
        batch_size: int = 8,
        prompts_key: str = keys.PROMPTS,
        output_key: str = keys.RACE,
    ):
        names = [modules] if isinstance(modules, str) else list(modules)
        self.modules = list(dict.fromkeys(canonical_module(m) for m in names))
        bad = [m for m in self.modules if m not in (MLP, SELF_ATTN)]
        if bad or not self.modules:
            raise ValueError(
                f"modules must be 'mlp', 'self_attn' or both, got {modules!r}."
            )
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError(f"layers as a string must be 'all', got {layers!r}.")
            self.layers = list(range(model.n_layers))
        else:
            self.layers = list(layers)
            out = [i for i in self.layers if not 0 <= i < model.n_layers]
            if out or not self.layers:
                raise ValueError(
                    f"layers must be a non-empty list in [0, {model.n_layers}), "
                    f"got {layers!r}."
                )
        for name, value in (
            ("lambda_0", lambda_0),
            ("alpha_0", alpha_0),
            ("beta_0", beta_0),
        ):
            if not value > 0:
                raise ValueError(f"{name} must be > 0, got {value}.")
        if not 0.0 < gamma < 1.0:
            raise ValueError(
                f"gamma must be in (0, 1), got {gamma}; 0.05 is a 95% bound."
            )
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        starts = (
            [start_positions] if isinstance(start_positions, int) else start_positions
        )
        if starts is not None and any(int(s) < 0 for s in starts):
            raise ValueError(f"start_positions must be >= 0, got {start_positions!r}.")

        self.model = model
        self.prior = (mu_0, lambda_0, alpha_0, beta_0)
        self.gamma = gamma
        self.eps = eps
        self.start_positions = start_positions
        self.batch_size = batch_size
        self.prompts_key = prompts_key
        self.output_key = output_key
        self.reads = [prompts_key]
        self.read_types = {prompts_key: PromptBatch}
        self.writes = [output_key]
        self.write_types = {output_key: RACEResult}

    def _projection(self, node: Node) -> Any:
        """Return the raw output projection whose input channels are the neurons."""
        if node.module == SELF_ATTN:
            return self.model.raw_attn_out_proj(node.layer, SELF_ATTN)
        mlp = self.model.raw_module(node.layer, MLP)
        if hasattr(mlp, "experts"):
            raise NotImplementedError(
                f"RACE: {node} is a mixture-of-experts block; score a dense MLP or "
                f"pass modules='self_attn'."
            )
        for name in _DOWN_PROJ_NAMES:
            if hasattr(mlp, name):
                return getattr(mlp, name)
        raise NotImplementedError(
            f"RACE: {node} ({type(mlp).__name__}) has no down projection named one "
            f"of {_DOWN_PROJ_NAMES}; pass modules='self_attn' to skip the MLP."
        )

    def _starts(self, n_prompts: int) -> Tensor:
        """Per-prompt start position over the prompt's own tokens."""
        if self.start_positions is None:
            return zeros(n_prompts, dtype=long)
        if isinstance(self.start_positions, int):
            return full((n_prompts,), self.start_positions, dtype=long)
        starts = list(self.start_positions)
        if len(starts) != n_prompts:
            raise ValueError(
                f"start_positions has {len(starts)} entries for {n_prompts} prompts; "
                f"pass one per prompt or a single int."
            )
        return tensor(starts, dtype=long)

    def __call__(self, results: Results) -> Results:
        prompts = results[self.prompts_key].prompts
        nodes = [Node(layer, m) for layer in self.layers for m in self.modules]
        logger.info("RACE: %d prompts, %d modules", len(prompts), len(nodes))
        starts = self._starts(len(prompts))

        projections = {node: self._projection(node) for node in nodes}
        moments: dict[Node, tuple[int, Tensor, Tensor]] = {}
        mask: dict[str, Tensor] = {}

        def hook(node: Node):
            def fn(module: Any, args: tuple) -> None:
                a = args[0][mask["obs"].to(args[0].device)].float()
                # Built per call so only one module's float32 copy is alive at once.
                weight = _projection_weight(module)
                # Differs from Nexround/RACE@cf6fb23 analyzer.py
                # compute_ffn_evidence_batch: evidence in float32 from model-dtype
                # activations instead of the model dtype, because bf16 evidence
                # loses the precision the moments are pooled at.
                e = _evidence(a, weight, self.eps).cpu()
                if node not in moments:
                    empty = zeros(e.shape[1], dtype=float64)
                    moments[node] = (0, empty, empty.clone())
                moments[node] = _merge_moments(*moments[node], e)

            return fn

        offset = 0
        handles = [p.register_forward_pre_hook(hook(n)) for n, p in projections.items()]
        try:
            with no_grad():
                for batch in _batched(prompts, self.batch_size):
                    # Right padding, as the reference's prefill_recorder.py does,
                    # so a left-padding tokenizer cannot shift absolute positions.
                    tok = self.model.tokenizer
                    prev_side = tok.padding_side
                    tok.padding_side = "right"
                    try:
                        tokens = tok(
                            batch,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            return_token_type_ids=False,
                        )
                    finally:
                        tok.padding_side = prev_side
                    attention_mask = tokens["attention_mask"]
                    # Differs from Nexround/RACE@cf6fb23 packages/race/src/race/llm/
                    # recording/prefill_recorder.py: observes every non-padding
                    # token from start_positions on, with no is_correct filter or
                    # last_k token mode, because the step teacher-forces given texts
                    # instead of the model's own graded generations.
                    start = starts[offset : offset + len(batch)].unsqueeze(1)
                    mask["obs"] = attention_mask.bool() & (
                        _rank_positions(attention_mask) >= start
                    )
                    offset += len(batch)
                    self.model.forward_logits(tokens, fn=None)
        finally:
            for handle in handles:
                handle.remove()

        n = next(iter(moments.values()))[0] if moments else 0
        mu_0, lambda_0, alpha_0, beta_0 = self.prior
        results[self.output_key] = RACEResult(
            n=n,
            mean={node: moments[node][1] for node in nodes if node in moments},
            ss={node: moments[node][2] for node in nodes if node in moments},
            mu_0=mu_0,
            lambda_0=lambda_0,
            alpha_0=alpha_0,
            beta_0=beta_0,
            gamma=self.gamma,
            metadata={"n_prompts": len(prompts), "start_positions": starts.tolist()},
        )
        logger.info("RACE: pooled %d observations", n)
        return results
