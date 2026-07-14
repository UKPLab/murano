"""LogitAttribution step: decompose a logit (difference) into per-component parts.

Direct logit attribution (DLA) answers "how does each component push the answer."
It splits the model's final residual into the pieces that wrote it (each attention
head, each MLP, the embedding), applies the final normalization with the scale
*frozen* from the full residual, and dots each piece with the unembedding
direction for the answer. Freezing the norm scale keeps the decomposition linear,
so the per-component contributions sum back to the model's true logit difference;
that completeness is checked and reported rather than assumed.

Per-head contributions are read off the attention output projection directly
(``o_proj(mask_head(z)) - o_proj(0)``) so the math is independent of how a given
architecture lays out that weight (e.g. gpt2's transposed ``Conv1D``). The
arithmetic runs in float32 so the completeness check is tight even on bf16 models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from torch import Tensor, arange  # pyright: ignore[reportPrivateImportUsage]
from torch import float32, no_grad  # pyright: ignore[reportPrivateImportUsage]
from torch import stack, zeros_like  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import MLP, SELF_ATTN, Node, NodeDict
from murano.results import Results
from murano.steps.base import Step
from murano.steps.metrics import (
    _answer_logits,
    _answer_positions,
    _answer_token_ids,
    _check_token_ids,
    _gather_answer,
)
from murano._proxy import unwrap_traced

if TYPE_CHECKING:
    from murano.backend import ModelBackend


@dataclass
class LogitAttributionResult:
    """Per-component contributions to a target logit (difference).

    Each value is the signed amount that component adds to the target at the
    answer position, in logit units, averaged over the batch. The named
    contributions plus ``embed_contribution`` and ``other_contribution`` sum to
    ``total`` up to ``completeness_error``.

    Attributes:
        contributions: ``{Node: float}`` mean contribution of each attention head
            (``Node(layer, "self_attn", head=h)``) and MLP (``Node(layer, "mlp")``).
        embed_contribution: Mean contribution of the embedding (pre-layer-0
            residual).
        other_contribution: Mean contribution of everything not named above:
            output-projection biases, the final-norm bias, any unembedding bias,
            and any layers excluded from ``layers``.
        target: ``"logit_diff"`` when an incorrect answer was given, else
            ``"logit"``.
        total: Mean true target value at the answer position (the quantity being
            attributed).
        completeness_error: Maximum absolute difference, over the batch, between
            the summed contributions and the true target. Near zero confirms the
            frozen-norm decomposition reproduces the model.
        per_example: Optional ``{Node: list[float]}`` per-example contributions
            for the named components.
        metadata: Free-form metadata (answer positions, norm type, target).
    """

    contributions: dict[Node, float]
    embed_contribution: float
    other_contribution: float
    target: str
    total: float
    completeness_error: float
    per_example: dict[Node, list[float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.contributions = NodeDict(self.contributions)
        if self.per_example is not None:
            self.per_example = NodeDict(self.per_example)


class LogitAttribution(Step):
    """Decompose a target logit (difference) into per-component contributions.

    Runs the prompts, splits the final residual into the embedding, each
    attention head's output, and each MLP's output, applies the final norm with
    its scale frozen from the full residual, and projects each piece onto the
    answer direction ``W_U[correct] - W_U[incorrect]`` (or ``W_U[correct]`` when
    no incorrect answer is given). The contributions sum to the model's true
    logit difference, which is verified and exposed as ``completeness_error``.

    Reads from results:
        results['prompts']: PromptBatch.

    Writes to results:
        results[output_key]: LogitAttributionResult.

    Args:
        model: Model backend to run.
        correct: Correct-answer token spec (token id, string, per-example
            sequence, or tensor), as accepted by the metric steps.
        incorrect: Optional incorrect-answer spec. When given, the target is the
            logit difference; when ``None``, the target is the correct token's
            logit.
        layers: Layer indices to attribute, or ``"all"``. Layers left out fold
            into ``other_contribution``.
        include_mlp: Attribute each layer's MLP output when True.
        include_embed: Attribute the embedding (pre-layer-0 residual) when True;
            otherwise it folds into ``other_contribution``.
        positions: Optional explicit answer position(s); otherwise the last real
            token is used (via the attention mask).
        per_example: Keep per-example contributions on the result when True.
        output_key: Results key to write the result under.

    Raises:
        ValueError: If a target token id is outside the vocabulary.
    """

    reads = [keys.PROMPTS]
    read_types = {keys.PROMPTS: PromptBatch}

    def __init__(
        self,
        model: ModelBackend,
        correct: int | str | Sequence[int] | Sequence[str] | Tensor,
        incorrect: int | str | Sequence[int] | Sequence[str] | Tensor | None = None,
        layers: list[int] | str = "all",
        *,
        include_mlp: bool = True,
        include_embed: bool = True,
        positions: int | Sequence[int] | Tensor | None = None,
        per_example: bool = False,
        output_key: str = keys.LOGIT_ATTRIBUTION,
    ):
        self.model = model
        self.correct = correct
        self.incorrect = incorrect
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError(f"layers as string must be 'all', got {layers!r}")
            self.layers: list[int] = list(range(model.n_layers))
        else:
            self.layers = list(layers)
        self.include_mlp = include_mlp
        self.include_embed = include_embed
        self.positions = positions
        self.per_example = per_example
        self.output_key = output_key
        self.writes = [output_key]
        self.write_types = {output_key: LogitAttributionResult}

    def __call__(self, results: Results) -> Results:
        prompt_batch: PromptBatch = results[keys.PROMPTS]
        prompts = prompt_batch.prompts
        logger.info(
            "LogitAttribution: %d prompts, %d layers", len(prompts), len(self.layers)
        )

        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        attention_mask: Tensor = tokens["attention_mask"]
        param = next(self.model.hf_model.parameters())
        device, model_dtype = param.device, param.dtype

        with no_grad():
            captured = self._capture(tokens, device, model_dtype)

            # True logits run in the model dtype, then everything below is float32
            # so the completeness check stays tight on bf16 models.
            true_logits = self.model.project_on_vocab(captured["resid"]).float()
            positions = _answer_positions(true_logits, attention_mask, self.positions)
            batch, _, vocab = true_logits.shape

            correct_ids = _answer_token_ids(self.correct, batch, self.model, device)
            _check_token_ids(correct_ids, vocab, "correct")
            incorrect_ids = None
            if self.incorrect is not None:
                incorrect_ids = _answer_token_ids(
                    self.incorrect, batch, self.model, device
                )
                _check_token_ids(incorrect_ids, vocab, "incorrect")

            direction = self._direction(correct_ids, incorrect_ids, device)
            true_target = self._true_target(
                true_logits, positions, correct_ids, incorrect_ids
            )

            r_pos = _at_positions(captured["resid"], positions).float()
            scale, gamma, beta = self._frozen_norm_params(r_pos)

            per_example, named_resid = self._named_contributions(
                captured, positions, direction, scale, gamma, beta, device, model_dtype
            )
            embed_pe, embed_resid = self._embed_contribution(
                captured, positions, direction, scale, gamma, beta
            )
            other_pe = self._other_contribution(
                r_pos, named_resid + embed_resid, direction, scale, gamma, beta
            )

        # Per-example completeness: the worst-case reconstruction error over the
        # batch, so per-example cancellation cannot hide behind the batch mean.
        reconstructed = embed_pe + other_pe
        for value in per_example.values():
            reconstructed = reconstructed + value
        completeness = float((reconstructed - true_target).abs().max().item())
        named_means = {node: float(v.mean().item()) for node, v in per_example.items()}

        results[self.output_key] = LogitAttributionResult(
            contributions=named_means,
            embed_contribution=float(embed_pe.mean().item()),
            other_contribution=float(other_pe.mean().item()),
            target="logit_diff" if incorrect_ids is not None else "logit",
            total=float(true_target.mean().item()),
            completeness_error=completeness,
            per_example=(
                {node: v.tolist() for node, v in per_example.items()}
                if self.per_example
                else None
            ),
            metadata={
                "positions": positions.tolist(),
                "norm": "layernorm" if beta is not None else "rmsnorm",
                "target": "logit_diff" if incorrect_ids is not None else "logit",
            },
        )
        logger.info(
            "LogitAttribution: total=%.4f completeness_error=%.2e",
            float(true_target.mean().item()),
            completeness,
        )
        return results

    def _capture(self, tokens: Any, device: Any, model_dtype: Any) -> dict[str, Any]:
        """Capture residual, embedding, per-head z, and MLP outputs from ``tokens``.

        Uses one trace per module kind (mirroring Record) so nnsight does not hit
        out-of-order saves on nested submodules. Tensors are returned on
        ``device`` in the model dtype; head z keeps the ``[B, S, H, Dh]`` shape.
        """
        last = self.model.n_layers - 1
        captured: dict[str, Any] = {}

        resid_saved: dict[str, Any] = {}
        with self.model.trace(tokens):
            resid_saved["r"] = self.model.layer(last).output.save()
        captured["resid"] = unwrap_traced(resid_saved["r"]).to(
            device=device, dtype=model_dtype
        )

        # The embedding (pre-layer-0 residual) is captured in its own trace:
        # nnsight rejects saving a layer's input alongside a later layer's output
        # in one trace as an out-of-order access. unwrap_traced takes the first
        # positional arg, i.e. the hidden_states the block runs on.
        if self.include_embed:
            embed_saved: dict[str, Any] = {}
            with self.model.trace(tokens):
                embed_saved["e"] = self.model.layer(0).input.save()
            captured["embed"] = unwrap_traced(embed_saved["e"]).to(
                device=device, dtype=model_dtype
            )

        attn_saved: dict[int, Any] = {}
        with self.model.trace(tokens):
            for layer in self.layers:
                attn_saved[layer] = self.model.attn_out_proj(
                    layer, SELF_ATTN
                ).input.save()
        captured["attn"] = {}
        width = self.model.n_heads * self.model.head_dim
        for layer in self.layers:
            value = unwrap_traced(attn_saved[layer])
            b, s, d = value.shape
            if d != width:
                raise ValueError(
                    f"attention output-projection input width {d} at layer {layer} "
                    f"is not n_heads*head_dim ({self.model.n_heads}*"
                    f"{self.model.head_dim}={width}); cannot split per head."
                )
            captured["attn"][layer] = value.reshape(
                b, s, self.model.n_heads, self.model.head_dim
            ).to(device=device, dtype=model_dtype)

        captured["mlp"] = {}
        if self.include_mlp:
            mlp_saved: dict[int, Any] = {}
            with self.model.trace(tokens):
                for layer in self.layers:
                    mlp_saved[layer] = self.model.resolve_module(
                        layer, MLP
                    ).output.save()
            for layer in self.layers:
                captured["mlp"][layer] = unwrap_traced(mlp_saved[layer]).to(
                    device=device, dtype=model_dtype
                )
        return captured

    def _direction(
        self, correct_ids: Tensor, incorrect_ids: Tensor | None, device: Any
    ) -> Tensor:
        """Return the float32 unembedding direction ``[B, d]`` for the answer (set)."""
        w_u = self.model.unembed_weight.to(device=device, dtype=float32)
        direction = _rows_for(w_u, correct_ids)
        if incorrect_ids is not None:
            direction = direction - _rows_for(w_u, incorrect_ids)
        return direction

    def _true_target(
        self,
        true_logits: Tensor,
        positions: Tensor,
        correct_ids: Tensor,
        incorrect_ids: Tensor | None,
    ) -> Tensor:
        """Gather the model's true target value ``[B]`` at the answer position."""
        answer_logits = _answer_logits(true_logits, positions)
        target = _gather_answer(answer_logits, correct_ids)
        if incorrect_ids is not None:
            target = target - _gather_answer(answer_logits, incorrect_ids)
        return target

    def _uses_unit_offset_norm(self, ln: Any) -> bool:
        """Whether the final norm scales by ``(1 + weight)`` instead of ``weight``.

        Gemma-family RMSNorm (``Gemma2RMSNorm`` etc.) applies ``x_normed * (1 +
        weight)``; the frozen-norm decomposition must add 1 to the stored weight
        or the reconstruction misses the identity term and completeness breaks.
        Detected by the norm class name or the model type, both robust to the
        nnsight Envoy wrapper.
        """
        try:
            if "Gemma" in type(getattr(ln, "_module", ln)).__name__:
                return True
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            config = getattr(self.model.hf_model, "config", None)
            return str(getattr(config, "model_type", "")).startswith("gemma")
        except Exception:  # pragma: no cover - defensive
            return False

    def _frozen_norm_params(
        self, r_pos: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Return ``(scale[B], gamma[d], beta[d] or None)`` frozen from ``r_pos``.

        LayerNorm centers and divides by the residual's standard deviation;
        RMSNorm divides by its root-mean-square. ``beta`` is returned only for
        LayerNorm, where it is a constant added once to the normalized vector.
        Gemma-family RMSNorm scales by ``(1 + weight)``, so its effective gamma
        adds one to the stored weight.
        """
        ln = self.model.final_norm
        gamma = ln.weight.to(device=r_pos.device, dtype=float32)
        if self._uses_unit_offset_norm(ln):
            gamma = gamma + 1.0
        eps = getattr(ln, "eps", None)
        if eps is None:
            eps = getattr(ln, "variance_epsilon", 1e-5)
        # ln is an nnsight Envoy, so isinstance(ln, LayerNorm) is always False;
        # detect by the LayerNorm-only ``normalized_shape`` attribute, which
        # proxies through. LayerNorm centers and has a bias; RMSNorm does neither.
        if hasattr(ln, "normalized_shape"):
            centered = r_pos - r_pos.mean(dim=-1, keepdim=True)
            scale = (centered.pow(2).mean(dim=-1) + eps).rsqrt()
            beta = (
                ln.bias.to(device=r_pos.device, dtype=float32)
                if ln.bias is not None
                else None
            )
            return scale, gamma, beta
        scale = (r_pos.pow(2).mean(dim=-1) + eps).rsqrt()
        return scale, gamma, None

    def _frozen_norm(
        self, c: Tensor, scale: Tensor, gamma: Tensor, beta: Tensor | None
    ) -> Tensor:
        """Apply the frozen norm to a component ``[B, d]`` (beta is shared, added once)."""
        if beta is not None:
            c = c - c.mean(dim=-1, keepdim=True)
        return c * scale.unsqueeze(-1) * gamma

    def _project(
        self,
        c_resid: Tensor,
        direction: Tensor,
        scale: Tensor,
        gamma: Tensor,
        beta: Tensor | None,
    ) -> Tensor:
        """Frozen-norm a component then dot it with the direction: ``[B]``."""
        return (self._frozen_norm(c_resid, scale, gamma, beta) * direction).sum(dim=-1)

    def _named_contributions(
        self,
        captured: dict[str, Any],
        positions: Tensor,
        direction: Tensor,
        scale: Tensor,
        gamma: Tensor,
        beta: Tensor | None,
        device: Any,
        model_dtype: Any,
    ) -> tuple[dict[Node, Tensor], Tensor]:
        """Per-example contributions for each head and MLP, and their summed residual."""
        per_example: dict[Node, Tensor] = {}
        # named_resid accumulates each piece's raw residual; the caller subtracts
        # it from r to get the "other" remainder for the completeness identity.
        named_resid = zeros_like(direction)

        for layer in self.layers:
            z_pos = _at_positions(captured["attn"][layer], positions)  # [B, H, Dh]
            head_resid = self._head_residuals(
                layer, z_pos, model_dtype
            )  # [B, H, d] f32
            for h in range(head_resid.shape[1]):
                c = head_resid[:, h, :]
                per_example[Node(layer, SELF_ATTN, head=h)] = self._project(
                    c, direction, scale, gamma, beta
                )
                named_resid = named_resid + c
            if self.include_mlp:
                m = _at_positions(captured["mlp"][layer], positions).float()
                per_example[Node(layer, MLP)] = self._project(
                    m, direction, scale, gamma, beta
                )
                named_resid = named_resid + m
        return per_example, named_resid

    def _head_residuals(self, layer: int, z_pos: Tensor, model_dtype: Any) -> Tensor:
        """Per-head residual contributions ``[B, H, d]`` (float32) via the o_proj module.

        Calls the output projection on a per-head-masked input and subtracts the
        projection of zeros, so the result is each head's linear contribution (the
        shared projection bias is excluded and folds into ``other``). Using the
        live module (the same one ``_capture`` reads) keeps this correct
        regardless of the weight layout and needs no separate resolution path.
        """
        o_proj = self.model.attn_out_proj(layer, SELF_ATTN)
        batch, n_heads, head_dim = z_pos.shape
        flat = z_pos.reshape(batch, n_heads * head_dim).to(model_dtype)
        bias = o_proj(zeros_like(flat))
        contribs = []
        for h in range(n_heads):
            masked = zeros_like(flat)
            sl = slice(h * head_dim, (h + 1) * head_dim)
            masked[:, sl] = flat[:, sl]
            contribs.append((o_proj(masked) - bias).float())
        return stack(contribs, dim=1)

    def _embed_contribution(
        self,
        captured: dict[str, Any],
        positions: Tensor,
        direction: Tensor,
        scale: Tensor,
        gamma: Tensor,
        beta: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Per-example embedding contribution ``[B]`` and its residual ``[B, d]``."""
        if not self.include_embed:
            return direction.new_zeros(direction.shape[0]), zeros_like(direction)
        embed_pos = _at_positions(captured["embed"], positions).float()
        value = self._project(embed_pos, direction, scale, gamma, beta)
        return value, embed_pos

    def _other_contribution(
        self,
        r_pos: Tensor,
        named_resid: Tensor,
        direction: Tensor,
        scale: Tensor,
        gamma: Tensor,
        beta: Tensor | None,
    ) -> Tensor:
        """Per-example contribution ``[B]`` of everything not named.

        Covers projection biases, the final-norm bias, and any excluded layers.
        Computed from the real residual remainder (``r`` minus the named pieces)
        plus the final-norm bias, so the named contributions plus this term
        reconstruct the true target. An unembedding (lm_head) bias, which the
        target architectures do not have, would instead show up in
        ``completeness_error``.
        """
        value = self._project(r_pos - named_resid, direction, scale, gamma, beta)
        if beta is not None:
            value = value + (beta * direction).sum(dim=-1)
        return value


def _at_positions(activations: Tensor, positions: Tensor) -> Tensor:
    """Gather one position per example along dim 1: ``[B, S, ...] -> [B, ...]``."""
    rows = arange(activations.shape[0], device=activations.device)
    return activations[rows, positions.to(activations.device)]


def _rows_for(weight: Tensor, ids: Tensor) -> Tensor:
    """Gather rows for token ids, averaging over a token set: ``[B, d]``."""
    rows = weight[ids]
    if rows.ndim == 3:
        rows = rows.mean(dim=1)
    return rows
