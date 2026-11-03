"""InformationFlow step: the routes information takes to a prediction.

Information flow routes (Ferrando & Voita, 2024) answer "which heads and MLPs
does this prediction actually go through", from one forward pass and with no
contrastive template, patching or gradients. Every residual-stream vector is a
sum of the vectors that wrote it: after attention, the incoming residual plus one
vector per (head, source token); after the FFN, the incoming residual plus the
FFN output. Each part gets an importance, its L1 proximity to the whole,
normalized over the parts (Eq. 1). Starting from the prediction and walking down,
the routes keep every edge whose importance exceeds a threshold.

The result's ``contributions`` holds, per head and per MLP, the share of prompts
whose routes go through that component. :class:`~murano.steps.select.SelectComponents`
can rank it directly, and :class:`~murano.steps.patch.Patch` can then test the
selection causally. The proximity normalization is :func:`_importance`, kept
module-level so the same decomposition can serve ALTI-style input attribution.

Attribution:
    Method: Ferrando & Voita (2024), "Information Flow Routes: Automatically
    Interpreting Language Models at Scale", EMNLP 2024. https://arxiv.org/abs/2403.00824
    Reference code: facebookresearch/llm-transparency-tool@f1340f0757b959c75c139f7aa91aef16eddced67
    (CC-BY-NC-4.0), used as oracle only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from torch import Tensor, cat, einsum, no_grad, stack  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import (
    MLP,
    RESID_MID,
    RESID_POST,
    RESID_PRE,
    SELF_ATTN,
    Node,
    NodeDict,
)
from murano.results import Results
from murano.steps.attention import (
    _capture_patterns,
    _head_ov_weights,
    _head_value_block,
    _positions,
)
from murano.steps.base import Step
from murano.steps.sae import _is_sandwich_norm
from murano._proxy import unwrap_traced

if TYPE_CHECKING:
    from murano.backend import ModelBackend

_TINY = 1e-12


# ── Importance (shared with ALTI-style attribution) ──────────────────


def _importance(parts: Tensor, whole: Tensor, order: int = 1) -> Tensor:
    """Return each part's share of ``whole``, normalized over the part axis (Eq. 1).

    ``proximity(z, y) = max(||y|| - ||y - z||, 0)``, then each proximity is
    divided by their sum. A zero part (a masked future token) gets exactly 0.

    Args:
        parts: ``[..., n_parts, d]`` vectors that sum to (roughly) ``whole``.
        whole: ``[..., d]``.
        order: Norm order of the distance, 1 (the paper's) or 2.

    Returns:
        ``[..., n_parts]`` importances summing to 1 (0 everywhere if every
        proximity is 0).
    """
    whole = whole.unsqueeze(-2)
    proximity = whole.norm(p=order, dim=-1) - (whole - parts).norm(p=order, dim=-1)
    # Differs from facebookresearch/llm-transparency-tool@f1340f0 routes/contributions.py:
    # clips the proximity at 0 instead of EPS=1e-5, because the paper's Eq. 1 uses
    # max(., 0) and an EPS floor gives masked future tokens a nonzero edge.
    proximity = proximity.clamp(min=0)
    return proximity / proximity.sum(dim=-1, keepdim=True).clamp(min=_TINY)


def _threshold(importance: Tensor, threshold: float) -> Tensor:
    """Zero importances at or below ``threshold`` and renormalize the rest (§2.2)."""
    kept = importance * (importance > threshold)
    return kept / kept.sum(dim=-1, keepdim=True).clamp(min=_TINY)


# ── Result ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteEdge:
    """One edge of an information flow route.

    Residual-stream nodes are ``Node(l, "resid_mid")`` (after layer ``l``'s
    attention) and ``Node(l, "resid_post")`` (after its FFN); the embeddings are
    ``Node(0, "resid_pre")``. An FFN edge goes ``resid_mid -> Node(l, "mlp") ->
    resid_post``, both halves carrying the FFN importance. An attention edge from
    a token to itself also carries the residual's importance (Eq. 4).

    Attributes:
        source: Lower node.
        source_pos: Token index of the source, into the prompt's real tokens.
        dest: Upper node.
        dest_pos: Token index of the destination.
        importance: The edge's importance in its destination node, summed over
            heads for attention edges.
    """

    source: Node
    source_pos: int
    dest: Node
    dest_pos: int
    importance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Node.coerce(self.source))
        object.__setattr__(self, "dest", Node.coerce(self.dest))


@dataclass
class InformationFlowResult:
    """Information flow routes to each prompt's prediction, and per-component readouts.

    Attributes:
        contributions: ``{Node: float}`` share of prompts whose routes pass
            through each head (``Node(l, "self_attn", head=h)``: one of its
            sub-edges is in a route) and each MLP (``Node(l, "mlp")``). It is an
            activation frequency that depends on ``threshold`` and the batch,
            not a causal effect.
        head_importance: ``{Node: float}`` mean importance of each head's
            sub-edges inside the routes, 0 for a head that is never in one.
        edges: Per prompt, the route's :class:`RouteEdge` list.
        tokens: Per prompt, the decoded real (unpadded) tokens; edge positions
            index into these.
        positions: Per prompt, the token index the routes start from.
        threshold: The importance threshold ``tau`` used.
        metadata: Free-form provenance (norm order, renormalization).
    """

    contributions: dict[Node, float]
    head_importance: dict[Node, float]
    edges: list[list[RouteEdge]]
    tokens: list[list[str]]
    positions: list[int]
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.contributions = NodeDict(self.contributions)
        self.head_importance = NodeDict(self.head_importance)


# ── Step ──────────────────────────────────────────────────────────────


def _head_outputs(attn_in: Tensor, w_v: Tensor, b_v: Tensor, w_o: Tensor) -> Tensor:
    """Return ``f^h(x_j) = (W_V LN(x_j) + b_V) W_O`` for every token and head ``[S, H, d]``.

    ``attn_in`` ``[S, d]`` is the attention module's (normed) input, so the layer
    norm is linearized with the scale the forward pass used (App. B.1).
    Weighting row ``j`` by ``alpha[h, pos, j]`` gives head ``h``'s part from
    token ``j`` in ``resid_mid`` at ``pos``.
    """
    # Differs from the paper (App. B.1): reads LN(x_j) as the attention module's
    # input instead of building the linearized norm matrix L, because with sigma
    # taken from the forward pass the two give the same vector.
    # Differs from the paper (Eq. 2), which ignores biases: b_V stays inside each
    # head's part, because the attention weights sum to 1, so b_V W_O is exactly
    # what the head adds from its bias.
    values = einsum("sd,hkd->shk", attn_in, w_v) + b_v
    return einsum("shk,hdk->shd", values, w_o)


class InformationFlow(Step):
    """Extract the information flow routes to each prompt's prediction.

    Runs the prompts once, splits every residual-stream vector into the parts
    that wrote it, scores each part's importance (:func:`_importance`), and walks
    down from the target token's last-layer residual keeping each edge whose
    importance exceeds ``threshold``. Requires a model loaded with
    ``enable_attention_probs=True``.

    A head's part for source token ``j`` is ``alpha[pos, j] * (W_V LN(x_j) +
    b_V) W_O``, using the attention module's actual normed input, so the layer
    norm is linearized with the scale the forward pass used.

    Reads from results:
        results[prompts_key]: PromptBatch (default ``prompts``).

    Writes to results:
        results[output_key]: InformationFlowResult.

    Args:
        model: Model backend loaded with attention weights enabled.
        positions: Token position(s) to start the routes from, in the
            :func:`~murano.steps.metrics._answer_positions` form. They index the
            padded batch, so on a padded batch ``-1`` can land on padding;
            ``None`` uses each prompt's last real token whatever the padding.
        threshold: Importance threshold ``tau`` in ``[0, 1)``. The paper uses
            0.03 for head-frequency plots and 0.04 for route drawings.
        renormalize: When True (the paper's experiments), each node's head
            sub-edges and residual at or below ``threshold`` are zeroed and the
            rest renormalized before summing over heads. When False, the
            head-summed edge is thresholded directly.
        norm_order: Distance norm of the proximity, 1 (the paper's) or 2.
        prompts_key: Results key to read the prompt batch from.
        output_key: Results key to write the result under.

    Raises:
        ValueError: If ``threshold`` is outside ``[0, 1)``, ``norm_order`` is
            not 1 or 2, a start position is padding, or the model uses sandwich
            norms (e.g. Gemma-2), whose residual cannot be split this way.
        NotImplementedError: If the attention layout exposes no value
            projection (e.g. GPT-NeoX).

    Note:
        The paper ignores biases. Here ``b_V`` stays inside each head's part,
        and ``b_O`` belongs to no part though the whole includes it, so on
        models with an output bias (GPT-2) the parts sum to ``resid_mid - b_O``.
    """

    reads = [keys.PROMPTS]
    read_types = {keys.PROMPTS: PromptBatch}
    writes = [keys.INFORMATION_FLOW]

    def __init__(
        self,
        model: ModelBackend,
        *,
        positions: int | Sequence[int] | Tensor | None = None,
        threshold: float = 0.03,
        renormalize: bool = True,
        norm_order: int = 1,
        prompts_key: str = keys.PROMPTS,
        output_key: str = keys.INFORMATION_FLOW,
    ):
        if not 0 <= threshold < 1:
            raise ValueError(
                f"threshold must be in [0, 1), got {threshold}; the paper uses "
                "0.03 to 0.04."
            )
        if norm_order not in (1, 2):
            raise ValueError(
                f"norm_order must be 1 (L1, the paper's) or 2, got {norm_order}."
            )
        self.model = model
        self.positions = positions
        self.threshold = threshold
        self.renormalize = renormalize
        self.norm_order = norm_order
        self.prompts_key = prompts_key
        self.output_key = output_key
        self.reads = [prompts_key]
        self.read_types = {prompts_key: PromptBatch}
        self.writes = [output_key]
        self.write_types = {output_key: InformationFlowResult}

    @no_grad()
    def __call__(self, results: Results) -> Results:
        self.model.require_attention_probs()
        if _is_sandwich_norm(self.model, 0):
            raise ValueError(
                "InformationFlow needs a plain pre-norm model: sandwich-norm "
                "layouts (e.g. Gemma-2) normalize the attention output before the "
                "residual add, so resid_mid is not resid_pre + attn_out."
            )
        prompts = results[self.prompts_key].prompts
        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        input_ids = cast(Tensor, tokens["input_ids"])
        mask = cast(Tensor, tokens["attention_mask"])
        targets = _positions(self.positions, mask)
        n_layers, n_heads = self.model.n_layers, self.model.n_heads
        logger.info(
            "InformationFlow: %d prompts, %d layers, threshold=%.3f",
            len(prompts),
            n_layers,
            self.threshold,
        )

        resid, patterns = self._capture(tokens)
        weights = [self._value_output_weights(layer) for layer in range(n_layers)]

        heads = [
            Node(layer, SELF_ATTN, head=h)
            for layer in range(n_layers)
            for h in range(n_heads)
        ]
        mlps = [Node(layer, MLP) for layer in range(n_layers)]
        active = {node: 0 for node in heads + mlps}
        sub_sum = {node: 0.0 for node in heads}
        sub_count = {node: 0 for node in heads}
        all_edges: list[list[RouteEdge]] = []
        all_tokens: list[list[str]] = []
        starts: list[int] = []

        for b in range(input_ids.shape[0]):
            real = mask[b].nonzero().squeeze(-1)
            hits = (real == targets[b]).nonzero()
            if hits.numel() == 0:
                raise ValueError(
                    f"start position {int(targets[b])} of prompt {b} is padding. "
                    "Positions index the padded batch; pass positions=None to "
                    "start from each prompt's last real token."
                )
            start = int(hits.item())
            layers = []
            for layer in range(n_layers):
                pre, attn_in, mid, post = (r[b, real].float() for r in resid[layer])
                pattern = patterns[layer][b][:, real][:, :, real].to(pre.device)
                layers.append(
                    self._layer_importances(
                        pre, attn_in, mid, post, pattern, *weights[layer]
                    )
                )
            edges, used_heads, used_mlps = self._routes(layers, start)
            for node, importance in used_heads.items():
                active[node] += 1
                sub_sum[node] += sum(importance)
                sub_count[node] += len(importance)
            for node in used_mlps:
                active[node] += 1
            all_edges.append(edges)
            all_tokens.append(
                [self.model.tokenizer.decode([int(t)]) for t in input_ids[b, real]]
            )
            starts.append(start)

        n = max(len(all_edges), 1)
        results[self.output_key] = InformationFlowResult(
            contributions={node: count / n for node, count in active.items()},
            head_importance={
                node: sub_sum[node] / sub_count[node] if sub_count[node] else 0.0
                for node in heads
            },
            edges=all_edges,
            tokens=all_tokens,
            positions=starts,
            threshold=self.threshold,
            metadata={
                "prompts_key": self.prompts_key,
                "renormalize": self.renormalize,
                "norm_order": self.norm_order,
            },
        )
        return results

    def _capture(
        self, tokens: Any
    ) -> tuple[list[tuple[Tensor, ...]], dict[int, Tensor]]:
        """Return per layer ``(resid_pre, attn_input, resid_mid, resid_post)`` and the patterns.

        The four residual tensors are ``[B, S, d]``; ``attn_input`` is what the
        attention module ran on (the normed residual). Saves follow execution
        order within one trace.
        """
        saved: dict[int, tuple[Any, ...]] = {}
        with self.model.trace(tokens):
            for layer in range(self.model.n_layers):
                attn = self.model.resolve_module(layer, SELF_ATTN)
                saved[layer] = (
                    self.model.layer(layer).input.save(),
                    attn.input.save(),
                    attn.output.save(),
                    self.model.layer(layer).output.save(),
                )
        resid = []
        for layer in range(self.model.n_layers):
            pre, attn_in, attn_out, post = (unwrap_traced(s) for s in saved[layer])
            # Differs from the paper (Eq. 3), which ignores biases: the whole
            # resid_mid includes b_O while no part does, because b_O belongs to no
            # head or token and the whole should be the model's actual residual.
            resid.append((pre, attn_in, pre + attn_out, post))
        patterns = _capture_patterns(
            self.model, tokens, list(range(self.model.n_layers))
        )
        return resid, patterns

    def _value_output_weights(self, layer: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(W_V [H, head_dim, d], b_V [H, head_dim], W_O [H, d, head_dim])``."""
        w_v, w_o = zip(
            *(_head_ov_weights(self.model, layer, h) for h in range(self.model.n_heads))
        )
        b_v = [
            _head_value_block(self.model, layer, h)[1]
            for h in range(self.model.n_heads)
        ]
        device = w_v[0].device
        return stack(w_v), stack(b_v).to(device), stack(w_o)

    def _layer_importances(
        self,
        pre: Tensor,
        attn_in: Tensor,
        mid: Tensor,
        post: Tensor,
        pattern: Tensor,
        w_v: Tensor,
        b_v: Tensor,
        w_o: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score one layer of one prompt.

        Returns:
            ``(heads [S, S, H], attn_edge [S, S], ffn [S, 2])``: each head's
            sub-edge importance ``[query, key, head]``, the head-summed attention
            edge with the residual added on the diagonal (Eq. 4), and the
            ``(FFN, residual)`` importances in ``resid_post``.
        """
        seq = pre.shape[0]
        n_heads = w_v.shape[0]
        device = pre.device
        w_v, b_v, w_o = w_v.to(device), b_v.to(device), w_o.to(device)
        head_out = _head_outputs(attn_in, w_v, b_v, w_o)
        rows = []
        # One query position at a time keeps memory at [S, H, d] instead of [S, S, H, d].
        for pos in range(seq):
            parts = pattern[:, pos, :].T.unsqueeze(-1) * head_out  # [S_key, H, d]
            parts = cat([parts.reshape(seq * n_heads, -1), pre[pos : pos + 1]])
            rows.append(_importance(parts, mid[pos], self.norm_order))
        attn = stack(rows)  # [S, S*H + 1]
        ffn = _importance(stack([post - mid, mid], dim=1), post, self.norm_order)
        if self.renormalize:
            attn = _threshold(attn, self.threshold)
            # Differs from the paper (§2.2), which describes renormalizing only the
            # attention sub-edges: the FFN pair is renormalized too, because a
            # node's incoming importances then sum to 1 at every node. It changes
            # weights, never which FFN edges pass the threshold.
            ffn = _threshold(ffn, self.threshold)
        heads = attn[:, :-1].reshape(seq, seq, n_heads)
        edge = heads.sum(dim=-1)
        edge.diagonal().add_(attn[:, -1])
        return heads.cpu(), edge.cpu(), ffn.cpu()

    def _routes(
        self, layers: list[tuple[Tensor, Tensor, Tensor]], start: int
    ) -> tuple[list[RouteEdge], dict[Node, list[float]], set[Node]]:
        """Walk down from ``resid_post`` of the last layer at ``start`` (§2.2).

        Keeps each edge whose importance exceeds the threshold and recurses into
        its source. Returns the edges, each used head's in-route sub-edge
        importances, and the used MLPs.
        """
        tau = self.threshold
        edges: list[RouteEdge] = []
        used_heads: dict[Node, list[float]] = {}
        used_mlps: set[Node] = set()
        reached = {start}  # positions reached at resid_post of the current layer
        for layer in reversed(range(len(layers))):
            heads, attn_edge, ffn = layers[layer]
            mid_node, post_node, mlp_node = (
                Node(layer, RESID_MID),
                Node(layer, RESID_POST),
                Node(layer, MLP),
            )
            mids = set()
            for pos in sorted(reached):
                ffn_w, res_w = float(ffn[pos, 0]), float(ffn[pos, 1])
                if ffn_w > tau:
                    edges.append(RouteEdge(mid_node, pos, mlp_node, pos, ffn_w))
                    edges.append(RouteEdge(mlp_node, pos, post_node, pos, ffn_w))
                    used_mlps.add(mlp_node)
                    mids.add(pos)
                if res_w > tau:
                    edges.append(RouteEdge(mid_node, pos, post_node, pos, res_w))
                    mids.add(pos)
            source = Node(layer - 1, RESID_POST) if layer > 0 else Node(0, RESID_PRE)
            reached = set()
            for pos in sorted(mids):
                for key in range(pos + 1):
                    weight = float(attn_edge[pos, key])
                    if weight <= tau:
                        continue
                    edges.append(RouteEdge(source, key, mid_node, pos, weight))
                    reached.add(key)
                    for h, sub in enumerate(heads[pos, key].tolist()):
                        # Differs from the paper (§3), which says a head is active
                        # when one of its sub-edges is in the routes: a sub-edge
                        # counts here when it also exceeds tau, because with
                        # renormalize=False sub-edges are never thresholded and
                        # every head with a non-zero share would count.
                        if sub > tau:
                            used_heads.setdefault(
                                Node(layer, SELF_ATTN, head=h), []
                            ).append(sub)
        return edges, used_heads, used_mlps
