"""Attention-pattern analysis and intervention.

Captures the per-head softmax attention weights ``[batch, n_heads, query, key]``
that :class:`~murano.steps.record.Record` cannot reach (it captures head
outputs, not the weights), and offers task-agnostic readouts over them: entropy,
attention-sink mass, mean attention distance, and the attention along a fixed
query-to-key offset. These are the generic building blocks; task-specific
interpretation (labelling induction, name-mover, or duplicate-token heads) is
left to the caller, who picks the offsets and applies the labels.

The module also exposes the head's OV map (:func:`ov_circuit`) and an
intervention step (:class:`AblateAttention`) that overwrites attention weights
through nnterp's settable accessor. All of it requires a model loaded with
``enable_attention_probs=True`` (eager attention); the steps fail with a clear
message otherwise.

Attributes:
    RecordAttention: Capture attention patterns for chosen layers.
    AttentionResult: Captured patterns plus the generic reduction methods.
    AblateAttention: Overwrite attention weights and read the resulting logits.
    ov_circuit: The effective per-head value-output map ``W_O_h @ W_V_h``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from torch import Tensor, arange, as_tensor  # pyright: ignore[reportPrivateImportUsage]
from torch import equal, stack, zeros  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import SELF_ATTN, AddressLike, Node, NodeSet
from murano.results import Results
from murano.steps.ablate import _coerce_targets
from murano.steps.base import Step
from murano.steps.metrics import _answer_positions
from murano._proxy import unwrap_traced

if TYPE_CHECKING:
    from murano.backend import ModelBackend

_ATTN_METHODS = ("zero", "mean", "resample")


# ── Capture ───────────────────────────────────────────────────────────


def _capture_patterns(
    model: ModelBackend, tokens: Any, layers: Sequence[int]
) -> dict[int, Tensor]:
    """Capture per-head attention weights at each layer for ``tokens``.

    Reads nnterp's standardized ``attention_probabilities`` accessor inside one
    trace and returns the patterns on CPU as float32.

    Returns:
        ``{layer: tensor}`` with each tensor shaped ``[batch, n_heads, query,
        key]``.
    """
    # nnterp's source-based accessor requires layers to be read in execution
    # (ascending) order within one trace; the returned dict is keyed by layer so
    # the caller's requested order is preserved downstream.
    saved = {}
    with model.trace(tokens):
        for layer in sorted(layers):
            saved[layer] = model.attention_probabilities[layer].save()
    return {
        layer: unwrap_traced(saved[layer]).detach().float().cpu() for layer in layers
    }


def _natural_lengths(model: ModelBackend, prompts: Sequence[str]) -> Tensor:
    """Return each prompt's untruncated token count ``[len(prompts)]``."""
    mask = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_token_type_ids=False,
    )["attention_mask"]
    return mask.sum(dim=1)


def _align_source_tokens(
    model: ModelBackend, prompts: Sequence[str], base_prompts: Sequence[str], seq: int
) -> Any:
    """Tokenize the resample source, aligned position-for-position to the base.

    Pads the source to the base sequence length ``seq`` and checks the source
    and base share per-example untruncated token lengths, so position ``s``
    carries the same role in both runs (the same requirement the
    :class:`~murano.steps.ablate.Ablate` resample enforces).

    Raises:
        ValueError: If the prompt counts or per-example natural lengths differ.
    """
    if len(prompts) != len(base_prompts):
        raise ValueError(
            f"resample source has {len(prompts)} prompts but the batch has "
            f"{len(base_prompts)}."
        )
    if not equal(
        _natural_lengths(model, prompts), _natural_lengths(model, base_prompts)
    ):
        raise ValueError(
            "resample source prompts must be token-length-matched to the base "
            "prompts per example so attention positions align; got mismatched "
            "lengths."
        )
    return model.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq,
        return_token_type_ids=False,
    )


# ── Generic reductions ────────────────────────────────────────────────
#
# Each reduction takes one layer's pattern [B, H, Q, K] and the [B, S] valid
# token mask and returns a per-head vector [H], averaging only over valid
# query positions so padding never enters the statistic.


def _valid_query_mean(per_query: Tensor, query_mask: Tensor) -> Tensor:
    """Average a per-query statistic ``[B, H, Q]`` over valid queries to ``[H]``."""
    weight = query_mask.to(per_query.dtype).unsqueeze(1)  # [B, 1, Q]
    total = (per_query * weight).sum(dim=(0, 2))
    count = weight.sum().clamp(min=1)
    return total / count


def _pattern_entropy(pattern: Tensor, mask: Tensor) -> Tensor:
    """Return the per-head mean Shannon entropy (nats) of the attention rows."""
    p = pattern.clamp(min=0)
    per_query = -(p * (p + 1e-12).log()).sum(dim=-1)  # [B, H, Q]
    return _valid_query_mean(per_query, mask.bool())


def _pattern_sink(pattern: Tensor, mask: Tensor, index: int) -> Tensor:
    """Return the per-head mean attention mass on the ``index``-th real key.

    The key is taken relative to each example's first real token, not absolute
    column ``index``: Murano's tokenizers left-pad by default, so column 0 is a
    padding token for the shorter sequences in a batch. ``index=0`` is the usual
    attention sink (the first real token).
    """
    batch, _, _, k_len = pattern.shape
    first_real = mask.long().argmax(dim=1)  # first key with mask == 1, per example
    key = (first_real + index).clamp(min=0, max=k_len - 1)
    gathered = pattern.gather(
        -1,
        key.view(batch, 1, 1, 1).expand(batch, pattern.shape[1], pattern.shape[2], 1),
    ).squeeze(-1)
    return _valid_query_mean(gathered, mask.bool())


def _pattern_distance(pattern: Tensor, mask: Tensor) -> Tensor:
    """Return the per-head mean attention distance ``query - key`` (in tokens)."""
    _, _, q_len, k_len = pattern.shape
    offset = (
        arange(q_len).view(1, 1, q_len, 1) - arange(k_len).view(1, 1, 1, k_len)
    ).to(pattern.dtype)
    per_query = (pattern * offset).sum(dim=-1)  # [B, H, Q]
    return _valid_query_mean(per_query, mask.bool())


def _pattern_at_offset(pattern: Tensor, mask: Tensor, offset: int) -> Tensor:
    """Return the per-head mean attention along the diagonal ``key - query == offset``.

    This is the un-labelled readout the caller turns into a previous-token
    (``offset=-1``), duplicate-token (``offset=-period``), or induction
    (``offset=1-period``) score by choosing the offset for their input.
    """
    diagonal = pattern.diagonal(offset=offset, dim1=-2, dim2=-1)  # [B, H, D]
    valid = mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1)  # [B, Q, K]
    valid_diagonal = valid.diagonal(offset=offset, dim1=-2, dim2=-1)  # [B, D]
    weight = valid_diagonal.to(diagonal.dtype).unsqueeze(1)  # [B, 1, D]
    total = (diagonal * weight).sum(dim=(0, 2))
    count = weight.sum().clamp(min=1)
    return total / count


def _positions(positions, mask: Tensor) -> Tensor:
    """Resolve query/key ``positions`` to a per-example index ``[B]`` (long).

    Delegates to :func:`~murano.steps.metrics._answer_positions`: an explicit
    ``positions`` wins (negatives from the end), otherwise the last real token per
    example is read from ``mask``.
    """
    dummy = zeros(mask.shape[0], mask.shape[1], 1)
    return _answer_positions(dummy, mask, positions)


@dataclass
class AttentionResult:
    """Captured per-head attention weights and the generic readouts over them.

    The reduction methods each return a ``[n_captured_layers, n_heads]`` tensor,
    rows in :attr:`layers` order, averaged over the batch and valid query
    positions.

    Attributes:
        patterns: ``{layer: tensor}`` of attention weights ``[batch, n_heads,
            query, key]`` on CPU.
        attention_mask: Tensor ``[batch, seq]`` marking real (1) vs padding (0)
            positions, used to exclude padding from the reductions.
        str_tokens: Decoded input tokens, shape ``[batch][seq]``, for labelling a
            pattern heatmap.
        layers: The captured layer indices, in row order of every reduction.
        addresses: One :class:`Node` per captured layer (``self_attn`` nodes).
        metadata: Free-form provenance (e.g. the prompts key read).
    """

    patterns: dict[int, Tensor]
    attention_mask: Tensor
    str_tokens: list[list[str]]
    layers: list[int]
    addresses: list[Node]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.addresses = [Node.coerce(address) for address in self.addresses]

    def _stack(self, reduce) -> Tensor:
        return stack([reduce(self.patterns[layer]) for layer in self.layers])

    def entropy(self) -> Tensor:
        """Return the per-head mean attention entropy ``[n_layers, n_heads]``."""
        return self._stack(lambda p: _pattern_entropy(p, self.attention_mask))

    def sink(self, index: int = 0) -> Tensor:
        """Return the per-head mean attention mass on a sink key ``[n_layers, n_heads]``.

        ``index`` counts from each example's first real token (default 0, the
        usual attention sink), so padding never shifts the sink column.
        """
        return self._stack(lambda p: _pattern_sink(p, self.attention_mask, index))

    def distance(self) -> Tensor:
        """Return the per-head mean attention distance ``[n_layers, n_heads]``."""
        return self._stack(lambda p: _pattern_distance(p, self.attention_mask))

    def at_offset(self, offset: int) -> Tensor:
        """Return the per-head mean attention at ``key - query == offset`` ``[n_layers, n_heads]``."""
        return self._stack(lambda p: _pattern_at_offset(p, self.attention_mask, offset))

    def attention_to(self, query=None, key=None) -> Tensor:
        """Return per-head mean attention from ``query`` to ``key`` ``[n_layers, n_heads]``.

        Averages ``pattern[query, key]`` over the batch for each head, so it reads
        "how much does this head, at the query position, attend to the key
        position." Both positions accept the
        :func:`~murano.steps.metrics._answer_positions` form (an int, a per-example
        sequence or tensor, negatives from the end); ``None`` uses each example's
        last real token (the natural query for next-token prediction). Explicit
        positive positions are absolute columns, so under Murano's default left
        padding use ``-1`` for the last real token and prefer negative indices.
        """
        q = _positions(query, self.attention_mask)
        k = _positions(key, self.attention_mask)
        rows = arange(self.attention_mask.shape[0])
        return self._stack(lambda p: p[rows, :, q, k].mean(dim=0))

    def head_pattern(self, layer: int, head: int, index: int = 0) -> Tensor:
        """Return one head's attention matrix ``[query, key]`` for input ``index``."""
        return self.patterns[layer][index, head]


class RecordAttention(Step):
    """Capture per-head attention weights across layers.

    Runs one trace over the batched prompts and stores each requested layer's
    ``[batch, n_heads, query, key]`` softmax weights in an
    :class:`AttentionResult`, ready for the generic reductions or a pattern
    heatmap. Requires a model loaded with ``enable_attention_probs=True``.

    Reads from results:
        results[prompts_key]: PromptBatch (default ``prompts``).

    Writes to results:
        results[output_key]: AttentionResult.

    Args:
        model: Model backend loaded with attention weights enabled.
        layers: Layer indices to capture, or ``"all"`` for every layer.
        prompts_key: Results key to read the prompt batch from.
        output_key: Results key to write the AttentionResult under.

    Raises:
        ValueError: If ``layers`` is a string other than ``"all"``.

    Note:
        Every captured layer's full ``[batch, n_heads, seq, seq]`` pattern is
        held in memory, so capturing all layers on a large model or long inputs
        is memory-bound; restrict ``layers`` or the prompt count when it is.
    """

    reads = [keys.PROMPTS]
    read_types = {keys.PROMPTS: PromptBatch}

    def __init__(
        self,
        model: ModelBackend,
        layers: list[int] | str = "all",
        *,
        prompts_key: str = keys.PROMPTS,
        output_key: str = keys.ATTENTION_PATTERN,
    ):
        self.model = model
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError(f"layers as string must be 'all', got {layers!r}")
            self.layers: list[int] = list(range(model.n_layers))
        else:
            self.layers = list(layers)
        self.prompts_key = prompts_key
        self.output_key = output_key
        self.reads = [prompts_key]
        self.read_types = {prompts_key: PromptBatch}
        self.writes = [output_key]
        self.write_types = {output_key: AttentionResult}

    def __call__(self, results: Results) -> Results:
        self.model.require_attention_probs()
        prompt_batch: PromptBatch = results[self.prompts_key]
        prompts = prompt_batch.prompts
        logger.info(
            "RecordAttention: %d prompts, %d layers", len(prompts), len(self.layers)
        )
        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        input_ids = cast(Tensor, tokens["input_ids"])
        attention_mask = cast(Tensor, tokens["attention_mask"])

        patterns = _capture_patterns(self.model, tokens, self.layers)
        str_tokens = [
            [self.model.tokenizer.decode([int(t)]) for t in seq.tolist()]
            for seq in input_ids
        ]
        results[self.output_key] = AttentionResult(
            patterns=patterns,
            attention_mask=attention_mask.detach().cpu(),
            str_tokens=str_tokens,
            layers=list(self.layers),
            addresses=[Node(layer, SELF_ATTN) for layer in self.layers],
            metadata={"prompts_key": self.prompts_key},
        )
        return results


# ── OV circuit ────────────────────────────────────────────────────────


def _projection_weight(module: Any) -> Tensor:
    """Return a projection's weight in ``[out_features, in_features]`` orientation.

    ``torch.nn.Linear`` already stores ``[out, in]``; the ``Conv1D`` used by GPT-2
    (and other OpenAI-style models) stores the transpose ``[in, out]`` and applies
    ``x @ weight``, so it is transposed to a common orientation.
    """
    from transformers.pytorch_utils import Conv1D

    weight = module.weight.detach().float()
    if isinstance(module, Conv1D):
        return weight.t()
    return weight


def _head_ov_weights(
    model: ModelBackend, layer: int, head: int
) -> tuple[Tensor, Tensor]:
    """Return one head's value and output weights ``(W_V_h [head_dim, d], W_O_h [d, head_dim])``.

    Handles both a separate value projection (``self_attn.v_proj``, grouped-query
    aware) and GPT-2's fused ``c_attn`` (query, key, value concatenated in that
    order). The output projection is resolved through the same name list as
    per-head capture. Biases are ignored (the OV circuit is the linear map).

    Raises:
        NotImplementedError: If neither a ``v_proj`` nor a contiguous ``c_attn``
            value block is present (e.g. GPT-NeoX's interleaved
            ``query_key_value``).
    """
    head_dim = model.head_dim
    attn = model.resolve_module(layer, SELF_ATTN)._module  # pyright: ignore[reportAttributeAccessIssue]
    w_o = _projection_weight(model.attn_out_proj(layer, SELF_ATTN)._module)  # pyright: ignore[reportAttributeAccessIssue]
    w_o_head = w_o[:, head * head_dim : (head + 1) * head_dim]  # [d, head_dim]

    if hasattr(attn, "v_proj"):
        w_v = _projection_weight(attn.v_proj)  # [n_kv_heads * head_dim, d]
        n_kv_heads = w_v.shape[0] // head_dim
        kv_head = head // (model.n_heads // n_kv_heads)
        w_v_head = w_v[kv_head * head_dim : (kv_head + 1) * head_dim, :]
        return w_v_head, w_o_head
    if hasattr(attn, "c_attn"):
        # Fused q|k|v: the value block is the last third of the output rows.
        d = model.d_model
        w_v = _projection_weight(attn.c_attn)[2 * d : 3 * d, :]  # [d, d]
        w_v_head = w_v[head * head_dim : (head + 1) * head_dim, :]
        return w_v_head, w_o_head
    raise NotImplementedError(
        "ov_circuit needs a separate value projection (self_attn.v_proj) or a "
        "contiguous fused c_attn (GPT-2 style); this architecture's attention "
        "module exposes neither (e.g. GPT-NeoX interleaves q/k/v)."
    )


def ov_circuit(model: ModelBackend, layer: int, head: int) -> Tensor:
    """Return the effective per-head OV map ``W_O_h @ W_V_h``, shape ``[d_model, d_model]``.

    The OV (value-output) circuit is the linear transform a head applies to the
    residual stream it reads before writing it back, ignoring the attention
    weighting. Composing it with the unembedding (a caller's step) gives the
    head's copy/writing behavior. The projection bias, when present, is omitted.

    Works for separate q/k/v projections (grouped-query aware, mapping the query
    head to its shared key/value head) and for GPT-2's fused ``c_attn``. GPT-NeoX
    style interleaved ``query_key_value`` is not supported and raises.

    Args:
        model: Model backend to read weights from.
        layer: Decoder layer index.
        head: Query-head index in ``[0, n_heads)``.

    Returns:
        The OV matrix on CPU as float32, mapping a residual vector ``x`` to the
        head's output contribution ``OV @ x``.

    Raises:
        ValueError: If ``head`` is out of range.
        NotImplementedError: If the value weights cannot be resolved for the
            architecture.
    """
    if not 0 <= head < model.n_heads:
        raise ValueError(f"head {head} out of range [0, {model.n_heads}).")
    w_v_head, w_o_head = _head_ov_weights(model, layer, head)
    return (w_o_head @ w_v_head).cpu()


# ── Intervention ──────────────────────────────────────────────────────


def _group_attention_targets(
    nodes: list[Node], n_layers: int, n_heads: int
) -> dict[int, list[int] | None]:
    """Group target heads by layer and validate them for attention ablation.

    Returns:
        ``{layer: heads}`` where ``heads`` is a sorted list of head indices, or
        ``None`` to mean every head at that layer (from a ``head``-less target).

    Raises:
        ValueError: If ``nodes`` is empty, a target is not an attention head,
            carries a head-side or per-node position, or names an out-of-range
            layer or head.
    """
    if not nodes:
        raise ValueError("AblateAttention needs at least one target head.")
    grouped: dict[int, set[int | None]] = {}
    for node in nodes:
        if node.module != SELF_ATTN:
            raise ValueError(
                f"attention ablation targets an attention head (module "
                f"{SELF_ATTN!r}); got {node}."
            )
        if node.side is not None:
            raise ValueError(
                f"head-side (Q/K/V/O) ablation is out of scope for attention "
                f"weights; got {node}."
            )
        if node.position is not None:
            raise ValueError(
                f"per-node position addressing is out of scope; use the step's "
                f"positions= argument for query selection. Got {node}."
            )
        if not 0 <= node.layer < n_layers:
            raise ValueError(f"layer {node.layer} out of range [0, {n_layers}).")
        if node.head is not None and not 0 <= node.head < n_heads:
            raise ValueError(f"head {node.head} out of range [0, {n_heads}).")
        grouped.setdefault(node.layer, set()).add(node.head)

    # A head-less target (``None``) selects every head, overriding any explicit
    # heads named for the same layer.
    return {
        layer: None if None in heads else sorted(cast("set[int]", heads))
        for layer, heads in grouped.items()
    }


def _attention_write_mask(
    batch: int,
    seq: int,
    heads: list[int] | None,
    query_positions: Tensor | None,
    n_heads: int,
) -> Tensor:
    """Build the float mask selecting the heads and query rows to overwrite.

    The mask multiplies the replacement into the pattern (1 where overwritten,
    0 where kept) and is returned broadcastable to ``[batch, n_heads, query,
    key]``, so a whole-head, all-query edit is a small ``[1, n_heads, 1, 1]``
    tensor.

    Args:
        batch: Batch size.
        seq: Sequence length.
        heads: Sorted head indices to overwrite, or ``None`` for every head.
        query_positions: Per-example query row to overwrite ``[batch]``, or
            ``None`` for every query row.
        n_heads: Number of attention heads.
    """
    head_mask = zeros(n_heads)
    if heads is None:
        head_mask[:] = 1.0
    else:
        head_mask[heads] = 1.0
    head_mask = head_mask.view(1, n_heads, 1, 1)
    if query_positions is None:
        return head_mask
    query_mask = zeros(batch, seq)
    query_mask[arange(batch), query_positions] = 1.0
    return query_mask.view(batch, 1, seq, 1) * head_mask


class AblateAttention(Step):
    """Overwrite per-head attention weights and read the resulting logits.

    Runs one intervened forward pass: the addressed heads' attention weights are
    zeroed, replaced by the batch-mean pattern, or resampled from a second run,
    and the model's output logits are stored. Comparing them against a clean
    :class:`~murano.steps.logits.Logits` run with a metric step scores how much
    those heads' attention pattern carries the behavior. Requires a model loaded
    with ``enable_attention_probs=True``.

    Zeroing a head's weights makes it read nothing (it writes a zero-weighted sum
    of values, i.e. nothing); ``mean``/``resample`` keep the rows normalized. The
    edit is restricted to the query rows in ``positions`` (every row by default);
    the full key axis of each edited row is replaced.

    Reads from results:
        results[prompts_key]: PromptBatch (default ``prompts``), the batch to run.
        results[source_key]: PromptBatch, only with ``source_key`` set: the
            resample source whose patterns are injected.

    Writes to results:
        results[logits_key]: Tensor [B, S, vocab] of the intervened logits.
        results[mask_key]: Tensor [B, S] marking real (1) vs padding (0).

    Args:
        model: Model backend loaded with attention weights enabled.
        targets: Heads to ablate: a :class:`NodeSet`, a single address, or an
            iterable. Each must be an attention node ``Node(layer, "self_attn",
            head=h)``; a head-less ``Node(layer, "self_attn")`` selects every head
            at that layer.
        method: ``"zero"``, ``"mean"``, or ``"resample"``.
        source: For ``method="resample"``, corrupted prompts paired one-to-one
            with the inputs, token-length-matched so positions align.
        source_key: For ``method="resample"``, a Results key naming the source
            :class:`~murano.artifacts.PromptBatch`; mutually exclusive with
            ``source``.
        positions: Query position(s) to overwrite, in the
            :func:`~murano.steps.metrics._answer_positions` form (an int or
            per-example sequence, negatives allowed); ``None`` overwrites every
            query row. These are absolute columns (negatives count from the end),
            so under Murano's default left padding use ``-1`` for the last real
            token.
        prompts_key: Results key to read the batch to run from.
        logits_key: Results key to write the intervened logits under.
        mask_key: Results key to write the attention mask under.

    Raises:
        ValueError: If ``method`` is unknown, ``targets`` is empty or malformed,
            a method-specific argument is misused, or ``resample`` has no source.
    """

    reads = [keys.PROMPTS]
    read_types = {keys.PROMPTS: PromptBatch}

    def __init__(
        self,
        model: ModelBackend,
        targets: NodeSet | AddressLike | Iterable[AddressLike],
        method: Literal["zero", "mean", "resample"] = "zero",
        *,
        source: Sequence[str] | None = None,
        source_key: str | None = None,
        positions: int | Sequence[int] | Tensor | None = None,
        prompts_key: str = keys.PROMPTS,
        logits_key: str = keys.ATTN_ABLATED_LOGITS,
        mask_key: str = keys.ATTN_ABLATED_MASK,
    ):
        if method not in _ATTN_METHODS:
            raise ValueError(f"method must be one of {_ATTN_METHODS}, got {method!r}")
        if source is not None and method != "resample":
            raise ValueError("source= is only valid for method='resample'.")
        if source_key is not None and method != "resample":
            raise ValueError("source_key= is only valid for method='resample'.")
        if source is not None and source_key is not None:
            raise ValueError(
                "pass either source= (raw prompts) or source_key= (a Results "
                "prompt batch), not both."
            )
        if method == "resample" and source is None and source_key is None:
            raise ValueError(
                "method='resample' needs a source= or source_key= to draw the "
                "replacement pattern from."
            )

        self.model = model
        self.sites = _group_attention_targets(
            _coerce_targets(targets), model.n_layers, model.n_heads
        )
        self.layers = sorted(self.sites)
        self.method = method
        self.source = list(source) if source is not None else None
        self.source_key = source_key
        self.positions = positions
        self.prompts_key = prompts_key
        self.logits_key = logits_key
        self.mask_key = mask_key
        self.reads = [prompts_key] + ([source_key] if source_key is not None else [])
        self.read_types = {
            prompts_key: PromptBatch,
            **({source_key: PromptBatch} if source_key is not None else {}),
        }
        self.writes = [logits_key, mask_key]
        self.write_types = {logits_key: Tensor, mask_key: Tensor}

    def __call__(self, results: Results) -> Results:
        self.model.require_attention_probs()
        prompts = results[self.prompts_key].prompts
        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        attention_mask: Tensor = tokens["attention_mask"]
        batch, seq = tokens["input_ids"].shape

        query_positions: Tensor | None = None
        if self.positions is not None:
            dummy = zeros(int(batch), int(seq), 1)
            query_positions = _answer_positions(dummy, None, self.positions).cpu()

        base_patterns: dict[int, Tensor] | None = None
        source_patterns: dict[int, Tensor] | None = None
        if self.method == "mean":
            base_patterns = _capture_patterns(self.model, tokens, self.layers)
        elif self.method == "resample":
            if self.source is not None:
                source_prompts: Sequence[str] = self.source
            else:
                # __init__ guarantees a source_key when source is unset.
                assert self.source_key is not None
                source_prompts = results[self.source_key].prompts
            source_tokens = _align_source_tokens(
                self.model, source_prompts, prompts, int(seq)
            )
            source_patterns = _capture_patterns(self.model, source_tokens, self.layers)

        edits: dict[int, tuple[Tensor, Tensor]] = {}
        for layer in self.layers:
            mask = _attention_write_mask(
                int(batch),
                int(seq),
                self.sites[layer],
                query_positions,
                self.model.n_heads,
            )
            replacement = self._replacement(layer, base_patterns, source_patterns)
            # forward_logits_attention places these on the model's device and dtype.
            edits[layer] = (replacement, mask)

        logger.info("AblateAttention(%s): %d layer(s)", self.method, len(self.sites))
        results[self.logits_key] = self.model.forward_logits_attention(tokens, edits)
        results[self.mask_key] = attention_mask.detach().cpu()
        return results

    def _replacement(
        self,
        layer: int,
        base_patterns: dict[int, Tensor] | None,
        source_patterns: dict[int, Tensor] | None,
    ) -> Tensor:
        """Return the replacement pattern for one layer, broadcastable to ``[B, H, S, S]``."""
        if self.method == "zero":
            return as_tensor(0.0)
        if self.method == "mean":
            assert base_patterns is not None
            # Average the head's pattern over the batch: meaningful when the
            # inputs are position-aligned, matching Ablate's mean baseline.
            return base_patterns[layer].mean(dim=0, keepdim=True)
        assert source_patterns is not None
        return source_patterns[layer]
