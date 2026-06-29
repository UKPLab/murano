"""Ablate step: zero, mean, or resample a component and read the logits.

Turns a component off and measures what breaks. The step rewrites one or more
target activations during a single forward pass and stores the resulting logits,
so the existing metric steps (logit-diff, KL, recovered) can score the clean run
against the ablated run and quantify the component's causal contribution.

Three replacement methods share one mechanism and differ only in the value that
overwrites the target:

- ``"zero"`` writes zeros (the component contributes nothing).
- ``"mean"`` writes the component's mean activation (it contributes only its
  average, removing the input-specific signal).
- ``"resample"`` writes another input's activation at the same site (the
  within-batch permutation, or a provided corrupted batch), the noising baseline
  used by activation patching.

Targets are :class:`~murano.nodes.Node` addresses. A whole-component target
(``head is None``) ablates the module output; an attention head target
(``Node(layer, "self_attn", head=h)``) ablates only that head's slice of the
attention output projection input. One :class:`Ablate` call is a single mode:
all whole-component or all per-head, not a mix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import torch
from torch import Generator, arange  # pyright: ignore[reportPrivateImportUsage]
from torch import as_tensor, equal, long  # pyright: ignore[reportPrivateImportUsage]
from torch import randperm, tensor, zeros  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import (
    RESID_MID,
    RESID_PRE,
    SELF_ATTN,
    AddressLike,
    Node,
    NodeDict,
    NodeSet,
)
from murano.results import Results
from murano.steps.base import Step
from murano.steps.metrics import _answer_positions
from murano.steps.record import _unwrap_traced

if TYPE_CHECKING:
    from murano.backend import ModelBackend

# A site is one hooked component, keyed by its (layer, canonical-module) pair so
# the intervention function can look targets up by the Node the model passes it.
Site = tuple[int, str]

_METHODS = ("zero", "mean", "resample")
_MEAN_OVER = ("all", "position")


def _coerce_targets(
    targets: NodeSet | AddressLike | Iterable[AddressLike],
) -> list[Node]:
    """Return ``targets`` as a list of canonical :class:`Node` addresses.

    Accepts a :class:`NodeSet`, a single address (a layer int, a
    ``(layer, module)`` tuple, an address string, or a :class:`Node`), or any
    iterable of those.
    """
    if isinstance(targets, NodeSet):
        return list(targets)
    if isinstance(targets, (Node, int, str, tuple)):
        return [Node.coerce(cast(AddressLike, targets))]
    return [Node.coerce(t) for t in targets]


def _group_targets(nodes: list[Node]) -> tuple[dict[Site, list[int] | None], bool]:
    """Group target nodes into per-site selections and infer the ablation mode.

    A target carrying a ``head`` selects per-head mode; one without selects
    whole-component mode. The two cannot be combined in a single call, since they
    hook different points (the module output vs the attention projection input).

    Returns:
        A pair ``(sites, per_head)``. ``sites`` maps each ``(layer, module)`` to
        a sorted list of head indices in per-head mode, or to ``None`` in
        whole-component mode. ``per_head`` is True when any target is a head.

    Raises:
        ValueError: If ``nodes`` is empty, mixes whole-component and per-head
            targets, addresses a head on a non-attention module, uses head-side
            (Q/K/V/O) or per-node position addressing, or names a residual point
            with no single output hook (resid_pre / resid_mid).
    """
    if not nodes:
        raise ValueError("Ablate needs at least one target node.")
    per_head = any(node.head is not None for node in nodes)
    sites: dict[Site, list[int] | None] = {}

    if per_head:
        for node in nodes:
            if node.head is None or node.module != SELF_ATTN:
                raise ValueError(
                    f"per-head ablation requires every target to be an attention "
                    f"head (module {SELF_ATTN!r} with a head index); got {node}. "
                    f"Mixing whole-component and per-head targets in one Ablate is "
                    f"not supported; run two steps."
                )
            if node.side is not None or node.position is not None:
                raise ValueError(
                    f"per-head ablation targets a whole head; head-side (Q/K/V/O) "
                    f"and per-node position addressing are out of scope here. Use "
                    f"the step's positions= argument for token selection; got {node}."
                )
            heads = sites.setdefault((node.layer, node.module), [])
            assert heads is not None  # per-head mode never stores None
            if node.head not in heads:
                heads.append(node.head)
        for heads in sites.values():
            assert heads is not None
            heads.sort()
        return sites, True

    for node in nodes:
        if node.side is not None:
            raise ValueError(
                f"head-side (Q/K/V/O) ablation is out of scope here; got {node}."
            )
        if node.position is not None:
            raise ValueError(
                f"per-node position addressing is out of scope; use the step's "
                f"positions= argument instead. Got {node}."
            )
        if node.module in (RESID_PRE, RESID_MID):
            raise ValueError(
                f"module {node.module!r} has no single output hook to ablate: "
                f"resid_pre is the block input and resid_mid is a derived sum. "
                f"Ablate resid_post, mlp, or self_attn."
            )
        sites[(node.layer, node.module)] = None
    return sites, False


def _component_mean(
    captured: torch.Tensor, mask: torch.Tensor, mean_over: str
) -> torch.Tensor:
    """Return the mean activation over the masked positions of a captured site.

    Args:
        captured: Full-position activations ``[B, S, ...]`` (the trailing dims
            are ``d_model`` for a whole component or ``n_heads, head_dim`` per
            head).
        mask: Valid-token mask ``[B, S]`` (1 = real, 0 = padding).
        mean_over: ``"all"`` pools over batch and valid positions, giving one
            mean per feature ``[...]``; ``"position"`` averages over batch only,
            giving a per-position mean ``[S, ...]`` (meaningful when the inputs
            are position-aligned).

    Returns:
        The mean activation; rank ``[...]`` for ``"all"`` or ``[S, ...]`` for
        ``"position"``.
    """
    # Accumulate in float32: a bf16/fp16 sum over many tokens loses precision,
    # which would bias the mean-ablation replacement. The fn casts back to the
    # activation dtype when it writes.
    captured = captured.float()
    weight = mask.float()
    while weight.dim() < captured.dim():
        weight = weight.unsqueeze(-1)
    if mean_over == "all":
        return (captured * weight).sum(dim=(0, 1)) / weight.sum(dim=(0, 1)).clamp(min=1)
    return (captured * weight).sum(dim=0) / weight.sum(dim=0).clamp(min=1)


def _site_mask(
    batch: int,
    seq: int,
    pos_idx: torch.Tensor | None,
    heads: list[int] | None,
    n_heads: int,
) -> torch.Tensor:
    """Build the float write-mask selecting the positions and heads to ablate.

    The mask multiplies the replacement into the activation: 1 where the value
    is overwritten, 0 where the original is kept. It is returned broadcastable to
    the activation shape, so a whole-batch, all-position ablation is a scalar.

    Args:
        batch: Batch size.
        seq: Sequence length.
        pos_idx: Per-example position to ablate ``[B]``, or ``None`` for every
            position.
        heads: Sorted head indices to ablate, or ``None`` for the whole
            component (no head axis).
        n_heads: Number of attention heads, used to size the head axis.

    Returns:
        A float mask broadcastable to the activation, with a trailing size-1
        feature axis so it spans ``d_model`` (or ``head_dim``).
    """
    pos: torch.Tensor | None = None
    if pos_idx is not None:
        pos = zeros(batch, seq)
        pos[arange(batch), pos_idx] = 1.0

    if heads is None:
        if pos is None:
            return tensor(1.0)
        return pos.unsqueeze(-1)

    head_mask = zeros(n_heads)
    head_mask[heads] = 1.0
    head_mask = head_mask.view(1, 1, n_heads, 1)
    if pos is None:
        return head_mask
    return pos.view(batch, seq, 1, 1) * head_mask


def _ablation_fn(
    replacement: dict[Site, tuple[torch.Tensor, torch.Tensor]],
) -> Callable[[torch.Tensor, Node], torch.Tensor]:
    """Return the intervention applied per target site during the forward pass.

    The returned function rewrites a target activation with ``A * (1 - m) +
    R * m`` (``R`` the replacement, ``m`` the write-mask), which touches only the
    masked positions and heads and leaves every other site unchanged. The blend
    uses only elementwise arithmetic so it composes with the nnsight proxy the
    same way the steering interventions do.

    Args:
        replacement: ``{site: (R, m)}`` mapping each hooked ``(layer, module)``
            to its replacement tensor and write-mask, both broadcastable to the
            activation.
    """

    def fn(activation: torch.Tensor, key: Node) -> torch.Tensor:
        entry = replacement.get((key.layer, key.module))
        if entry is None:
            return activation
        r, m = entry
        m = m.to(device=activation.device, dtype=activation.dtype)
        r = r.to(device=activation.device, dtype=activation.dtype)
        return activation * (1 - m) + r * m

    return fn


class Ablate(Step):
    """Ablate target components and write the resulting forward-pass logits.

    Runs one intervened forward pass over the batched prompts: each target
    component's activation is replaced by zeros, its mean, or a resampled
    activation, and the model's output logits are stored under ``logits_key``.
    Comparing those logits against a clean :class:`~murano.steps.logits.Logits`
    run with a metric step quantifies the component's contribution.

    Reads from results:
        results['prompts']: PromptBatch.

    Writes to results:
        results[logits_key]: Tensor [B, S, vocab] of ablated logits.
        results[mask_key]: Tensor [B, S] marking real (1) vs padding (0), so a
            downstream metric step can locate the answer position.

    Args:
        model: Model backend to run.
        targets: Components to ablate: a :class:`NodeSet`, a single address, or
            an iterable of addresses. A whole-component target ablates the module
            output; a head target (``Node(layer, "self_attn", head=h)``) ablates
            that head. One call is a single mode (all whole-component or all
            per-head).
        method: ``"zero"``, ``"mean"``, or ``"resample"``.
        mean_over: For ``method="mean"``, whether the mean pools over batch and
            positions (``"all"``) or is taken per position (``"position"``).
        means: For ``method="mean"``, an optional precomputed ``{address:
            tensor}`` of mean vectors (e.g. a corpus mean) used instead of the
            current batch's mean. Whole-component only.
        source: For ``method="resample"``, an optional sequence of corrupted
            prompts paired one-to-one with the inputs; each must be
            token-length-matched to its clean prompt so positions align. Without
            it, resample permutes within the batch.
        permutation: For within-batch resample, an explicit rearrangement of
            ``range(batch)`` (for reproducibility); otherwise a random one.
        seed: Seed for the random within-batch permutation when ``permutation``
            is not given.
        positions: Token position(s) to ablate, in the
            :func:`~murano.steps.metrics._answer_positions` form (an int or
            per-example sequence, negatives allowed); ``None`` ablates every
            position.
        logits_key: Results key to write the ablated logits under.
        mask_key: Results key to write the attention mask under.

    Raises:
        ValueError: If ``method``/``mean_over`` is unknown, ``targets`` is
            empty or mixes modes, or a method-specific argument is misused.
        NotImplementedError: If precomputed ``means`` are given for per-head
            ablation.
    """

    reads = [keys.PROMPTS]
    read_types = {keys.PROMPTS: PromptBatch}

    def __init__(
        self,
        model: ModelBackend,
        targets: NodeSet | AddressLike | Iterable[AddressLike],
        method: Literal["zero", "mean", "resample"] = "zero",
        *,
        mean_over: Literal["all", "position"] = "all",
        means: dict[AddressLike, torch.Tensor] | None = None,
        source: Sequence[str] | None = None,
        permutation: Sequence[int] | None = None,
        seed: int | None = None,
        positions: int | Sequence[int] | torch.Tensor | None = None,
        logits_key: str = keys.ABLATED_LOGITS,
        mask_key: str = keys.ATTENTION_MASK,
    ):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        if mean_over not in _MEAN_OVER:
            raise ValueError(
                f"mean_over must be one of {_MEAN_OVER}, got {mean_over!r}"
            )
        if means is not None and method != "mean":
            raise ValueError("means= is only valid for method='mean'.")
        if means is not None and mean_over != "all":
            raise ValueError(
                "means= supplies one per-feature vector per site, so mean_over "
                "does not apply; drop mean_over or omit means."
            )
        if source is not None and method != "resample":
            raise ValueError("source= is only valid for method='resample'.")
        if (permutation is not None or seed is not None) and method != "resample":
            raise ValueError("permutation=/seed= are only valid for method='resample'.")
        if permutation is not None and seed is not None:
            raise ValueError("pass either permutation= or seed=, not both.")

        self.sites, self.per_head = _group_targets(_coerce_targets(targets))
        if means is not None and self.per_head:
            raise NotImplementedError(
                "precomputed means= for per-head ablation is not supported; omit "
                "means to use the batch mean, or ablate whole components."
            )

        self.model = model
        self.method = method
        self.mean_over = mean_over
        self.means = NodeDict(means) if means is not None else None
        if self.means is not None:
            self._validate_means(self.means)
        self.source = list(source) if source is not None else None
        self.permutation = list(permutation) if permutation is not None else None
        self.seed = seed
        self.positions = positions
        self.logits_key = logits_key
        self.mask_key = mask_key
        self.writes = [logits_key, mask_key]
        self.write_types = {logits_key: torch.Tensor, mask_key: torch.Tensor}

    def _validate_means(self, means: NodeDict) -> None:
        """Check the precomputed mean table covers every site with a right-sized vector.

        Raises:
            ValueError: If a whole-component site has no entry, or its vector's
                last dim is not ``d_model``.
        """
        for layer, module in self.sites:
            node = Node(layer, module)
            if node not in means:
                raise ValueError(
                    f"means= has no entry for target {node}; provide a mean vector "
                    f"for every ablated site."
                )
            vector = means[node]
            if vector.shape[-1] != self.model.d_model:
                raise ValueError(
                    f"means= vector for {node} has last dim {vector.shape[-1]}, "
                    f"expected d_model={self.model.d_model}."
                )

    def __call__(self, results: Results) -> Results:
        prompt_batch: PromptBatch = results[keys.PROMPTS]
        prompts = prompt_batch.prompts
        tokens = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        attention_mask: torch.Tensor = tokens["attention_mask"]
        batch, seq = tokens["input_ids"].shape

        # zero needs no capture; mean from a precomputed table needs none either.
        captured: dict[Site, torch.Tensor] = {}
        if self.method == "resample" or (self.method == "mean" and self.means is None):
            captured = self._capture(tokens)

        source_captured: dict[Site, torch.Tensor] | None = None
        if self.method == "resample" and self.source is not None:
            source_captured = self._capture(
                self._tokenize_source(attention_mask, int(seq))
            )

        perm: torch.Tensor | None = None
        if self.method == "resample" and self.source is None:
            perm = self._permutation(int(batch))
            # Resample is position-wise: replacing example b's activation at
            # absolute index s with the partner's index s only aligns when both
            # have the same real-token count (same padding). Enforce it rather
            # than silently overwriting a real token with a partner's padding.
            counts = attention_mask.sum(dim=1)
            if not equal(counts, counts[perm]):
                raise ValueError(
                    "within-batch resample requires equal-length prompts so "
                    "positions align; got mismatched real-token counts. Match the "
                    "prompt lengths, or pass a token-length-matched source=."
                )

        pos_idx: torch.Tensor | None = None
        if self.positions is not None:
            dummy = zeros(int(batch), int(seq), 1)
            pos_idx = _answer_positions(dummy, None, self.positions).cpu()

        replacement: dict[Site, tuple[torch.Tensor, torch.Tensor]] = {}
        for site, heads in self.sites.items():
            mask = _site_mask(int(batch), int(seq), pos_idx, heads, self.model.n_heads)
            value = self._replacement(
                site, captured, source_captured, perm, attention_mask
            )
            replacement[site] = (value, mask)

        fn = _ablation_fn(replacement)
        layers = sorted({layer for layer, _ in self.sites})
        modules = sorted({module for _, module in self.sites})
        logger.info(
            "Ablate(%s): %d site(s), per_head=%s",
            self.method,
            len(self.sites),
            self.per_head,
        )
        results[self.logits_key] = self.model.forward_logits(
            tokens, fn=fn, layers=layers, modules=modules, per_head=self.per_head
        )
        results[self.mask_key] = attention_mask.detach().cpu()
        return results

    def _replacement(
        self,
        site: Site,
        captured: dict[Site, torch.Tensor],
        source_captured: dict[Site, torch.Tensor] | None,
        perm: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the replacement value for one site, broadcastable to its activation."""
        if self.method == "zero":
            return as_tensor(0.0)
        if self.method == "mean":
            if self.means is not None:
                vector = self.means[Node(site[0], site[1])]
                # A precomputed per-feature mean carries no batch or sequence
                # axis, so lift it to [1, 1, ...] to broadcast over both.
                return vector.unsqueeze(0).unsqueeze(0)
            mean = _component_mean(captured[site], attention_mask, self.mean_over)
            if self.mean_over == "all":
                return mean.unsqueeze(0).unsqueeze(0)
            return mean.unsqueeze(0)
        # resample: a paired corrupted batch, else the within-batch permutation.
        if source_captured is not None:
            return source_captured[site]
        assert perm is not None
        return captured[site][perm]

    def _capture(self, tokens: Any) -> dict[Site, torch.Tensor]:
        """Capture full-position activations at every target site for ``tokens``.

        Runs one trace per module (mirroring :class:`~murano.steps.record.Record`
        to avoid nnsight out-of-order saves on nested submodules) and returns the
        activations on CPU, reshaped to expose the head axis in per-head mode.
        """
        layers_by_module: dict[str, list[int]] = {}
        for layer, module in self.sites:
            layers_by_module.setdefault(module, []).append(layer)

        captured: dict[Site, torch.Tensor] = {}
        for module, layers in layers_by_module.items():
            saved = {}
            with self.model.trace(tokens):
                for layer in layers:
                    if self.per_head:
                        saved[layer] = self.model.attn_out_proj(
                            layer, module
                        ).input.save()
                    else:
                        saved[layer] = self.model.resolve_module(
                            layer, module
                        ).output.save()
            for layer in layers:
                value = _unwrap_traced(saved[layer])
                if self.per_head:
                    b, s, d = value.shape
                    value = value.reshape(b, s, self.model.n_heads, self.model.head_dim)
                captured[(layer, module)] = value.detach().cpu()
        return captured

    def _tokenize_source(self, clean_mask: torch.Tensor, seq: int) -> Any:
        """Tokenize the corrupted resample prompts, aligned to the clean batch.

        Pads to the clean sequence length and checks that each corrupted prompt
        has the same real-token count as its clean counterpart, since resample
        is position-wise: a position must carry the same role in both runs.

        Raises:
            ValueError: If the prompt count or per-example token lengths differ.
        """
        assert self.source is not None
        if len(self.source) != clean_mask.shape[0]:
            raise ValueError(
                f"resample source has {len(self.source)} prompts but the batch "
                f"has {clean_mask.shape[0]}."
            )
        tokens = self.model.tokenizer(
            self.source,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq,
            return_token_type_ids=False,
        )
        source_mask: torch.Tensor = tokens["attention_mask"]
        if not equal(source_mask.sum(dim=1), clean_mask.sum(dim=1)):
            raise ValueError(
                "resample source prompts must be token-length-matched to the clean "
                "prompts per example so positions align; got mismatched lengths."
            )
        return tokens

    def _permutation(self, batch: int) -> torch.Tensor:
        """Return the within-batch source permutation for resample ablation.

        Raises:
            ValueError: If an explicit ``permutation`` is not a rearrangement of
                ``range(batch)``.
        """
        if self.permutation is not None:
            perm = tensor(self.permutation, dtype=long)
            if perm.shape[0] != batch or sorted(perm.tolist()) != list(range(batch)):
                raise ValueError(
                    f"permutation must be a rearrangement of range({batch}); got "
                    f"{self.permutation}."
                )
            return perm
        if batch == 1:
            logger.warning(
                "resample on a batch of 1 replaces each example with itself "
                "(a no-op); use a larger batch or a corrupted source."
            )
        generator = Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        return randperm(batch, generator=generator)
