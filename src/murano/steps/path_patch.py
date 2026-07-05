"""PathPatch step: isolate the direct sender-to-receiver effect.

Activation patching (:class:`~murano.steps.patch.Patch`) replaces one site and lets
the change propagate through every downstream route, so it measures a component's
*total* effect. Path patching measures the *direct* effect: it injects the sender's
activation from a second run but freezes every other component at its base value, so
the perturbation can only reach the receiver along the direct residual path. This is
the primitive behind circuit localization (name movers, s-inhibition, backup name
movers in Wang et al.'s IOI work); the algorithm is the one from Goldowsky-Dill et
al., "Localizing Model Behavior with Path Patching".

For senders ``S`` and receiver ``R``, with ``base`` = the run to measure and
``source`` = the run supplying the sender activations:

1. Run ``source``; cache each sender's activation.
2. Run ``base``; cache every head's per-head output and every MLP output.
3. Run ``base`` again with every attention head frozen to its base output, MLPs
   frozen too when ``freeze_mlps`` (otherwise they recompute), and each sender's
   output overwritten with its ``source`` value at ``positions``. Capture ``R``.
4. When ``R`` is the final residual, step 3's logits are the result. Otherwise run
   ``base`` once more with ``R`` patched to its step-3 value and everything
   downstream of ``R`` free, and read the logits.

Freezing head *outputs* (the attention output projection's per-head input) plus MLP
outputs is equivalent to freezing the reference's q/k/v and matches what murano can
write. The default direction (base = clean ``prompts``, source = corrupt
``corrupt_prompts``) injects the corrupt sender into the clean run, so a metric step
reads how much of the behaviour the direct sender path carries.

nnsight's own per-token trace idiom differs across versions, so the freeze, inject,
and capture are done with plain torch forward hooks on the underlying modules, the
same mechanism :meth:`~murano.model.MuranoModel._register_generation_intervention`
uses; it is independent of the nnsight tracing layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, arange, zeros  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import PromptBatch
from murano.logging import logger
from murano.nodes import (
    MLP,
    RESID_POST,
    SELF_ATTN,
    AddressLike,
    Edge,
    Node,
    NodeSet,
)
from murano.results import Results
from murano.steps.ablate import _coerce_targets
from murano.steps.base import Step
from murano.steps.metrics import _answer_positions

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from murano.backend import ModelBackend


def _group_senders(nodes: list[Node]) -> tuple[dict[int, list[int]], set[int]]:
    """Split sender nodes into per-layer attention heads and whole MLP layers.

    Returns:
        A pair ``(attn_senders, mlp_senders)``: ``attn_senders`` maps a layer to
        its sorted sender head indices; ``mlp_senders`` is the set of layers whose
        MLP output is a sender.

    Raises:
        ValueError: If a sender is neither an attention head nor an MLP, an
            attention sender omits its head, or a head-side (Q/K/V/O) or per-node
            position sender is given (out of scope).
    """
    attn_heads: dict[int, set[int]] = {}
    mlp_senders: set[int] = set()
    for node in nodes:
        # Token selection is the step's positions= argument, not a per-node
        # position, so reject the latter to match Ablate and avoid it silently
        # having no effect.
        if node.position is not None:
            raise ValueError(
                f"per-node position addressing is out of scope; use the step's "
                f"positions= argument instead. Got {node}."
            )
        if node.module == SELF_ATTN:
            if node.head is None:
                raise ValueError(
                    f"an attention sender must name a head "
                    f"(Node(layer, {SELF_ATTN!r}, head=h)); got {node}."
                )
            if node.side is not None:
                raise ValueError(
                    f"head-side (Q/K/V/O) senders are out of scope; got {node}."
                )
            attn_heads.setdefault(node.layer, set()).add(node.head)
        elif node.module == MLP:
            mlp_senders.add(node.layer)
        else:
            raise ValueError(
                f"a sender must be an attention head or an MLP; got module "
                f"{node.module!r} in {node}."
            )
    attn_senders = {layer: sorted(heads) for layer, heads in attn_heads.items()}
    return attn_senders, mlp_senders


class PathPatch(Step):
    """Path-patch senders into a receiver and write the resulting logits.

    Runs the base prompts while injecting the source prompts' sender activations
    along only the direct path to the receiver (every other component frozen), then
    stores the logits so a metric step can score the direct effect. With the
    defaults (base = clean ``prompts``, source = corrupt ``corrupt_prompts``) this
    reads how much of the behaviour the direct sender path carries.

    Reads from results:
        results[base_key]: PromptBatch (default ``prompts``), the run to measure.
        results[source_key]: PromptBatch (default ``corrupt_prompts``), the run
            supplying the sender activations.

    Writes to results:
        results[logits_key]: Tensor [B, S, vocab] of path-patched logits.
        results[mask_key]: Tensor [B, S] marking real (1) vs padding (0) for the
            base batch, so a downstream metric step can locate the answer position.

    Args:
        model: Model backend to run.
        senders: The sending sites, as a :class:`~murano.nodes.NodeSet`, an
            address, an iterable of addresses, or an :class:`~murano.nodes.Edge`
            (its source is the sender, its dest the receiver). Each sender is an
            attention head (``Node(layer, "self_attn", head=h)``) or an MLP
            (``Node(layer, "mlp")``).
        receiver: The receiving site, a residual-stream ``Node(layer,
            "resid_post")``. Defaults to the final residual (the last layer). Only
            residual-stream receivers are supported; a head/side or component-output
            receiver raises.
        base_key: Results key of the batch to measure (default ``prompts``).
        source_key: Results key of the batch supplying sender activations (default
            ``corrupt_prompts``).
        positions: Token position(s) to inject the senders at, in the
            :func:`~murano.steps.metrics._answer_positions` form; ``None`` injects
            at every position.
        freeze_mlps: If True, freeze MLP outputs to their base values too, so only
            the attention-mediated direct path survives; if False (default), MLPs
            recompute and mediate the effect.
        logits_key: Results key to write the path-patched logits under.
        mask_key: Results key to write the base attention mask under.

    Raises:
        ValueError: If ``senders`` is empty or malformed, a sender/receiver layer
            or sender head is out of range, both an ``Edge`` and a ``receiver`` are
            given, or the base and source prompts are not token-length-matched per
            pair.
        NotImplementedError: If ``receiver`` is not a residual-stream site.
    """

    def __init__(
        self,
        model: ModelBackend,
        senders: NodeSet | AddressLike | Iterable[AddressLike] | Edge,
        receiver: AddressLike | None = None,
        *,
        base_key: str = keys.PROMPTS,
        source_key: str = keys.CORRUPT_PROMPTS,
        positions: int | Sequence[int] | Tensor | None = None,
        freeze_mlps: bool = False,
        logits_key: str = keys.PATH_PATCHED_LOGITS,
        mask_key: str = keys.PATH_PATCHED_MASK,
    ):
        if isinstance(senders, Edge):
            if receiver is not None:
                raise ValueError("pass an Edge OR (senders, receiver), not both.")
            receiver = senders.dest
            senders = senders.source

        self.attn_senders, self.mlp_senders = _group_senders(_coerce_targets(senders))
        if not self.attn_senders and not self.mlp_senders:
            raise ValueError("PathPatch needs at least one sender (a head or an MLP).")

        receiver_node = (
            Node(model.n_layers - 1, RESID_POST)
            if receiver is None
            else Node.coerce(receiver)
        )
        if (
            receiver_node.module != RESID_POST
            or receiver_node.head is not None
            or receiver_node.side is not None
        ):
            raise NotImplementedError(
                f"PathPatch supports residual-stream receivers "
                f"(Node(layer, {RESID_POST!r})); component-output and head/side "
                f"receivers (e.g. a head's value input) are not yet supported. "
                f"Got {receiver_node}."
            )
        # Catch out-of-range layers/heads here rather than as an opaque IndexError
        # (or a silently dropped sender) deep in the forward passes.
        self._check_bounds(model, receiver_node)
        self.receiver = receiver_node
        self._receiver_is_final = receiver_node.layer == model.n_layers - 1

        self.model = model
        self.base_key = base_key
        self.source_key = source_key
        self.positions = positions
        self.freeze_mlps = freeze_mlps
        self.logits_key = logits_key
        self.mask_key = mask_key
        self.reads = [base_key, source_key]
        self.read_types = {base_key: PromptBatch, source_key: PromptBatch}
        self.writes = [logits_key, mask_key]
        self.write_types = {logits_key: Tensor, mask_key: Tensor}

    def _check_bounds(self, model: ModelBackend, receiver: Node) -> None:
        """Reject layers/heads outside the model's range with a clear error.

        Raises:
            ValueError: If the receiver layer, a sender layer, or a sender head is
                out of range for ``model``.
        """
        n_layers, n_heads = model.n_layers, model.n_heads
        if receiver.layer >= n_layers:
            raise ValueError(
                f"receiver layer {receiver.layer} is out of range for a "
                f"{n_layers}-layer model."
            )
        for layer, heads in self.attn_senders.items():
            if layer >= n_layers:
                raise ValueError(
                    f"sender layer {layer} is out of range for a {n_layers}-layer model."
                )
            for head in heads:
                if head >= n_heads:
                    raise ValueError(
                        f"sender head {head} is out of range for a {n_heads}-head model."
                    )
        for layer in self.mlp_senders:
            if layer >= n_layers:
                raise ValueError(
                    f"sender layer {layer} is out of range for a {n_layers}-layer model."
                )

    def __call__(self, results: Results) -> Results:
        base_prompts = results[self.base_key].prompts
        source_prompts = results[self.source_key].prompts
        base_tokens, source_tokens, attention_mask = self._tokenize_pair(
            base_prompts, source_prompts
        )
        batch, seq = base_tokens["input_ids"].shape

        pos_idx: Tensor | None = None
        if self.positions is not None:
            pos_idx = _answer_positions(
                zeros(batch, seq, 1), None, self.positions
            ).cpu()

        n_layers = self.model.n_layers
        want_mlp = self.freeze_mlps or bool(self.mlp_senders)
        clean_attn, clean_mlp = self._capture(
            base_tokens, range(n_layers), range(n_layers) if want_mlp else ()
        )
        source_attn, source_mlp = self._capture(
            source_tokens, sorted(self.attn_senders), sorted(self.mlp_senders)
        )

        frozen_attn = self._build_frozen_attn(clean_attn, source_attn, pos_idx)
        frozen_mlp = (
            self._build_frozen_mlp(clean_mlp, source_mlp, pos_idx)
            if self.freeze_mlps
            else None
        )

        logger.info(
            "PathPatch: %d attn sender(s) + %d mlp sender(s) -> %s, freeze_mlps=%s",
            sum(len(h) for h in self.attn_senders.values()),
            len(self.mlp_senders),
            self.receiver,
            self.freeze_mlps,
        )

        logits, captured = self._run_frozen(
            base_tokens, frozen_attn, frozen_mlp, source_mlp, pos_idx
        )
        if self._receiver_is_final:
            results[self.logits_key] = logits
        else:
            results[self.logits_key] = self._run_patched_receiver(
                base_tokens, captured, pos_idx
            )
        results[self.mask_key] = attention_mask.detach().cpu()
        return results

    # ── Capture / freeze construction ─────────────────────────────────

    def _tokenize_pair(self, base_prompts, source_prompts):
        """Tokenize the base and source prompts, requiring aligned lengths.

        Path patching injects the source sender activation at the base token
        positions, so the two runs must have the same per-example token count;
        otherwise a base position would be filled with the partner's padding. The
        check compares the natural (untruncated) lengths: comparing after
        truncation would wrongly pass a genuinely mismatched pair that both
        truncate to the model's max length.

        Raises:
            ValueError: If a base/source pair has different natural token counts.
        """
        base_lengths = self._natural_lengths(base_prompts)
        source_lengths = self._natural_lengths(source_prompts)
        if base_lengths != source_lengths:
            raise ValueError(
                "PathPatch needs token-length-matched base/source pairs so the "
                "injected positions align; got mismatched lengths "
                f"{base_lengths} vs {source_lengths}."
            )
        kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "return_token_type_ids": False,
        }
        base = self.model.tokenizer(base_prompts, **kwargs)
        source = self.model.tokenizer(source_prompts, **kwargs)
        return base, source, base["attention_mask"]

    def _natural_lengths(self, prompts) -> list[int]:
        """Return each prompt's token count with no padding or truncation."""
        encoded = self.model.tokenizer(prompts, return_token_type_ids=False)[
            "input_ids"
        ]
        return [len(ids) for ids in encoded]

    def _capture(self, tokens, attn_layers, mlp_layers):
        """Run one forward pass, caching per-head attention inputs and MLP outputs.

        Returns:
            A pair ``(attn, mlp)`` of ``{layer: tensor}`` maps: ``attn`` holds each
            requested layer's concatenated per-head output ``[B, S, d]`` (the
            attention output projection's input), ``mlp`` each requested layer's
            output ``[B, S, d]``.
        """
        attn: dict[int, Tensor] = {}
        mlp: dict[int, Tensor] = {}
        handles = []

        def save_attn(layer):
            def hook(module, args):
                attn[layer] = args[0].detach().clone()

            return hook

        def save_mlp(layer):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                mlp[layer] = out.detach().clone()

            return hook

        for layer in attn_layers:
            proj = self.model.attn_out_proj(layer, SELF_ATTN)._module  # pyright: ignore[reportAttributeAccessIssue]
            handles.append(proj.register_forward_pre_hook(save_attn(layer)))
        for layer in mlp_layers:
            mod = self.model.resolve_module(layer, MLP)._module  # pyright: ignore[reportAttributeAccessIssue]
            handles.append(mod.register_forward_hook(save_mlp(layer)))
        try:
            self.model.forward_logits(tokens, fn=None)
        finally:
            for handle in handles:
                handle.remove()
        return attn, mlp

    def _build_frozen_attn(self, clean_attn, source_attn, pos_idx):
        """Build each layer's frozen o_proj input: base heads, sender heads swapped in.

        Every head is pinned to its base activation; each sender head's slice is
        overwritten with the source activation at ``pos_idx`` (or all positions when
        ``pos_idx`` is None), so only the sender's direct contribution changes.
        """
        heads = self.model.n_heads
        frozen: dict[int, Tensor] = {}
        for layer, base in clean_attn.items():
            if layer not in self.attn_senders:
                frozen[layer] = base
                continue
            b, s, d = base.shape
            edited = base.clone().reshape(b, s, heads, d // heads)
            source = source_attn[layer].reshape(b, s, heads, d // heads)
            self._swap_positions(edited, source, self.attn_senders[layer], pos_idx)
            frozen[layer] = edited.reshape(b, s, d)
        return frozen

    def _build_frozen_mlp(self, clean_mlp, source_mlp, pos_idx):
        """Build each layer's frozen MLP output: base, sender layers swapped in."""
        frozen: dict[int, Tensor] = {}
        for layer, base in clean_mlp.items():
            if layer not in self.mlp_senders:
                frozen[layer] = base
                continue
            edited = base.clone()
            self._swap_positions(edited, source_mlp[layer], None, pos_idx)
            frozen[layer] = edited
        return frozen

    @staticmethod
    def _swap_positions(dest, source, heads, pos_idx):
        """Overwrite ``dest`` with ``source`` at the sender heads and positions.

        ``heads`` is a list of head indices (for a per-head ``[B, S, H, Dh]``
        tensor) or None (a whole ``[B, S, d]`` MLP output). ``pos_idx`` selects one
        position per example, or None for all positions. Edits ``dest`` in place.
        """
        source = source.to(dest.device, dest.dtype)
        if pos_idx is None:
            if heads is None:
                dest[:] = source
            else:
                for head in heads:
                    dest[:, :, head, :] = source[:, :, head, :]
            return
        rows = arange(dest.shape[0], device=dest.device)
        cols = pos_idx.to(dest.device)
        if heads is None:
            dest[rows, cols] = source[rows, cols]
        else:
            for head in heads:
                dest[rows, cols, head, :] = source[rows, cols, head, :]

    # ── Frozen run + optional receiver patch ──────────────────────────

    def _run_frozen(self, tokens, frozen_attn, frozen_mlp, source_mlp, pos_idx):
        """Run the base batch with the frozen attention/MLP set, capturing R.

        Registers torch hooks that replace every o_proj input with its frozen
        tensor, freeze MLP outputs when ``freeze_mlps`` else inject any MLP sender
        into the recomputed output, and capture the receiver's residual (unless the
        receiver is the final residual, where the logits are the result).

        Returns:
            A pair ``(logits, captured)``: the forward logits, and the captured
            receiver residual ``[B, S, d]`` (or None for the final-residual case).
        """
        captured: dict[str, Tensor] = {}
        handles = []

        def freeze_attn(layer):
            def hook(module, args):
                return (frozen_attn[layer].to(args[0].device, args[0].dtype),)

            return hook

        def freeze_mlp(layer):
            def hook(module, args, output):
                base = output[0] if isinstance(output, tuple) else output
                frozen = frozen_mlp[layer].to(base.device, base.dtype)
                return (frozen, *output[1:]) if isinstance(output, tuple) else frozen

            return hook

        def inject_mlp(layer):
            def hook(module, args, output):
                base = output[0] if isinstance(output, tuple) else output
                edited = base.clone()
                self._swap_positions(edited, source_mlp[layer], None, pos_idx)
                return (edited, *output[1:]) if isinstance(output, tuple) else edited

            return hook

        def capture_receiver(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            captured["r"] = out.detach().clone()

        for layer in frozen_attn:
            proj = self.model.attn_out_proj(layer, SELF_ATTN)._module  # pyright: ignore[reportAttributeAccessIssue]
            handles.append(proj.register_forward_pre_hook(freeze_attn(layer)))
        if self.freeze_mlps and frozen_mlp is not None:
            for layer in frozen_mlp:
                mod = self.model.resolve_module(layer, MLP)._module  # pyright: ignore[reportAttributeAccessIssue]
                handles.append(mod.register_forward_hook(freeze_mlp(layer)))
        else:
            for layer in self.mlp_senders:
                mod = self.model.resolve_module(layer, MLP)._module  # pyright: ignore[reportAttributeAccessIssue]
                handles.append(mod.register_forward_hook(inject_mlp(layer)))
        if not self._receiver_is_final:
            recv = self.model.layer(self.receiver.layer)._module  # pyright: ignore[reportAttributeAccessIssue]
            handles.append(recv.register_forward_hook(capture_receiver))

        try:
            logits = self.model.forward_logits(tokens, fn=None)
        finally:
            for handle in handles:
                handle.remove()
        return logits, captured.get("r")

    def _run_patched_receiver(self, tokens, captured, pos_idx):
        """Run the base batch with the receiver residual patched, nothing frozen.

        Transplants the frozen run's receiver residual into an otherwise-normal
        forward pass, so everything downstream of the receiver recomputes from the
        direct-path value, and returns the logits.
        """
        handles = []

        def patch(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            edited = hidden.clone()
            self._swap_positions(edited, captured, None, pos_idx)
            if isinstance(output, tuple):
                return (edited, *output[1:])
            return edited

        recv = self.model.layer(self.receiver.layer)._module  # pyright: ignore[reportAttributeAccessIssue]
        handles.append(recv.register_forward_hook(patch))
        try:
            logits = self.model.forward_logits(tokens, fn=None)
        finally:
            for handle in handles:
                handle.remove()
        return logits
