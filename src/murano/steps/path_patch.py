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

``R`` is a residual-stream node, or a specific head's query/key/value input (a
``Node(layer, "self_attn", head=h, side=Q|K|V)``): the latter isolates edges into a
head, such as the S-inhibition heads writing to a name mover's query in IOI. For a
Q/K/V receiver, step 3 captures the head's projection output and step 4 patches it,
so the model's own rotary, softmax, and downstream layers recompute from it.

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

from typing import TYPE_CHECKING, Any

from torch import Tensor, arange, tensor, zeros  # pyright: ignore[reportPrivateImportUsage]

from murano import keys
from murano.artifacts import ComponentSelection, PromptBatch
from murano.logging import logger
from murano.nodes import (
    MLP,
    RESID_POST,
    SELF_ATTN,
    AddressLike,
    Edge,
    Node,
    NodeSet,
    Side,
)
from murano.results import Results
from murano.steps.ablate import _coerce_targets
from murano.steps.attention import _projection_weight
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


def _receiver_slot(model: ModelBackend, receiver: Node) -> tuple[Any, int, int]:
    """Return ``(module, col_start, col_end)`` locating a head's Q/K/V activation.

    The receiver head's query, key, or value is a contiguous column range on the
    output of the attention module's projection: the separate query/key/value
    projection (``self_attn.q_proj``/``k_proj``/``v_proj``) when they exist, or the
    fused ``c_attn`` (GPT-2), where query, key and value are concatenated in that
    order. Slicing the projection output means the receiver is the pre-rotary q/k
    (the standard "path patch into a head's query" seam); the model's own rotary,
    softmax, and downstream layers then run on the patched value.

    On a grouped-query model a key or value receiver resolves to the shared
    key/value head (the query head's kv group), so patching it affects every query
    head in that group.

    Args:
        model: Model backend to resolve the attention module from.
        receiver: A ``Node(layer, "self_attn", head=h, side=Q|K|V)`` receiver.

    Returns:
        The projection module to hook and the ``[col_start, col_end)`` slice of its
        output holding the receiver head's ``head_dim`` activations.

    Raises:
        NotImplementedError: If the architecture exposes neither separate q/k/v
            projections nor a contiguous fused ``c_attn`` (e.g. GPT-NeoX interleaves
            q/k/v in a single ``query_key_value``).
    """
    head_dim = model.head_dim
    head, side = receiver.head, receiver.side
    # The caller only reaches here for a validated Q/K/V receiver; assert the
    # invariant so the head/side arithmetic below is type-narrowed.
    assert head is not None and side is not None
    attn = model.raw_module(receiver.layer, SELF_ATTN)

    if hasattr(attn, "q_proj"):
        if side == Side.Q:
            return attn.q_proj, head * head_dim, (head + 1) * head_dim
        proj = attn.k_proj if side == Side.K else attn.v_proj
        n_kv_heads = _projection_weight(proj).shape[0] // head_dim
        kv_head = head // (model.n_heads // n_kv_heads)
        return proj, kv_head * head_dim, (kv_head + 1) * head_dim
    if hasattr(attn, "c_attn"):
        # Fused q|k|v in that order; each block is n_heads*head_dim wide. The only
        # fused-c_attn architecture (GPT-2) is multi-head, so key/value need no
        # kv-group remap and the block width always equals d_model.
        block = {Side.Q: 0, Side.K: 1, Side.V: 2}[side]
        base = block * model.n_heads * head_dim + head * head_dim
        return attn.c_attn, base, base + head_dim
    raise NotImplementedError(
        "path patching into a head's Q/K/V needs separate q/k/v projections "
        "(self_attn.q_proj/k_proj/v_proj) or a contiguous fused c_attn (GPT-2 "
        "style); this architecture's attention module exposes neither (e.g. "
        "GPT-NeoX interleaves q/k/v in a single query_key_value)."
    )


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
            (``Node(layer, "mlp")``). Pass this or ``senders_key``, not both.
        senders_key: Results key holding a
            :class:`~murano.artifacts.ComponentSelection` a discovery step wrote,
            read at run time so attribute-then-path-patch composes in one pipeline.
            The ``receiver`` is still fixed at construction. Pass this or
            ``senders``, not both.
        receiver: The receiving site. A residual-stream ``Node(layer,
            "resid_post")`` (default: the final residual, the last layer), or a
            head's query/key/value input ``Node(layer, "self_attn", head=h,
            side=Q|K|V)`` to path-patch into that head's Q/K/V (e.g. an S-inhibition
            head into a name mover's query). The output side (O) is a head's output,
            addressable only as a sender; an MLP or side-less head receiver raises.
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
        ValueError: If neither or both of ``senders`` and ``senders_key`` are
            given, ``senders`` is empty or malformed, a sender/receiver layer or
            sender/receiver head is out of range, a per-node receiver position is
            given, both an ``Edge`` and a ``receiver`` are given, an ``Edge`` is
            combined with ``senders_key``, or the base and source prompts are not
            token-length-matched per pair.
        NotImplementedError: If ``receiver`` is neither a residual-stream site nor a
            head's Q/K/V input, or if a head's Q/K/V receiver is requested on an
            architecture with interleaved fused q/k/v (e.g. GPT-NeoX).
    """

    def __init__(
        self,
        model: ModelBackend,
        senders: NodeSet | AddressLike | Iterable[AddressLike] | Edge | None = None,
        receiver: AddressLike | None = None,
        *,
        senders_key: str | None = None,
        base_key: str = keys.PROMPTS,
        source_key: str = keys.CORRUPT_PROMPTS,
        positions: int | Sequence[int] | Tensor | None = None,
        freeze_mlps: bool = False,
        logits_key: str = keys.PATH_PATCHED_LOGITS,
        mask_key: str = keys.PATH_PATCHED_MASK,
    ):
        if (senders is None) == (senders_key is None):
            raise ValueError(
                "Pass exactly one of senders= (addresses or an Edge) or "
                "senders_key= (a ComponentSelection a discovery step wrote)."
            )
        if isinstance(senders, Edge):
            # An Edge is a senders= form, so senders_key= is already rejected by the
            # exactly-one guard above; here only the receiver can conflict.
            if receiver is not None:
                raise ValueError("pass an Edge OR (senders, receiver), not both.")
            receiver = senders.dest
            senders = senders.source

        # With senders_key the senders are resolved from Results in __call__; hold
        # valid-but-empty placeholders so _check_bounds validates the receiver here
        # and the sender bounds are checked once the selection is read.
        if senders is not None:
            self.attn_senders, self.mlp_senders = _group_senders(
                _coerce_targets(senders)
            )
            if not self.attn_senders and not self.mlp_senders:
                raise ValueError(
                    "PathPatch needs at least one sender (a head or an MLP)."
                )
        else:
            self.attn_senders, self.mlp_senders = {}, set()
        self.senders_key = senders_key

        receiver_node = (
            Node(model.n_layers - 1, RESID_POST)
            if receiver is None
            else Node.coerce(receiver)
        )
        # Token selection is the step's positions= argument; a per-node receiver
        # position would be silently ignored (as it is for senders), so reject it.
        if receiver_node.position is not None:
            raise ValueError(
                f"per-node receiver position is out of scope; use the step's "
                f"positions= argument instead. Got {receiver_node}."
            )
        self._receiver_kind = self._classify_receiver(receiver_node)
        # Catch out-of-range layers/heads here rather than as an opaque IndexError
        # (or a silently dropped sender) deep in the forward passes.
        self._check_bounds(model, receiver_node)
        self.receiver = receiver_node
        # Resolve the Q/K/V slot now so an unsupported architecture (interleaved
        # fused q/k/v) fails at construction, not mid-run, and is resolved once.
        self._recv_slot = (
            _receiver_slot(model, receiver_node)
            if self._receiver_kind == "qkv"
            else None
        )
        # A head's Q/K/V receiver always has the softmax and everything downstream
        # to recompute, so it never short-circuits to the frozen run's logits.
        self._receiver_is_final = (
            self._receiver_kind == "resid" and receiver_node.layer == model.n_layers - 1
        )

        self.model = model
        self.base_key = base_key
        self.source_key = source_key
        self.positions = positions
        self.freeze_mlps = freeze_mlps
        self.logits_key = logits_key
        self.mask_key = mask_key
        self.reads = [base_key, source_key] + (
            [senders_key] if senders_key is not None else []
        )
        self.read_types = {
            base_key: PromptBatch,
            source_key: PromptBatch,
            **({senders_key: ComponentSelection} if senders_key is not None else {}),
        }
        self.writes = [logits_key, mask_key]
        self.write_types = {logits_key: Tensor, mask_key: Tensor}

    @staticmethod
    def _classify_receiver(receiver: Node) -> str:
        """Return the receiver kind, ``"resid"`` or ``"qkv"``.

        Raises:
            NotImplementedError: If ``receiver`` is neither a bare residual-stream
                node nor a head's Q/K/V input.
        """
        if (
            receiver.module == RESID_POST
            and receiver.head is None
            and receiver.side is None
        ):
            return "resid"
        if (
            receiver.module == SELF_ATTN
            and receiver.head is not None
            and receiver.side in (Side.Q, Side.K, Side.V)
        ):
            return "qkv"
        raise NotImplementedError(
            f"PathPatch receivers are a residual-stream node "
            f"(Node(layer, {RESID_POST!r})) or a head's query/key/value input "
            f"(Node(layer, {SELF_ATTN!r}, head=h, side=Q|K|V)). The output side (O) "
            f"is the head's output, addressable only as a sender. Got {receiver}."
        )

    def _check_bounds(self, model: ModelBackend, receiver: Node) -> None:
        """Reject layers/heads outside the model's range with a clear error.

        Raises:
            ValueError: If the receiver layer or head, a sender layer, or a sender
                head is out of range for ``model``.
        """
        n_layers, n_heads = model.n_layers, model.n_heads
        if receiver.layer >= n_layers:
            raise ValueError(
                f"receiver layer {receiver.layer} is out of range for a "
                f"{n_layers}-layer model."
            )
        if (
            self._receiver_kind == "qkv"
            and receiver.head is not None
            and receiver.head >= n_heads
        ):
            raise ValueError(
                f"receiver head {receiver.head} is out of range for a "
                f"{n_heads}-head model."
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
        if self.senders_key is not None:
            # Resolve the senders a discovery step wrote before running, so
            # attribute-then-path-patch is one pipeline. Group them, then check
            # the sender bounds now that they are known (the receiver was already
            # bounds-checked at construction).
            self.attn_senders, self.mlp_senders = _group_senders(
                _coerce_targets(results[self.senders_key])
            )
            if not self.attn_senders and not self.mlp_senders:
                raise ValueError(
                    "PathPatch needs at least one sender (a head or an MLP); the "
                    f"selection at '{self.senders_key}' was empty."
                )
            self._check_bounds(self.model, self.receiver)
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
        if base_lengths.tolist() != source_lengths.tolist():
            raise ValueError(
                "PathPatch needs token-length-matched base/source pairs so the "
                "injected positions align; got mismatched lengths "
                f"{base_lengths.tolist()} vs {source_lengths.tolist()}."
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

    def _natural_lengths(self, prompts) -> Tensor:
        """Return each prompt's token count with no padding or truncation."""
        encoded = self.model.tokenizer(prompts, return_token_type_ids=False)[
            "input_ids"
        ]
        return tensor([len(ids) for ids in encoded])

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
            proj = self.model.raw_attn_out_proj(layer, SELF_ATTN)
            handles.append(proj.register_forward_pre_hook(save_attn(layer)))
        for layer in mlp_layers:
            mod = self.model.raw_module(layer, MLP)
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
        into the recomputed output, and capture the receiver: its residual for a
        residual receiver, or its head's Q/K/V activation for a Q/K/V receiver
        (unless the receiver is the final residual, where the logits are the result).

        Returns:
            A pair ``(logits, captured)``: the forward logits, and the captured
            receiver value (residual ``[B, S, d]`` or head Q/K/V ``[B, S, head_dim]``;
            None for the final-residual case).
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

        def capture_qkv(start, end):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                captured["r"] = out[:, :, start:end].detach().clone()

            return hook

        for layer in frozen_attn:
            proj = self.model.raw_attn_out_proj(layer, SELF_ATTN)
            handles.append(proj.register_forward_pre_hook(freeze_attn(layer)))
        if self.freeze_mlps and frozen_mlp is not None:
            for layer in frozen_mlp:
                mod = self.model.raw_module(layer, MLP)
                handles.append(mod.register_forward_hook(freeze_mlp(layer)))
        else:
            for layer in self.mlp_senders:
                mod = self.model.raw_module(layer, MLP)
                handles.append(mod.register_forward_hook(inject_mlp(layer)))
        if self._receiver_kind == "qkv":
            assert self._recv_slot is not None
            module, start, end = self._recv_slot
            handles.append(module.register_forward_hook(capture_qkv(start, end)))
        elif not self._receiver_is_final:
            recv = self.model.raw_layer(self.receiver.layer)
            handles.append(recv.register_forward_hook(capture_receiver))

        try:
            logits = self.model.forward_logits(tokens, fn=None)
        finally:
            for handle in handles:
                handle.remove()
        return logits, captured.get("r")

    @staticmethod
    def _swap_qkv_slot(output, captured, start, end, pos_idx):
        """Return ``output`` with columns ``[start:end)`` set to ``captured``.

        ``output`` is a projection output ``[B, S, F]`` whose ``[start:end)`` columns
        hold the receiver head's Q/K/V. ``pos_idx`` selects one position per example,
        or None for all positions. Does not mutate ``output``.
        """
        edited = output.clone()
        value = captured.to(edited.device, edited.dtype)
        if pos_idx is None:
            edited[:, :, start:end] = value
            return edited
        rows = arange(edited.shape[0], device=edited.device)
        cols = pos_idx.to(edited.device)
        edited[rows, cols, start:end] = value[rows, cols]
        return edited

    def _run_patched_receiver(self, tokens, captured, pos_idx):
        """Run the base batch with the receiver patched, nothing frozen.

        Transplants the frozen run's receiver value (the residual for a residual
        receiver, or the head's Q/K/V for a Q/K/V receiver) into an otherwise-normal
        forward pass, so everything downstream of the receiver recomputes from the
        direct-path value, and returns the logits.
        """
        handles = []

        if self._receiver_kind == "qkv":
            assert self._recv_slot is not None
            module, start, end = self._recv_slot

            def patch_qkv(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                edited = self._swap_qkv_slot(out, captured, start, end, pos_idx)
                return (edited, *output[1:]) if isinstance(output, tuple) else edited

            handles.append(module.register_forward_hook(patch_qkv))
        else:

            def patch(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                edited = hidden.clone()
                self._swap_positions(edited, captured, None, pos_idx)
                if isinstance(output, tuple):
                    return (edited, *output[1:])
                return edited

            recv = self.model.raw_layer(self.receiver.layer)
            handles.append(recv.register_forward_hook(patch))

        try:
            logits = self.model.forward_logits(tokens, fn=None)
        finally:
            for handle in handles:
                handle.remove()
        return logits
