"""MuranoModel: nnterp-based model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import torch
from torch import Tensor, bfloat16  # pyright: ignore[reportPrivateImportUsage]
from torch import dtype as TorchDtype  # pyright: ignore[reportPrivateImportUsage]
from nnterp import StandardizedTransformer

from murano import keys
from murano.logging import logger
from murano.nodes import (
    RESID_MID,
    RESID_POST,
    RESID_PRE,
    Node,
    NodeDict,
    canonical_module,
)

if TYPE_CHECKING:
    from murano.steps.record import ActivationStore
    from murano.steps.train import SteeringResult


# Known names for an attention module's output projection, tried in order when
# resolving the per-head split point. A general per-architecture resolver is
# future work; until then unknown architectures raise from ``attn_out_proj``.
_ATTN_OUT_PROJ_NAMES = ("o_proj", "out_proj", "c_proj", "dense", "wo")


def _ensure_downloaded(model_id: str) -> str:
    """Ensure the model is fully downloaded and return the local snapshot path.

    Tries offline first (no API calls) to avoid rate limits.
    Falls back to online download if the model isn't cached yet.
    """
    local_path = Path(model_id)
    if local_path.exists():
        return str(local_path)

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return snapshot_download(model_id)


class MuranoModel:
    """Thin wrapper around nnterp StandardizedTransformer for mechanistic interpretability.

    Provides cross-architecture access to model layers, tokenizer, and metadata.
    All analysis logic lives in pipeline steps, not here.

    Args:
        model_id: HuggingFace model identifier.
        device_map: Device placement strategy.
        dtype: Model weight dtype.

    Example:
        model = MuranoModel("meta-llama/Llama-3.2-1B-Instruct")
        print(model.n_layers, model.d_model)
    """

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        dtype: TorchDtype = bfloat16,
    ):
        self.model_id = model_id

        # Download the model first (no-op if already cached), then load
        # from the local path. This prevents nnsight/transformers from
        # making repeated HF API calls that trigger rate limits.
        load_path = _ensure_downloaded(model_id)

        # device_map="auto" with nnsight can produce zero/NaN activations
        # for layers beyond the first. Use a single GPU when available.
        if device_map == "auto" and torch.cuda.is_available():
            device_map = "cuda:0"

        try:
            self._lm = StandardizedTransformer(
                load_path,
                device_map=device_map,
                dtype=dtype,
                dispatch=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e
        self.tokenizer = self._lm.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = self._lm.config
        if config is None:
            raise RuntimeError(f"Loaded model {model_id} has no config.")
        self.n_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.n_heads = config.num_attention_heads
        # Prefer the config's explicit head_dim: on architectures like Gemma it
        # is set independently and n_heads * head_dim != hidden_size, so deriving
        # head_dim by division would be wrong. Fall back to the derived value
        # (and its divisibility guard) only when the config omits it.
        config_head_dim = getattr(config, "head_dim", None)
        if config_head_dim:
            self.head_dim = int(config_head_dim)
        else:
            if self.d_model % self.n_heads != 0:
                raise RuntimeError(
                    f"Model {model_id} has d_model={self.d_model} not divisible by "
                    f"n_heads={self.n_heads}; cannot derive head_dim."
                )
            self.head_dim = self.d_model // self.n_heads
        # Bind nnsight's trace directly instead of wrapping it in a method.
        # nnsight inspects the caller's frame to locate the `with` block, so a
        # wrapper method would sit between the step's `with model.trace(...)`
        # and nnsight and hide the block, which fails on some Python versions
        # (WithBlockNotFoundError).
        self.trace = self._lm.trace
        logger.info(
            "Loaded %s (%d layers, d=%d)", model_id, self.n_layers, self.d_model
        )

    def layer(self, idx: int):
        """Return the nnterp module proxy for a decoder layer."""
        return self._lm.layers[idx]  # pyright: ignore[reportIndexIssue,reportArgumentType]

    def resolve_module(self, layer_idx: int, module: str):
        """Resolve a submodule proxy by name at a given layer.

        Args:
            layer_idx: Decoder layer index.
            module: Module name to resolve, e.g. ``"residual"``, ``"mlp"``, or
                a dotted path like ``"mlp.gate_proj"``.

        Returns:
            nnsight proxy for the requested submodule.

        Raises:
            ValueError: If any part of the dotted path does not exist.
        """
        return self._resolve_module(self.layer(layer_idx), module)

    def attn_out_proj(self, layer_idx: int, module: str):
        """Resolve an attention module's output projection for per-head capture.

        The input to this projection is the concatenated per-head outputs, so
        callers reshape it to recover per-head activations.

        Args:
            layer_idx: Decoder layer index.
            module: Module name expected to resolve to an attention module.

        Returns:
            nnsight proxy for the attention output projection.

        Raises:
            NotImplementedError: If the module exposes no known output
                projection (e.g. it is not attention, or the architecture is
                unsupported for per-head capture).
        """
        attn = self._resolve_module(self.layer(layer_idx), module)
        for name in _ATTN_OUT_PROJ_NAMES:
            if hasattr(attn, name):
                return getattr(attn, name)
        raise NotImplementedError(
            f"per_head capture requires an attention module exposing a known "
            f"output projection {_ATTN_OUT_PROJ_NAMES}; module {module!r} has "
            f"none on this architecture."
        )

    def project_on_vocab(self, hidden: Tensor) -> Tensor:
        """Project hidden states onto the vocabulary.

        Applies the standardized final norm and unembedding,
        ``lm_head(ln_final(hidden))``, matching the logit-lens computation.

        Args:
            hidden: Hidden states ``[..., d_model]``.

        Returns:
            Vocabulary logits ``[..., vocab_size]``.
        """
        return self._lm.lm_head(self._lm.ln_final(hidden))

    @property
    def unembed_weight(self) -> Tensor:
        """The unembedding (lm_head) weight ``[vocab, d_model]``.

        The unembedding lives on the causal-LM wrapper, not the base module
        returned by :attr:`hf_model`, so it is exposed here for attribution.
        """
        return self._lm.lm_head.weight  # pyright: ignore[reportReturnType]

    @property
    def final_norm(self):
        """The standardized final-normalization module (before the unembedding).

        Exposed via nnterp's standardized ``ln_final`` accessor, matching
        :meth:`project_on_vocab`; used by attribution to freeze the norm scale.
        """
        return self._lm.ln_final

    def forward_logits(
        self,
        tokens: Any,
        fn: Callable[[Tensor, Node], Tensor] | None = None,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        per_head: bool = False,
    ) -> Tensor:
        """Run a forward pass and return the model's output logits ``[B, S, V]``.

        Returns the model's true unembedding output via nnterp's standardized
        ``.logits`` accessor, not a re-projection of a captured hidden state
        (contrast :meth:`project_on_vocab`). The result is detached, cast to
        float32, and moved to CPU to match the other stores and to keep
        bf16/fp16 models safe for downstream cross-entropy and argmax.

        With ``fn`` given, the pass is intervened: ``fn`` rewrites each target
        module's activation before the logits are read, so an ablation or patch
        flows through to the output (the forward-pass analogue of
        :meth:`generate_with_hooks`). ``fn=None`` is the plain forward pass.

        Args:
            tokens: Tokenizer output (or anything :attr:`trace` accepts) to run.
            fn: ``(activation, key) -> activation`` applied to each target
                module's activation, keyed by a :class:`Node` at
                ``(layer, module)``; ``None`` runs unmodified. The function is
                expected to no-op on sites it does not target.
            layers: Layer indices to hook, or ``"all"``. Only used when ``fn``
                is given.
            modules: Module name(s) to hook at each layer. Only used when ``fn``
                is given.
            per_head: If True, hook the attention output projection's input and
                pass ``fn`` the per-head activation ``[B, S, n_heads, head_dim]``
                (the concatenated head outputs), so ``fn`` can rewrite a single
                head's slice. The rewritten tensor is reshaped back before the
                projection runs. Requires ``tokens`` to expose ``input_ids`` for
                the batch and sequence dimensions.

        Returns:
            Output logits ``[batch, seq, vocab_size]`` on CPU as float32.
        """
        if fn is None:
            with self.trace(tokens):
                # Inside a trace, nnterp's `.logits` is an nnsight proxy exposing
                # `.save()`; its stub types it as a plain Tensor, hence the ignore.
                saved = self._lm.logits.save()  # pyright: ignore[reportAttributeAccessIssue]
            value = saved.value if hasattr(saved, "value") else saved
            return value.detach().float().cpu()

        module_list = [modules] if isinstance(modules, str) else list(modules)
        batch = seq = 0
        if per_head:
            try:
                input_ids = tokens["input_ids"]
            except (TypeError, KeyError) as exc:
                raise ValueError(
                    "per_head=True needs tokens to expose 'input_ids' for the "
                    "batch and sequence dimensions."
                ) from exc
            batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])

        with self.trace(tokens):
            for layer_idx in self._layer_indices(layers):
                for mod_str in module_list:
                    key = Node(layer_idx, mod_str)
                    if per_head:
                        # The o_proj input is the concatenated per-head outputs;
                        # reshape to expose the head axis, let fn rewrite a head
                        # slice, then fold back so the projection sees its normal
                        # [batch, seq, n_heads*head_dim] input. nnsight's input
                        # setter swaps the first positional arg (Envoy.input
                        # postprocess), so assigning the folded tensor is enough.
                        proj = self.attn_out_proj(layer_idx, mod_str)
                        split = proj.input.reshape(
                            batch, seq, self.n_heads, self.head_dim
                        )
                        proj.input = fn(split, key).reshape(  # pyright: ignore[reportAttributeAccessIssue]
                            batch, seq, self.n_heads * self.head_dim
                        )
                    else:
                        mod_proxy = self._resolve_module(self.layer(layer_idx), mod_str)
                        h = mod_proxy.output
                        # Decoder-layer outputs are (hidden_states, ...) tuples,
                        # while submodule outputs (mlp, self_attn) are plain
                        # tensors; edit the hidden-states element either way. fn
                        # returns the same object on sites it does not target, so
                        # only write back when it actually changed the activation
                        # and the loop never touches a non-target site.
                        if isinstance(h, tuple):
                            new = fn(h[0], key)
                            if new is not h[0]:
                                mod_proxy.output = (new, *h[1:])  # pyright: ignore[reportArgumentType]
                        else:
                            new = fn(h, key)
                            if new is not h:
                                mod_proxy.output = new  # pyright: ignore[reportArgumentType]
            saved = self._lm.logits.save()  # pyright: ignore[reportAttributeAccessIssue]
        value = saved.value if hasattr(saved, "value") else saved
        return value.detach().float().cpu()

    @property
    def hf_model(self):
        """Underlying HuggingFace module.

        Note:
            Transitional accessor so weight-level steps need not import
            ``_lm``. It still exposes HF module internals, so it does not make
            weight editing backend-neutral.
        """
        return self._lm.model

    def _coerce_texts(self, text: str | Sequence[str]) -> tuple[list[str], bool]:
        if isinstance(text, str):
            return [text], True
        return list(text), False

    def _coerce_directions(self, direction_like: Any) -> dict[Node, Tensor]:
        if hasattr(direction_like, "direction_per_layer"):
            directions = direction_like.direction_per_layer
        elif isinstance(direction_like, dict):
            directions = direction_like
        else:
            raise TypeError(
                "Expected a SteeringResult or {address: tensor} mapping for the "
                "intervention directions."
            )
        # Normalize every key to a canonical Node so shorthand (a bare layer int,
        # a (layer, module) tuple, an address string) resolves to the same
        # address the generation hooks key by. An uncoercible key fails loudly
        # here rather than silently never matching during generation.
        try:
            return NodeDict(directions)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Intervention direction keys must be addresses (an int layer, a "
                f"(layer, module) tuple, an address string, or a Node); got an "
                f"uncoercible key ({exc})."
            ) from exc

    def _check_directions_reachable(
        self,
        directions: dict[Node, Tensor],
        layers: list[int] | str,
        modules: str | list[str],
    ) -> None:
        """Fail loudly if no intervention address matches a hooked site.

        A direction keyed at an address that is never hooked would silently do
        nothing (e.g. a bare-int key resolves to ``resid_post`` while
        ``modules="mlp"``). An empty ``directions`` is treated as an intentional
        no-op and passes.

        Raises:
            ValueError: If ``directions`` is non-empty yet none of its addresses
                fall on a hooked (layer, module) site.
        """
        if not directions:
            return
        module_list = [modules] if isinstance(modules, str) else list(modules)
        hooked_modules = {Node(0, mod).module for mod in module_list}
        if isinstance(layers, str):
            # "all" hooks every layer, so a matching module is sufficient.
            reachable = any(node.module in hooked_modules for node in directions)
        else:
            hooked = {Node(layer, mod) for layer in layers for mod in module_list}
            reachable = bool(set(directions) & hooked)
        if not reachable:
            raise ValueError(
                f"No intervention address matches a hooked site "
                f"(layers={layers!r}, modules={modules!r}); the intervention "
                f"would do nothing. Addresses: {sorted(directions)}. A bare-int "
                f"direction key targets 'resid_post'."
            )

    def _layer_indices(self, layers: list[int] | str) -> list[int]:
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError(f"layers as string must be 'all', got {layers!r}")
            return list(range(self.n_layers))
        return list(layers)

    @staticmethod
    def _resolve_module(layer_proxy, mod_str: str):
        """Resolve a submodule from a layer proxy by (canonical) name.

        Accepts the canonical :class:`~murano.Node` module names and their
        aliases so a stored ``Node``'s module resolves back to a live hook:
        the block output (:data:`~murano.nodes.RESID_POST`, alias ``residual``)
        is the layer proxy itself; ``mlp``/``self_attn`` (aliases ``mlp_out``/
        ``attn_out``) and dotted paths (e.g. ``mlp.gate_proj``) resolve by
        attribute lookup. The proxy's ``.output`` is the named activation.

        Args:
            layer_proxy: nnsight proxy for a decoder layer.
            mod_str: Module name to resolve.

        Returns:
            nnsight proxy whose ``.output`` is the requested activation.

        Raises:
            ValueError: If ``mod_str`` is a residual-stream point with no single
                output hook here (``resid_pre``/``resid_mid``), or any part of a
                dotted path does not exist.
        """
        canonical = canonical_module(mod_str)
        if canonical == RESID_POST:
            return layer_proxy
        if canonical in (RESID_PRE, RESID_MID):
            raise ValueError(
                f"module {mod_str!r} ({canonical}) has no single output hook on "
                f"this path: resid_pre is the block input and resid_mid is a "
                f"derived sum. Record/intervene on resid_post, mlp, or self_attn "
                f"here; the SAE path handles resid_pre/resid_mid."
            )
        current = layer_proxy
        for part in canonical.split("."):
            try:
                current = getattr(current, part)
            except AttributeError:
                raise ValueError(
                    f"Could not resolve submodule {mod_str!r} on layer proxy: "
                    f"attribute {part!r} not found. "
                    f"Available attributes depend on the model architecture."
                ) from None
        return current

    def _generate_single(
        self,
        text: str,
        fn: Callable[[Tensor, Node], Tensor] | None = None,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        input_ids = cast(Tensor, tokens["input_ids"])
        input_len = input_ids.shape[1]
        generation_kwargs = gen_kwargs or {"max_new_tokens": 256, "do_sample": False}

        module_list = [modules] if isinstance(modules, str) else modules

        with self._lm.generate(tokens, **generation_kwargs):
            if fn is not None:
                for layer_idx in self._layer_indices(layers):
                    for mod_str in module_list:
                        mod_proxy = self._resolve_module(self.layer(layer_idx), mod_str)
                        h = mod_proxy.output
                        key = Node(layer_idx, mod_str)
                        # Decoder-layer outputs are (hidden_states, ...) tuples,
                        # while submodule outputs (mlp, self_attn) are plain
                        # tensors. Edit the hidden-states element and write the
                        # same structure back so the intervention works on either.
                        if isinstance(h, tuple):
                            mod_proxy.output = (fn(h[0], key), *h[1:])  # pyright: ignore[reportArgumentType]
                        else:
                            mod_proxy.output = fn(h, key)  # pyright: ignore[reportArgumentType]
            output_ids = self._lm.generator.output.save()

        out = output_ids.value if hasattr(output_ids, "value") else output_ids
        generated = out[0, input_len:]
        # nnsight returns a proxy; tokenizer.decode accepts it at runtime.
        return cast(str, self.tokenizer.decode(generated, skip_special_tokens=True))  # pyright: ignore[reportArgumentType]

    def generate_with_hooks(
        self,
        text: str,
        fn: Callable[[Tensor, Node], Tensor] | None = None,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Generate from ``text``, optionally applying ``fn`` per layer/module.

        Args:
            text: Prompt to generate from.
            fn: ``(activation, key) -> activation`` applied to each target
                module's output during generation; ``None`` runs unmodified.
            layers: Layer indices to hook, or ``"all"``.
            modules: Module name(s) to hook at each layer.
            gen_kwargs: Forwarded to the underlying generation call.

        Returns:
            The decoded continuation, excluding the prompt.
        """
        return self._generate_single(
            text,
            fn=fn,
            layers=layers,
            modules=modules,
            gen_kwargs=gen_kwargs,
        )

    def record(
        self,
        text: str | Sequence[str],
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        position: str | int = "last",
        batch_size: int = 8,
        per_head: bool = False,
    ) -> ActivationStore:
        """Record activations on one or more texts.

        Args:
            text: Single string or sequence of strings to record from.
            layers: Layer indices to record at, or ``"all"`` for every layer.
            position: Token position to record. One of ``"last"``, ``"first"``,
                ``"mean"``, an integer index, or ``"none"`` to keep every
                position.
            batch_size: Forward-pass batch size.
            per_head: If True, split attention activations per head (attention
                modules only).

        Returns:
            ActivationStore with per-layer activations under ``positive``;
            ``negative`` is empty since this is a single-class call.
        """
        from murano.dataset import MuranoDataset
        from murano.results import Results
        from murano.steps.record import Record

        texts, _ = self._coerce_texts(text)
        results = Results()
        results[keys.DATASET] = MuranoDataset(positive_texts=texts, negative_texts=[])
        results = Record(
            self,
            layers=layers,
            modules=modules,
            position=position,
            batch_size=batch_size,
            per_head=per_head,
        )(results)
        return results[keys.RECORD]

    def logits(self, text: str | Sequence[str]) -> Tensor:
        """Return the model's output logits ``[B, S, V]`` for one or more texts.

        Quick-API counterpart to :meth:`record` and :meth:`generate`: tokenizes
        ``text`` and runs a single forward pass through the ``Logits`` step.

        Args:
            text: A single string or a sequence of strings.

        Returns:
            Output logits ``[batch, seq, vocab_size]`` on CPU as float32, one
            row per input text.
        """
        from murano.artifacts import PromptBatch
        from murano.results import Results
        from murano.steps.logits import Logits

        texts, _ = self._coerce_texts(text)
        results = Results()
        results[keys.PROMPTS] = PromptBatch(prompts=texts)
        results = Logits(self)(results)
        return results[keys.FINAL_LOGITS]

    def find_direction(
        self,
        positive: Sequence[str],
        negative: Sequence[str],
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        position: str | int = "last",
        batch_size: int = 8,
        normalize: bool = True,
    ) -> SteeringResult:
        """Find a contrastive steering direction between two text sets.

        Args:
            positive: Texts in the positive class.
            negative: Texts in the negative class.
            layers: Layer indices to record at, or ``"all"`` for every layer.
            position: Token position to record. One of ``"last"``, ``"first"``,
                ``"mean"``, or an integer index.
            batch_size: Forward-pass batch size.
            normalize: If True, normalize each per-layer direction to unit norm.

        Returns:
            SteeringResult with one direction per layer plus the best-scoring
            layer.
        """
        from murano.dataset import MuranoDataset
        from murano.results import Results
        from murano.steps.record import Record
        from murano.steps.train import SteeringVector

        results = Results()
        results[keys.DATASET] = MuranoDataset.contrastive(
            positive=list(positive),
            negative=list(negative),
        )
        results = Record(
            self,
            layers=layers,
            modules=modules,
            position=position,
            batch_size=batch_size,
        )(results)
        results = SteeringVector(normalize=normalize)(results)
        return results[keys.STEERING]

    def generate(
        self,
        text: str | Sequence[str],
        ablate: Any | None = None,
        steer: tuple[Any, float] | None = None,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str | list[str]:
        """Generate text, optionally with activation-space steering or ablation.

        Args:
            text: Single prompt or sequence of prompts.
            ablate: SteeringResult or ``{layer: tensor}`` mapping; if given, the
                direction is projected out of the residual stream at each
                target layer during generation.
            steer: ``(direction_like, alpha)`` tuple; if given, ``alpha *
                direction`` is added to the residual stream at each target
                layer. Pass either ``ablate`` or ``steer``, not both.
            layers: Layer indices to apply the intervention at, or ``"all"``.
            gen_kwargs: Forwarded to the underlying generation call. Defaults
                to ``{"max_new_tokens": 256, "do_sample": False}``.

        Returns:
            A single string when ``text`` is a single string, otherwise a list.

        Raises:
            ValueError: If both ``ablate`` and ``steer`` are passed.
        """
        from murano.steps.intervene import ablate_direction, steer_direction

        if ablate is not None and steer is not None:
            raise ValueError("Pass either 'ablate' or 'steer', not both.")

        prompts, is_single = self._coerce_texts(text)
        fn = None
        if ablate is not None:
            directions = self._coerce_directions(ablate)
            self._check_directions_reachable(directions, layers, modules)
            fn = ablate_direction(directions)
        elif steer is not None:
            direction_like, alpha = steer
            directions = self._coerce_directions(direction_like)
            self._check_directions_reachable(directions, layers, modules)
            fn = steer_direction(directions, alpha)

        outputs = [
            self._generate_single(
                prompt,
                fn=fn,
                layers=layers,
                modules=modules,
                gen_kwargs=gen_kwargs,
            )
            for prompt in prompts
        ]
        return outputs[0] if is_single else outputs

    def chat_template(self, messages: list[dict]) -> str:
        """Apply the tokenizer's chat template to a list of messages.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.

        Returns:
            The rendered prompt string with a generation prompt appended.
        """
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
        )

    def __repr__(self) -> str:
        return (
            f"MuranoModel({self.model_id!r}, layers={self.n_layers}, d={self.d_model})"
        )
