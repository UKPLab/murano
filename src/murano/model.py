"""MuranoModel: nnterp-based model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import torch
from torch import Tensor, bfloat16  # pyright: ignore[reportPrivateImportUsage]
from torch import dtype as TorchDtype  # pyright: ignore[reportPrivateImportUsage]
from nnterp import StandardizedTransformer

from murano.logging import logger
from murano.steps.record import ActivationKey

if TYPE_CHECKING:
    from murano.steps.record import ActivationStore
    from murano.steps.train import SteeringResult


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


def _apply_intervention(
    output: Any,
    fn: Callable[[Tensor, ActivationKey], Tensor],
    key: ActivationKey,
) -> Any:
    """Apply an intervention ``fn`` to a module's output during generation.

    Decoder layers (and attention submodules) return ``(hidden_states, ...)``
    tuples rather than a bare tensor, so the intervention must transform only
    the hidden-states element and leave the rest of the tuple (e.g. cached
    key/values) untouched. When the output is already a tensor (some submodules
    and some architectures), ``fn`` is applied directly. This mirrors the
    ``isinstance(output, tuple)`` unwrapping the record and logit-lens read
    paths already use.
    """
    if isinstance(output, tuple):
        return (fn(output[0], key),) + tuple(output[1:])
    return fn(output, key)


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
        logger.info(
            "Loaded %s (%d layers, d=%d)", model_id, self.n_layers, self.d_model
        )

    def layer(self, idx: int):
        """Return the nnterp module proxy for a decoder layer."""
        return self._lm.layers[idx]  # pyright: ignore[reportIndexIssue,reportArgumentType]

    def _coerce_texts(self, text: str | Sequence[str]) -> tuple[list[str], bool]:
        if isinstance(text, str):
            return [text], True
        return list(text), False

    def _coerce_directions(self, direction_like: Any) -> dict[ActivationKey, Tensor]:
        if hasattr(direction_like, "direction_per_layer"):
            return direction_like.direction_per_layer
        if isinstance(direction_like, dict):
            return direction_like
        raise TypeError(
            "Expected a SteeringResult or {key: tensor} mapping for the "
            "intervention directions."
        )

    def _layer_indices(self, layers: list[int] | str) -> list[int]:
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError(f"layers as string must be 'all', got {layers!r}")
            return list(range(self.n_layers))
        return list(layers)

    @staticmethod
    def _resolve_module(layer_proxy, mod_str: str):
        """Resolve a submodule from a layer proxy by name.

        Handles ``"residual"`` (returns the layer proxy itself),
        direct children (e.g. ``"mlp"``), and dotted paths
        (e.g. ``"mlp.gate_proj"``).

        Args:
            layer_proxy: nnsight proxy for a decoder layer.
            mod_str: Module name to resolve.

        Returns:
            nnsight proxy for the requested submodule.

        Raises:
            ValueError: If any part of the dotted path does not exist.
        """
        if mod_str == "residual":
            return layer_proxy
        current = layer_proxy
        for part in mod_str.split("."):
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
        fn: Callable[[Tensor, int | tuple[int, str]], Tensor] | None = None,
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
                        key: int | tuple[int, str] = (
                            layer_idx if len(module_list) == 1 else (layer_idx, mod_str)
                        )
                        mod_proxy.output = _apply_intervention(  # pyright: ignore[reportAttributeAccessIssue]
                            mod_proxy.output, fn, key
                        )
            output_ids = self._lm.generator.output.save()

        out = output_ids.value if hasattr(output_ids, "value") else output_ids
        generated = out[0, input_len:]
        # nnsight returns a proxy; tokenizer.decode accepts it at runtime.
        return cast(str, self.tokenizer.decode(generated, skip_special_tokens=True))  # pyright: ignore[reportArgumentType]

    def record(
        self,
        text: str | Sequence[str],
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        position: str | int = "last",
        batch_size: int = 8,
    ) -> ActivationStore:
        """Record activations on one or more texts.

        Args:
            text: Single string or sequence of strings to record from.
            layers: Layer indices to record at, or ``"all"`` for every layer.
            position: Token position to record. One of ``"last"``, ``"first"``,
                ``"mean"``, or an integer index.
            batch_size: Forward-pass batch size.

        Returns:
            ActivationStore with per-layer activations under ``positive``;
            ``negative`` is empty since this is a single-class call.
        """
        from murano.dataset import MuranoDataset
        from murano.results import Results
        from murano.steps.record import Record

        texts, _ = self._coerce_texts(text)
        results = Results()
        results["dataset"] = MuranoDataset(positive_texts=texts, negative_texts=[])
        results = Record(
            self,
            layers=layers,
            modules=modules,
            position=position,
            batch_size=batch_size,
        )(results)
        return results["record"]

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
        results["dataset"] = MuranoDataset.contrastive(
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
        return results["steering"]

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
            fn = ablate_direction(self._coerce_directions(ablate))
        elif steer is not None:
            direction_like, alpha = steer
            fn = steer_direction(self._coerce_directions(direction_like), alpha)

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
