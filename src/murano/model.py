"""MuranoModel — nnsight-based model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from torch import Tensor
from nnsight import LanguageModel

from murano.logging import logger

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


class MuranoModel:
    """Thin wrapper around nnsight LanguageModel for mechanistic interpretability.

    Provides access to model layers, tokenizer, and metadata.
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
        dtype: torch.dtype = torch.bfloat16,
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

        kwargs = dict(device_map=device_map, dtype=dtype, dispatch=True)
        try:
            self._lm = LanguageModel(load_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e
        self.tokenizer = self._lm.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.n_layers = self._lm.config.num_hidden_layers
        self.d_model = self._lm.config.hidden_size
        logger.info(
            "Loaded %s (%d layers, d=%d)", model_id, self.n_layers, self.d_model
        )

    def layer(self, idx: int):
        """Returns the nnsight module proxy for a decoder layer."""
        return self._lm.model.layers[idx]

    def _coerce_texts(self, text: str | Sequence[str]) -> tuple[list[str], bool]:
        if isinstance(text, str):
            return [text], True
        return list(text), False

    def _coerce_directions(self, direction_like: Any) -> dict[int, Tensor]:
        if hasattr(direction_like, "direction_per_layer"):
            return direction_like.direction_per_layer
        if isinstance(direction_like, dict):
            return direction_like
        raise TypeError(
            "Expected a SteeringResult or {layer: tensor} mapping for the "
            "intervention directions."
        )

    def _layer_indices(self, layers: list[int] | str) -> list[int]:
        return list(range(self.n_layers)) if layers == "all" else list(layers)

    def _generate_single(
        self,
        text: str,
        fn: Callable[[Tensor, int], Tensor] | None = None,
        layers: list[int] | str = "all",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        input_len = tokens["input_ids"].shape[1]
        generation_kwargs = gen_kwargs or {"max_new_tokens": 256, "do_sample": False}

        with self._lm.generate(tokens, **generation_kwargs):
            if fn is not None:
                for layer_idx in self._layer_indices(layers):
                    h = self.layer(layer_idx).output
                    self.layer(layer_idx).output = fn(h, layer_idx)
            output_ids = self._lm.generator.output.save()

        out = output_ids.value if hasattr(output_ids, "value") else output_ids
        generated = out[0, input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def record(
        self,
        text: str | Sequence[str],
        layers: list[int] | str = "all",
        position: str | int = "last",
        batch_size: int = 8,
    ) -> ActivationStore:
        """Record activations on one or more texts."""
        from murano.dataset import MuranoDataset
        from murano.results import Results
        from murano.steps.record import Record

        texts, _ = self._coerce_texts(text)
        results = Results()
        results["dataset"] = MuranoDataset(positive_texts=texts, negative_texts=[])
        results = Record(
            self,
            layers=layers,
            position=position,
            batch_size=batch_size,
        )(results)
        return results["record"]

    def find_direction(
        self,
        positive: Sequence[str],
        negative: Sequence[str],
        layers: list[int] | str = "all",
        position: str | int = "last",
        batch_size: int = 8,
        normalize: bool = True,
    ) -> SteeringResult:
        """Find a contrastive steering direction between two text sets."""
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
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str | list[str]:
        """Generate text, optionally with activation-space steering or ablation."""
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
                gen_kwargs=gen_kwargs,
            )
            for prompt in prompts
        ]
        return outputs[0] if is_single else outputs

    def chat_template(self, messages: list[dict]) -> str:
        """Apply the tokenizer's chat template to a list of messages."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def __repr__(self) -> str:
        return (
            f"MuranoModel({self.model_id!r}, layers={self.n_layers}, d={self.d_model})"
        )
