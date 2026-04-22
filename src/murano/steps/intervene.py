"""Intervene step — applies activation-space interventions during generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor
from tqdm import tqdm

from murano.artifacts import GenerationComparison, PromptBatch
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.model import MuranoModel


class InterveneResult(GenerationComparison):
    """Output of Intervene step.

    Attributes:
        clean_generations: Model responses without intervention.
        modified_generations: Model responses with intervention applied.
    """

    def __init__(
        self,
        clean_generations: list[str],
        modified_generations: list[str],
        prompts: list[str] | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(
            baseline_generations=clean_generations,
            modified_generations=modified_generations,
            prompts=prompts,
            baseline_label="clean",
            modified_label="modified",
            metadata={} if metadata is None else metadata,
        )


def _normalize_directions(directions: dict[int, Tensor]) -> dict[int, Tensor]:
    normalized: dict[int, Tensor] = {}
    for layer, direction in directions.items():
        norm = direction.norm()
        if not torch.isfinite(norm).item() or norm.item() < 1e-10:
            logger.warning(
                "Skipping non-finite or near-zero direction at layer %d",
                layer,
            )
            continue
        normalized[layer] = direction / norm
    return normalized


def ablate_direction(directions: dict[int, Tensor]) -> Callable:
    """Returns an intervention function that projects out a direction.

    Removes the component along the direction from the residual stream.

    Args:
        directions: {layer_idx: tensor [d_model]} directions to ablate.

    Returns:
        Callable(activation, layer_idx) -> modified activation.
    """
    normalized = _normalize_directions(directions)

    def fn(activation: Tensor, layer: int) -> Tensor:
        if layer not in normalized:
            return activation
        d_hat = normalized[layer].to(activation.device, activation.dtype)
        proj = (activation @ d_hat).unsqueeze(-1) * d_hat
        return activation - proj

    return fn


def steer_direction(directions: dict[int, Tensor], alpha: float) -> Callable:
    """Returns an intervention function that adds a scaled direction.

    Adds alpha * direction to the residual stream at each layer.

    Args:
        directions: {layer_idx: tensor [d_model]} directions to add.
        alpha: Scaling factor. Positive = strengthen, negative = suppress.

    Returns:
        Callable(activation, layer_idx) -> modified activation.
    """
    normalized = _normalize_directions(directions)

    def fn(activation: Tensor, layer: int) -> Tensor:
        if layer not in normalized:
            return activation
        d_hat = normalized[layer].to(activation.device, activation.dtype)
        return activation + alpha * d_hat

    return fn


class Intervene(Step):
    """Applies an intervention function during model generation.

    Generates both clean (no intervention) and modified outputs for comparison.

    Reads from results:
        results['prompts']: PromptBatch

    Writes to results:
        results['intervene']: InterveneResult

    Args:
        model: MuranoModel to generate with.
        fn: Callable(activation, layer_idx) -> activation.
        layers: Which layers to apply the intervention at.
        gen_kwargs: Keyword arguments for model.generate().
    """

    reads = ["prompts"]
    writes = ["intervene"]
    write_types = {"intervene": InterveneResult}

    def expected_read_types(self, results=None, available_types=None):
        return {"prompts": PromptBatch}

    def __init__(
        self,
        model: MuranoModel,
        fn: Callable,
        layers: list[int] | str = "all",
        gen_kwargs: dict | None = None,
    ):
        self.model = model
        self.fn = fn
        self.layers = (
            list(range(model.n_layers)) if layers == "all" else list(layers)
        )
        self.gen_kwargs = gen_kwargs or {"max_new_tokens": 256, "do_sample": False}

    def __call__(self, results: Results) -> Results:
        prompt_batch = results["prompts"]
        prompts = prompt_batch.prompts
        clean_gens = []
        modified_gens = []

        for prompt in tqdm(prompts, desc="Intervene"):
            clean_gens.append(self._generate_clean(prompt))
            modified_gens.append(self._generate_ablated(prompt))

        results['intervene'] = InterveneResult(
            clean_generations=clean_gens,
            modified_generations=modified_gens,
            prompts=(
                list(prompt_batch.raw_prompts)
                if prompt_batch.raw_prompts is not None
                else list(prompt_batch.prompts)
            ),
            metadata={
                "prompt_source": prompt_batch.source,
                **prompt_batch.metadata,
            },
        )
        return results

    def _generate_clean(self, text: str) -> str:
        """Generate without any intervention."""
        return self.model._generate_single(text, gen_kwargs=self.gen_kwargs)

    def _generate_ablated(self, text: str) -> str:
        """Generate with the intervention applied at each layer."""
        return self.model._generate_single(
            text,
            fn=self.fn,
            layers=self.layers,
            gen_kwargs=self.gen_kwargs,
        )
