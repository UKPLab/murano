"""Intervene step: applies activation-space interventions during generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from torch import Tensor, isfinite  # pyright: ignore[reportPrivateImportUsage]
from tqdm import tqdm

from murano import keys
from murano.artifacts import GenerationComparison, PromptBatch
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step
from murano.steps.record import ActivationKey

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


def _normalize_directions(
    directions: dict[ActivationKey, Tensor],
) -> dict[ActivationKey, Tensor]:
    normalized: dict[ActivationKey, Tensor] = {}
    for key, direction in directions.items():
        norm = direction.norm()
        if not isfinite(norm).item() or norm.item() < 1e-10:
            logger.warning(
                "Skipping non-finite or near-zero direction at key %s",
                key,
            )
            continue
        normalized[key] = direction / norm
    return normalized


def ablate_direction(directions: dict[ActivationKey, Tensor]) -> Callable:
    """Return an intervention function that projects out a direction.

    Removes the component along the direction from the residual stream.

    Args:
        directions: {key: tensor [d_model]} directions to ablate.
                    Keys are ``(layer, module_name)`` tuples.

    Returns:
        Callable(activation, key) -> modified activation.
    """
    normalized = _normalize_directions(directions)

    def fn(activation: Tensor, key: ActivationKey) -> Tensor:
        if key not in normalized:
            return activation
        d_hat = normalized[key].to(activation.device, activation.dtype)
        proj = (activation @ d_hat).unsqueeze(-1) * d_hat
        return activation - proj

    return fn


def steer_direction(directions: dict[ActivationKey, Tensor], alpha: float) -> Callable:
    """Return an intervention function that adds a scaled direction.

    Adds alpha * direction to the residual stream at each layer/module.

    Args:
        directions: {key: tensor [d_model]} directions to add.
                    Keys are ``(layer, module_name)`` tuples.
        alpha: Scaling factor. Positive = strengthen, negative = suppress.

    Returns:
        Callable(activation, key) -> modified activation.
    """
    normalized = _normalize_directions(directions)

    def fn(activation: Tensor, key: ActivationKey) -> Tensor:
        if key not in normalized:
            return activation
        d_hat = normalized[key].to(activation.device, activation.dtype)
        return activation + alpha * d_hat

    return fn


class Intervene(Step):
    """Apply an intervention function during model generation.

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

    reads = [keys.PROMPTS]
    writes = [keys.INTERVENE]
    write_types = {keys.INTERVENE: InterveneResult}

    def expected_read_types(self, results=None, available_types=None):
        """Return ``{"prompts": PromptBatch}``."""
        return {keys.PROMPTS: PromptBatch}

    def __init__(
        self,
        model: MuranoModel,
        fn: Callable,
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        gen_kwargs: dict | None = None,
    ):
        self.model = model
        self.fn = fn
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError(f"layers as string must be 'all', got {layers!r}")
            self.layers: list[int] = list(range(model.n_layers))
        else:
            self.layers = list(layers)
        self.modules: list[str] = [modules] if isinstance(modules, str) else modules
        self.gen_kwargs = gen_kwargs or {"max_new_tokens": 256, "do_sample": False}

    def __call__(self, results: Results) -> Results:
        prompt_batch = results[keys.PROMPTS]
        prompts = prompt_batch.prompts
        clean_gens = []
        modified_gens = []

        for prompt in tqdm(prompts, desc="Intervene"):
            clean_gens.append(self._generate_clean(prompt))
            modified_gens.append(self._generate_ablated(prompt))

        results[keys.INTERVENE] = InterveneResult(
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
        """Generate with the intervention applied at each layer/module."""
        return self.model._generate_single(
            text,
            fn=self.fn,
            layers=self.layers,
            modules=self.modules,
            gen_kwargs=self.gen_kwargs,
        )
