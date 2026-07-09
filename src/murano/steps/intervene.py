"""Intervene step: applies activation-space interventions during generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from torch import Tensor, isfinite  # pyright: ignore[reportPrivateImportUsage]
from tqdm import tqdm

from murano import keys
from murano.artifacts import GenerationComparison, PromptBatch
from murano.logging import logger
from murano.nodes import Node, NodeDict
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.backend import ModelBackend


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


def _normalize_directions(directions: dict[Node, Tensor]) -> NodeDict:
    normalized = NodeDict()
    for key, direction in NodeDict(directions).items():
        norm = direction.norm()
        if not isfinite(norm).item() or norm.item() < 1e-10:
            logger.warning(
                "Skipping non-finite or near-zero direction at key %s",
                key,
            )
            continue
        normalized[key] = direction / norm
    return normalized


def ablate_direction(directions: dict[Node, Tensor]) -> Callable:
    """Return an intervention function that projects out a direction.

    Removes the component along the direction from the residual stream.

    Args:
        directions: ``{address: tensor [d_model]}`` directions to ablate. Keys
            are coerced to canonical :class:`Node` addresses, so shorthand
            (a layer int, a ``(layer, module)`` tuple, a Node) is accepted.

    Returns:
        Callable(activation, node) -> modified activation.
    """
    normalized = _normalize_directions(directions)

    def fn(activation: Tensor, key: Node) -> Tensor:
        if key not in normalized:
            return activation
        d_hat = normalized[key].to(activation.device, activation.dtype)
        proj = (activation @ d_hat).unsqueeze(-1) * d_hat
        return activation - proj

    return fn


def steer_direction(directions: dict[Node, Tensor], alpha: float) -> Callable:
    """Return an intervention function that adds a scaled direction.

    Adds alpha * direction to the residual stream at each layer/module.

    Args:
        directions: ``{address: tensor [d_model]}`` directions to add. Keys are
            coerced to canonical :class:`Node` addresses.
        alpha: Scaling factor. Positive = strengthen, negative = suppress.

    Returns:
        Callable(activation, node) -> modified activation.
    """
    normalized = _normalize_directions(directions)

    def fn(activation: Tensor, key: Node) -> Tensor:
        if key not in normalized:
            return activation
        d_hat = normalized[key].to(activation.device, activation.dtype)
        return activation + alpha * d_hat

    return fn


class Intervene(Step):
    """Apply an activation-space intervention during generation and compare.

    Generates a clean (unmodified) and an intervened completion for each prompt.
    The intervention can be supplied two ways:

    - ``fn``: a prebuilt ``Callable(activation, node) -> activation``. The
      direction is fixed when the step is constructed.
    - ``direction_key``: the name of a Results key holding a ``SteeringResult``
      an upstream step wrote. The direction is read at run time and turned into a
      steer or ablate intervention, so deriving a direction and applying it
      compose in a single pipeline.

    With ``direction_key``, ``direction_layers`` chooses which of the recorded
    per-layer directions to apply. A ``SteeringResult`` carries one direction per
    recorded layer; adding all of them at once over-steers deep models into
    degenerate text, so ``"best"`` (the single best-separating layer) or an
    explicit layer list keeps the flagship one-pipeline steer coherent, while
    ``"all"`` preserves the every-layer behavior.

    Reads from results:
        results['prompts']: PromptBatch
        results[direction_key]: SteeringResult, when ``direction_key`` is set.

    Writes to results:
        results['intervene']: InterveneResult

    Args:
        model: Model to generate with.
        fn: Prebuilt intervention ``Callable(activation, node) -> activation``.
            Pass this or ``direction_key``, not both.
        direction_key: Results key holding the ``SteeringResult`` to apply at run
            time. Pass this or ``fn``, not both.
        mode: With ``direction_key``, ``"steer"`` adds the direction and
            ``"ablate"`` projects it out. Ignored when ``fn`` is given.
        alpha: Steering strength for ``mode="steer"``.
        direction_layers: With ``direction_key``, which recorded directions to
            apply: ``"all"`` every layer, ``"best"`` only the best-separating
            layer (``SteeringResult.best_layer``), or an explicit list of layer
            indices. Only valid with ``direction_key``.
        layers: Layers to apply the intervention at, or ``"all"``.
        modules: Module name(s) to apply the intervention at each layer.
        gen_kwargs: Keyword arguments forwarded to generation.

    Raises:
        ValueError: If neither or both of ``fn`` and ``direction_key`` are given,
            if ``mode`` is not ``"steer"`` or ``"ablate"``, or if
            ``direction_layers`` is given without ``direction_key`` or is not
            ``"all"``, ``"best"``, or a list of layer indices.
    """

    writes = [keys.INTERVENE]
    write_types = {keys.INTERVENE: InterveneResult}

    def expected_read_types(self, results=None, available_types=None):
        """Return the expected types for the keys this step reads."""
        # Imported lazily: train.py pulls in step modules, so a top-level import
        # here would be circular.
        from murano.steps.train import SteeringResult

        expected: dict[str, type] = {keys.PROMPTS: PromptBatch}
        if self.direction_key is not None:
            expected[self.direction_key] = SteeringResult
        return expected

    def __init__(
        self,
        model: ModelBackend,
        fn: Callable | None = None,
        *,
        direction_key: str | None = None,
        mode: str = "steer",
        alpha: float = 1.0,
        direction_layers: str | list[int] = "all",
        layers: list[int] | str = "all",
        modules: str | list[str] = "residual",
        gen_kwargs: dict | None = None,
    ):
        if (fn is None) == (direction_key is None):
            raise ValueError(
                "Pass exactly one of fn= (a prebuilt intervention) or "
                "direction_key= (read a SteeringResult from Results and "
                "steer/ablate it)."
            )
        if direction_key is not None and mode not in ("steer", "ablate"):
            raise ValueError(f"mode must be 'steer' or 'ablate', got {mode!r}")
        if fn is not None and direction_layers != "all":
            raise ValueError(
                "direction_layers only applies with direction_key=; a prebuilt "
                "fn= already fixes which layers it touches."
            )
        if isinstance(direction_layers, str):
            if direction_layers not in ("all", "best"):
                raise ValueError(
                    f"direction_layers as a string must be 'all' or 'best', got "
                    f"{direction_layers!r}; pass a list of layer indices to choose "
                    f"specific layers."
                )
            self.direction_layers: str | list[int] = direction_layers
        else:
            self.direction_layers = list(direction_layers)
        self.model = model
        self.fn = fn
        self.direction_key = direction_key
        self.mode = mode
        self.alpha = alpha
        # Declare the direction source as a read so the pipeline's pre-flight
        # check fails when no upstream step produces it, instead of silently
        # generating without the intended intervention.
        self.reads = (
            [keys.PROMPTS] if direction_key is None else [keys.PROMPTS, direction_key]
        )
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
        fn = self._resolve_fn(results)
        clean_gens = []
        modified_gens = []

        for prompt in tqdm(prompts, desc="Intervene"):
            clean_gens.append(self._generate_clean(prompt))
            modified_gens.append(self._generate_modified(prompt, fn))

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

    def _resolve_fn(self, results: Results) -> Callable:
        """Return the intervention to apply, building it from Results if needed."""
        if self.fn is not None:
            return self.fn
        # __init__ guarantees exactly one of fn / direction_key, so direction_key
        # is set on this branch.
        assert self.direction_key is not None
        # Build the intervention now, from the direction an upstream step wrote,
        # so the derive-then-apply arc is one pipeline. steer_direction /
        # ablate_direction normalize the direction keys to canonical Nodes.
        steering = results[self.direction_key]
        directions = self._select_directions(steering)
        if self.mode == "steer":
            return steer_direction(directions, self.alpha)
        return ablate_direction(directions)

    def _select_directions(self, steering: object) -> dict[Node, Tensor]:
        """Return the subset of recorded directions ``direction_layers`` selects.

        ``steering`` is the ``SteeringResult`` read from Results (its
        ``direction_per_layer`` is a ``{Node: tensor}`` map); a bare mapping is
        also accepted so a caller can write directions directly under the key.

        Raises:
            ValueError: If ``direction_layers="best"`` but ``steering`` carries no
                ``best_layer``, or an explicit layer list matches no recorded layer.
        """
        directions = NodeDict(getattr(steering, "direction_per_layer", steering))
        if self.direction_layers == "all":
            return directions
        if self.direction_layers == "best":
            best = getattr(steering, "best_layer", None)
            if best is None:
                raise ValueError(
                    "direction_layers='best' needs a SteeringResult with a "
                    "best_layer; the direction source has none."
                )
            return NodeDict({best: directions[best]})
        wanted = set(self.direction_layers)
        subset = NodeDict(
            {node: vec for node, vec in directions.items() if node.layer in wanted}
        )
        if not subset:
            available = sorted({node.layer for node in directions})
            raise ValueError(
                f"direction_layers={self.direction_layers} selected no recorded "
                f"layer; the direction source covers layers {available}."
            )
        return subset

    def _generate_clean(self, text: str) -> str:
        """Generate without any intervention."""
        return self.model.generate_with_hooks(text, gen_kwargs=self.gen_kwargs)

    def _generate_modified(self, text: str, fn: Callable) -> str:
        """Generate with ``fn`` applied at each target layer and module."""
        return self.model.generate_with_hooks(
            text,
            fn=fn,
            layers=self.layers,
            modules=self.modules,
            gen_kwargs=self.gen_kwargs,
        )
