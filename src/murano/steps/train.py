"""Train step — computes directions from recorded activations."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step
from murano.steps.record import ActivationStore


@dataclass
class SteeringResult:
    """Output of SteeringVector step.

    Attributes:
        direction_per_layer: {layer_idx: tensor [d_model]} normalized direction.
        separation_scores: {layer_idx: float} how well the direction separates classes.
        best_layer: Layer index with highest separation score.
    """

    direction_per_layer: dict[int, Tensor]
    separation_scores: dict[int, float]
    best_layer: int


class SteeringVector(Step):
    """Finds a steering direction via contrastive mean difference.

    Reads from results:
        results['record']: ActivationStore (must have .positive and .negative)

    Writes to results:
        results['steering']: SteeringResult
    """

    reads = ["record"]
    writes = ["steering"]
    read_types = {"record": ActivationStore}
    write_types = {"steering": SteeringResult}

    def __init__(self, method: str = "contrastive_mean_diff", normalize: bool = True):
        if method != "contrastive_mean_diff":
            raise ValueError(
                "SteeringVector currently only supports method='contrastive_mean_diff'."
            )
        self.method = method
        self.normalize = normalize

    def __call__(self, results: Results) -> Results:
        store = results["record"]
        directions: dict[int, Tensor] = {}
        scores: dict[int, float] = {}

        for layer in store.positive:
            pos_acts = store.positive[layer].float()
            neg_acts = store.negative[layer].float()

            direction = pos_acts.mean(0) - neg_acts.mean(0)

            # Separation score: normalized distance between projected means
            pos_proj = pos_acts @ direction
            neg_proj = neg_acts @ direction
            score = (pos_proj.mean() - neg_proj.mean()) / (
                pos_proj.std(unbiased=False) + neg_proj.std(unbiased=False) + 1e-8
            )

            if self.normalize:
                norm = direction.norm()
                if norm < 1e-10:
                    logger.warning(
                        "Near-zero direction at layer %d, skipping normalization", layer
                    )
                else:
                    direction = direction / norm

            directions[layer] = direction
            scores[layer] = score.item()

        if not scores:
            raise ValueError(
                "SteeringVector requires at least one layer with recorded "
                "positive and negative activations."
            )

        best_layer = max(scores, key=lambda k: scores[k])
        results["steering"] = SteeringResult(
            direction_per_layer=directions,
            separation_scores=scores,
            best_layer=best_layer,
        )
        logger.info("Best layer: %d (score=%.4f)", best_layer, scores[best_layer])
        return results
