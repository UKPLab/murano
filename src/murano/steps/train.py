"""Train step: computes directions from recorded activations."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from murano import keys
from murano.logging import logger
from murano.nodes import Node, NodeDict
from murano.results import Results
from murano.steps.base import Step
from murano.steps.record import ActivationStore, _require_reduced_store


@dataclass
class SteeringResult:
    """Output of SteeringVector step.

    Attributes:
        direction_per_layer: ``{Node: tensor [d_model]}`` normalized direction,
            keyed by component address.
        separation_scores: ``{Node: float}`` how well the direction separates
            classes.
        best_layer: :class:`Node` with the highest separation score.
    """

    direction_per_layer: dict[Node, Tensor]
    separation_scores: dict[Node, float]
    best_layer: Node

    def __post_init__(self) -> None:
        self.direction_per_layer = NodeDict(self.direction_per_layer)
        self.separation_scores = NodeDict(self.separation_scores)
        self.best_layer = Node.coerce(self.best_layer)


class SteeringVector(Step):
    """Find a steering direction via contrastive mean difference.

    Reads from results:
        results['record']: ActivationStore (must have .positive and .negative)

    Writes to results:
        results['steering']: SteeringResult

    Args:
        method: Estimation method. Currently only ``"contrastive_mean_diff"``
            is supported.
        normalize: If True, normalize each stored per-layer direction to unit
            norm. Note the intervention functions (``steer_direction`` /
            ``ablate_direction``) unit-normalize again at apply time, so
            ``normalize=False`` only affects the magnitude of the stored vector
            (e.g. if you read ``direction_per_layer`` yourself); it does not
            change steering or ablation strength.

    Raises:
        ValueError: If ``method`` is not a supported value.
    """

    reads = [keys.RECORD]
    writes = [keys.STEERING]
    read_types = {keys.RECORD: ActivationStore}
    write_types = {keys.STEERING: SteeringResult}

    def __init__(self, method: str = "contrastive_mean_diff", normalize: bool = True):
        if method != "contrastive_mean_diff":
            raise ValueError(
                "SteeringVector currently only supports method='contrastive_mean_diff'."
            )
        self.method = method
        self.normalize = normalize

    def __call__(self, results: Results) -> Results:
        store = results[keys.RECORD]
        _require_reduced_store(store, "SteeringVector")
        directions: dict[Node, Tensor] = {}
        scores: dict[Node, float] = {}

        for key in store.positive:
            pos_acts = store.positive[key].float()
            neg_acts = store.negative[key].float()

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
                        "Near-zero direction at key %s, skipping normalization", key
                    )
                else:
                    direction = direction / norm

            directions[key] = direction
            scores[key] = score.item()

        if not scores:
            raise ValueError(
                "SteeringVector requires at least one layer with recorded "
                "positive and negative activations."
            )

        best_key = max(scores, key=lambda k: scores[k])
        results[keys.STEERING] = SteeringResult(
            direction_per_layer=directions,
            separation_scores=scores,
            best_layer=best_key,
        )
        logger.info("Best key: %s (score=%.4f)", best_key, scores[best_key])
        return results
