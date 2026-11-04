"""Train step: computes directions from recorded activations.

Attribution:
    Method (``SteeringVector(method="lat")``): Zou, Phan, Chen, Campbell et al.
    (2023), "Representation Engineering: A Top-Down Approach to AI
    Transparency", arXiv. https://arxiv.org/abs/2310.01405
    Reference code: andyzoujm/representation-engineering@5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e
    (MIT), adapted: ``PCARepReader`` direction and sign rule
    (``repe/rep_readers.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
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


_METHODS = ("contrastive_mean_diff", "lat")


class SteeringVector(Step):
    """Find a steering direction from contrastive activations.

    ``method="contrastive_mean_diff"`` takes the mean of the differences
    ``D = positive - negative``. ``method="lat"`` is the reading vector of
    Linear Artificial Tomography (Zou et al., 2023): the top singular vector of
    the row-aligned pair differences ``D``, i.e. the top eigenvector of their
    second moment ``mean(D) mean(D)^T + Cov(D)``. With a consistent shift
    between the sides the two agree; when the differences also vary strongly
    along another axis, LAT follows that axis. The singular vector's sign is
    arbitrary, so LAT flips it when fewer than half of the pairs project
    positive above negative.

    Reads from results:
        results['record']: ActivationStore (must have .positive and .negative)

    Writes to results:
        results['steering']: SteeringResult

    Args:
        method: Estimation method, ``"contrastive_mean_diff"`` (default) or
            ``"lat"``. ``"lat"`` needs the i-th positive and i-th negative
            examples to form a pair.
        normalize: If True, normalize each stored per-layer direction to unit
            norm. A ``"lat"`` direction is a singular vector, so it is unit norm
            either way. Note the intervention functions (``steer_direction`` /
            ``ablate_direction``) unit-normalize again at apply time, so
            ``normalize=False`` only affects the magnitude of the stored vector
            (e.g. if you read ``direction_per_layer`` yourself); it does not
            change steering or ablation strength.

    Raises:
        ValueError: If ``method`` is not a supported value, or if
            ``method="lat"`` and a layer has different positive and negative
            counts.
    """

    reads = [keys.RECORD]
    writes = [keys.STEERING]
    read_types = {keys.RECORD: ActivationStore}
    write_types = {keys.STEERING: SteeringResult}

    def __init__(self, method: str = "contrastive_mean_diff", normalize: bool = True):
        if method not in _METHODS:
            raise ValueError(
                f"Unknown SteeringVector method {method!r}; pass "
                "method='contrastive_mean_diff' or method='lat'."
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

            if self.method == "lat":
                direction = _lat_direction(pos_acts, neg_acts, key)
            else:
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


def _lat_direction(pos_acts: Tensor, neg_acts: Tensor, key: Node) -> Tensor:
    """Sign-fixed top right singular vector of the pair differences."""
    if pos_acts.shape[0] != neg_acts.shape[0]:
        raise ValueError(
            f"SteeringVector(method='lat') needs row-aligned pairs, but {key} has "
            f"{pos_acts.shape[0]} positive and {neg_acts.shape[0]} negative rows. "
            "Record a dataset with one negative per positive, or use "
            "method='contrastive_mean_diff'."
        )
    diffs = pos_acts - neg_acts
    if diffs.norm() < 1e-10:
        # Differs from andyzoujm/representation-engineering@5455d8a
        # repe/rep_readers.py PCARepReader.get_rep_directions: returns a zero
        # vector, not the PCA's arbitrary unit vector, when no pair differs,
        # because a zero direction takes the near-zero warning path, as
        # mean-diff does.
        return diffs.new_zeros(diffs.shape[1])
    # Differs from andyzoujm/representation-engineering@5455d8a
    # repe/rep_readers.py PCARepReader.get_rep_directions: an uncentered SVD of
    # the fixed pos - neg differences, not a random within-pair shuffle
    # (examples/honesty/utils.py) followed by a centered sklearn PCA, because
    # the shuffle only exists to make the centered PCA act as this SVD, and the
    # SVD is deterministic.
    direction = torch.linalg.svd(diffs, full_matrices=False).Vh[0]
    # Differs from andyzoujm/representation-engineering@5455d8a
    # repe/rep_readers.py PCARepReader.get_signs: compares pos and neg through
    # their difference, not through raw states recentered on the training mean,
    # because the shared shift cancels in every within-pair comparison.
    proj = diffs @ direction
    # Differs from andyzoujm/representation-engineering@5455d8a
    # repe/rep_readers.py PCARepReader.get_signs: a tied pair counts against the
    # positive side, because the maintainer decision on #124 flips when fewer
    # than half of the pairs project positive.
    if 2 * (proj > 0).sum() < proj.numel():
        direction = -direction
    return direction
