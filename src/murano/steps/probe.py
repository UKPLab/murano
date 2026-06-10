"""Probe step: trains linear classifiers on recorded activations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from numpy import ndarray

from murano import keys
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step
from murano.steps.record import (
    ActivationKey,
    LabeledActivationStore,
    _require_reduced_store,
)


@dataclass
class ProbeResult:
    """Output of the Probe step.

    Attributes:
        accuracy_per_layer: {key: float} mean CV accuracy.
        cv_scores: {key: ndarray} per-fold accuracy scores.
        best_layer: Key with highest mean accuracy.
        classifiers: {key: fitted sklearn classifier} (only if refit=True).
        label_names: Human-readable label names (passed through from dataset).
    """

    accuracy_per_layer: dict[ActivationKey, float]
    cv_scores: dict[ActivationKey, ndarray]
    best_layer: ActivationKey
    classifiers: dict[ActivationKey, Any] = field(default_factory=dict)
    label_names: list[str] | None = None


class Probe(Step):
    """Train a linear probe per layer via cross-validation.

    Reads from results:
        results['record']: LabeledActivationStore
        results['dataset']: LabeledDataset (optional, for label_names)

    Writes to results:
        results['probe']: ProbeResult

    Args:
        classifier: sklearn classifier instance (default: LogisticRegression).
            Will be cloned per layer.
        cv: Number of cross-validation folds.
        refit: If True, fit a final classifier on all data per layer and store
               in ProbeResult.classifiers.
    """

    reads = [keys.RECORD]
    writes = [keys.PROBE]
    read_types = {keys.RECORD: LabeledActivationStore}
    write_types = {keys.PROBE: ProbeResult}

    def __init__(
        self,
        classifier: Any | None = None,
        cv: int = 5,
        refit: bool = False,
    ):
        self.cv = cv
        self.refit = refit
        self._classifier_template = classifier

    def __call__(self, results: Results) -> Results:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone

        store = results[keys.RECORD]
        if not isinstance(store, LabeledActivationStore):
            raise TypeError(
                f"Probe requires LabeledActivationStore in results['record'], "
                f"got {type(store).__name__}. "
                f"Did you use a LabeledDataset with the Load step?"
            )
        _require_reduced_store(store, "Probe")

        classifier = self._classifier_template or LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )

        labels = store.labels.numpy()

        # Validate that we have enough examples per class for CV
        class_counts = Counter(labels)
        min_count = min(class_counts.values())
        if min_count < self.cv:
            raise ValueError(
                f"Smallest class has {min_count} examples, but cv={self.cv} "
                f"requires at least {self.cv} per class. "
                f"Reduce cv or add more data."
            )

        accuracy_per_layer: dict[ActivationKey, float] = {}
        cv_scores: dict[ActivationKey, ndarray] = {}
        classifiers: dict[ActivationKey, Any] = {}

        for key in sorted(store.activations.keys()):
            X = store.activations[key].float().numpy()
            y = labels

            scores = cross_val_score(
                clone(classifier),
                X,
                y,
                cv=self.cv,
                scoring="accuracy",
            )
            accuracy_per_layer[key] = float(scores.mean())
            cv_scores[key] = scores

            logger.info(
                "Key %s: accuracy=%.4f (+/- %.4f)",
                key,
                scores.mean(),
                scores.std(),
            )

            if self.refit:
                clf: Any = clone(classifier)
                clf.fit(X, y)
                classifiers[key] = clf

        best_layer = max(accuracy_per_layer, key=lambda k: accuracy_per_layer[k])

        label_names = None
        dataset = results.get(keys.DATASET)
        if dataset is not None and hasattr(dataset, "label_names"):
            label_names = dataset.label_names

        results[keys.PROBE] = ProbeResult(
            accuracy_per_layer=accuracy_per_layer,
            cv_scores=cv_scores,
            best_layer=best_layer,
            classifiers=classifiers,
            label_names=label_names,
        )
        logger.info(
            "Best layer: %s (accuracy=%.4f)",
            best_layer,
            accuracy_per_layer[best_layer],
        )
        return results
