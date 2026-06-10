"""Load step: puts a dataset into the results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from murano import keys
from murano.artifacts import PromptBatch
from murano.dataset import LabeledDataset, MuranoDataset
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    pass


class Load(Step):
    """Load a dataset into the pipeline results.

    Writes to results:
        results['dataset']: MuranoDataset or LabeledDataset
        results['prompts']: PromptBatch derived from the dataset texts

    Args:
        dataset: Contrastive or labeled dataset to make available to
            downstream steps.
    """

    reads = []
    writes = [keys.DATASET, keys.PROMPTS]

    def __init__(self, dataset: MuranoDataset | LabeledDataset):
        self.dataset = dataset

    def expected_write_types(self, results=None, available_types=None):
        """Return ``{"dataset": <constructor dataset's type>, "prompts": PromptBatch}``."""
        return {
            keys.DATASET: type(self.dataset),
            keys.PROMPTS: PromptBatch,
        }

    def __call__(self, results: Results) -> Results:
        results[keys.DATASET] = self.dataset
        results[keys.PROMPTS] = self._prompt_batch()
        return results

    def _prompt_batch(self) -> PromptBatch:
        dataset = self.dataset
        metadata = {"dataset_type": type(dataset).__name__}

        if isinstance(dataset, LabeledDataset):
            return PromptBatch(
                prompts=list(dataset.texts),
                raw_prompts=(
                    list(dataset.raw_texts) if dataset.raw_texts is not None else None
                ),
                source="dataset.texts",
                metadata=metadata,
            )

        return PromptBatch(
            prompts=list(dataset.positive_texts),
            raw_prompts=(
                list(dataset.raw_positive) if dataset.raw_positive is not None else None
            ),
            source="dataset.positive_texts",
            metadata=metadata,
        )
