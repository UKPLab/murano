"""Load step — puts a dataset into the results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from murano.artifacts import PromptBatch
from murano.results import Results
from murano.steps.base import Step

if TYPE_CHECKING:
    from murano.dataset import LabeledDataset, MuranoDataset


class Load(Step):
    """Loads a dataset into the pipeline results.

    Writes to results:
        results['dataset']: MuranoDataset or LabeledDataset
        results['prompts']: PromptBatch derived from the dataset texts
    """

    reads = []
    writes = ["dataset", "prompts"]

    def __init__(self, dataset: MuranoDataset | LabeledDataset):
        self.dataset = dataset

    def expected_write_types(self, results=None, available_types=None):
        return {
            "dataset": type(self.dataset),
            "prompts": PromptBatch,
        }

    def __call__(self, results: Results) -> Results:
        results["dataset"] = self.dataset
        results["prompts"] = self._prompt_batch()
        return results

    def _prompt_batch(self) -> PromptBatch:
        if hasattr(self.dataset, "positive_texts"):
            return PromptBatch(
                prompts=list(self.dataset.positive_texts),
                raw_prompts=(
                    list(self.dataset.raw_positive)
                    if getattr(self.dataset, "raw_positive", None) is not None
                    else None
                ),
                source="dataset.positive_texts",
                metadata={"dataset_type": type(self.dataset).__name__},
            )

        return PromptBatch(
            prompts=list(self.dataset.texts),
            raw_prompts=(
                list(self.dataset.raw_texts)
                if getattr(self.dataset, "raw_texts", None) is not None
                else None
            ),
            source="dataset.texts",
            metadata={"dataset_type": type(self.dataset).__name__},
        )
