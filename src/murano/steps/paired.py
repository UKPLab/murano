"""LoadPaired step: put a clean/corrupt paired dataset into the results."""

from __future__ import annotations

from murano import keys
from murano.artifacts import PromptBatch
from murano.dataset import CleanCorruptDataset
from murano.results import Results
from murano.steps.base import Step


class LoadPaired(Step):
    """Load a clean/corrupt paired dataset into the pipeline results.

    Splits the matched pairs into two prompt batches so the clean and corrupt
    sides can each be run through the model: the clean prompts land under
    ``prompts`` (the input every step reads by default) and the corrupt prompts
    under ``corrupt_prompts``. The dataset itself is kept under ``dataset`` for
    provenance.

    Writes to results:
        results['dataset']: the CleanCorruptDataset.
        results['prompts']: PromptBatch of the clean prompts.
        results['corrupt_prompts']: PromptBatch of the corrupt prompts.

    Run the clean side through a default ``Logits`` step and the corrupt side
    through a second ``Logits(prompts_key=keys.CORRUPT_PROMPTS,
    logits_key=keys.CORRUPT_LOGITS, mask_key=keys.CORRUPT_MASK)`` so the two
    runs land under distinct keys; the metric steps then compare them.

    Args:
        dataset: The paired dataset to load.
    """

    reads = []
    writes = [keys.DATASET, keys.PROMPTS, keys.CORRUPT_PROMPTS]

    def __init__(self, dataset: CleanCorruptDataset):
        self.dataset = dataset

    def expected_write_types(self, results=None, available_types=None):
        """Return the dataset type plus a PromptBatch for each side."""
        return {
            keys.DATASET: type(self.dataset),
            keys.PROMPTS: PromptBatch,
            keys.CORRUPT_PROMPTS: PromptBatch,
        }

    def __call__(self, results: Results) -> Results:
        dataset = self.dataset
        metadata = {"dataset_type": type(dataset).__name__}
        results[keys.DATASET] = dataset
        results[keys.PROMPTS] = PromptBatch(
            prompts=list(dataset.clean),
            raw_prompts=(
                list(dataset.raw_clean) if dataset.raw_clean is not None else None
            ),
            source="dataset.clean",
            metadata=metadata,
        )
        results[keys.CORRUPT_PROMPTS] = PromptBatch(
            prompts=list(dataset.corrupt),
            raw_prompts=(
                list(dataset.raw_corrupt) if dataset.raw_corrupt is not None else None
            ),
            source="dataset.corrupt",
            metadata=metadata,
        )
        return results
