"""Prompt-loading helpers for generation-style experiments."""

from __future__ import annotations

from murano.artifacts import PromptBatch
from murano.results import Results
from murano.steps.base import Step


class LoadPrompts(Step):
    """Loads prompts directly into pipeline results."""

    reads = []
    writes = ["prompts"]

    def __init__(
        self,
        prompts: PromptBatch | list[str],
        raw_prompts: list[str] | None = None,
        source: str = "manual",
    ):
        if isinstance(prompts, PromptBatch):
            self.prompts = prompts
        else:
            self.prompts = PromptBatch(
                prompts=list(prompts),
                raw_prompts=raw_prompts,
                source=source,
            )

    def expected_write_types(self, results=None, available_types=None):
        return {"prompts": PromptBatch}

    def __call__(self, results: Results) -> Results:
        results["prompts"] = self.prompts
        return results
