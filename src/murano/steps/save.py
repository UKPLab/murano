"""Save step: persists pipeline results to disk."""

from __future__ import annotations

from pathlib import Path

from murano import keys
from murano.io import save_results
from murano.results import Results
from murano.steps.base import Step


class Save(Step):
    """Save all available results to organized subdirectories.

    Output structure:
        output_dir/
        ├── direction/steering.pt
        ├── evaluation/generations.json
        └── metadata.json

    Reads from results:
        Whatever keys are present (steering, intervene).

    Writes to results:
        results['output_dir']: Path to the output directory.

    Args:
        output_dir: Base directory for outputs. Defaults to ``murano_outputs``
            in the current working directory.
        model_id: Model identifier for metadata.
        run_name: Optional subdirectory inside ``output_dir`` for this run. By
            default results are written directly into ``output_dir`` and a
            re-run overwrites them; set ``run_name`` to keep runs separate.
    """

    reads = []
    writes = [keys.OUTPUT_DIR]
    write_types = {keys.OUTPUT_DIR: Path}

    def __init__(
        self,
        output_dir: str = keys.DEFAULT_OUTPUT_DIR,
        model_id: str = "",
        run_name: str | None = None,
    ):
        self.output_dir = output_dir
        self.model_id = model_id
        self.run_name = run_name

    def __call__(self, results: Results) -> Results:
        out_dir = save_results(
            results,
            output_dir=self.output_dir,
            model_id=self.model_id,
            run_name=self.run_name,
        )
        results[keys.OUTPUT_DIR] = out_dir
        return results
