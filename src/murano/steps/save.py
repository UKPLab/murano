"""Save step: persists pipeline results to disk."""

from __future__ import annotations

from pathlib import Path

from murano.io import save_results
from murano.results import Results
from murano.steps.base import Step


class Save(Step):
    """Save all available results to organized subdirectories.

    Output structure:
        output_dir/
        ├── direction/steering.pt
        ├── evaluation/{generations.json, eval.json}
        └── metadata.json

    Reads from results:
        Whatever keys are present (steering, intervene, eval).

    Writes to results:
        results['output_dir']: Path to the output directory.

    Args:
        output_dir: Base directory for outputs.
        model_id: Model identifier for metadata.
    """

    reads = []
    writes = ["output_dir"]
    write_types = {"output_dir": Path}

    def __init__(self, output_dir: str = "murano_outputs", model_id: str = ""):
        self.output_dir = output_dir
        self.model_id = model_id

    def __call__(self, results: Results) -> Results:
        out_dir = save_results(
            results,
            output_dir=self.output_dir,
            model_id=self.model_id,
        )
        results["output_dir"] = out_dir
        return results
