"""Visualize layer-wise next-token predictions with the logit lens.

Demonstrates:
  1. Loading prompts via LoadPrompts.
  2. Computing per-layer next-token distributions with the LogitLens step.
  3. Rendering the result as a heatmap with plot_logit_lens.

Cross-architecture: works on any model nnterp can standardize (Llama, GPT-2,
Mistral, Qwen, OPT, etc.). The example below uses Llama-3.2-1B, but you can
swap the model_id for any HuggingFace causal-LM.
"""

from __future__ import annotations

from murano import MuranoModel, Pipeline
from murano.plotting import plot_logit_lens
from murano.steps.logit_lens import LogitLens
from murano.steps.prompts import LoadPrompts


def main():
    model = MuranoModel("meta-llama/Llama-3.2-1B-Instruct")

    pipe = Pipeline(
        [
            LoadPrompts(["The Eiffel Tower is in the city of"]),
            LogitLens(model),
        ]
    )
    results = pipe.run()
    fig = plot_logit_lens(results["logit_lens"], title="Logit Lens: Llama-3.2-1B")
    fig.write_html("/tmp/logit_lens.html")
    print("Saved /tmp/logit_lens.html")


if __name__ == "__main__":
    main()
