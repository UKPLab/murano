"""Inspect SAE features via top-activating contexts.

Pipeline:
    1. Load prompts
    2. Encode prompts through an SAE loaded from HuggingFace via sae-lens
    3. For each feature, find the top-K (text, token) positions where it activated
    4. Save results to disk and print the table

Defaults to Gemma 2 2B + gemma-scope layer-8 residual SAEs. The target layer
and hook point are read from the SAE's own config.
"""

from __future__ import annotations

from murano import MuranoModel, Pipeline
from murano.steps import SAEEncode, SAETopActivations, Save
from murano.steps.prompts import LoadPrompts


MODEL_ID = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_8/width_16k/canonical"
K = 10
OUTPUT_DIR = "/tmp/murano_sae_example"


PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "Photosynthesis converts sunlight into",
    "She baked a cherry pie for",
    "The Roman Empire fell in the year",
    "Quantum computers store information in",
    "Mozart composed his first symphony at the age of",
    "Mount Everest is the tallest mountain in",
    "DNA carries the genetic instructions of",
]


def main() -> None:
    model = MuranoModel(MODEL_ID)

    results = Pipeline(
        [
            # -> 'prompts': the input texts wrapped as a PromptBatch
            LoadPrompts(PROMPTS),
            # -> 'sae_record': per-token raw SAE feature activations
            SAEEncode(model, release=SAE_RELEASE, sae_id=SAE_ID),
            # -> 'feature_examples': top-K activating contexts for the first K features
            SAETopActivations(model, k=K, feat_ids=list(range(K))),
            # -> 'output_dir': persists everything above under OUTPUT_DIR
            Save(output_dir=OUTPUT_DIR, model_id=MODEL_ID),
        ]
    ).run()

    examples = results["feature_examples"]
    for feat_id in examples.feat_ids:
        print(f"\n── feature {feat_id} ──")
        for context, token, act_val in zip(
            examples.contexts[feat_id],
            examples.tokens[feat_id],
            examples.act_vals[feat_id],
        ):
            print(f"  {act_val:8.4f}  [{token!r:>14}]  {context}")
    print(f"\nArtifacts saved to {results['output_dir']}")


if __name__ == "__main__":
    main()
