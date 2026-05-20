"""Inspect SAE features via top-activating contexts and the logit lens.

Pipeline:
    1. Load prompts
    2. Load SAE weights from HuggingFace
    3. Encode prompts through the SAE to get activation records
    4. For each feature, find the top-K activating contexts (token + leftward text) across the prompts
    5. Run the logit lens on the same prompts to get per-layer next-token predictions
    6. Save results to disk and print top contexts.

Set ``SAE_REPO`` and ``LAYER`` below to match a real SAE on the Hub for
the chosen base model.
"""

from __future__ import annotations

import os

from murano import MuranoModel, Pipeline
from murano.steps import LogitLens, Save, SAEEncode, SAETopKContexts
from murano.steps.prompts import LoadPrompts


MODEL_ID = os.environ.get("MURANO_EXAMPLE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
SAE_REPO = "<set me to a Hub repo with SAE weights for the chosen layer>"
LAYER = 8
TOP_K = 10
N_FEATURES_TO_TRACK = 5
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
            LoadPrompts(PROMPTS),
            SAEEncode(model, sae_repo=SAE_REPO, layer=LAYER),
            SAETopKContexts(model, k=TOP_K, feat_ids=list(range(N_FEATURES_TO_TRACK))),
            LogitLens(model, layers=[LAYER]),
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

    lens = results["logit_lens"]
    print(f"\n── Logit-lens next-token predictions at layer {LAYER} ──")
    for i, prompt in enumerate(PROMPTS):
        valid_len = int(lens.attention_mask[i].sum().item())
        pred = lens.predicted_words[0][i][valid_len - 1]
        print(f"  {prompt!r:50}  →  {pred!r}")

    print(f"\nArtifacts saved to {results['output_dir']}")


if __name__ == "__main__":
    main()
