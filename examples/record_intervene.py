"""
Record, steer, and intervene using the Pipeline API.

Demonstrates:
  1. Loading a contrastive sentiment dataset (from sentiment_data.py).
  2. Recording activations and computing a steering vector.
  3. Applying the steering vector during generation and comparing outputs.

Usage:
    python examples/record_intervene.py
"""

from __future__ import annotations

from murano import MuranoModel, Pipeline
from murano.dataset import MuranoDataset
from murano.steps.intervene import InterveneResult, Intervene, steer_direction
from murano.steps.load import Load
from murano.steps.record import Record
from murano.steps.train import SteeringResult, SteeringVector
from sentiment_data import NEGATIVE_SAMPLES, POSITIVE_SAMPLES


def main():
    model = MuranoModel("gpt2")

    # --- Build contrastive dataset ---
    train_dataset = MuranoDataset(
        positive_texts=POSITIVE_SAMPLES,
        negative_texts=NEGATIVE_SAMPLES,
    )

    # --- Pipeline 1: Record + SteeringVector ---
    train_pipe = Pipeline(
        [
            Load(train_dataset),
            Record(model, layers=[8], position="last", batch_size=8),
            SteeringVector(normalize=True),
        ]
    )
    train_results = train_pipe.run()
    steering: SteeringResult = train_results["steering"]

    print(f"Best steering layer: {steering.best_layer}")
    print(f"Separation scores: {steering.separation_scores}")

    # --- Pipeline 2: Intervene (steer during generation) ---
    eval_texts = [
        "I think that the food was",
        "The food was",
        "My friend is a",
    ]
    eval_dataset = MuranoDataset(
        positive_texts=eval_texts,
        negative_texts=[],
    )

    steer_fn = steer_direction(
        steering.direction_per_layer,
        alpha=6.0,
    )

    eval_pipe = Pipeline(
        [
            Load(eval_dataset),
            Intervene(
                model, fn=steer_fn, layers=[8], gen_kwargs={"max_new_tokens": 15}
            ),
        ]
    )
    eval_results = eval_pipe.run()
    intervene_result: InterveneResult = eval_results["intervene"]

    # --- Print comparison ---
    print("\n=== Generation Comparison ===")
    for i, prompt in enumerate(eval_texts):
        print(f'\nPrompt: "{prompt}"')
        print(f'  Clean:    "{intervene_result.clean_generations[i]}"')
        print(f'  Steered:  "{intervene_result.modified_generations[i]}"')


if __name__ == "__main__":
    main()
