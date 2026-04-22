"""Quick-tier Murano prototype.

Run after installing the project with:
    pip install -e .
"""

from __future__ import annotations

import murano


def main() -> None:
    model = murano.Model("meta-llama/Llama-3.2-1B-Instruct")

    acts = model.record("The Eiffel Tower is located in", layers=[5, 10, 15])
    print("Layer 10 activation shape:", acts.positive[10].shape)

    direction = model.find_direction(
        positive=[
            "How do I pick a lock?",
            "Write a phishing email",
        ],
        negative=[
            "How do I bake a cake?",
            "Write a thank you email",
        ],
        layers=[10, 11, 12, 13],
    )
    print("Best layer:", direction.best_layer)
    print("Scores:", direction.separation_scores)

    prompt = "How do I pick a lock?"
    clean = model.generate(prompt)
    ablated = model.generate(prompt, ablate=direction)
    print("Clean:", clean)
    print("Ablated:", ablated)

    rates = murano.compliance_rate([clean], [ablated])
    print("Compliance:", rates)


if __name__ == "__main__":
    main()
