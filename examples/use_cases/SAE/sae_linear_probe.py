"""
Example demonstrating SAE (Sparse Autoencoder) activation extraction and linear probe training.

This example shows how to:
1. Load a pre-trained SAE from sae_lens (optional)
2. Extract hidden state activations from residual stream using nnsight
3. Train linear probes on raw hidden state activations for classification
4. Optionally use SAE activations directly without training
5. Evaluate and visualize probe performance
6. Save trained probes for reuse
"""

import torch
from datasets import Dataset

from murano import SAEMuranoModel, LinearProbe


# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create synthetic dataset (similar to Untitled-1 example)
    data = [
        {
            "text": f"Mary is born on the {i+1} of August.",
            "label": i % 2,  # Binary classification
            "metadata": {"id": i, "source": "synthetic"},
        }
        for i in range(16)
    ]

    dataset = Dataset.from_list(data)

    # Example 1: Extract raw hidden states and train linear probe
    print("\n=== Example 1: Training linear probe on raw hidden states ===")
    model = SAEMuranoModel(
        model_name="gpt2",
        hook_points=["resid_pre", "resid_post"],
        layers=[2, 4, 6],
    )

    # Extract raw activations
    artifact = model.run_task(dataset, batch_size=4, use_sae=False)
    print(f"Extracted activations shape: {artifact['activations'].shape}")

    # Train linear probe on raw activations
    probe = LinearProbe(
        input_dim=artifact["activations"].shape[-1],
        num_classes=2,
        loss_fn="cross_entropy",
    )

    training_history = probe.train_probe(
        activations=artifact["activations"],
        labels=artifact["labels"],
        learning_rate=1e-3,
        epochs=10,
        validation_split=0.2,
    )

    # Evaluate
    metrics = probe.evaluate(artifact["activations"], artifact["labels"])
    probe.visualize(training_history, metrics)

    # Save complete workflow output as pickle
    output_path = probe.save_workflow_output(
        activations=artifact["activations"],
        labels=artifact["labels"],
        training_history=training_history,
        metrics=metrics,
        artifact=artifact,
        save_dir="outputs",
        prefix="sae_linear_probe",
    )
    print(f"\nComplete workflow output saved to: {output_path}")

    # Example 2: Use SAE activations directly (optional, requires SAE)
    print("\n=== Example 2: Using SAE activations directly (optional) ===")
    print("To use SAE activations directly without training:")
    print("1. Load SAE: model = SAEMuranoModel.from_pretrained(")
    print("   model_name='gpt2',")
    print("   hook_points=['resid_pre'],")
    print("   layers=[2, 4, 6],")
    print("   sae_release='gpt2-small-res-jb',")
    print("   sae_id='blocks.7.hook_resid_pre')")
    print("2. Extract SAE features: artifact = model.run_task(dataset, use_sae=True)")
    print("3. Use SAE features directly for downstream tasks")
    print("\nSee SAE_LINEAR_PROBE_PLAN.md for API documentation.")
