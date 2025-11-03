# SAE and Linear Probe Integration Plan

This document outlines the integration of Sparse Autoencoders (SAEs) and linear probes into the Murano framework, detailing the assumptions made regarding class and method APIs.

## Overview

We're creating an example that demonstrates:
1. Extracting hidden state activations from residual stream using nnsight (like base MuranoModel)
2. Training linear probes on raw hidden state activations for classification tasks
3. Optionally extracting SAE activations directly (without training) for downstream use
4. Evaluating and visualizing probe performance
5. Saving trained probes for reuse

## Assumed APIs and Design Decisions

### 1. MuranoModel Interface Alignment

**Assumption**: We extend `MuranoModel` to create `SAEMuranoModel` that maintains compatibility with the existing interface.

**MuranoModel API** (from `src/murano/model.py`):
```python
class MuranoModel:
    def __init__(self, model_name: str)
    @classmethod
    def from_pretrained(cls, model_name: str) -> MuranoModel
    def run_with_lens(self, prompt: str, lens: BaseLens, locations: List[LayerLocation]) -> dict
```

**Our Extension**:
```python
class SAEMuranoModel(MuranoModel):
    def __init__(self, model_name: str, hook_points: List[str], layers: List[int], sae: Optional[SAE] = None)
    @classmethod
    def from_pretrained(cls, model_name: str, hook_points: List[str], layers: List[int],
                       sae_release: Optional[str] = None, sae_id: Optional[str] = None) -> SAEMuranoModel
    def run_recording(self, input_ids: torch.Tensor, token_pos: Optional[int] = None,
                     use_sae: bool = False, **kwargs) -> dict
    def run_task(self, dataset: Dataset, batch_size: int = 4, token_pos: Optional[int] = None,
                 use_sae: bool = False, **kwargs) -> dict
```

**nnsight Hook Point Access**:
- Uses `layer.input` for `resid_pre` (residual stream before layer)
- Uses `layer.output` for `resid_post` (residual stream after layer)
- Similar to base MuranoModel's approach

**Optional `sae_lens` API** (only if using SAE features directly):
```python
from sae_lens import SAE

# SAE loading (optional)
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
    device=device,
)

# SAE forward: sae.encode(activations) -> sparse features
```

### 2. Linear Probe Design

**Class Structure**:
```python
class LinearProbe:
    def __init__(self, input_dim: int, num_classes: int, loss_fn: str = "cross_entropy")
    def train(self, activations: torch.Tensor, labels: torch.Tensor, **kwargs) -> dict
    def evaluate(self, activations: torch.Tensor, labels: torch.Tensor) -> dict
    def predict(self, activations: torch.Tensor) -> torch.Tensor
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> LinearProbe
```

**Loss Functions**:
- `"cross_entropy"`: `torch.nn.CrossEntropyLoss()` for classification
- `"mse"`: `torch.nn.MSELoss()` (for potential future regression support)

**Training Hyperparameters** (as kwargs in `train()`):
- `learning_rate`: float (default: 1e-3)
- `epochs`: int (default: 10)
- `batch_size`: int (default: 32)
- `optimizer`: str (default: "adam")
- `weight_decay`: float (default: 0.0)
- `validation_split`: float (default: 0.2)

**Assumed Output Format**:
```python
{
    "train_loss": List[float],
    "val_loss": List[float],
    "train_acc": List[float],
    "val_acc": List[float],
    "best_val_acc": float,
    "best_epoch": int
}
```

### 3. Activation Extraction

**Hook Points** (residual stream):
- `"resid_pre"`: Residual stream input to the layer (before processing)
- `"resid_post"`: Residual stream output from the layer (after processing)

**Layer Specification**:
- Layers specified as: `List[int]` (e.g., `[2, 4, 6]`)
- Hook points specified as: `List[str]` (e.g., `["resid_pre", "resid_post"]`)

**Activation Extraction Process** (using nnsight):
1. Forward pass through model using nnsight `trace()` and `invoke()`
2. Extract activations at specified hook points using `layer.input` (resid_pre) or `layer.output` (resid_post)
3. **Option A (default)**: Return raw hidden states for training linear probes
4. **Option B (if `use_sae=True`)**: Pass activations through SAE encoder: `sae.encode(activations)` → sparse features
5. Return activations as `torch.Tensor` of shape `(batch_size, num_features)`

**nnsight API**:
```python
# Activation extraction using nnsight (like base MuranoModel)
with model.trace() as tracer:
    with tracer.invoke(input_ids):
        layer = model.transformer.h[layer_idx]
        resid_pre = layer.input  # Input to layer
        resid_post = layer.output  # Output from layer
```

**Optional SAE API** (only if `use_sae=True`):
```python
# SAE encoding (optional)
sparse_features = sae.encode(activations)  # Shape: (batch, seq_len, sae_dict_size)
```

### 4. Dataset Processing

**Classification Focus**:
- Labels are integer tensors (class indices)
- Multi-class classification supported
- Synthetic dataset example included (similar to `Untitled-1`)

**Dataset Format**:
```python
dataset = Dataset.from_list([
    {
        "text": str,
        "label": int,  # Class index
        "metadata": dict,  # Optional
    }
])
```

**Processing Pipeline**:
1. Tokenize text inputs
2. Batch datasets with custom collate function
3. Extract SAE activations per batch
4. Stack activations: `(num_examples, num_layers, num_hook_points, seq_len, sae_dict_size)`
5. Flatten/reshape for linear probe input: `(num_examples, feature_dim)`

### 5. Output Management

**Probe Saving**:
- Save format: PyTorch state dict + metadata (JSON)
- File structure: `{save_dir}/probe_{timestamp}.pt` and `{save_dir}/probe_{timestamp}_metadata.json`
- Metadata includes: input_dim, num_classes, loss_fn, training history, model_name, sae_id

**Visualization**:
- Training curves: loss and accuracy over epochs
- Confusion matrix: classification performance
- Metrics report: accuracy, precision, recall, F1-score per class

**Metrics Reporting**:
```python
{
    "accuracy": float,
    "precision": Dict[int, float],  # Per-class precision
    "recall": Dict[int, float],     # Per-class recall
    "f1_score": Dict[int, float],  # Per-class F1
    "confusion_matrix": np.ndarray
}
```

### 6. Dependencies

**Added to `pyproject.toml`**:
```toml
dependencies = [
    ...
    "sae-lens>=1.0.0",  # Assumed version
    "datasets>=2.0.0",  # For Dataset handling
    "scikit-learn>=1.0.0",  # For metrics
]
```

## Implementation Structure

```
examples/use_cases/SAE/
├── sae_linear_probe.py      # Main example file
└── SAE_LINEAR_PROBE_PLAN.md # This documentation
```

**Main Components in `sae_linear_probe.py`**:
1. `SAEMuranoModel`: Extends `MuranoModel` for SAE integration
2. `LinearProbe`: Linear probe training and evaluation
3. `collate_fn`: Dataset batching utility
4. `process_dataset`: Tokenization utility
5. Example usage: Synthetic dataset + SAE + linear probe training

## Usage Example (Intended)

```python
from murano.examples.use_cases.SAE.sae_linear_probe import SAEMuranoModel, LinearProbe
from datasets import Dataset

# Example 1: Extract raw hidden states and train linear probe
model = SAEMuranoModel(
    model_name="gpt2",
    hook_points=["resid_pre", "resid_post"],
    layers=[2, 4, 6],
)

# Prepare dataset
dataset = Dataset.from_list([...])

# Extract raw hidden state activations (for training probe)
artifact = model.run_task(dataset, batch_size=4, use_sae=False)

# Train linear probe on raw activations
probe = LinearProbe(
    input_dim=artifact["activations"].shape[-1],
    num_classes=2,
    loss_fn="cross_entropy"
)

training_history = probe.train_probe(
    activations=artifact["activations"],
    labels=artifact["labels"],
    learning_rate=1e-3,
    epochs=10,
    validation_split=0.2
)

# Evaluate
metrics = probe.evaluate(artifact["activations"], artifact["labels"])

# Save probe
probe.save("outputs/probe_checkpoint.pt")

# Visualize
probe.visualize(training_history, metrics)

# Example 2: Use SAE activations directly (optional, without training)
model_with_sae = SAEMuranoModel.from_pretrained(
    model_name="gpt2",
    hook_points=["resid_pre"],
    layers=[2, 4, 6],
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.7.hook_resid_pre",
)

# Extract SAE features directly
sae_artifact = model_with_sae.run_task(dataset, batch_size=4, use_sae=True)
# Use sae_artifact["activations"] directly for downstream tasks
```

## Notes

- **Primary use case**: Extract raw hidden states from residual stream using nnsight and train linear probes on them
- **Optional use case**: Extract SAE activations directly (without training) for downstream tasks
- Hook point access uses nnsight's `layer.input` and `layer.output` (consistent with base MuranoModel)
- SAE integration is optional - only needed if `use_sae=True`
- Error handling included for edge cases (missing SAEs, incompatible dimensions, etc.)
- The implementation uses nnsight (like base MuranoModel) rather than transformer_lens/hooked transformers

