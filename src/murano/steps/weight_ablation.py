"""Weight-level ablation: projects out directions from model weights.

Implements the approach from Arditi et al. 2024: instead of hooking activations
during generation, directly modify weight matrices using an orthogonal projection
P = I - dd^T. This removes the direction from ALL computations (not just the
residual stream at specific hook points), yielding stronger ablation.

Read matrices (input from residual stream):  W_new = W @ P
Write matrices (output to residual stream): W_new = P @ W
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, eye, float32, isfinite  # pyright: ignore[reportPrivateImportUsage]
from tqdm import tqdm

from murano.artifacts import GenerationComparison, PromptBatch
from murano.logging import logger
from murano.results import Results
from murano.steps.base import Step
from murano.steps.intervene import InterveneResult
from murano.steps.train import SteeringResult

if TYPE_CHECKING:
    from murano.model import MuranoModel


class ProjectionOperator:
    """Orthogonal projection P = I - RR^T for ablating directions from weights.

    Supports single direction (vector) or multiple directions (matrix).
    Directions are normalized and orthonormalized automatically so the
    resulting projection is valid for both single- and multi-direction
    ablations. All projection math runs on CPU in float32 for numerical
    stability.

    Args:
        directions: Either [d_model] for single direction,
                    or [n_dirs, d_model] for multiple directions.
    """

    def __init__(self, directions: Tensor):
        if directions.dim() == 1:
            directions = directions.unsqueeze(0)  # [1, d_model]
        if directions.dim() != 2:
            raise ValueError(
                "directions must have shape [d_model] or [n_dirs, d_model], "
                f"got {tuple(directions.shape)}"
            )

        normalized_rows = []
        for idx, direction in enumerate(directions):
            norm = direction.norm()
            if not isfinite(norm).item() or norm.item() < 1e-10:
                logger.warning(
                    "Skipping non-finite or near-zero projection direction %d",
                    idx,
                )
                continue
            normalized_rows.append((direction / norm).float().cpu())

        if not normalized_rows:
            raise ValueError(
                "ProjectionOperator requires at least one non-zero direction"
            )

        basis = torch.stack(normalized_rows).T  # [d_model, n_dirs]
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        self.P = eye(basis.shape[0], dtype=float32) - basis @ basis.T
        self.n_dirs = basis.shape[1]
        self.d_model = basis.shape[0]

    def project_read(self, W_data: Tensor) -> Tensor:
        """Apply ``W @ P`` for read matrices (embed, q/k/v, gate_proj, up_proj)."""
        orig_device = W_data.device
        W = W_data.cpu().float()
        return (W @ self.P).to(W_data.dtype).to(orig_device)

    def project_write(self, W_data: Tensor) -> Tensor:
        """Apply ``P @ W`` for write matrices (o_proj, down_proj)."""
        orig_device = W_data.device
        W = W_data.cpu().float()
        return (self.P @ W).to(W_data.dtype).to(orig_device)


def ablate_model_weights(model: MuranoModel, proj_op: ProjectionOperator) -> int:
    """Apply directional projection to all relevant weight matrices in-place.

    Handles embedding, attention (q/k/v read, o_proj write), and
    dense MLP (gate/up read, down write).

    Args:
        model: MuranoModel to modify.
        proj_op: ProjectionOperator built from the direction(s) to ablate.

    Returns:
        Number of weight matrices modified.
    """
    # hf_model is typed as an nnsight Envoy union; at runtime it exposes
    # the plain transformer parameters we mutate below. Cast to Any so the
    # attribute / dtype checks don't fight with nnsight's proxy types.
    hf_model: Any = model._lm.model
    n_modified = 0

    with torch.no_grad():
        # Embedding (reads from residual stream → W @ P)
        hf_model.embed_tokens.weight.data = proj_op.project_read(
            hf_model.embed_tokens.weight.data
        )
        n_modified += 1

        for layer_idx in range(model.n_layers):
            layer = hf_model.layers[layer_idx]

            # Attention read matrices (q, k, v)
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = getattr(layer.self_attn, proj_name)
                proj.weight.data = proj_op.project_read(proj.weight.data)
                n_modified += 1

            # Attention write matrix (o_proj)
            layer.self_attn.o_proj.weight.data = proj_op.project_write(
                layer.self_attn.o_proj.weight.data
            )
            n_modified += 1

            # Dense MLP
            if hasattr(layer.mlp, "gate_proj"):
                layer.mlp.gate_proj.weight.data = proj_op.project_read(
                    layer.mlp.gate_proj.weight.data
                )
                n_modified += 1
            if hasattr(layer.mlp, "up_proj"):
                layer.mlp.up_proj.weight.data = proj_op.project_read(
                    layer.mlp.up_proj.weight.data
                )
                n_modified += 1
            if hasattr(layer.mlp, "down_proj"):
                layer.mlp.down_proj.weight.data = proj_op.project_write(
                    layer.mlp.down_proj.weight.data
                )
                n_modified += 1

    return n_modified


def save_weights(model: MuranoModel) -> dict[str, Tensor]:
    """Save a copy of all model weights for later restoration."""
    hf_model: Any = model._lm.model
    return {name: param.data.clone() for name, param in hf_model.named_parameters()}


def restore_weights(model: MuranoModel, saved: dict[str, Tensor]) -> None:
    """Restore model weights from a saved copy."""
    hf_model: Any = model._lm.model
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            param.data.copy_(saved[name])


class WeightAblationResult(GenerationComparison):
    """Output of WeightAblation step.

    Attributes:
        clean_generations: Model responses before weight ablation.
        modified_generations: Model responses after weight ablation.
        n_modified: Number of weight matrices modified.
    """

    def __init__(
        self,
        clean_generations: list[str],
        modified_generations: list[str],
        n_modified: int,
        prompts: list[str] | None = None,
        metadata: dict | None = None,
    ):
        payload = {} if metadata is None else dict(metadata)
        payload.setdefault("n_modified", n_modified)
        super().__init__(
            baseline_generations=clean_generations,
            modified_generations=modified_generations,
            prompts=prompts,
            baseline_label="clean",
            modified_label="ablated",
            metadata=payload,
        )
        self.n_modified = n_modified


class WeightAblation(Step):
    """Ablate a direction from model weights, then generate.

    Unlike activation-level ablation (Intervene step), this modifies the actual
    weight matrices using orthogonal projection P = I - dd^T. This removes the
    direction from ALL computations, not just the residual stream at hook points.

    Uses the best layer's direction from SteeringResult, applied globally to
    all weight matrices across all layers.

    Reads from results:
        results['prompts']: PromptBatch
        results['steering']: SteeringResult (uses best_layer direction)

    Writes to results:
        results['intervene']: InterveneResult
        results['weight_ablation']: WeightAblationResult

    Args:
        model: MuranoModel to generate with.
        save_dir: If provided, save the ablated model in HF format to this directory.
        gen_kwargs: Keyword arguments for generation.
    """

    reads = ["prompts", "steering"]
    writes = ["intervene", "weight_ablation"]
    write_types = {
        "intervene": InterveneResult,
        "weight_ablation": WeightAblationResult,
    }

    def expected_read_types(self, results=None, available_types=None):
        return {
            "prompts": PromptBatch,
            "steering": SteeringResult,
        }

    def __init__(
        self,
        model: MuranoModel,
        save_dir: str | None = None,
        gen_kwargs: dict | None = None,
    ):
        self.model = model
        self.save_dir = save_dir
        self.gen_kwargs = gen_kwargs or {"max_new_tokens": 256, "do_sample": False}

    def __call__(self, results: Results) -> Results:
        prompt_batch = results["prompts"]
        steering = results["steering"]
        prompts = prompt_batch.prompts

        # Use the best layer's direction for the projection
        direction = steering.direction_per_layer[steering.best_layer]
        proj_op = ProjectionOperator(direction)

        # Step 1: Generate clean outputs (original weights)
        clean_gens = []
        for prompt in tqdm(prompts, desc="WeightAblation (clean)"):
            clean_gens.append(self._generate(prompt))

        # Step 2: Save weights → ablate → (save model) → generate → restore
        saved = save_weights(self.model)
        try:
            n_modified = ablate_model_weights(self.model, proj_op)
            logger.info("Modified %d weight matrices", n_modified)

            if self.save_dir:
                from murano.io import save_ablated_model

                save_ablated_model(self.model, self.save_dir)

            modified_gens = []
            for prompt in tqdm(prompts, desc="WeightAblation (ablated)"):
                modified_gens.append(self._generate(prompt))
        finally:
            restore_weights(self.model, saved)

        # Write results
        result = WeightAblationResult(
            clean_generations=clean_gens,
            modified_generations=modified_gens,
            n_modified=n_modified,
            prompts=(
                list(prompt_batch.raw_prompts)
                if prompt_batch.raw_prompts is not None
                else list(prompt_batch.prompts)
            ),
            metadata={
                "prompt_source": prompt_batch.source,
                **prompt_batch.metadata,
            },
        )
        results["weight_ablation"] = result
        results["intervene"] = InterveneResult(
            clean_generations=clean_gens,
            modified_generations=modified_gens,
            prompts=result.prompts,
            metadata={
                "prompt_source": prompt_batch.source,
                **prompt_batch.metadata,
            },
        )
        return results

    def _generate(self, text: str) -> str:
        """Generate text with current model weights."""
        return self.model._generate_single(text, gen_kwargs=self.gen_kwargs)
