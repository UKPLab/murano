"""Native loader for dictionary-learning ReLU SAE checkpoints used by FEGA."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from murano.steps.sae import SAEModel


class DictionarySAEModel(SAEModel):
    """Load a dictionary-learning ReLU SAE without a SAEBench dependency.

    Args:
        repo_id: Hugging Face repository containing the checkpoint.
        location: Repository directory containing ``config.json`` and ``ae.pt``.
        model_name: Expected language-model identifier from the training config.
        device: Device used for encode and decode.
        dtype: Runtime tensor dtype; defaults to float32.
        local_dir: Optional already-downloaded checkpoint directory. It may be
            either ``location`` itself or a repository root containing it.
    """

    def __init__(
        self,
        repo_id: str,
        location: str,
        model_name: str,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        local_dir: str | Path | None = None,
    ) -> None:
        """Store checkpoint identity and defer all file access until first use."""
        # Reuse SAEModel's public identity while replacing only its loading path.
        super().__init__(repo_id, location.strip("/"), device=device)
        self.model_name = model_name
        self.dtype = dtype
        self.local_dir = None if local_dir is None else Path(local_dir)

    def _ensure_loaded(self) -> None:
        """Resolve, validate, normalize, and materialize the source ReLU SAE."""
        # Load only once so both FEGA intervention phases can share the same weights.
        if self._sae is not None:
            return
        config_path = self._resolve("config.json")
        weights_path = self._resolve("ae.pt")
        config = json.loads(config_path.read_text())
        trainer = config.get("trainer", {})
        trainer_class = trainer.get("trainer_class")
        if trainer_class not in {
            "StandardTrainer",
            "StandardTrainerAprilUpdate",
            "PAnnealTrainer",
        }:
            raise ValueError(
                f"DictionarySAEModel supports ReLU checkpoints, got {trainer_class!r}"
            )
        trained_model = str(trainer.get("lm_name", ""))
        if (
            self.model_name not in trained_model
            and trained_model not in self.model_name
        ):
            raise ValueError(
                f"SAE was trained for {trained_model!r}, not {self.model_name!r}"
            )
        layer = int(trainer["layer"])
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        required = {"bias", "encoder.weight", "encoder.bias", "decoder.weight"}
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("dictionary SAE checkpoint has unsupported weights")

        # Match SAEBench's load order: cast first, then normalize the cast weights.
        w_enc = state["encoder.weight"].T.to(device=self.device, dtype=self.dtype)
        w_dec = state["decoder.weight"].T.to(device=self.device, dtype=self.dtype)
        b_enc = state["encoder.bias"].to(device=self.device, dtype=self.dtype)
        b_dec = state["bias"].to(device=self.device, dtype=self.dtype)
        norms = torch.linalg.vector_norm(w_dec, dim=1)
        if torch.any(~torch.isfinite(norms)) or torch.any(norms == 0):
            raise ValueError("dictionary SAE decoder contains invalid feature norms")
        tolerance = 1.0e-2 if self.dtype in {torch.bfloat16, torch.float16} else 1.0e-5
        if not torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            # Normalize in float32 after runtime quantization, then restore runtime dtype.
            w_enc = w_enc.float()
            w_dec = w_dec.float()
            b_enc = b_enc.float()
            b_dec = b_dec.float()
            norms = torch.linalg.vector_norm(w_dec, dim=1)
            w_dec = w_dec / norms[:, None]
            w_enc = w_enc * norms
            b_enc = b_enc * norms
        self._sae = _DictionaryReLUSAE(
            w_enc.to(dtype=self.dtype),
            w_dec.to(dtype=self.dtype),
            b_enc.to(dtype=self.dtype),
            b_dec.to(dtype=self.dtype),
            layer,
        )

    def _resolve(self, filename: str) -> Path:
        """Resolve one checkpoint file locally or through huggingface_hub."""
        # Prefer explicit local files so cluster runs need no repeated download.
        if self.local_dir is not None:
            candidates = (
                self.local_dir / filename,
                self.local_dir / self.sae_id / filename,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=self.release,
                filename=f"{self.sae_id}/{filename}",
                local_dir=None if self.local_dir is None else str(self.local_dir),
            )
        )


class _DictionaryReLUSAE:
    """Minimal in-memory interface shared with :class:`SAEModel`."""

    def __init__(
        self,
        w_enc: Tensor,
        w_dec: Tensor,
        b_enc: Tensor,
        b_dec: Tensor,
        layer: int,
    ) -> None:
        """Store normalized weights and SAEModel-compatible metadata."""
        # Expose the same fields SAEModel reads from a SAE Lens object.
        self.W_enc = w_enc
        self.W_dec = w_dec
        self.b_enc = b_enc
        self.b_dec = b_dec
        metadata = SimpleNamespace(
            hook_name=f"blocks.{layer}.hook_resid_post", hook_layer=layer
        )
        self.cfg = SimpleNamespace(d_sae=w_dec.shape[0], metadata=metadata)

    def encode(self, residual: Tensor) -> Tensor:
        """Encode residual rows with the source centered ReLU transform."""
        # Center by the decoder bias before applying the normalized encoder.
        return torch.relu((residual - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, activations: Tensor) -> Tensor:
        """Decode feature activations with the normalized dictionary."""
        # Restore the decoder bias after the feature-weighted reconstruction.
        return activations @ self.W_dec + self.b_dec
