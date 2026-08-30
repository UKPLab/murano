"""Tests for the ModelBackend interface.

Behavioral tests exercise the interface members on a real (tiny) model; the
isinstance check is a cheap structural smoke test. A scoped lint test then
asserts that no pipeline step (or io.py) reaches into the model internals the
interface replaces.
"""

from __future__ import annotations

import tokenize
from pathlib import Path

import pytest
import torch

from murano.backend import ModelBackend


def _build_tiny_model(path):
    """Write a tiny randomly-initialized Llama model + tokenizer to ``path``."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
        "good": 6,
        "bad": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=64,
    )
    fast_tokenizer.save_pretrained(path)

    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_backend_model")
    _build_tiny_model(path)
    from murano.model import MuranoModel

    return MuranoModel(str(path), device_map="cpu", dtype=torch.float32)


# ── Behavioral tests ──────────────────────────────────────────────────


class TestModelBackendBehavior:
    def test_metadata(self, tiny_model):
        assert tiny_model.n_heads == 4
        assert tiny_model.head_dim == tiny_model.d_model // tiny_model.n_heads

    def test_resolve_module_delegates(self, tiny_model):
        mlp = tiny_model.resolve_module(0, "mlp")
        assert hasattr(mlp, "gate_proj")

    def test_attn_out_proj_resolves_attention(self, tiny_model):
        proj = tiny_model.attn_out_proj(0, "self_attn")
        assert proj is not None

    def test_attn_out_proj_raises_on_non_attention(self, tiny_model):
        with pytest.raises(NotImplementedError):
            tiny_model.attn_out_proj(0, "mlp")

    def test_raw_accessors_return_hookable_modules(self, tiny_model):
        # Steps use these to register native torch hooks and read weights, so
        # they must return real nn.Modules, not nnsight proxies.
        from torch.nn import Module
        from torch.utils.hooks import RemovableHandle

        layer = tiny_model.raw_layer(0)
        mlp = tiny_model.raw_module(0, "mlp")
        out_proj = tiny_model.raw_attn_out_proj(0, "self_attn")
        assert isinstance(layer, Module)
        assert isinstance(mlp, Module) and hasattr(mlp, "gate_proj")
        assert isinstance(out_proj, Module) and hasattr(out_proj, "weight")
        handle = out_proj.register_forward_hook(lambda *args: None)
        assert isinstance(handle, RemovableHandle)
        handle.remove()

    def test_trace_yields_savable_output(self, tiny_model):
        tokens = tiny_model.tokenizer(
            ["hello world"], return_tensors="pt", return_token_type_ids=False
        )
        with tiny_model.trace(tokens):
            saved = tiny_model.layer(0).output.save()
        value = saved.value if hasattr(saved, "value") else saved
        if isinstance(value, tuple):
            value = value[0]
        assert value.shape[0] == 1
        assert value.shape[-1] == tiny_model.d_model

    def test_project_on_vocab_shape(self, tiny_model):
        hidden = torch.randn(1, 3, tiny_model.d_model)
        logits = tiny_model.project_on_vocab(hidden)
        assert logits.shape[:-1] == hidden.shape[:-1]
        assert logits.shape[-1] == tiny_model._lm.config.vocab_size

    def test_forward_logits_shape_dtype_device(self, tiny_model):
        tokens = tiny_model.tokenizer(
            ["hello world", "good"],
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        logits = tiny_model.forward_logits(tokens)
        assert logits.shape[0] == 2
        assert logits.shape[-1] == tiny_model._lm.config.vocab_size
        assert logits.dtype == torch.float32
        assert logits.device.type == "cpu"

    def test_forward_logits_fn_intervenes(self, tiny_model):
        from murano.nodes import Node

        tokens = tiny_model.tokenizer(
            ["hello world"],
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        base = tiny_model.forward_logits(tokens)

        def zero_layer0(activation, key: Node):
            return activation * 0 if key.layer == 0 else activation

        out = tiny_model.forward_logits(
            tokens, fn=zero_layer0, layers=[0], modules="residual"
        )
        assert out.shape == base.shape
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"
        assert not torch.allclose(out, base)

        # fn=None preserves the plain forward pass exactly.
        again = tiny_model.forward_logits(tokens, fn=None)
        assert torch.allclose(again, base)

    def test_forward_logits_per_head_intervenes(self, tiny_model):
        from murano.nodes import SELF_ATTN, Node

        tokens = tiny_model.tokenizer(
            ["hello world"],
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        base = tiny_model.forward_logits(tokens)

        def zero_head0(activation, key: Node):
            # activation is [B, S, n_heads, head_dim]; drop head 0 only.
            head_mask = torch.ones(tiny_model.n_heads, 1)
            head_mask[0] = 0.0
            return activation * head_mask

        out = tiny_model.forward_logits(
            tokens, fn=zero_head0, layers=[0], modules=SELF_ATTN, per_head=True
        )
        assert out.shape == base.shape
        assert out.dtype == torch.float32
        assert not torch.allclose(out, base)

    def test_hf_model_is_underlying(self, tiny_model):
        assert tiny_model.hf_model is tiny_model._lm.model

    def test_raw_model_includes_output_embedding(self, tiny_model):
        """Expose the complete causal LM for notebook-local custom analyses."""
        # The generic accessor must retain the output embedding omitted by hf_model.
        assert tiny_model.raw_model is tiny_model._lm._model
        assert tiny_model.raw_model.get_output_embeddings() is not None

    def test_generate_with_hooks_returns_str(self, tiny_model):
        out = tiny_model.generate_with_hooks(
            "hello world",
            gen_kwargs={"max_new_tokens": 2, "do_sample": False},
        )
        assert isinstance(out, str)

    def test_chat_template_delegates(self, tiny_model):
        tiny_model.tokenizer.chat_template = (
            "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        )
        rendered = tiny_model.chat_template([{"role": "user", "content": "hello"}])
        assert isinstance(rendered, str)
        assert "hello" in rendered

    def test_isinstance_model_backend(self, tiny_model):
        # Cheap structural smoke check: presence of the protocol surface, not
        # a substitute for the behavioral tests above.
        assert isinstance(tiny_model, ModelBackend)


# ── Scoped lint: steps and io must not touch model internals ──────────

_BANNED_INTERNALS = {"_lm", "_module", "_resolve_module", "_generate_single"}


_MURANO_ROOT = Path(__file__).resolve().parents[1] / "src" / "murano"


def _scoped_source_files() -> list[Path]:
    files = sorted((_MURANO_ROOT / "steps").rglob("*.py"))
    files.append(_MURANO_ROOT / "io.py")
    return files


def _identifier_names(path: Path) -> set[str]:
    """Return the set of identifier tokens in ``path``.

    Operating on NAME tokens ignores comments, docstrings, and string
    literals, so a legitimate mention of an internal name in prose does not
    trip the lint.
    """
    names: set[str] = set()
    with tokenize.open(path) as handle:
        for tok in tokenize.generate_tokens(handle.readline):
            if tok.type == tokenize.NAME:
                names.add(tok.string)
    return names


@pytest.mark.parametrize(
    "path",
    _scoped_source_files(),
    ids=lambda p: str(p.relative_to(_MURANO_ROOT)),
)
def test_no_model_internals_in_steps_or_io(path):
    leaked = _BANNED_INTERNALS & _identifier_names(path)
    assert not leaked, (
        f"{path.name} references model internals {sorted(leaked)} directly; "
        f"use the ModelBackend interface instead."
    )
