"""Tests for the direct-logit-attribution step (LogitAttribution).

The completeness check (contributions sum to the true logit difference) validates
the frozen-norm decomposition. Completeness alone is partly self-fulfilling, since
any partition of the residual sums back, so on the tiny Llama fixture (no biases,
RMSNorm) we additionally assert ``other_contribution`` is ~0: that proves the
per-head ``z @ W_O`` plus MLP plus embedding pieces actually reconstruct the
residual. The fixture is RMSNorm, so the LayerNorm branch is covered by the
real-model validation, not here.
"""

from __future__ import annotations

import pytest
import torch

from murano import Pipeline, keys
from murano.artifacts import PromptBatch
from murano.nodes import MLP, SELF_ATTN, Node
from murano.results import Results
from murano.steps.logit_attribution import LogitAttribution, LogitAttributionResult
from murano.steps.metrics import _answer_positions
from murano.steps.prompts import LoadPrompts
from murano._proxy import unwrap_traced
from murano.steps.save import Save

PROMPTS = ["hello world", "good world"]


def _run(murano_model, **kwargs) -> LogitAttributionResult:
    results = Results()
    results[keys.PROMPTS] = PromptBatch(prompts=PROMPTS)
    out = LogitAttribution(murano_model, **kwargs)(results)
    return out[keys.LOGIT_ATTRIBUTION]


def _independent_head_contribution(model, layer, head, correct, incorrect) -> float:
    """Compute one head's mean contribution from the o_proj WEIGHT, independently.

    Uses ``z_head @ W_O_head`` (the weight, not the step's o_proj-masking path) so
    it cross-checks the per-head decomposition rather than restating it. Assumes a
    bias-free RMSNorm model (the tiny Llama fixture).
    """
    tok = model.tokenizer
    tokens = tok(
        PROMPTS, return_tensors="pt", padding=True, return_token_type_ids=False
    )
    with torch.no_grad():
        z_saved = {}
        with model.trace(tokens):
            z_saved["z"] = model.attn_out_proj(layer, SELF_ATTN).input.save()
        r_saved = {}
        with model.trace(tokens):
            r_saved["r"] = model.layer(model.n_layers - 1).output.save()

        z = unwrap_traced(z_saved["z"]).float()
        r = unwrap_traced(r_saved["r"]).float()
        true_logits = model.project_on_vocab(
            r.to(next(model.hf_model.parameters()).dtype)
        )
        pos = _answer_positions(true_logits, tokens["attention_mask"], None)
        rows = torch.arange(z.shape[0])
        r_pos = r[rows, pos]
        head_dim = model.head_dim
        sl = slice(head * head_dim, (head + 1) * head_dim)
        z_head = z[rows, pos][:, sl]
        w_o = model.hf_model.layers[layer].self_attn.o_proj.weight.float()
        head_resid = z_head @ w_o[:, sl].T

        ln = model.hf_model.ln_final
        eps = getattr(ln, "variance_epsilon", 1e-6)
        scale = (r_pos.pow(2).mean(dim=-1) + eps).rsqrt()
        normed = head_resid * scale.unsqueeze(-1) * ln.weight.float()
        w_u = model.unembed_weight.float()
        direction = w_u[correct] - w_u[incorrect]
        return float((normed * direction).sum(dim=-1).mean().item())


# ── Completeness ──────────────────────────────────────────────────────


class TestCompleteness:
    def test_logit_diff_completeness(self, murano_model):
        result = _run(murano_model, correct=5, incorrect=6)
        assert result.target == "logit_diff"
        assert result.completeness_error < 1e-3

    def test_logit_target_completeness(self, murano_model):
        result = _run(murano_model, correct=5)
        assert result.target == "logit"
        assert result.completeness_error < 1e-3

    def test_decomposition_reconstructs_residual(self, murano_model):
        # The tiny Llama has no projection/MLP/unembed biases and RMSNorm (no
        # beta), so the named pieces must reconstruct the residual exactly and the
        # catch-all "other" term collapses to ~0. A wrong per-head split would
        # leave a nonzero remainder here even while completeness still held.
        result = _run(murano_model, correct=5, incorrect=6)
        assert abs(result.other_contribution) < 1e-3

    def test_head_matches_independent_computation(self, murano_model):
        # Pin one head against a weight-based z @ W_O computation, independent of
        # the step's o_proj-masking path. This catches a wrong-but-summing split
        # (e.g. heads reshuffled) that completeness and other~0 would miss.
        result = _run(murano_model, correct=5, incorrect=6)
        last = murano_model.n_layers - 1
        expected = _independent_head_contribution(murano_model, last, 0, 5, 6)
        assert result.contributions[Node(last, SELF_ATTN, head=0)] == pytest.approx(
            expected, abs=1e-3
        )


# ── LayerNorm architecture ────────────────────────────────────────────


class TestLayerNorm:
    def test_completeness_on_gpt2(self, gpt2_model):
        # The tiny Llama fixture is RMSNorm; gpt2 exercises the LayerNorm branch
        # (centering, beta, normalized_shape detection) and transposed Conv1D
        # projections on CPU.
        results = Results()
        results[keys.PROMPTS] = PromptBatch(prompts=PROMPTS)
        out = LogitAttribution(gpt2_model, correct=5, incorrect=6)(results)
        result = out[keys.LOGIT_ATTRIBUTION]
        assert result.metadata["norm"] == "layernorm"
        assert result.completeness_error < 1e-3


# ── Contract ──────────────────────────────────────────────────────────


class TestContract:
    def test_writes_result_with_all_components(self, murano_model):
        result = _run(murano_model, correct=5, incorrect=6)
        assert isinstance(result, LogitAttributionResult)
        for layer in range(murano_model.n_layers):
            assert Node(layer, MLP) in result.contributions
            for head in range(murano_model.n_heads):
                assert Node(layer, SELF_ATTN, head=head) in result.contributions
        expected = murano_model.n_layers * (murano_model.n_heads + 1)
        assert len(result.contributions) == expected

    def test_validate_passes(self, murano_model):
        pipe = Pipeline(
            [
                LoadPrompts(PROMPTS),
                LogitAttribution(murano_model, correct=5, incorrect=6),
            ]
        )
        assert keys.LOGIT_ATTRIBUTION in pipe.validate()

    def test_chain_without_prompts_fails(self, murano_model):
        pipe = Pipeline([LogitAttribution(murano_model, correct=5)])
        with pytest.raises(KeyError, match="prompts"):
            pipe.validate()

    def test_invalid_layers_string_raises(self, murano_model):
        with pytest.raises(ValueError, match="layers as string must be 'all'"):
            LogitAttribution(murano_model, correct=5, layers="last")

    def test_layers_subset(self, murano_model):
        # A subset attributes only those layers; the rest fold into "other" and
        # completeness still holds.
        result = _run(murano_model, correct=5, incorrect=6, layers=[0])
        assert all(node.layer == 0 for node in result.contributions)
        assert result.completeness_error < 1e-3

    def test_per_example_matches_mean(self, murano_model):
        result = _run(murano_model, correct=5, incorrect=6, per_example=True)
        assert result.per_example is not None
        node = Node(murano_model.n_layers - 1, SELF_ATTN, head=0)
        mean = sum(result.per_example[node]) / len(result.per_example[node])
        assert mean == pytest.approx(result.contributions[node])

    def test_exclude_mlp_drops_mlp_nodes(self, murano_model):
        result = _run(murano_model, correct=5, incorrect=6, include_mlp=False)
        assert not any(node.module == MLP for node in result.contributions)
        # The MLPs now fold into "other", but the total still reconstructs.
        assert result.completeness_error < 1e-3

    def test_exclude_embed_zeroes_embed(self, murano_model):
        result = _run(murano_model, correct=5, incorrect=6, include_embed=False)
        assert result.embed_contribution == 0.0
        assert result.completeness_error < 1e-3


# ── Answer spec ───────────────────────────────────────────────────────


class TestAnswerSpec:
    def test_string_answers(self, murano_model):
        result = _run(murano_model, correct="world", incorrect="good")
        assert result.completeness_error < 1e-3

    def test_per_example_answers(self, murano_model):
        result = _run(murano_model, correct=[5, 6], incorrect=[6, 5])
        assert result.completeness_error < 1e-3

    def test_token_set_answers(self, murano_model):
        # [B, k] token sets: the direction averages W_U rows over the set and the
        # target averages the corresponding logits; completeness must still hold.
        result = _run(
            murano_model,
            correct=torch.tensor([[5, 6], [5, 6]]),
            incorrect=torch.tensor([[6, 7], [6, 7]]),
        )
        assert result.completeness_error < 1e-3

    def test_explicit_positions(self, murano_model):
        result = _run(murano_model, correct=5, incorrect=6, positions=[1, 1])
        assert result.completeness_error < 1e-3
        assert result.metadata["positions"] == [1, 1]


# ── Serialization ─────────────────────────────────────────────────────


class TestSerialization:
    def test_roundtrip(self, murano_model, tmp_path):
        from murano.io import load_logit_attribution

        results = Pipeline(
            [
                LoadPrompts(PROMPTS),
                LogitAttribution(
                    murano_model, correct=5, incorrect=6, per_example=True
                ),
                Save(output_dir=str(tmp_path)),
            ]
        ).run()
        original = results[keys.LOGIT_ATTRIBUTION]
        loaded = load_logit_attribution(
            tmp_path / "logit_attribution" / "logit_attribution.json"
        )
        assert isinstance(loaded, LogitAttributionResult)
        assert loaded.target == original.target
        assert loaded.total == pytest.approx(original.total)
        assert loaded.completeness_error == pytest.approx(original.completeness_error)
        # String addresses coerce back to Node keys.
        node = Node(murano_model.n_layers - 1, SELF_ATTN, head=0)
        assert loaded.contributions[node] == pytest.approx(original.contributions[node])


# ── Plot ──────────────────────────────────────────────────────────────


class TestPlot:
    def test_plot_returns_figure(self, murano_model):
        pytest.importorskip("plotly")
        from murano.plotting import plot_logit_attribution

        result = _run(murano_model, correct=5, incorrect=6)
        fig = plot_logit_attribution(result, top_k=5)
        assert fig.data
        assert len(fig.data[0].y) <= 5


# ── Gemma-family RMSNorm (1 + weight) regression ──────────────────────


class TestLogitAttributionGemmaNorm:
    """Gemma RMSNorm scales by (1 + weight); the frozen-norm DLA must match.

    Without the unit-offset correction the reconstruction misses the identity
    term and completeness_error blows up (seen as ~18 on real gemma-2-2b). This
    builds a tiny Gemma-2 and asserts completeness holds.
    """

    @pytest.fixture(scope="class")
    def gemma_model(self, tmp_path_factory):
        from pathlib import Path

        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            Gemma2Config,
            Gemma2ForCausalLM,
            PreTrainedTokenizerFast,
        )

        from murano.model import MuranoModel

        vocab = {
            "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
            "hello": 4, "world": 5, "good": 6, "bad": 7,
        }
        path = Path(tmp_path_factory.mktemp("tiny_gemma2"))
        tok = Tokenizer(WordLevel(vocab=dict(vocab), unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
            bos_token="<s>", eos_token="</s>", model_max_length=64,
        ).save_pretrained(path)
        config = Gemma2Config(
            vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64, sliding_window=64,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        torch.manual_seed(0)
        Gemma2ForCausalLM(config).save_pretrained(path)
        return MuranoModel(str(path), device_map="cpu", dtype=torch.float32)

    def test_completeness_holds_on_gemma(self, gemma_model):
        result = _run(gemma_model, correct=4, incorrect=5)
        assert result.metadata["norm"] == "rmsnorm"
        # With the (1 + weight) correction this is ~machine-epsilon; without it,
        # it is large (order 1-20).
        assert result.completeness_error < 1e-2, (
            f"gemma DLA completeness_error={result.completeness_error} — the "
            f"frozen norm is not reconstructing Gemma's (1 + weight) RMSNorm"
        )

    def test_unit_offset_norm_detected(self, gemma_model):
        step = LogitAttribution(gemma_model, correct=4)
        assert step._uses_unit_offset_norm(gemma_model.final_norm) is True
