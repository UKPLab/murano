"""Tests for the RACE step, RACEResult and its neuron selection.

The evidence and posterior math is checked on hand-built tensors against the
pinned reference functions (Nexround/RACE@cf6fb23, ``analyzer.py``) and against
the module outputs of the tiny fixtures. Batching, padding and the start
position are checked through the step itself.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from murano import Pipeline, keys
from murano._proxy import unwrap_traced
from murano.artifacts import NeuronSelection, PromptBatch
from murano.io import load_race
from murano.nodes import MLP, SELF_ATTN, Neuron, Node
from murano.results import Results
from murano.steps.prompts import LoadPrompts
from murano.steps.race import (
    RACE,
    RACEResult,
    _evidence,
    _merge_moments,
    _t_ppf,
)
from murano.steps.save import Save

# Fixture tokenizer vocab: hello world good bad prompt response (+ specials).
PROMPTS = [
    "hello world",
    "good world bad",
    "bad",
    "prompt response hello world good",
    "world good",
    "response",
    "hello bad prompt",
    "good good good world",
    "bad response",
    "world",
]


def _run(model, prompts=PROMPTS, **kwargs) -> RACEResult:
    results = Results()
    results[keys.PROMPTS] = PromptBatch(prompts=list(prompts))
    return RACE(model, **kwargs)(results)[keys.RACE]


def _from_evidence(e: torch.Tensor, node="L0.mlp", **prior) -> RACEResult:
    """A RACEResult pooled from an explicit ``[N, C]`` evidence matrix."""
    zeros = torch.zeros(e.shape[1], dtype=torch.float64)
    n, mean, ss = _merge_moments(0, zeros, zeros.clone(), e)
    return RACEResult(n=n, mean={node: mean}, ss={node: ss}, **prior)


def _module_outputs(model, prompts, layer, module, side="right"):
    """Per-position module outputs ``[B, S, d]`` and the attention mask."""
    tok = model.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = side
    try:
        tokens = tok(
            prompts, return_tensors="pt", padding=True, return_token_type_ids=False
        )
    finally:
        tok.padding_side = prev_side
    with model.trace(tokens):
        saved = model.resolve_module(layer, module).output.save()
    return unwrap_traced(saved), tokens["attention_mask"].bool()


# ── Contract ──────────────────────────────────────────────────────────


class TestRACEContract:
    def test_default_keys(self, murano_model):
        step = RACE(murano_model)
        assert step.reads == [keys.PROMPTS]
        assert step.writes == [keys.RACE]
        assert step.write_types[keys.RACE] is RACEResult

    def test_custom_keys(self, murano_model):
        step = RACE(murano_model, prompts_key="p", output_key="r")
        assert step.reads == ["p"]
        assert step.writes == ["r"]

    def test_pipeline_shapes_and_count(self, murano_model):
        out = Pipeline([LoadPrompts(PROMPTS), RACE(murano_model)]).run()
        result = out[keys.RACE]
        n_tokens = sum(len(p.split()) for p in PROMPTS)
        assert result.n == n_tokens
        assert set(result.mean) == {
            Node(layer, m) for layer in range(2) for m in (MLP, SELF_ATTN)
        }
        assert result.mean[Node(0, MLP)].shape == (64,)  # intermediate_size
        assert result.mean[Node(1, SELF_ATTN)].shape == (32,)  # n_heads * head_dim
        assert result.mean[Node(0, MLP)].dtype == torch.float64

    def test_module_and_layer_subset(self, murano_model):
        result = _run(murano_model, modules="mlp_out", layers=[1])
        assert set(result.mean) == {Node(1, MLP)}

    def test_start_positions_skip_tokens(self, murano_model):
        result = _run(murano_model, start_positions=1)
        assert result.n == sum(len(p.split()) - 1 for p in PROMPTS)

    def test_per_prompt_start_positions(self, murano_model):
        starts = [0, 1, 2, 0, 3, 1, 0, 2, 1, 0]
        result = _run(murano_model, start_positions=starts, batch_size=2)
        # Word counts 2 3 1 5 2 1 3 4 2 1 minus the starts, floored at zero.
        assert result.n == 2 + 2 + 0 + 5 + 0 + 0 + 3 + 2 + 1 + 1
        assert result.metadata["start_positions"] == starts
        # Precondition: a zero start keeps every token.
        assert _run(murano_model, start_positions=0).n == 24

    def test_select_returns_neurons(self, murano_model):
        selection = _run(murano_model).select(top_k=3)
        assert isinstance(selection, NeuronSelection)
        assert len(selection) == 3 * 4
        assert all(isinstance(n, Neuron) for n in selection)
        assert set(selection.scores) == set(selection.neurons)

    def test_neuron_selection_canonicalises_addresses(self):
        selection = NeuronSelection(
            neurons=["L0.mlp#3", Neuron(Node(1, SELF_ATTN), 2)],
            scores={"L0.mlp#3": 1.5},
        )
        assert selection.neurons == [Neuron(Node(0, MLP), 3), Neuron("L1.self_attn", 2)]
        assert selection.scores == {Neuron(Node(0, MLP), 3): 1.5}
        assert "L0.mlp#3" in selection
        assert Neuron(Node(0, MLP), 3) in selection
        assert "L0.mlp#4" not in selection
        assert "not a neuron" not in selection
        assert 3 not in selection

    def test_gpt2_runs(self, gpt2_model):
        result = _run(gpt2_model)
        assert result.mean[Node(0, MLP)].shape == (128,)  # GPT-2 4 * n_embd


# ── Mechanism ─────────────────────────────────────────────────────────


class TestRACEMechanism:
    def test_evidence_sums_to_update_norm(self):
        # Dossier oracle 1: sum_j e_j = ||Wa||^2 / (||Wa|| + eps), Eqs. 4-9.
        torch.manual_seed(0)
        a = torch.randn(50, 6, dtype=torch.float64)
        w = torch.randn(4, 6, dtype=torch.float64)
        norm = (a @ w.T).norm(dim=-1)
        expected = norm.pow(2) / (norm + 1e-8)
        assert torch.allclose(_evidence(a, w).sum(-1), expected, atol=1e-8)
        # Precondition: the identity is specific to W; a rescaled column breaks it.
        w_bad = w.clone()
        w_bad[:, 2] *= 3.0
        norm_bad = (a @ w_bad.T).norm(dim=-1)
        assert not torch.allclose(_evidence(a, w).sum(-1), norm_bad, atol=1e-2)

    @pytest.mark.parametrize("module", [MLP, SELF_ATTN])
    def test_mean_evidence_equals_mean_update_norm(self, murano_model, module):
        # Dossier oracle 2: per non-padding position the evidence sums to the
        # module output norm (Llama has no projection bias), so their means agree.
        prompts = PROMPTS[:4]
        result = _run(murano_model, prompts=prompts, modules=module, layers=[1])
        out, mask = _module_outputs(murano_model, prompts, 1, module)
        norms = out.float().norm(dim=-1)
        expected = norms[mask].mean().item()
        assert result.mean[Node(1, module)].sum().item() == pytest.approx(
            expected, rel=1e-4
        )
        # Precondition: a one-position shift of the observation set changes it.
        shifted = torch.zeros_like(mask)
        shifted[:, 1:] = mask[:, :-1]
        assert norms[shifted].mean().item() != pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("module", [MLP, SELF_ATTN])
    def test_gpt2_conv1d_orientation(self, gpt2_model, module):
        # GPT-2's Conv1D stores W transposed and adds a bias the evidence excludes.
        prompts = PROMPTS[:4]
        proj = (
            gpt2_model.raw_attn_out_proj(0, SELF_ATTN)
            if module == SELF_ATTN
            else gpt2_model.raw_module(0, MLP).c_proj
        )
        original = proj.bias.detach().clone()
        torch.manual_seed(0)
        torch.nn.init.normal_(proj.bias)  # make the bias visible
        try:
            out, mask = _module_outputs(gpt2_model, prompts, 0, module)
            result = _run(gpt2_model, prompts=prompts, modules=module, layers=[0])
            bias = proj.bias.detach().clone()
        finally:
            with torch.no_grad():
                proj.bias.copy_(original)
        expected = (out - bias).norm(dim=-1)[mask].mean().item()
        assert result.mean[Node(0, module)].sum().item() == pytest.approx(
            expected, rel=1e-4
        )
        # Precondition: the bias is large enough that keeping it would fail.
        with_bias = out.norm(dim=-1)[mask].mean().item()
        assert with_bias != pytest.approx(expected, rel=1e-2)

    def test_posterior_matches_reference(self):
        # Dossier oracle 3: Nexround/RACE@cf6fb23 analyzer.py NIGPosterior on
        # compute_ffn_evidence_batch, seed 0, float64 draws, default prior.
        torch.manual_seed(0)
        a = torch.randn(50, 6, dtype=torch.float64)
        w = torch.randn(4, 6, dtype=torch.float64)
        result = _from_evidence(_evidence(a, w))
        node = Node(0, MLP)
        mu_n = [1.8420393, 0.6905398, 0.4143091, 0.5692523, 0.7400377, 0.2129089]
        beta_n = [109.89747, 52.00200, 12.018635, 37.232346, 33.923232, 6.6752]
        cam = [1.3599181, 0.3588955, 0.2548717, 0.2886297, 0.4721757, 0.0940869]
        expected = torch.tensor(mu_n, dtype=torch.float64)
        assert torch.allclose(result.posterior_mean[node], expected, rtol=1e-4)
        expected = torch.tensor(beta_n, dtype=torch.float64)
        assert torch.allclose(result.beta_n[node], expected, rtol=1e-4)
        expected = torch.tensor(cam, dtype=torch.float64)
        assert torch.allclose(result.cam()[node], expected, rtol=1e-4)
        assert torch.all(result.neg_cam()[node] == 0)

    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (3, 2.3533634348018233),
            (12, 1.7822875556493194),
            (52, 1.6746891537260247),
            (100002, 1.6448688644802159),
        ],
    )
    def test_t_quantile_matches_scipy(self, df, expected):
        # Dossier oracle 4: scipy 1.17.1 t.ppf(0.95, df).
        assert _t_ppf(0.95, df) == pytest.approx(expected, rel=1e-9)
        assert _t_ppf(0.05, df) == pytest.approx(-expected, rel=1e-9)

    def test_batching_and_order_invariance(self, murano_model):
        # Dossier oracle 5: streamed moments do not depend on batch size or order.
        one = _run(murano_model, batch_size=1)
        eight = _run(murano_model, prompts=PROMPTS[::-1], batch_size=8)
        assert one.n == eight.n
        for node in one.mean:
            assert torch.allclose(one.mean[node], eight.mean[node], atol=1e-5)
            assert torch.allclose(one.ss[node], eight.ss[node], atol=1e-5, rtol=1e-5)
        # Precondition: dropping the second batch changes the count.
        assert _run(murano_model, prompts=PROMPTS[:8]).n != one.n

    def test_default_prior_shrinks_mean(self):
        # Dossier oracle 6: with mu_0 = 0, lambda_0 = 1, mu_n = n / (n + 1) * mean.
        torch.manual_seed(1)
        result = _from_evidence(torch.randn(20, 8, dtype=torch.float64))
        node = Node(0, MLP)
        mean = result.mean[node]
        assert torch.allclose(result.posterior_mean[node], 20 / 21 * mean)
        assert torch.equal(
            torch.argsort(result.posterior_mean[node]), torch.argsort(mean)
        )
        # Precondition: a non-zero prior mean moves it.
        moved = _from_evidence(torch.randn(20, 8, dtype=torch.float64), mu_0=1.0)
        assert not torch.allclose(
            moved.posterior_mean[node], 20 / 21 * moved.mean[node]
        )

    def test_step_default_prior(self, murano_model):
        # Dossier defaults (§3.1): mu_0, lambda_0, alpha_0, beta_0 = 0, 1, 1, 1.
        assert RACE(murano_model).prior == (0.0, 1.0, 1.0, 1.0)
        assert _run(murano_model, prompts=PROMPTS[:2]).lambda_0 == 1.0

    def test_hand_nig_update_with_prior_mean(self):
        # e = 1, 2, 3: n = 3, mean 2, centred sum of squares 2. With mu_0 = 1 and
        # lambda_0 = 2: mu_n = (2 + 6) / 5 = 1.6, beta_n = 1 + 1 + 2*3*1 / 10 = 2.6,
        # alpha_n = 2.5, lambda_n = 5 (Eqs. 13-16).
        e = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        result = _from_evidence(e, mu_0=1.0, lambda_0=2.0)
        node = Node(0, MLP)
        assert result.posterior_mean[node].item() == pytest.approx(1.6)
        assert result.beta_n[node].item() == pytest.approx(2.6)
        t_q = 2.015048373333023  # scipy t.ppf(0.95, 5)
        cam = 1.6 - t_q * math.sqrt(2.6 / (2.5 * 5))
        assert result.cam()[node].item() == pytest.approx(cam, rel=1e-9)
        # Precondition: with mu_0 = 0 the discrepancy term changes beta_n.
        zero_prior = _from_evidence(e, lambda_0=2.0)
        assert zero_prior.beta_n[node].item() == pytest.approx(4.4)

    def test_cam_bounds_and_negative_symmetry(self):
        # Dossier oracle 7, Eq. 19.
        torch.manual_seed(2)
        e = torch.randn(40, 16, dtype=torch.float64) + torch.linspace(-2, 2, 16)
        result = _from_evidence(e)
        node = Node(0, MLP)
        mu, cam = result.posterior_mean[node], result.cam()[node]
        assert (mu < 0).any() and (cam > 0).any()  # precondition: both signs
        assert torch.all(cam >= 0)
        assert torch.all(cam[mu <= 0] == 0)
        assert torch.all(cam <= mu.clamp(min=0))
        flipped = _from_evidence(-e)
        assert torch.allclose(flipped.cam()[node], result.neg_cam()[node])

    def test_noisier_neuron_ranks_lower(self):
        # Dossier oracle 8: equal mean, different variance -> same mu_n, lower CAM.
        steady = torch.full((30,), 1.0, dtype=torch.float64)
        steady[::2] += 0.1
        steady[1::2] -= 0.1
        noisy = torch.full((30,), 1.0, dtype=torch.float64)
        noisy[::2] += 1.5
        noisy[1::2] -= 1.5
        result = _from_evidence(torch.stack([noisy, steady], dim=1))
        node = Node(0, MLP)
        assert result.mean[node][0] == pytest.approx(result.mean[node][1])
        assert result.posterior_mean[node][0] == pytest.approx(
            result.posterior_mean[node][1]
        )
        cam = result.cam()[node]
        assert cam[0] < cam[1]
        assert result.select(top_k=1).neurons == [Neuron(node, 1)]

    def test_reference_set_filtering(self, caplog):
        # Dossier oracle 9, Eq. 20, on hand-built posterior means.
        node = Node(0, MLP)

        def built(means):
            m = torch.tensor(means, dtype=torch.float64)
            return RACEResult(n=10, mean={node: m}, ss={node: torch.zeros_like(m)})

        target = built([5.0, 4.0, 3.0, 3.0, 2.0, 1.0])
        reference = built([0.0, 9.0, 8.0, 0.0, 0.0, 0.0])
        kw = {"score": "posterior_mean"}
        plain = [n.index for n in target.select(top_k=3, **kw)]
        assert plain == [0, 1, 2]  # ties (2, 3) break by ascending index
        assert [n.index for n in target.select(top_k=3, exclude=None, **kw)] == plain
        # Reference top 2 = {1, 2}; walk 0, 1, 2, 3, ... skipping them.
        filtered = target.select(top_k=3, exclude=reference, exclude_top_k=2, **kw)
        assert [n.index for n in filtered] == [0, 3, 4]
        # Reference budget defaults to the target budget: top 3 = {1, 2, 0}.
        default = target.select(top_k=3, exclude=reference, **kw)
        assert [n.index for n in default] == [3, 4, 5]
        # Too few left: fewer neurons and a warning.
        with caplog.at_level(logging.WARNING, logger="murano"):
            short = target.select(top_k=3, exclude=reference, exclude_top_k=4, **kw)
        assert [n.index for n in short] == [4, 5]
        assert "keeps 2/3" in caplog.text

    def test_fraction_budget_rounds_with_floor_of_one(self):
        node = Node(0, MLP)
        m = torch.arange(250, dtype=torch.float64)
        result = RACEResult(n=5, mean={node: m}, ss={node: torch.zeros_like(m)})
        assert len(result.select(fraction=0.01)) == 2  # round(2.5) -> 2
        assert len(result.select(fraction=0.001)) == 1  # floor of one

    def test_padding_does_not_change_statistics(self, murano_model):
        # Dossier oracle 10: a short prompt padded next to a long one.
        prompts = ["hello", "prompt response hello world good"]
        padded = _run(murano_model, prompts=prompts, batch_size=2)
        alone = _run(murano_model, prompts=prompts, batch_size=1)
        assert padded.n == alone.n == 6
        for node in alone.mean:
            assert torch.allclose(padded.mean[node], alone.mean[node], atol=1e-5)
            assert torch.allclose(padded.ss[node], alone.ss[node], atol=1e-5)

    def test_left_padding_tokenizer_is_forced_right(self, gpt2_model):
        # GPT-2 has absolute positions, so left padding would shift the short
        # prompt's tokens and change its activations.
        prompts = ["hello", "prompt response hello world good"]
        alone = _run(gpt2_model, prompts=prompts, batch_size=1)
        tok = gpt2_model.tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            padded = _run(gpt2_model, prompts=prompts, batch_size=2)
            assert tok.padding_side == "left"  # restored after the step
        finally:
            tok.padding_side = prev_side
        for node in alone.mean:
            assert torch.allclose(padded.mean[node], alone.mean[node], atol=1e-5)
            assert torch.allclose(padded.ss[node], alone.ss[node], atol=1e-5)
        # Precondition: left padding really moves the short prompt's activations.
        right, right_mask = _module_outputs(gpt2_model, prompts, 0, MLP, "right")
        left, left_mask = _module_outputs(gpt2_model, prompts, 0, MLP, "left")
        assert not torch.allclose(
            left[0][left_mask[0]], right[0][right_mask[0]], atol=1e-3
        )


# ── Save ──────────────────────────────────────────────────────────────


class TestRACESave:
    def test_round_trip(self, murano_model, tmp_path):
        Pipeline(
            [
                LoadPrompts(PROMPTS[:3]),
                RACE(murano_model, gamma=0.1, lambda_0=2.0),
                Save(output_dir=str(tmp_path)),
            ]
        ).run()
        path = next(tmp_path.rglob("race.pt"))
        loaded = load_race(path)
        original = _run(murano_model, prompts=PROMPTS[:3], gamma=0.1, lambda_0=2.0)
        assert loaded.n == original.n
        assert (loaded.gamma, loaded.lambda_0) == (0.1, 2.0)
        for node in original.mean:
            assert torch.equal(loaded.mean[node], original.mean[node])
            assert torch.equal(loaded.ss[node], original.ss[node])
            assert torch.equal(loaded.cam()[node], original.cam()[node])


# ── Plot ──────────────────────────────────────────────────────────────


class TestRACEPlot:
    def test_figure(self, murano_model):
        pytest.importorskip("plotly")
        from murano.plotting import plot_race_layers

        result = _run(murano_model, gamma=0.5)
        fig = plot_race_layers(result).to_dict()
        assert fig["data"]
        assert {t["orientation"] for t in fig["data"]} == {"h"}
        assert fig["layout"]["xaxis"]["title"]["text"] == "log10 CAM score"
        # Each trace holds quantiles of the layer's positive scores, in log10.
        cam = result.cam()[Node(0, MLP)]
        first = fig["data"][0]["x"]
        assert max(first) == pytest.approx(cam.max().log10().item())

    def test_plot_step_writes_file(self, murano_model, tmp_path):
        pytest.importorskip("plotly")
        from murano.steps.plot import Plot

        results = Results()
        results[keys.RACE] = _run(murano_model, gamma=0.5)
        Plot(output_dir=str(tmp_path), show=False, save_format="html")(results)
        assert (tmp_path / "plots" / "race_layers.html").exists()


# ── Errors ────────────────────────────────────────────────────────────


class TestRACEErrors:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"modules": "resid_post"}, "modules must be"),
            ({"modules": []}, "modules must be"),
            ({"layers": "some"}, "must be 'all'"),
            ({"layers": [5]}, "layers must be"),
            ({"gamma": 0.0}, "gamma must be"),
            ({"alpha_0": 0.0}, "alpha_0 must be > 0"),
            ({"lambda_0": -1.0}, "lambda_0 must be > 0"),
            ({"batch_size": 0}, "batch_size must be"),
            ({"start_positions": -1}, "start_positions must be"),
        ],
    )
    def test_bad_arguments(self, murano_model, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RACE(murano_model, **kwargs)

    def test_start_positions_length(self, murano_model):
        with pytest.raises(ValueError, match="one per prompt"):
            _run(murano_model, start_positions=[0, 1])

    def test_unknown_down_projection(self, murano_model, monkeypatch):
        monkeypatch.setattr("murano.steps.race._DOWN_PROJ_NAMES", ("nope",))
        with pytest.raises(NotImplementedError, match="no down projection"):
            _run(murano_model, modules="mlp")

    def test_select_arguments(self):
        node = Node(0, MLP)
        m = torch.ones(4, dtype=torch.float64)
        result = RACEResult(n=5, mean={node: m}, ss={node: m})
        other = RACEResult(n=5, mean={"L1.mlp": m}, ss={"L1.mlp": m})
        with pytest.raises(ValueError, match="top_k must be"):
            result.select(top_k=0)
        with pytest.raises(ValueError, match="fraction must be"):
            result.select(fraction=0.0)
        with pytest.raises(ValueError, match="not both"):
            result.select(exclude=result, exclude_top_k=1, exclude_fraction=0.1)
        with pytest.raises(ValueError, match="same modules"):
            result.select(exclude=other)
        with pytest.raises(ValueError, match="score must be"):
            result.select(score="lcb")
