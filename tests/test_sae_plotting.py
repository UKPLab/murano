"""Tests for the single-feature SAE Plotly visualizations."""

import pytest
import torch

from murano.plotting.sae import (
    plot_sae_feature_logit_effects,
    plot_sae_token_activations,
)


# ── plot_sae_feature_logit_effects ────────────────────────────────────


class TestFeatureLogitEffects:
    @pytest.fixture
    def decoder_unembedding(self):
        torch.manual_seed(0)
        n_features, d_model, vocab = 6, 8, 20
        decoder = torch.randn(n_features, d_model)
        unembedding = torch.randn(d_model, vocab)  # [d_model, vocab]
        return decoder, unembedding, n_features, vocab

    def test_returns_figure(self, decoder_unembedding):
        import plotly.graph_objects as go

        decoder, unembedding, *_ = decoder_unembedding
        fig = plot_sae_feature_logit_effects(
            2, decoder=decoder, unembedding=unembedding
        )
        assert isinstance(fig, go.Figure)

    def test_top_positive_token_matches_manual_projection(self, decoder_unembedding):
        decoder, unembedding, *_ = decoder_unembedding
        feature_id = 3
        expected_top = int((decoder[feature_id] @ unembedding).argmax())
        fig = plot_sae_feature_logit_effects(
            feature_id, decoder=decoder, unembedding=unembedding, num_tokens=1
        )
        # Table columns are [neg_token, neg_effect, pos_token, pos_effect].
        pos_tokens = list(fig.data[0].cells.values[2])
        assert pos_tokens[0] == str(expected_top)

    def test_transposed_unembedding_orientation(self, decoder_unembedding):
        import plotly.graph_objects as go

        decoder, unembedding, *_ = decoder_unembedding
        fig = plot_sae_feature_logit_effects(
            1, decoder=decoder, unembedding=unembedding.T
        )
        assert isinstance(fig, go.Figure)

    def test_token_labels_are_used(self, decoder_unembedding):
        decoder, unembedding, _, vocab = decoder_unembedding
        labels = [f"tok{i}" for i in range(vocab)]
        fig = plot_sae_feature_logit_effects(
            0,
            decoder=decoder,
            unembedding=unembedding,
            token_labels=labels,
            num_tokens=1,
        )
        pos_tokens = list(fig.data[0].cells.values[2])
        assert pos_tokens[0].startswith("tok")

    def test_rejects_non_positive_num_tokens(self, decoder_unembedding):
        decoder, unembedding, *_ = decoder_unembedding
        with pytest.raises(ValueError):
            plot_sae_feature_logit_effects(
                0, decoder=decoder, unembedding=unembedding, num_tokens=0
            )

    def test_rejects_non_positive_bins(self, decoder_unembedding):
        decoder, unembedding, *_ = decoder_unembedding
        with pytest.raises(ValueError):
            plot_sae_feature_logit_effects(
                0, decoder=decoder, unembedding=unembedding, bins=0
            )

    def test_rejects_out_of_range_feature(self, decoder_unembedding):
        decoder, unembedding, n_features, _ = decoder_unembedding
        with pytest.raises(ValueError):
            plot_sae_feature_logit_effects(
                n_features + 5, decoder=decoder, unembedding=unembedding
            )


# ── plot_sae_token_activations ────────────────────────────────────────


class TestTokenActivations:
    @pytest.fixture
    def examples(self):
        return [
            {"tokens": ["a", "great", "film"], "activations": [0.1, 0.9, 0.2]},
            {"tokens": ["so", "bad"], "activations": [0.3, 0.7]},
        ]

    def test_returns_figure_from_examples(self, examples):
        import plotly.graph_objects as go

        fig = plot_sae_token_activations(examples)
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_raw_activations(self):
        import plotly.graph_objects as go

        torch.manual_seed(0)
        activations = torch.rand(4, 5, 3)  # [N, seq, features]
        token_ids = torch.randint(0, 50, (4, 5))
        fig = plot_sae_token_activations(
            activations=activations, token_ids=token_ids, feature_id=1, num_examples=3
        )
        assert isinstance(fig, go.Figure)

    def test_requires_examples_or_raw_activations(self):
        with pytest.raises(ValueError):
            plot_sae_token_activations()

    def test_rejects_non_positive_num_examples(self, examples):
        with pytest.raises(ValueError):
            plot_sae_token_activations(examples, num_examples=0)

    def test_rejects_misaligned_tokens_and_activations(self):
        with pytest.raises(ValueError):
            plot_sae_token_activations([{"tokens": ["a", "b"], "activations": [0.1]}])

    def test_feature_id_required_for_3d_activations(self):
        activations = torch.rand(2, 4, 3)
        token_ids = torch.randint(0, 10, (2, 4))
        with pytest.raises(ValueError):
            plot_sae_token_activations(activations=activations, token_ids=token_ids)
