"""Shape and contract tests for pipeline steps.

Tests run on CPU with synthetic data — no model loading required.
These verify tensor shapes, step contracts, and pipeline validation.
"""

import math

import pytest
import torch

from murano.artifacts import MetricResult, PromptBatch
from murano.results import Results
from murano.pipeline import Pipeline
from murano.steps.base import Step
from murano.steps.load import Load
from murano.steps.prompts import LoadPrompts
from murano.steps.evaluate import GenerationMetric
from murano.steps.record import (
    ActivationStore,
    LabeledActivationStore,
    Record,
    _select_token_activations,
)
from murano.steps.train import SteeringVector, SteeringResult
from murano.steps.intervene import (
    InterveneResult,
    ablate_direction,
    steer_direction,
)
from murano.steps.weight_ablation import ProjectionOperator
from murano.steps.refusal.evaluate import (
    ComplianceRate,
    EvalResult,
)
from murano.dataset import MuranoDataset, LabeledDataset
from murano.steps.probe import Probe, ProbeResult


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def n_layers():
    return 4


@pytest.fixture(params=[1, 8, 32])
def n_examples(request):
    return request.param


@pytest.fixture
def activation_store(n_examples, n_layers, d_model):
    """Synthetic ActivationStore with random activations."""
    # Add a slight mean shift to positive so steering vector is non-degenerate
    positive = {
        layer: torch.randn(n_examples, d_model) + 0.5 for layer in range(n_layers)
    }
    negative = {
        layer: torch.randn(n_examples, d_model) - 0.5 for layer in range(n_layers)
    }
    return ActivationStore(positive=positive, negative=negative)


@pytest.fixture
def contrastive_dataset():
    """Minimal contrastive dataset."""
    return MuranoDataset(
        positive_texts=["positive text"] * 5,
        negative_texts=["negative text"] * 5,
    )


@pytest.fixture
def results_with_activations(activation_store, contrastive_dataset):
    """Results with dataset and activations pre-loaded."""
    r = Results()
    r["dataset"] = contrastive_dataset
    r["record"] = activation_store
    return r


# ── Step Protocol Tests ───────────────────────────────────────────────


class TestStepProtocol:
    """Test that all steps follow the Step base class contract."""

    def test_step_has_reads_writes(self):
        """Every Step subclass must declare reads and writes."""
        from murano.steps import (
            Load,
            Save,
            SteeringVector,
            ComplianceRate,
            Plot,
        )

        for cls in [Load, Save, SteeringVector, ComplianceRate, Plot]:
            assert hasattr(cls, "reads"), f"{cls.__name__} missing reads"
            assert hasattr(cls, "writes"), f"{cls.__name__} missing writes"
            assert isinstance(cls.reads, list), f"{cls.__name__}.reads must be list"
            assert isinstance(cls.writes, list), f"{cls.__name__}.writes must be list"

    def test_step_validate_catches_missing_key(self):
        """Step.validate raises KeyError when required key is missing."""

        class NeedsRecord(Step):
            reads = ["record"]
            writes = ["output"]

            def __call__(self, results):
                return results

        step = NeedsRecord()
        with pytest.raises(KeyError, match="record"):
            step.validate(Results())

    def test_step_validate_passes_when_key_present(self):
        """Step.validate succeeds when required key exists."""

        class NeedsRecord(Step):
            reads = ["record"]
            writes = ["output"]

            def __call__(self, results):
                return results

        r = Results()
        r["record"] = "something"
        step = NeedsRecord()
        step.validate(r)  # should not raise

    def test_step_validate_catches_type_mismatch(self):
        """Step.validate raises TypeError when a required artifact has the wrong type."""

        class NeedsActivationStore(Step):
            reads = ["record"]
            writes = ["output"]
            read_types = {"record": ActivationStore}

            def __call__(self, results):
                return results

        r = Results()
        r["record"] = "not an activation store"
        with pytest.raises(TypeError, match="ActivationStore"):
            NeedsActivationStore().validate(r)


# ── Pipeline Validation Tests ─────────────────────────────────────────


class TestPipelineValidation:
    """Test that Pipeline.validate catches broken step chains."""

    def test_validate_correct_chain(self, contrastive_dataset):
        """Valid step chain passes validation."""
        pipe = Pipeline(
            [
                Load(contrastive_dataset),
                SteeringVector(),  # reads record — but Load writes dataset, not record
            ]
        )
        # This should fail because SteeringVector reads 'record' which Load doesn't write
        with pytest.raises(KeyError, match="record"):
            pipe.validate()

    def test_validate_returns_keys(self, contrastive_dataset):
        """Pipeline.validate returns list of keys produced."""
        pipe = Pipeline([Load(contrastive_dataset)])
        keys = pipe.validate()
        assert "dataset" in keys

    def test_validate_rejects_probe_after_contrastive_record(self, contrastive_dataset):
        class DummyModel:
            n_layers = 1

        pipe = Pipeline(
            [
                Load(contrastive_dataset),
                Record(DummyModel(), layers=[0], position="mean"),
                Probe(cv=2),
            ]
        )

        with pytest.raises(TypeError, match="LabeledActivationStore"):
            pipe.validate()

    def test_validate_rejects_steering_after_labeled_record(self, labeled_dataset):
        class DummyModel:
            n_layers = 1

        pipe = Pipeline(
            [
                Load(labeled_dataset),
                Record(DummyModel(), layers=[0], position="first"),
                SteeringVector(),
            ]
        )

        with pytest.raises(TypeError, match="ActivationStore"):
            pipe.validate()

    def test_validate_accepts_labeled_record_to_probe(self, labeled_dataset):
        class DummyModel:
            n_layers = 1

        pipe = Pipeline(
            [
                Load(labeled_dataset),
                Record(DummyModel(), layers=[0], position=0),
                Probe(cv=2),
            ]
        )

        keys = pipe.validate()
        assert "probe" in keys


# ── Load Step Tests ───────────────────────────────────────────────────


class TestLoad:
    def test_writes_dataset(self, contrastive_dataset):
        results = Load(contrastive_dataset)(Results())
        assert "dataset" in results
        assert results["dataset"] is contrastive_dataset
        assert isinstance(results["prompts"], PromptBatch)
        assert results["prompts"].prompts == contrastive_dataset.positive_texts

    def test_labeled_dataset_writes_prompts(self, labeled_dataset):
        results = Load(labeled_dataset)(Results())
        assert results["prompts"].prompts == labeled_dataset.texts

    def test_record_accepts_supported_positions(self):
        class DummyModel:
            n_layers = 2

        Record(DummyModel(), position="last")
        Record(DummyModel(), position="first")
        Record(DummyModel(), position="mean")
        Record(DummyModel(), position=0)
        Record(DummyModel(), position=-1)

    def test_record_rejects_unsupported_position(self):
        class DummyModel:
            n_layers = 2

        with pytest.raises(
            ValueError, match="position must be 'last', 'first', 'mean'"
        ):
            Record(DummyModel(), position="middle")


class TestLoadPrompts:
    def test_load_prompts_accepts_list(self):
        results = LoadPrompts(["a", "b"])(Results())
        assert isinstance(results["prompts"], PromptBatch)
        assert results["prompts"].prompts == ["a", "b"]

    def test_load_prompts_accepts_prompt_batch(self):
        prompt_batch = PromptBatch(
            prompts=["hi"], raw_prompts=["raw hi"], source="manual"
        )
        results = LoadPrompts(prompt_batch)(Results())
        assert results["prompts"] is prompt_batch


class TestRecordTokenSelection:
    def test_selects_first_last_mean_and_indexed_tokens(self):
        output = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [0.0, 0.0]],
                [[0.0, 0.0], [4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [0, 1, 1, 1],
            ]
        )

        assert torch.equal(
            _select_token_activations(output, attention_mask, "first"),
            torch.tensor([[1.0, 10.0], [4.0, 40.0]]),
        )
        assert torch.equal(
            _select_token_activations(output, attention_mask, "last"),
            torch.tensor([[3.0, 30.0], [6.0, 60.0]]),
        )
        assert torch.equal(
            _select_token_activations(output, attention_mask, 1),
            torch.tensor([[2.0, 20.0], [5.0, 50.0]]),
        )
        assert torch.equal(
            _select_token_activations(output, attention_mask, -1),
            torch.tensor([[3.0, 30.0], [6.0, 60.0]]),
        )
        assert torch.allclose(
            _select_token_activations(output, attention_mask, "mean"),
            torch.tensor([[2.0, 20.0], [5.0, 50.0]]),
        )

    def test_indexed_position_rejects_out_of_bounds(self):
        output = torch.randn(2, 3, 4)
        attention_mask = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
            ]
        )

        with pytest.raises(ValueError, match="out of bounds"):
            _select_token_activations(output, attention_mask, 2)


# ── SteeringVector Step Tests ─────────────────────────────────────────


class TestSteeringVector:
    """Shape tests for SteeringVector (contrastive mean diff)."""

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="contrastive_mean_diff"):
            SteeringVector(method="pca")

    def test_output_type(self, results_with_activations):
        results = SteeringVector()(results_with_activations)
        assert isinstance(results["steering"], SteeringResult)

    def test_direction_shapes(self, results_with_activations, n_layers, d_model):
        results = SteeringVector()(results_with_activations)
        steering = results["steering"]

        assert len(steering.direction_per_layer) == n_layers
        for layer, direction in steering.direction_per_layer.items():
            assert direction.shape == (d_model,), (
                f"Layer {layer}: expected ({d_model},), got {direction.shape}"
            )

    def test_directions_are_normalized(self, results_with_activations, n_layers):
        results = SteeringVector(normalize=True)(results_with_activations)
        for layer, direction in results["steering"].direction_per_layer.items():
            norm = direction.norm().item()
            assert abs(norm - 1.0) < 1e-5, f"Layer {layer}: norm={norm}, expected 1.0"

    def test_directions_unnormalized(self, results_with_activations):
        results = SteeringVector(normalize=False)(results_with_activations)
        # At least one direction should not have unit norm
        norms = [
            d.norm().item() for d in results["steering"].direction_per_layer.values()
        ]
        assert not all(abs(n - 1.0) < 1e-5 for n in norms)

    def test_separation_scores(self, results_with_activations, n_layers):
        results = SteeringVector()(results_with_activations)
        steering = results["steering"]
        assert len(steering.separation_scores) == n_layers
        assert all(isinstance(v, float) for v in steering.separation_scores.values())

    def test_single_example_score_is_finite(self, d_model):
        r = Results()
        r["record"] = ActivationStore(
            positive={0: torch.randn(1, d_model) + 0.5},
            negative={0: torch.randn(1, d_model) - 0.5},
        )
        results = SteeringVector()(r)
        assert math.isfinite(results["steering"].separation_scores[0])

    def test_best_layer_is_valid(self, results_with_activations, n_layers):
        results = SteeringVector()(results_with_activations)
        assert results["steering"].best_layer in range(n_layers)

    def test_best_layer_has_highest_score(self, results_with_activations, n_examples):
        if n_examples < 2:
            pytest.skip("Separation score requires n >= 2 (std undefined for n=1)")
        results = SteeringVector()(results_with_activations)
        steering = results["steering"]
        best_score = steering.separation_scores[steering.best_layer]
        assert best_score == max(steering.separation_scores.values())


# ── Intervention Function Tests ───────────────────────────────────────


class TestInterventionFunctions:
    """Shape tests for ablate_direction and steer_direction."""

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (4, 10), (8, 32)])
    def test_ablate_preserves_shape(self, batch_size, seq_len, d_model, n_layers):
        directions = {layer: torch.randn(d_model) for layer in range(n_layers)}
        fn = ablate_direction(directions)

        activation = torch.randn(batch_size, seq_len, d_model)
        for layer in range(n_layers):
            result = fn(activation, layer)
            assert result.shape == activation.shape

    @pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (4, 10), (8, 32)])
    def test_steer_preserves_shape(self, batch_size, seq_len, d_model, n_layers):
        directions = {layer: torch.randn(d_model) for layer in range(n_layers)}
        fn = steer_direction(directions, alpha=1.5)

        activation = torch.randn(batch_size, seq_len, d_model)
        for layer in range(n_layers):
            result = fn(activation, layer)
            assert result.shape == activation.shape

    def test_ablate_removes_direction_component(self, d_model):
        direction = torch.randn(d_model)
        direction = direction / direction.norm()
        fn = ablate_direction({0: direction})

        # Activation with known component along direction
        activation = direction.unsqueeze(0).unsqueeze(0) * 5.0  # [1, 1, d_model]
        result = fn(activation, 0)

        # Component along direction should be near zero
        component = (result @ direction).item()
        assert abs(component) < 1e-4, f"Residual component: {component}"

    def test_steer_adds_direction(self, d_model):
        direction = torch.randn(d_model)
        direction = direction / direction.norm()
        alpha = 2.0
        fn = steer_direction({0: direction}, alpha=alpha)

        activation = torch.zeros(1, 1, d_model)
        result = fn(activation, 0)

        # Result should be alpha * direction
        expected = alpha * direction
        diff = (result.squeeze() - expected).norm().item()
        assert diff < 1e-4, f"Expected alpha*direction, diff={diff}"

    def test_intervention_on_absent_layer_is_identity(self, d_model):
        directions = {0: torch.randn(d_model)}
        fn = ablate_direction(directions)

        activation = torch.randn(2, 5, d_model)
        result = fn(activation, 99)  # layer not in directions
        assert torch.equal(result, activation)

    def test_zero_direction_is_skipped(self, d_model):
        fn = ablate_direction({0: torch.zeros(d_model)})
        activation = torch.randn(2, 5, d_model)
        result = fn(activation, 0)
        assert torch.equal(result, activation)


# ── ProjectionOperator Tests ──────────────────────────────────────────


class TestProjectionOperator:
    """Shape and correctness tests for ProjectionOperator."""

    @pytest.mark.parametrize("d_model", [64, 256, 768])
    def test_projection_shape(self, d_model):
        direction = torch.randn(d_model)
        proj_op = ProjectionOperator(direction)
        assert proj_op.P.shape == (d_model, d_model)

    def test_projection_is_idempotent(self, d_model):
        """P @ P == P for orthogonal projection."""
        direction = torch.randn(d_model)
        direction = direction / direction.norm()
        proj_op = ProjectionOperator(direction)
        P2 = proj_op.P @ proj_op.P
        assert torch.allclose(proj_op.P, P2, atol=1e-5)

    @pytest.mark.parametrize(
        "in_features,out_features", [(64, 64), (64, 256), (256, 64)]
    )
    def test_project_read_shape(self, in_features, out_features):
        direction = torch.randn(in_features)
        proj_op = ProjectionOperator(direction)
        W = torch.randn(out_features, in_features)
        result = proj_op.project_read(W)
        assert result.shape == W.shape

    @pytest.mark.parametrize(
        "in_features,out_features", [(64, 64), (64, 256), (256, 64)]
    )
    def test_project_write_shape(self, in_features, out_features):
        direction = torch.randn(in_features)
        proj_op = ProjectionOperator(direction)
        W = torch.randn(in_features, out_features)
        result = proj_op.project_write(W)
        assert result.shape == W.shape

    def test_projection_removes_direction(self, d_model):
        direction = torch.randn(d_model)
        direction = direction / direction.norm()
        proj_op = ProjectionOperator(direction)

        # Project a vector that is purely along the direction
        v = direction.unsqueeze(0)  # [1, d_model]
        result = proj_op.project_read(v)  # v @ P
        assert result.norm().item() < 1e-4


# ── ComplianceRate Tests ──────────────────────────────────────────────


class TestComplianceRate:
    """Tests for compliance rate evaluation."""

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method='keyword'"):
            ComplianceRate(method="unsupported")

    def test_output_type(self):
        r = Results()
        r["intervene"] = InterveneResult(
            clean_generations=["I'm sorry, I can't help"],
            modified_generations=["Sure, here's how"],
        )
        results = ComplianceRate()(r)
        assert isinstance(results["eval"], EvalResult)
        assert results["eval"].baseline_scores == [0.0]
        assert results["eval"].modified_scores == [1.0]

    def test_refusal_detected(self):
        r = Results()
        r["intervene"] = InterveneResult(
            clean_generations=["I'm sorry, I cannot help with that."],
            modified_generations=["Sure, here is the information."],
        )
        results = ComplianceRate()(r)
        assert results["eval"].clean_compliance == 0.0
        assert results["eval"].ablated_compliance == 1.0

    def test_all_compliant(self):
        r = Results()
        r["intervene"] = InterveneResult(
            clean_generations=["Sure thing!", "Here you go."],
            modified_generations=["Of course!", "No problem."],
        )
        results = ComplianceRate()(r)
        assert results["eval"].clean_compliance == 1.0
        assert results["eval"].ablated_compliance == 1.0

    def test_empty_generations(self):
        r = Results()
        r["intervene"] = InterveneResult(
            clean_generations=[],
            modified_generations=[],
        )
        results = ComplianceRate()(r)
        assert results["eval"].clean_compliance == 0.0
        assert results["eval"].ablated_compliance == 0.0

    def test_works_with_weight_ablation_result(self):
        """ComplianceRate should work with WeightAblationResult too,
        since WeightAblation writes an InterveneResult to results['intervene']."""
        r = Results()
        r["intervene"] = InterveneResult(
            clean_generations=["I refuse to help"],
            modified_generations=["Here's how to do it"],
        )
        results = ComplianceRate()(r)
        assert results["eval"].clean_compliance == 0.0
        assert results["eval"].ablated_compliance == 1.0

    def test_context_window(self):
        """Refusal phrase after context_window should not be detected."""
        # Phrase at position 10 with context_window=5 should not be caught
        r = Results()
        text = "A" * 310 + " i'm sorry"
        r["intervene"] = InterveneResult(
            clean_generations=[text],
            modified_generations=["ok"],
        )
        results = ComplianceRate(context_window=300)(r)
        # The refusal phrase is past the 300-char window
        assert results["eval"].clean_compliance == 1.0


class TestGenerationMetric:
    def test_output_type(self):
        r = Results()
        r["intervene"] = InterveneResult(
            clean_generations=["aa", "bbbb"],
            modified_generations=["aaaa", "bb"],
        )
        results = GenerationMetric(
            metric_name="mean_length",
            score_fn=lambda texts: sum(len(text) for text in texts) / len(texts),
            item_score_fn=lambda text: float(len(text)),
        )(r)
        assert isinstance(results["metric"], MetricResult)
        assert results["metric"].metric_name == "mean_length"
        assert results["metric"].baseline_score == pytest.approx(3.0)
        assert results["metric"].modified_score == pytest.approx(3.0)
        assert results["metric"].baseline_scores == [2.0, 4.0]


# ── Results Tests ─────────────────────────────────────────────────────


class TestResults:
    def test_getitem_raises_helpful_error(self):
        r = Results()
        r["dataset"] = "foo"
        with pytest.raises(KeyError, match="record"):
            r["record"]

    def test_contains(self):
        r = Results()
        r["key"] = "val"
        assert "key" in r
        assert "other" not in r

    def test_copy_is_independent(self):
        r = Results()
        r["x"] = 1
        r2 = r.copy()
        r2["y"] = 2
        assert "y" not in r

    def test_save_accepts_run_name(self, tmp_path):
        r = Results()
        out_dir = r.save(output_dir=str(tmp_path), run_name="demo")
        assert out_dir == tmp_path / "demo"
        assert (out_dir / "metadata.json").exists()

    def test_save_serializes_prompt_and_metric_artifacts(self, tmp_path):
        r = Results()
        r["prompts"] = PromptBatch(
            prompts=["templated prompt"],
            raw_prompts=["raw prompt"],
            source="manual",
        )
        r["intervene"] = InterveneResult(
            clean_generations=["clean answer"],
            modified_generations=["modified answer"],
            prompts=["raw prompt"],
        )
        r["metric"] = MetricResult(
            metric_name="toy_metric",
            baseline_score=0.25,
            modified_score=0.75,
        )

        out_dir = r.save(output_dir=str(tmp_path), run_name="artifacts")
        assert (out_dir / "prompts" / "prompts.json").exists()
        assert (out_dir / "evaluation" / "generations.json").exists()
        assert (out_dir / "metrics" / "metric.json").exists()


# ── Probing Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def labeled_dataset():
    """Minimal labeled dataset for probing tests."""
    return LabeledDataset(
        texts=["positive text"] * 10 + ["negative text"] * 10,
        labels=[0] * 10 + [1] * 10,
        label_names=["class_0", "class_1"],
    )


@pytest.fixture
def labeled_activation_store(n_examples, n_layers, d_model):
    """Synthetic LabeledActivationStore with well-separated classes."""
    n_per_class = max(n_examples, 5)  # ensure enough for CV
    activations = {
        layer: torch.cat(
            [
                torch.randn(n_per_class, d_model) + 2.0,
                torch.randn(n_per_class, d_model) - 2.0,
            ]
        )
        for layer in range(n_layers)
    }
    labels = torch.tensor([0] * n_per_class + [1] * n_per_class)
    return LabeledActivationStore(activations=activations, labels=labels)


# ── LabeledDataset Tests ─────────────────────────────────────────────


class TestLabeledDataset:
    def test_creation(self):
        ds = LabeledDataset(texts=["a", "b"], labels=[0, 1])
        assert len(ds) == 2
        assert ds.label_names is None

    def test_label_names(self):
        ds = LabeledDataset(
            texts=["a", "b"],
            labels=[0, 1],
            label_names=["neg", "pos"],
        )
        assert ds.label_names == ["neg", "pos"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            LabeledDataset(texts=["a"], labels=[0, 1])

    def test_repr(self):
        ds = LabeledDataset(texts=["a", "b", "c"], labels=[0, 1, 0])
        assert "n=3" in repr(ds)
        assert "classes=2" in repr(ds)


# ── LabeledActivationStore Tests ─────────────────────────────────────


class TestLabeledActivationStore:
    def test_shapes(self, labeled_activation_store, n_layers, d_model):
        store = labeled_activation_store
        assert len(store.activations) == n_layers
        n_total = store.labels.shape[0]
        for layer in range(n_layers):
            assert store.activations[layer].shape == (n_total, d_model)

    def test_labels_shape(self, labeled_activation_store):
        store = labeled_activation_store
        n_total = sum(store.activations[0].shape[0] for _ in [None])
        assert store.labels.shape == (n_total,)
        assert store.labels.dtype == torch.long


# ── Probe Step Tests ─────────────────────────────────────────────────


class TestProbe:
    def test_output_type(self, labeled_activation_store, labeled_dataset):
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(cv=2)(r)
        assert isinstance(results["probe"], ProbeResult)

    def test_accuracy_per_layer_keys(
        self, labeled_activation_store, labeled_dataset, n_layers
    ):
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(cv=2)(r)
        assert len(results["probe"].accuracy_per_layer) == n_layers

    def test_best_layer_valid(
        self, labeled_activation_store, labeled_dataset, n_layers
    ):
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(cv=2)(r)
        assert results["probe"].best_layer in range(n_layers)

    def test_best_layer_has_highest_accuracy(
        self, labeled_activation_store, labeled_dataset
    ):
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(cv=2)(r)
        probe = results["probe"]
        best_acc = probe.accuracy_per_layer[probe.best_layer]
        assert best_acc == max(probe.accuracy_per_layer.values())

    def test_separable_data_high_accuracy(
        self, labeled_activation_store, labeled_dataset
    ):
        """With well-separated classes (offset +/-2), accuracy should be high."""
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(cv=2)(r)
        best_acc = max(results["probe"].accuracy_per_layer.values())
        assert best_acc > 0.7

    def test_refit_stores_classifiers(
        self, labeled_activation_store, labeled_dataset, n_layers
    ):
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(cv=2, refit=True)(r)
        assert len(results["probe"].classifiers) == n_layers

    def test_no_refit_empty_classifiers(
        self, labeled_activation_store, labeled_dataset
    ):
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(cv=2, refit=False)(r)
        assert len(results["probe"].classifiers) == 0

    def test_cv_scores_shape(self, labeled_activation_store, labeled_dataset, n_layers):
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        cv = 2
        results = Probe(cv=cv)(r)
        for layer in range(n_layers):
            assert len(results["probe"].cv_scores[layer]) == cv

    def test_label_names_passed_through(
        self, labeled_activation_store, labeled_dataset
    ):
        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(cv=2)(r)
        assert results["probe"].label_names == ["class_0", "class_1"]

    def test_custom_classifier(self, labeled_activation_store, labeled_dataset):
        from sklearn.linear_model import RidgeClassifier

        r = Results()
        r["dataset"] = labeled_dataset
        r["record"] = labeled_activation_store
        results = Probe(classifier=RidgeClassifier(), cv=2)(r)
        assert "probe" in results

    def test_wrong_store_type_raises(self, contrastive_dataset, activation_store):
        """Probe should raise TypeError if given ActivationStore instead of LabeledActivationStore."""
        r = Results()
        r["dataset"] = contrastive_dataset
        r["record"] = activation_store
        with pytest.raises(TypeError, match="LabeledActivationStore"):
            Probe(cv=2)(r)


class TestWeightAblation:
    def test_restores_weights_on_failure(self, monkeypatch):
        import murano.io as murano_io
        import murano.steps.weight_ablation as weight_ablation_module

        class DummyHFModel:
            def __init__(self):
                self.param = torch.nn.Parameter(torch.tensor([1.0]))

            def named_parameters(self):
                yield "param", self.param

        class DummyModel:
            def __init__(self):
                self.n_layers = 1
                self._lm = type("LM", (), {"model": DummyHFModel()})()

        model = DummyModel()
        step = weight_ablation_module.WeightAblation(model, save_dir="ablated")
        monkeypatch.setattr(step, "_generate", lambda _text: "ok")

        def fake_ablate(_model, _proj_op):
            _model._lm.model.param.data.mul_(2)
            return 1

        def fail_save(*_args, **_kwargs):
            raise RuntimeError("save failed")

        monkeypatch.setattr(weight_ablation_module, "ablate_model_weights", fake_ablate)
        monkeypatch.setattr(murano_io, "save_ablated_model", fail_save)

        r = Results()
        r["prompts"] = PromptBatch(prompts=["prompt"])
        r["steering"] = SteeringResult(
            direction_per_layer={0: torch.ones(4)},
            separation_scores={0: 1.0},
            best_layer=0,
        )

        with pytest.raises(RuntimeError, match="save failed"):
            step(r)

        assert model._lm.model.param.item() == pytest.approx(1.0)


# ── Backward Compatibility Tests ─────────────────────────────────────


class TestBackwardCompatibility:
    """Ensure probing additions don't break existing contrastive pipeline."""

    def test_contrastive_dataset_unchanged(self, contrastive_dataset):
        assert hasattr(contrastive_dataset, "positive_texts")
        assert hasattr(contrastive_dataset, "negative_texts")

    def test_activation_store_unchanged(self, activation_store):
        assert hasattr(activation_store, "positive")
        assert hasattr(activation_store, "negative")

    def test_steering_still_works(self, results_with_activations):
        results = SteeringVector()(results_with_activations)
        assert isinstance(results["steering"], SteeringResult)

    def test_load_accepts_labeled_dataset(self):
        ds = LabeledDataset(texts=["a", "b"], labels=[0, 1])
        results = Load(ds)(Results())
        assert results["dataset"] is ds
