"""Tests for the Intervene step and generation-time interventions.

The regression guarded here is that an activation intervention applied during
generation reaches every generated token, not only the prompt pass. nnsight binds
an edit to a module's first forward, so before the ``tracer.all()`` fix a steer or
ablation silently reverted after the first token, and every earlier generating test
used ``max_new_tokens=1``, which is why the gap went unnoticed.

The decisive check counts how many times the intervention fn runs during a multi-
token generation: the prefill-only bug calls it once, the fix once per decode step.
That signal is independent of the model's weights, so it holds on the random-init
CPU fixtures (``murano_model`` is RMSNorm, ``gpt2_model`` is LayerNorm). A companion
check forces a token at the last layer; because the forcing overrides the logits on
every step, the whole continuation is that token regardless of the random weights.
"""

from __future__ import annotations

import pytest
import torch

from murano import Pipeline, keys
from murano.dataset import MuranoDataset
from murano.steps.intervene import Intervene, InterveneResult, steer_direction
from murano.steps.load import Load

FIXTURES = ["murano_model", "gpt2_model"]


# ── Forcing helpers ───────────────────────────────────────────────────


def _force_last_layer(model, token_id: int):
    """Return a fn that pins the last-layer residual to one token's direction.

    Overwriting the whole residual with a large multiple of ``token_id``'s
    unembedding row makes the final norm renormalize to that direction, so greedy
    decoding emits a fixed token on every step the fn runs, whatever the model's
    weights are.
    """
    direction = model.unembed_weight[token_id].detach().float()
    direction = direction / direction.norm()

    def fn(activation, key):
        forced = direction.to(activation.device, activation.dtype)
        return torch.zeros_like(activation) + 10.0 * forced

    return fn


def _forced_token(model):
    """Return ``(token_id, fn)`` where ``fn`` forces the last layer to emit it.

    The forced token is the argmax the forcing produces; searching real-word ids
    (specials 0-3 would be stripped by ``skip_special_tokens`` and can end
    generation early) keeps the forced continuation a clean run of one word.
    """
    last = model.n_layers - 1
    tokens = model.tokenizer(
        "hello world", return_tensors="pt", return_token_type_ids=False
    )
    specials = set(model.tokenizer.all_special_ids or [])
    for seed in range(4, model.unembed_weight.shape[0]):
        fn = _force_last_layer(model, seed)
        logits = model.forward_logits(tokens, fn=fn, layers=[last], modules="residual")
        token_id = int(logits[0, -1].argmax())
        if token_id not in specials:
            return token_id, fn
    pytest.skip("no non-special forced token on this fixture")


# ── Generation reaches every token ────────────────────────────────────


@pytest.mark.parametrize("fixture", FIXTURES)
def test_intervention_runs_on_every_generated_token(fixture, request):
    # The prefill-only bug ran fn once; the fix runs it once per decode step. The
    # call count is weight-independent, and the forced continuation confirms the
    # edit shapes each token, not just the first.
    model = request.getfixturevalue(fixture)
    token_id, force = _forced_token(model)
    word = model.tokenizer.decode([token_id])
    n_new = 5
    last = model.n_layers - 1

    calls: list = []

    def fn(activation, key):
        calls.append(key)
        return force(activation, key)

    out = model.generate_with_hooks(
        "hello world",
        fn=fn,
        layers=[last],
        modules="residual",
        gen_kwargs={"max_new_tokens": n_new, "do_sample": False},
    )

    assert len(calls) == n_new
    assert out.split() == [word] * n_new, out


@pytest.mark.parametrize("fixture", FIXTURES)
def test_intervene_step_runs_on_every_generated_token(fixture, request):
    # The public Intervene step routes generation through the same path, so its
    # modified run must also touch every token and differ from the clean run.
    model = request.getfixturevalue(fixture)
    token_id, force = _forced_token(model)
    word = model.tokenizer.decode([token_id])
    n_new = 5
    last = model.n_layers - 1

    calls: list = []

    def fn(activation, key):
        calls.append(key)
        return force(activation, key)

    results = Pipeline(
        [
            Load(MuranoDataset(positive_texts=["hello world"], negative_texts=[])),
            Intervene(
                model,
                fn=fn,
                layers=[last],
                gen_kwargs={"max_new_tokens": n_new, "do_sample": False},
            ),
        ]
    ).run()
    intervene: InterveneResult = results[keys.INTERVENE]

    assert len(calls) == n_new
    assert intervene.modified_generations[0].split() == [word] * n_new


# ── No-intervention path stays intact ─────────────────────────────────


@pytest.mark.parametrize("fixture", FIXTURES)
def test_plain_generation_is_deterministic(fixture, request):
    # fn=None skips the per-token loop entirely; greedy generation must still run
    # and stay reproducible.
    model = request.getfixturevalue(fixture)
    gk = {"max_new_tokens": 5, "do_sample": False}

    first = model.generate_with_hooks("hello world", fn=None, gen_kwargs=gk)
    second = model.generate_with_hooks("hello world", fn=None, gen_kwargs=gk)

    assert isinstance(first, str)
    assert first == second


def test_steer_direction_is_usable_during_generation(murano_model):
    # A realistic steer (not a hard override) must run cleanly through the fixed
    # per-token loop without error and return a continuation.
    model = murano_model
    steer_fn = steer_direction({0: torch.ones(model.d_model)}, alpha=2.0)

    out = model.generate_with_hooks(
        "hello world",
        fn=steer_fn,
        layers=[0],
        gen_kwargs={"max_new_tokens": 4, "do_sample": False},
    )

    assert isinstance(out, str)
