---
name: murano-test
description: Write the tests for a new or changed Murano step, oracle first. Covers contract, mechanism (checked against invariants or reference numbers), save round-trip, plot and error paths, on the tiny CPU fixtures. Use when implementing a Murano method, when a review asks for missing or weak tests, or when asked to "write tests" for a step.
---

# Murano tests: oracle first

Adapted from `obra/superpowers` `skills/test-driven-development@2ce8aa1` (MIT).
We keep red-green. We drop "delete any code written before its test", because
porting a vetted reference implementation legitimately comes first. What stays
non-negotiable: **every mechanism test has been seen failing** against a wrong
implementation, or against a precondition that proves it can fail.

Rules: `AGENTS.md` §9. Model file: `tests/test_patch.py`.

## Order of work

1. Copy the oracles from the issue's dossier (`<!-- murano-dossier -->`
   comment, "Oracles" section). Each oracle becomes one `TestXMechanism` test.
2. Write the mechanism tests first and run them: `uv run pytest tests/test_<x>.py -x`.
   They must fail, and fail on the assertion, not on an import or typo.
3. Implement until they pass. Then add the contract, save, plot and error-path
   tests.
4. Mutation check: break the implementation on purpose in one line (flip a
   sign, drop a term, skip a layer) and confirm at least one mechanism test
   fails. Then revert.

## File layout: `tests/test_<method>.py`

```python
"""Tests for the <Method> step.

<One paragraph: what is tested here, and what the shared machinery already covers elsewhere.>
"""

from __future__ import annotations

import pytest
import torch

from murano import Pipeline, keys
from murano.results import Results
from murano.steps.<method> import <Method>, <Method>Result

# Fixture tokenizer vocab: hello world good bad prompt response (+ specials).
PROMPTS = ["hello world", "good world"]


class Test<Method>Contract:
    def test_default_keys(self, murano_model):
        step = <Method>(murano_model)
        assert step.reads == [keys.PROMPTS]
        assert step.writes == [keys.<KEY>]
        assert step.write_types[keys.<KEY>] is <Method>Result

    def test_custom_output_key(self, murano_model): ...


class Test<Method>Mechanism:
    def test_<invariant>(self, murano_model):
        # Precondition: the quantity under test is not trivially satisfied.
        assert not torch.allclose(a, b)
        # Oracle (dossier oracle 1, Eq. n): ...
        assert torch.allclose(result.x, expected, atol=1e-4)

    def test_matches_reference_value(self, gpt2_model):
        # Dossier oracle 2: <repo>@<sha>, seed 0, inputs ..., expected 0.4137 ± 1e-3
        ...


class Test<Method>Save:
    def test_round_trip(self, murano_model, tmp_path):
        out = Pipeline([..., Save(output_dir=tmp_path)]).run()
        loaded = load_<method>(tmp_path / ... )
        ...


class Test<Method>Plot:
    def test_figure(self, murano_model):
        pytest.importorskip("plotly")
        fig = plot_<method>(result)
        assert fig.to_dict()["data"]  # assert on the dict, never render


class Test<Method>Errors:
    def test_bad_argument(self, murano_model):
        with pytest.raises(ValueError, match="<the actionable part of the message>"):
            <Method>(murano_model, layers="some")
```

## Rules

- **Fixtures:** `murano_model` (tiny Llama, fp32), `gpt2_model` (LayerNorm and
  Conv1D paths), `*_attn` variants when attention probabilities are needed.
  Never re-declare `_build_tiny_local_model`, and never download from the Hub.
- **Two architectures:** if the step touches module internals, parametrize the
  mechanism test over `murano_model` and `gpt2_model`.
- **What counts as a mechanism test:** one that would pass for the right math
  and fail for plausible wrong math. Tests that only check shapes, dtypes or
  "it runs" belong in the contract class. No mechanism test may be one of
  those.
- **Frozen reference numbers:** put them in the test as literals, with a
  comment naming their source (paper table, or repo@sha plus seed). Use
  `torch.manual_seed(0)` wherever the step samples.
- **Tolerances:** `atol=1e-4` for fp32 identities. Anything looser needs a
  comment saying why.
- **Optional dependencies:** `pytest.importorskip("<module>")` at the top of
  the test, never at module level.
- Steps run either through `Pipeline([...]).run()`, which exercises
  validation, or directly as `Step(...)(Results())`. Have at least one test of
  each kind.

## Return

The test file path, the oracle → test mapping, and the mutation you tried
along with the test that caught it.
