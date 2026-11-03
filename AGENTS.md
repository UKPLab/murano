# Murano agent guidelines

Rules for any agent (or person) adding code to Murano. Murano is a composition
of small `Step`s over a shared `Results` dict. A new method fits when it reads
like the steps already in `src/murano/steps/`, reuses their helpers, and
registers itself in every place listed below. When this file and the code
disagree, the code wins. Fix this file in the same PR.

CONTRIBUTING.md covers setup, extras and `uv.lock`. This file covers how new
code has to look.

## 1. Golden rule: compose first

Before writing a step, try to express the method as a chain of existing ones
(`Record`, `Logits`, `Ablate`, `Patch`, `PathPatch`, `LogitAttribution`,
`SelectComponents`, `Sweep`, the metric steps). Write new code only for the part
no chain can express.

- If a method is a setting of an existing engine, write a thin preset.
  `Patch(Ablate)` in `steps/patch.py` is the model: about 70 lines, all
  defaults and docstring.

How much new code a method gets, smallest first. Take the first rung that
fits:

1. A chain of existing steps. No new code beyond a guide section or notebook.
2. A new value for an existing option on the closest step, when the method
   changes only the core computation and keeps that step's inputs,
   parameters and result fields. For example, LAT reading vectors are
   `SteeringVector(method="lat")`: the top singular vector of the paired
   differences instead of their mean. Accepting a new value is not a public
   signature change.
3. A thin preset subclass, like `Patch(Ablate)`.
4. A new step, only when the method needs new parameters, different inputs
   or new result fields.

To pick the rung, list every difference between the method (paper and
reference code) and the closest existing step, and sort each into one of two
kinds:

- **Instrumental:** it changes what gets computed (a different estimator,
  target or intervention).
- **Incidental:** an implementation artefact (a sign flip that makes a
  centered PCA act as a plain SVD), a paper-versus-code toggle the code
  already settles, or a diagnostic the existing result already covers.

Only instrumental differences justify code. Drop incidental ones and say so
in the dossier.
- If it needs a capability the backend lacks (gradients, a new hook point, a
  new model family), stop. That's a backend change, which needs its own
  human-approved issue (label `agent:blocked`).

## 2. Where things go

| What | Where | Example |
| --- | --- | --- |
| The step | `src/murano/steps/<method>.py` | `steps/logit_lens.py` |
| Its result type | `@dataclass <Name>Result` in the same module | `LogitLensResult` |
| A result shared by several steps | `src/murano/artifacts.py` | `ComponentSelection` |
| Results keys | `NAME: Final = "name"` in `src/murano/keys.py`, pinned in `tests/test_keys.py` | `keys.SELECTION` |
| Save / load | `save_<x>` + `load_<x>` in `src/murano/io.py`, plus a closure in `_serializer_registry()` | `save_logit_lens` |
| Plots | `plot_<x>(result, ...) -> go.Figure` in `src/murano/plotting/<domain>.py` | `plotting/logit_lens.py` |
| Toy data for tests and notebooks | `murano.tasks` (`ioi()`, `sentiment()`) | |
| A new dependency | an extra, following CONTRIBUTING "Adding a feature with its own dependency" | `sae` |

## 3. The step contract

- Subclass `murano.steps.base.Step` and implement `__call__(results) -> results`.
- Declare `reads`, `writes`, `read_types` and `write_types`. When keys are
  configurable, set them on the instance in `__init__` through `*_key` kwargs
  whose defaults come from `murano.keys` (`steps/logits.py:110`). Never
  advertise a key the step doesn't write.
- Validate arguments in `__init__` and raise `ValueError`, saying what to pass
  instead (`steps/select.py:105`).
- Model argument: `model: ModelBackend`, imported under `TYPE_CHECKING`.
  `import murano` must never import `murano.model` (`tests/test_imports.py`).
- Docstring: Google style with `Reads from results:`, `Writes to results:`,
  `Args:`, `Raises:`. The module docstring says why the step exists and how it
  composes (`steps/select.py:1`).
- Logging: `from murano.logging import logger`, `logger.info("StepName: ...")`.
  No `print`.
- Naming: steps are bare nouns or verbs (`LogitLens`, `Patch`), results are
  `<Name>Result`, activation containers are `<Name>Store`.

## 4. Model access

Steps talk to the model only through `murano.backend.ModelBackend`.
`tests/test_backend.py` (`_BANNED_INTERNALS`) fails the build when a step
reaches into private model internals.

- Prefer the high-level primitives: `forward_logits(tokens, fn=...)`,
  `generate_with_hooks`, `project_on_vocab`, `unembed_weight`, `final_norm`,
  `attention_probabilities`, `forward_logits_attention`.
- Inside a trace: `resolve_module(layer, module).output.save()`, then
  `murano._proxy.unwrap_traced`. One trace per module, layers saved in
  ascending order (`steps/record.py:347`, `steps/ablate.py:602`).
- Per-head work goes through `attn_out_proj(layer, module).input`, reshaped to
  `[B, S, n_heads, head_dim]`.
- Address components with `murano.nodes` (`Node`, `NodeSet`, `NodeDict`,
  `canonical_module`). A new kind of object (feature, neuron) is its own type
  composed on a `Node`. It never becomes a new `Node` field.
- `hf_model` is for weight reads only (`steps/weight_ablation.py`).

## 5. Reuse before you write

Grep for a helper before writing one. A duplicate helper always gets a
must-fix in review.

| Need | Use |
| --- | --- |
| Unwrap a saved proxy | `murano._proxy.unwrap_traced` |
| Targets to `list[Node]`, per-head grouping | `steps/ablate.py` `_coerce_targets`, `_group_targets` |
| Masked means, write masks, blend `fn` | `steps/ablate.py` `_component_mean`, `_site_mask`, `_ablation_fn` |
| Steer or project out a direction | `steps/intervene.py` `steer_direction`, `ablate_direction` |
| Answer positions, answer token ids, gathering | `steps/metrics.py` `_answer_positions`, `_answer_token_ids`, `_gather_answer` |
| A number out of a metric | `steps/metrics.py` `_metric_score` |
| Batching, token selection | `steps/record.py` `_batched`, `_select_token_activations` |
| Length-matched source prompts | `steps/attention.py` `_align_source_tokens` |
| Residual at resid_pre / resid_mid | `steps/sae.py` `_capture_residual` |
| JSON on disk | `io._write_json` (strict, NaN to null) |
| Heatmap, line chart, saving figures | `plotting/plotly_utils.py` `plot_heatmap`, `plot_line_chart`, `save_figure` |

## 6. Make results composable

- A per-component score goes in `contributions: dict[Node, float]`.
  `SelectComponents` can then rank it with no new code.
- A scalar outcome is a `MetricScore`, so `Sweep` can collect it.
- A set of components is iterable over `Node`, so `Patch(targets_key=...)` can
  read it.
- Canonicalize addresses in `__post_init__` with `Node.coerce` or `NodeDict`.
  Store tensors detached on CPU. Large fields are opt-in
  (`LogitLens(store_full_probs=...)`).

## 7. Don'ts

- No reaching into `nnsight` or `nnterp` internals, and no `model._lm`.
- No new base dependencies. Use an extra and `require_optional`, never a
  `try/except ImportError`.
- No new `MuranoModel` facade method unless an issue asks for one. Analysis
  logic lives in steps (`model.py:133`).
- No `Pipeline` built inside a function, and no causal step constructed inside
  a `for` loop. Use `Sweep` (`tests/test_notebook_structure.py`).
- No re-declaring `murano.tasks` content in notebooks or tests.
- No changes to the Step protocol, `ModelBackend` or existing public
  signatures inside a method PR.
- No drive-by refactors or reformatting of files the method doesn't need.
- No copied code without the license check in section 11.

## 8. Registration checklist

Each item is a place a new method must be added. A missing item is a review
must-fix. `tests/test_registration.py` enforces the starred ones.

1. `src/murano/steps/__init__.py`: import it and add it to `__all__`. ★
2. `src/murano/__init__.py`, headline names only: the `TYPE_CHECKING` import,
   the `_LAZY_ATTRS` entry and the `__all__` entry.
3. `src/murano/keys.py`, plus the pinned value in `tests/test_keys.py`.
4. `io.py`: save/load functions and the `_serializer_registry()` entry. Export
   the loader from `murano/__init__.py`.
5. `plotting/__init__.py` `__all__`, plus a `Plot` branch in `steps/plot.py`
   only when the result has one sensible default view.
6. `docs/scripts/gen_api_docs.py` `MODULES`, for every new module. ★
7. The step tables in `README.md` and `docs/src/content/docs/docs/guides/pipeline.md`.
8. `_CAUSAL_STEPS` in `tests/test_notebook_structure.py`, if the step runs an
   intervened forward pass.
9. A new notebook goes in `ORDER` in `docs/scripts/gen_notebook_docs.py` and in
   `notebooks/README.md` (test-enforced).
10. `CHANGELOG.md` `## [Unreleased]` → `### Added`: one bullet that opens with
    the backticked name and says what it lets you do.
11. A new extra: `pyproject.toml`, the `all` extra, `_optional.EXTRA_IMPORTS` ★,
    `_EXTRA_IMPLIED_BY` in the notebook structure test, and `uv lock`.

## 9. Tests

Procedure: `.claude/skills/murano-test/SKILL.md`. The short version:

- One `tests/test_<method>.py` with `TestXContract`, `TestXMechanism`,
  `TestXSave`, `TestXPlot` and the error paths, like `tests/test_patch.py`.
- Mechanism tests check the method's math against an oracle (a property that
  has to hold, or a number from the paper or reference code). Each one has a
  precondition showing it can fail. A test that only checks shapes isn't a
  mechanism test.
- Use the conftest fixtures (`murano_model`, `gpt2_model`, `*_attn`). Tests
  never download from the Hub, and CI has no GPU.

## 10. Docs, notebooks and prose

Run every piece of prose through the `humanizer` skill
(`.claude/skills/humanizer`, embedded mode): guide sections, notebook markdown,
CHANGELOG bullets, PR and issue bodies, review comments. Code, commands and
paths stay as they are.

- **Concise and specific.** A method gets one guide section that shows one
  usage: what question it answers, one runnable `Pipeline([...])`, and how to
  read the output. Stay under about 25 lines of prose, with no feature tours.
  Add it to the closest existing guide in `docs/src/content/docs/docs/guides/`.
  A new guide page (plus its `docs/astro.config.mjs` sidebar entry) is only for
  a new family of methods.
- **Application notebook** (`notebooks/applications/<method>.ipynb`): the
  template `tests/test_notebook_structure.py` enforces (header with
  `**Key questions**`, `**Structure**`, `**Model and data.**`,
  `**Requirements.**`; numbered sections; `## What next`;
  `OUTPUT_DIR = "murano_outputs/<stem>"`). GPT-2 small on CPU. Executed and
  committed with outputs. Markdown cells of 4 sentences or fewer.
- **Reproduction notebook** (`notebooks/reproductions/<author><year>_<slug>.ipynb`),
  when the headline result runs on a small model: a citation blockquote first,
  a recap table against the paper's numbers (say honestly where they differ),
  BibTeX at the end, a gallery page in `docs/src/pages/reproductions/<slug>.mdx`
  and an entry in `docs/src/pages/reproductions/index.astro`. Copy
  `notebooks/reproductions/wang2023_ioi.ipynb` and `docs/src/pages/reproductions/ioi.mdx`.

## 11. Attribution and licenses

Check a reference repo with `.claude/skills/murano-source/scripts/repo_check.py`
and pin the commit it prints.

- `adapt-ok` (MIT, BSD, Apache-2.0, ISC, CC0, Unlicense): you may adapt the
  code.
- `oracle-only` (GPL family, MPL, no license, NOASSERTION): you may read and
  run it to get test oracles, but write Murano's code from the paper. Say so in
  the attribution.

Every place where Murano computes something differently from the pinned
reference code (or from the paper, when the code is oracle-only) gets a
comment at that line, `# Differs from <repo>@<sha> <path>: <what>, because
<why>.`, and a bullet under "Differences from the reference code" in the PR
body. Incidental differences dropped on purpose count too.

Credit every method in three places:

1. An `Attribution:` paragraph at the end of the step module's docstring:
   ```
   Attribution:
       Method: <Authors> (<year>), "<title>", <venue>. <arXiv/DOI link>
       Reference code: <repo>@<sha> (<SPDX>), adapted: <what> | used as oracle only
   ```
2. A line under "Adapted implementations" in `NOTICE.txt`.
3. The notebook header and, for reproductions, the gallery page's "Original
   code" button.

## 12. Pull requests

- One method per PR. Branch `agent/<issue>-<slug>` (humans: any name), opened
  against `base_branch` in `.claude/agent-loop.json` (normally `main`).
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`).
- Fill in `.github/PULL_REQUEST_TEMPLATE.md`, starting with `Closes #<issue>`,
  using `.claude/skills/write-pr-description`: give the reviewer what the diff
  cannot. Keep everything above the collapsed checklist under 250 words. The
  rules are in `.claude/skills/murano-implement/SKILL.md` step 9.
- Before pushing, all of these pass:
  ```bash
  uv run ruff format --check . && uv run ruff check .
  uv run pyright src
  uv run pytest
  uv lock --check   # when pyproject.toml changed
  ```
- Agents never merge. A maintainer merges PRs labelled `agent:ready-to-merge`.

## 13. The autonomous loop

`/murano-loop` (`.claude/skills/murano-loop/SKILL.md`) is the single entry
point. Each run does one bounded tick of this label state machine:

```
issue  agent:todo ─source→ agent:sourced ─implement→ PR
                  └──────→ agent:blocked   (needs a human decision)
PR     agent:needs-review ─review→ agent:changes-requested ─revise→ agent:needs-review
                          └──────→ agent:ready-to-merge → a maintainer merges
any    agent:wip = claimed by a running tick · agent:hold = agents keep out
```

How humans steer it:

- To queue a paper, file it with the "Method proposal" form and add `agent:todo`.
- To reject a paper for good, close its issue as "not planned". The scout never
  proposes it again.
- To pause anything, add `agent:hold`.
- To require a change on an agent PR, comment on it. The next review counts
  unaddressed maintainer comments as must-fixes.
- To settle an open question, reply on the PR. The loop never waits for an
  answer: it ships a provisional decision, listed under "Open questions" in
  the PR body, and your reply overrides it in the next round.

Tunables (cadence, caps, thresholds, model tiers) live in
`.claude/agent-loop.json`. To change how often it runs, edit `schedule_cron` and
run `/schedule update` on the `murano-loop` routine. The scouting memory (every
paper ever screened) lives on the `agent-state` branch. Never merge that branch
into `main`.
