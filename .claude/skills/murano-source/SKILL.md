---
name: murano-source
description: Turn a queued Murano method issue (agent:todo) into an implementation dossier. Finds and vets the reference code, pins it, checks its license, extracts the algorithm and test oracles from the paper, maps the method onto Murano's steps, and posts the dossier on the issue (agent:sourced), or blocks it with the decision a human must make. Use when the murano loop sources an issue, or when asked to "source"/"find the code for" a paper.
---

# Murano source: paper → dossier

The dossier works as a durable agent brief: everything the implementer and
reviewer need, in one comment. The idea of an "agent brief" comes from
`mattpocock/skills` `skills/engineering/triage@4ae2317` (MIT). Two working
rules are inspired by `lingzhi227/agent-research-skills` `paper-to-code`, ideas
only since that repo has no license: never invent a hyperparameter, and flag
every ambiguity.

Script: `scripts/repo_check.py <github-url> --authors "A B,C D"` (needs `gh`).

## Procedure

1. **Claim.** Add `agent:wip` and read the issue, including its links and
   comments.
2. **Read the paper.** Use the arXiv HTML or PDF. Read the method section and
   the appendix in full, plus every equation, algorithm box and hyperparameter
   table the method uses. Note the section or equation number for each fact you
   pull out.
3. **Find reference code**, best source first:
   1. Code the paper links or the arXiv page lists
      (`../murano-scout/scripts/paper.py code-links <arxiv>`), or the authors'
      own GitHub accounts.
   2. Org-maintained libraries that already implement it: TransformerLens,
      nnsight tutorials, SAELens, pyvene, nnterp, or a lab release.
   3. An independent reimplementation that has tests and visible use.
   4. None. Implement from the paper.

   Never use gists, forks of the official repo, or unmaintained copies. Run
   `repo_check.py` on every repo you consider and keep its JSON. Prefer
   `owner_is_author: true`. Reject `fork: true`, and reject `archived: true`
   unless the paper itself links it.
4. **Read the chosen implementation** at the pinned SHA. Read only the files
   that implement the method. Write down where it differs from the paper. The
   code wins for behaviour, but the dossier names every difference.
5. **Get oracles.** These are the numbers and properties the tests will
   assert:
   - Invariants that follow from the math. For example: patching every site
     recovers the clean logits; a zero direction is a no-op; scores sum to the
     logit difference.
   - Numbers the paper reports that a small model can reproduce.
   - A reference run, when the repo is runnable on CPU with GPT-2 small or a
     tiny model within about 10 minutes. Clone it into a new empty directory
     in the scratchpad, never inside the Murano checkout. Give it its own
     virtualenv. Pass paths as arguments, run Python with `-I`, and skip
     anything that needs sudo or a GPU. Record the exact inputs, seed, model
     and outputs.
6. **Map onto Murano.** Read `AGENTS.md` §1–6, and read end to end the two
   existing steps closest to the method. First decide how much new code it
   needs (`AGENTS.md` §1): list every difference between the method and the
   closest step, mark each instrumental or incidental, and take the smallest
   rung the instrumental ones allow. A method that only swaps the core
   computation of an existing step, keeping its inputs, parameters and
   result fields, becomes a new option value on that step, not a new step.
   Then decide:
   - which existing steps it reuses
   - the new step(s), or the new option value on an existing step, with what
     each reads, writes and returns
   - the new keys
   - the Result fields, including whether it exposes `contributions` or a
     `MetricScore`
   - the default plot, if any
   - the extra needed, if any (usually none)

   If the method needs a backend extension, a Step-protocol change or a new
   base dependency, stop. Label the issue `agent:blocked` and post the
   decision a maintainer has to make.
7. **Check the license** with `repo_check.py`'s `verdict`:
   - `adapt-ok`: the implementer may adapt the code.
   - `oracle-only`: the implementer writes from the paper and uses the code
     only to produce the oracles in step 5.
8. **Post the dossier** (template below) through `humanizer`. Code, math,
   numbers and paths stay as they are. Then remove `agent:todo` and
   `agent:wip`, and add `agent:sourced`.

When the paper leaves out a detail and no reference code settles it, don't
block and don't invent a value silently. Take a provisional decision (the most
conservative option, or a value the paper uses elsewhere), write it into the
algorithm, and list it under "Open questions" in the dossier as
`<question> Provisional: <choice>, because <why>.` The implementer carries it
into the PR. Block (`agent:blocked`) only when no provisional choice can be
built: a backend extension, a Step-protocol change, a new base dependency, or
a license that forbids the work.

## Dossier template

```markdown
> *Generated by the murano agent loop.*
<!-- murano-dossier -->
## Dossier: <method name>

### Algorithm
<numbered pseudocode; each line cites (§x.y / Eq. n)>
Defaults: <hyperparameter = value (source)> …
Ambiguities: <what the paper leaves open, and how the reference code settles it>

### Provenance
| Role | Source | Pinned | License | Verdict |
| --- | --- | --- | --- | --- |
| Method | <citation + link> | | | |
| Reference code | <repo> | <sha> | <SPDX> | adapt-ok / oracle-only |
Differences, paper vs code: <list or "none">

### Murano mapping
Size: <chain of existing steps | `<Step>(<option>="<value>")` | preset of `<Step>` | new step>, because <the instrumental differences>
Instrumental differences from `<closest step>`: <what changes in the computation>
Incidental, dropped: <artefacts, settled toggles, redundant diagnostics; or "none">
Planned differences from the reference code: <what Murano does differently and why; or "none">
Reuses: <steps> · New: <`<StepName>` reads `<keys>` → writes `<key>`: `<Name>Result(<fields>)` | the option value and the lines it changes>
Composes with: <SelectComponents via contributions | Sweep via MetricScore | …>
Plot: <default view or "none, call plot_<x> directly"> · Extra: <none | name>

### Oracles
1. <invariant> → a test in TestXMechanism
2. <reference value: inputs, seed, model, expected ± tolerance, from where>

### Notebooks
Application: `notebooks/applications/<x>.ipynb`, showing <the one question it answers>
Reproduction: <`notebooks/reproductions/<author><year>_<slug>.ipynb` reproducing <figure/table>> | none (<why>)

### Open questions
<question> Provisional: <choice>, because <why>. | None.

### Risks
<what could make the implementation wrong or slow>
```

## Return

The issue number, the resulting label (`agent:sourced` or `agent:blocked`),
and the dossier comment URL or the reason it was blocked.
