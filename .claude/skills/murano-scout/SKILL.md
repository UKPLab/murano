---
name: murano-scout
description: Find mechanistic-interpretability methods worth implementing in Murano. Searches papers and posts published since the last scout, drops everything the ledger has already seen, screens the rest against hard gates and a score, and files the winners as GitHub issues labelled agent:todo. Use when the murano loop says scouting is due, or when asked to "scout for new methods/papers".
---

# Murano scout

Adapted from `k-dense-ai/scientific-agent-skills` `skills/literature-review`
(MIT; a reproducible search with a screening log of records identified,
screened and excluded, with reasons).

Scripts (stdlib, run with `python3 -I`):
- `scripts/paper.py`: `search`, `meta`, `impact`, `code-links`
- `scripts/ledger.py`: `filter`, `add`, `due`, `sync-issues`. Always pass
  `--ledger .agent-state/scouted.jsonl`.

Every source is keyless: the arXiv API (search, `meta`), OpenAlex (search,
citations, venues), Hugging Face papers and GitHub (the official repo and its
stars). There is no Semantic Scholar: keyless it is throttled to zero, and no
keyless source counts arXiv citations reliably. OpenAlex undercounts them
by about 10×, so `impact` treats its count as a lower bound and adds adoption
signals. Keyless OpenAlex has a daily budget of about 100 searches per IP
(shared with everything else on that IP, resets at midnight UTC), so the scout
spends it on `impact` only. `OPENALEX_API_KEY` (free, its own budget) and
`GITHUB_TOKEN` are optional.

To get an OpenAlex key when `OPENALEX_API_KEY` is unset, first ask the user
(AskUserQuestion) whether you may send their email address to OpenAlex. Send
it only after an explicit yes:
`curl -s -X POST https://api.openalex.org/api-keys -H 'Content-Type: application/json' -d '{"email": "<their email>"}'`.
OpenAlex emails them a link to the key; ask them to export `OPENALEX_API_KEY`
(for the cloud routine, add it to the routine's environment). Ask once per
session and not again after a no. When nobody can answer, as in the scheduled
routine, send nothing and note in the log that a key would help. When a paper comes back as
`{"error": "source_unavailable"}`, do not reject or defer it. If it was
already screened, keep its screening result in
`.agent-state/pending_impact.json` and score only Impact next scout. Otherwise
add it to `postponed` in `.agent-state/notes.md`.

## No paper is screened twice

The ledger runs before any model reads anything:

- Every paper that reaches screening gets a ledger row, whatever the outcome.
- `rejected` is permanent.
- `deferred` (Impact too low *because the paper is too new*, or the backlog is
  full) comes back only through `ledger.py due`.
- Issues closed as "not planned" sync in as human rejections.
- The window only covers papers since the last scout. Older classics come from
  a separate sweep every `cfg.classics_sweep_every_days`.

## Procedure

1. **Window.** Set `since` to `last_scout` from `.agent-state/notes.md`. On a
   first run, use 90 days ago.
2. **Collect IDs, not text.** Append to `candidates.jsonl` in the scratchpad:
   - For each query in `cfg.queries`, run `paper.py search "<q>" --since <since>`
     (arXiv, newest first). Pause 3 s between calls (arXiv's API etiquette).
     Run the loop under `bash`: zsh doesn't word-split `$args`, so flags kept
     in a variable arrive as one argument.
   - If the classics sweep is due, also run
     `paper.py search "<q>" --limit 100` with no `--since` (most relevant
     first; the ledger drops what was seen before).
   - Add the papers in `.agent-state/pending_impact.json`: already screened,
     they skip straight to step 6.
   - Add the IDs in `postponed` from the notes.
   - Use WebSearch, limited to the window, for method posts on
     transformer-circuits.pub, the Anthropic, Google DeepMind and OpenAI
     interpretability blogs, the Alignment Forum and LessWrong interpretability
     tags, and widely shared threads about a new interpretability method. Map
     each post to its arXiv ID when it has one. Otherwise add
     `{"url": ..., "title": ...}`.
   - Optional sources (alphaXiv, Scholar Gateway) are fine when their
     connectors are authorized.
3. **Dedup.**
   `ledger.py filter < candidates.jsonl > unseen.jsonl`, then
   `ledger.py due >> unseen.jsonl`.
4. **Title triage** (by you, no subagent). Drop titles that are plainly not
   about language-model internals, such as clinical, vision-only or
   chess-only work. Record each one with
   `ledger.py add --status rejected --gate G2 --reason "title triage: <why>"`.
5. **Screen** the rest in batches of 5, one cheap subagent per batch, at most
   `cfg.max_parallel_subagents` at a time. Each subagent reads the abstracts
   (`paper.py meta`, or the arXiv abstract page) and the method section when
   the abstract isn't enough. `paper.py meta` takes many IDs in one call. It
   returns per paper: G1–G4 and G6 as pass/fail with one line of evidence
   each; Generality, Fit and Cost scored 0–3; a three-sentence method summary;
   `paper.py code-links` output; and the official repo, if any, as
   `owner/name` (not a dependency such as TransformerLens).
6. **G5 and Impact** (by you, with scripts):
   - G5: grep `src/murano/steps/` and the README step table for the method,
     and search issues and PRs (`gh issue list --search "<key terms>" --state all`,
     same for `gh pr list`). Open work such as #119 (gradients) and #121
     (FEGA) counts.
   - Impact: one call for all gate-passing papers,
     `paper.py impact <id> <id> ... --repo <id>=<owner/name> ...`. Pass
     `--repo` for each paper whose screen named official code (without it,
     `impact` falls back to the repo the authors linked on Hugging Face). Add
     `--viral <id>` only when the screen found concrete evidence.
7. **Decide**, and write a ledger row for every paper:
   - **Accept**: all gates pass, total ≥ `cfg.accept.min_total`, Impact ≥
     `cfg.accept.min_impact`.
   - **Defer**: the gates pass but Impact is short and `impact` says
     `deferrable` (`--status deferred --recheck-days cfg.defer_recheck_days`).
   - **Reject**: anything else, with `--gate` naming the first gate that failed
     (`score` when only the total fell short).
8. **File issues.** Sort accepts by total. Open at most
   `min(cfg.max_new_issues_per_scout, cfg.backlog_cap - open backlog)` of them.
   Each one gets title `[METHOD] <method name> (<Author> et al., <year>)`,
   labels `enhancement` and `agent:todo`, and a body with the headings from
   `.github/ISSUE_TEMPLATE/method_proposal.yml` (see the template below),
   written through `humanizer`. Then run
   `ledger.py add ... --status accepted --issue <n>`. Accepts that didn't fit
   get `--status deferred --recheck-days 7 --reason "backlog full"`.
9. **Log.** Hand the loop the counts: identified, already seen, title-triaged,
   screened, accepted (issue links), deferred, rejected per gate, postponed.
   Also hand it the new `last_scout` (today), and `last_classics` if the sweep
   ran.

## Gates (all must pass)

| Gate | Passes when |
| --- | --- |
| G1 Method | The paper introduces a reusable procedure (attribution, patching variant, probe, steering, lens, SAE technique, circuit discovery) that applies beyond its own case study. Findings about one model or one task fail. |
| G2 Models | It works on decoder-only Hugging Face LMs that nnterp supports. Proprietary models, or training a large model from scratch, fail. |
| G3 Backend | It can be built with what `ModelBackend` exposes today (forward passes, hooks, attention probabilities, weights). Needing gradients or another backend extension fails, with the reason "backend extension". |
| G4 Testable | A tiny random CPU model can check it through an invariant or a numeric oracle. |
| G5 New | No Murano step, open issue, open PR or ledger row already covers it. |
| G6 Specified | There is official code, or the paper gives the algorithm completely (equations plus hyperparameters). |

## Score (0–3 each, accept at total ≥ 8 and Impact ≥ 2)

- **Impact** comes from `paper.py impact`, never by hand. Citations are the
  OpenAlex sum over every record with the paper's title, a lower bound. Stars
  are those of the official repo; the thresholds were calibrated on papers
  whose citation-based Impact is known.
  - 3: ≥ 300 citations, or under 18 months old at ≥ 10 citations per month,
    or ≥ 500 stars.
  - 2: ≥ 100 citations; or under 18 months old with ≥ 20 citations at ≥ 3.3
    per month; or a top venue (NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, TMLR,
    COLM) on any OpenAlex record; or the viral flag; or ≥ 150 stars.
  - 1: ≥ 20 citations, or ≥ 30 stars.
  - The viral flag needs concrete evidence: a post with ≥ 150 karma on the
    Alignment Forum or LessWrong, a transformer-circuits or major-lab post, or
    wide discussion you can link.
- **Generality**: 3 means usable on most models, tasks and analyses; 0 means
  tied to one setup.
- **Fit**: 3 means it composes from existing steps plus one small new step,
  and its result feeds `SelectComponents` or `Sweep`; 0 means it fights the
  Step/Results model.
- **Cost**: 3 means an afternoon; 0 means weeks or heavy dependencies.

## Issue body template

```markdown
> *Generated by the murano agent loop.*

### Paper
<Title>. <Authors>. <Venue, year>. <arXiv/DOI link>

### Links
<code repos from code-links, project page, notable posts>

### Method
<three sentences: what it computes, on which components, what you learn>

### Why it matters
Impact <n>/3 (<paper.py reasons>) · Generality <n> · Fit <n> · Cost <n> · total <n>/12
Gates: G1 ✓ <evidence> · G2 ✓ · G3 ✓ · G4 ✓ <oracle idea> · G5 ✓ · G6 ✓ <official code | fully specified>

### Murano mapping
<existing steps it reuses; the new step(s) and what they read and write>
```

## Return

The log counts from step 9, the new issue numbers, and the new values for
`last_scout`, `last_classics` and `postponed`.
