---
name: murano-scout
description: Find mechanistic-interpretability methods worth implementing in Murano. Searches papers and posts published since the last scout, drops everything the ledger has already seen, screens the rest against hard gates and a score, and files the winners as GitHub issues labelled agent:todo. Use when the murano loop says scouting is due, or when asked to "scout for new methods/papers".
---

# Murano scout

Adapted from `k-dense-ai/scientific-agent-skills` `skills/literature-review`
(MIT; a reproducible search with a screening log of records identified,
screened and excluded, with reasons).

Every deterministic step is a script (stdlib, run with `python3 -I`); you
make only the judgment calls: which titles are on-topic, G5 verdicts, the
viral flag, and the issue prose.
- `scripts/scout.py`: the procedure below, one subcommand per step. Run state
  lives in `.agent-state/run/`.
- `scripts/paper.py`: `search`, `meta`, `impact`, `code-links` (used by
  `scout.py`, and by screeners for reading papers).
- `scripts/ledger.py`: the ledger (`scout.py` writes every row through it).

Every source is keyless: the arXiv API (search, `meta`), OpenAlex (citations,
venues), Hugging Face papers and GitHub (the official repo and its stars).
There is no Semantic Scholar: keyless it is throttled to zero, and no keyless
source counts arXiv citations reliably. OpenAlex undercounts them by about
10×, so `impact` treats its count as a lower bound and adds adoption signals.
Keyless OpenAlex has a daily budget of about 100 searches per IP (shared with
everything else on that IP, resets at midnight UTC), which is why search uses
arXiv. `OPENALEX_API_KEY` (free, its own budget) and `GITHUB_TOKEN` are
optional; locally they can live in the `env` block of the git-ignored
`.claude/settings.local.json`.

To get an OpenAlex key when `OPENALEX_API_KEY` is unset, first ask the user
(AskUserQuestion) whether you may send their email address to OpenAlex. Send
it only after an explicit yes:
`curl -s -X POST https://api.openalex.org/api-keys -H 'Content-Type: application/json' -d '{"email": "<their email>"}'`.
The response holds the key, which works once the user clicks the activation
link OpenAlex emails them. Give them the key and ask them to export
`OPENALEX_API_KEY` (for the cloud routine, add it to the routine's
environment). Never commit it. Ask once per session and not again after a no.
When nobody can answer, as in the scheduled routine, send nothing and note in
the log that a key would help.

## No paper is screened twice

The ledger runs before any model reads anything:

- Every paper that reaches screening gets a ledger row, whatever the outcome.
- `rejected` is permanent.
- `deferred` (Impact too low *because the paper is too new*, or the backlog is
  full) comes back only through `ledger.py due`.
- A paper that passed screening but could not be scored (`source_unavailable`)
  waits in `.agent-state/pending_impact.json`; the next `score` picks it up
  without screening it again.
- Issues closed as "not planned" sync in as human rejections.
- The window only covers papers since the last scout. Older classics come from
  a separate sweep every `cfg.classics_sweep_every_days`.

## Procedure

`S` below means `python3 -I .claude/skills/murano-scout/scripts/scout.py`.

1. **Web posts** (optional, by you). WebSearch, limited to the window, for
   method posts on transformer-circuits.pub, the Anthropic, Google DeepMind
   and OpenAI interpretability blogs, the Alignment Forum and LessWrong
   interpretability tags, and widely shared threads about a new method. Write
   one JSON line per post to a file: `{"arxiv": "<id>"}` when it has an arXiv
   ID, else `{"url": ..., "title": ...}`.
2. **Collect.** `S collect [--extra posts.jsonl]`. It sets the window
   (`last_scout`, or 90 days back on a first run), decides whether the classics
   sweep is due, runs every `cfg.queries` search on arXiv at arXiv's pace, adds
   the postponed IDs, drops everything in the ledger or pending, adds deferred
   papers that are due, and fetches missing titles.
3. **Title triage** (by you, no subagent). `S triage` prints every unseen
   title with its index, likeliest on-topic first (`K` interpretability term,
   `L` LM term, `!` off-topic term). Read all of it. Pick at most
   `5 × (cfg.max_subagents_per_tick − subagents already used)` papers to
   screen, preferring method-proposal titles and cited classics. Postpone the
   other on-topic papers. Then `S triage --screen 0-9,14 --postpone 20-40
   [--used N]`: every index you didn't name is rejected as G2.
4. **Screen.** `S batches` writes one brief per 5 papers
   (`.agent-state/run/batch_<n>.md`, from `screener_prompt.md`). Spawn one
   cheap subagent per brief with the file's contents as its prompt, at most
   `cfg.max_parallel_subagents` at a time. Each writes its JSON to
   `run/out_<n>.json`. Then `S record` computes pass and the first failed
   gate for every paper.
5. **G5** (by you). For each passing paper, `S g5-search "<term>" ...` lists
   matching issues, PRs and source files. Open work such as #119 (gradients)
   and #121 (FEGA) counts, and so does a method that composes from existing
   steps with no new code. `S g5-fail <id> "<reason>"` records a failure.
6. **Impact.** `S score`, adding `--viral <id>` only when you verified the
   screener's `viral_evidence`. One call scores every gate-passing paper and
   every pending one, using the screener's `official_repo` for stars.
7. **Decide.** `S decide` applies the rules and writes the ledger rows: accept
   at total ≥ `cfg.accept.min_total` and Impact ≥ `cfg.accept.min_impact`,
   up to `min(cfg.max_new_issues_per_scout − filed this scout,
   cfg.backlog_cap − open backlog)` issues; extra accepts are deferred 7 days
   ("backlog full"); short Impact on a `deferrable` paper defers it
   `cfg.defer_recheck_days`; anything else is rejected with its first failed
   gate (`score` when only the total fell short). It prints `to_file`.
8. **File issues.** For each paper in `to_file`:
   `S draft-issue <id> > body.md` fills every computed field (paper line,
   links, scores, gates) and prints a suggested title. Fill the two
   `<!-- EDIT -->` lines (see "Issue body"), run the prose through
   `humanizer` (embedded mode), and file with
   `S file-issue <id> body.md --title "[METHOD] <method name> (<Author> et al., <year>)"`.
   It labels the issue `enhancement` and `agent:todo` and writes the accepted
   ledger row.
9. **Finish.** `S finish` stamps `last_scout` (and `last_classics`) in the
   notes and prints the counts block for the loop's log comment.

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
- **Fit**: 3 means it is a new option on an existing step, or composes from
  existing steps plus one small new step, and its result feeds
  `SelectComponents` or `Sweep`; 0 means it fights the Step/Results model.
- **Cost**: 3 means an afternoon; 0 means weeks or heavy dependencies.

## Issue body

A maintainer decides from the top of the issue whether the paper is worth
it, so `scout.py draft-issue` puts only the paper, its code, the method and
the Murano mapping above a collapsed "Scores and gates" block (Impact with
its reasons, the other scores, every gate's evidence, extra links). You fill
the two `<!-- EDIT -->` lines in plain words: Method in at most two
sentences, the mapping in one. `file-issue` runs
`.claude/skills/murano-loop/scripts/post.py check issue` and refuses a body
over 120 words above the block; cut points, never compress them into
symbols.

## Return

The counts block from `scout.py finish` and the new issue numbers.
