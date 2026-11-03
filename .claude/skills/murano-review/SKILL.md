---
name: murano-review
description: Strict review of a Murano pull request against AGENTS.md and the method's dossier. Runs the checks, fans out over correctness, tests, conventions and docs/attribution, fact-checks every finding, then posts one verdict comment listing must-fix changes and labels the PR agent:changes-requested or agent:ready-to-merge. Use when the murano loop reviews an agent PR, or when asked to "review PR #N" in Murano. Run it in a fresh context, never in the session that wrote the code.
---

# Murano review

Adapted from `pytorch/pytorch` `.agents/skills/pr-review/SKILL.md@4fb332d`
(BSD-3-Clause, © PyTorch contributors) for the review philosophy, the fan-out,
consolidation, fact-check passes and the maintainer-comment check. Also from
`obra/superpowers` `skills/requesting-code-review@2ce8aa1` (MIT): the reviewer
gets crafted context, never the author's session history. The PyTorch
checklists are replaced by `AGENTS.md`.

Inputs: the PR number. Read the linked issue's `<!-- murano-dossier -->`
comment and `AGENTS.md` yourself. Don't take the PR's own summary on trust.

## Philosophy

1. **Only report problems.** No praise and no "looks good". Leave out any
   section that has nothing to report.
2. **Everything is a must-fix.** There are no nits. If something is worth
   saying, it's worth fixing, because small inconsistencies are how the
   codebase drifts.
3. **Investigate, don't guess.** When unsure whether a rule applies, read the
   surrounding code or the closest existing step. A wrong finding costs a
   whole review round.
4. **Match the codebase.** A pattern that differs from how the neighbouring
   steps do the same thing is wrong, even if it works.
5. **Be specific.** Give file:line, name the helper, key or file to use
   instead, and say exactly what to change.
6. **Prose is held to the same rules.** Docs, notebook markdown and CHANGELOG
   that show `humanizer` tells (staged contrasts, one-line closers, inflated
   claims, bold labels everywhere) or go past `AGENTS.md` §10's length limits
   get a must-fix.

## Procedure

1. **Context.**
   - `gh pr view <n> --json title,body,files,headRefName,baseRefName,comments,reviews`
   - `gh pr diff <n>` saved to a file in the scratchpad
   - the dossier
   - the previous `<!-- murano-review -->` comments and the author's numbered
     replies to them
   - maintainer comments: check each commenter with
     `gh api repos/{owner}/{repo}/collaborators/<login>/permission`. Triage or
     higher counts as a maintainer. Note whether each comment is addressed at
     HEAD.
2. **Checks.** `gh pr checkout <n>`, then run every command in `AGENTS.md` §12
   and `gh pr checks <n>`. Each failure is a finding. Also run
   `.claude/skills/murano-source/scripts/repo_check.py` on the reference repo
   and compare its verdict and SHA with the attribution block.
3. **Deep review.** If you have the Agent tool, spawn these four reviewers in
   parallel (tier: mid). Each gets the diff path, the checkout path,
   `AGENTS.md`, the dossier and its own focus, and reports every candidate
   problem with file:line and evidence. Without the Agent tool, make the four
   passes yourself.
   - **Correctness:** does the code compute the dossier's algorithm (step by
     step against the equations), including edge cases (padding, batch of 1,
     per-head, GPT-2 tuple outputs, dtype and device)? Diff the computation
     against the reference code at the pinned SHA. Every difference needs a
     `# Differs from ...` comment at the line with its reason, and a matching
     bullet under "Differences from the reference code" in the PR body. A
     missing or unexplained one is a must-fix.
   - **Tests:** check against `murano-test`. Does each oracle have a test?
     Are there shape-only "mechanism" tests, missing error paths, or Hub
     downloads? Then run one full mutation sweep, not a sample, so every gap
     surfaces in this round instead of one per round:
     - Mutate every line the PR adds or changes under `src/`, one mutant at
       a time: flip each comparison (`>`/`>=`, `<`/`<=`, `==`/`!=`), zero or
       negate each arithmetic term, delete each call or assignment whose
       effect is used, and shift each index or slice bound by one.
     - After each mutant, run the PR's test module and the tests of any
       module that imports the changed one, then restore the file with
       `git checkout -- <file>`. Never commit a mutant.
     - Every mutant that survives is a gap. Group survivors into one finding
       per missing test, listing the mutated lines.
     - In round 2 and later, sweep only the lines changed since the previous
       review. Lines the earlier sweep passed are settled.
   - **Conventions and reuse:** check `AGENTS.md` §2–8. Grep for every new
     private helper and see whether an existing one does the same job. Check
     backend-only model access, key handling, docstring shape, every
     registration-checklist item, scope creep, and changes to public
     signatures. Check the size against `AGENTS.md` §1: a new step, key or
     result type where a new option value on an existing step would carry the
     instrumental differences is a must-fix.
   - **Docs, prose and attribution:** check `AGENTS.md` §10–11. Look for
     humanizer tells, notebook template compliance, whether the notebook was
     executed, the guide length, the Attribution block, NOTICE, the license
     verdict, and the CHANGELOG bullet. Run
     `python3 -I .claude/skills/murano-implement/scripts/pr_text.py check body <file>`
     on the PR body: every error it prints is one must-fix. Then read the
     visible part as a maintainer who skipped the paper would: jargon or
     symbols they can't parse, a file or test inventory, restated code
     comments, or an "Open questions" item that needs no human is a
     must-fix.
4. **Consolidate.** Flatten every candidate into `(file:line, claim)` pairs.
   One root cause or one fix makes one finding. The same line twice gets
   merged unless you can name two separate defects.
5. **Fact-check** the consolidated list, not the raw candidates. Group the
   findings by area. Each group gets its own cheap verifier (at most 3), which
   re-reads the code and returns valid / invalid / reword for each finding.
   Drop the invalid ones.
6. **Verdict.**
   - `READY_TO_MERGE` needs all of: zero findings, green checks, and every
     maintainer comment addressed.
   - Open questions don't block. A decision the PR records as an open
     question (PR body or a `<!-- murano-open-question -->` comment) with a
     provisional choice is not a finding, even when it departs from earlier
     guidance, as long as no maintainer has answered it. Check that the code
     does what the provisional choice says. Once a maintainer answers, the
     answer is a requirement. When you find a decision that needs a human and
     isn't recorded, the finding is to record it as an open question with a
     provisional choice, not to wait.
   - Anything else is `CHANGES_REQUESTED`.
   - An unexecuted notebook doesn't block the verdict when it's the only
     issue. List it under "Maintainer action before merge" instead.
7. **Post** the comment below with
   `python3 -I .claude/skills/murano-implement/scripts/pr_text.py post review <pr> <file>`
   (it refuses a finding over 45 words; shorten it, don't split it), and set
   the labels: remove `agent:needs-review`,
   add `agent:changes-requested` or `agent:ready-to-merge`. Use a comment and
   labels, not `gh pr review`, because GitHub won't let an author approve or
   request changes on their own PR.

## Comment format

```markdown
> *Generated by the murano agent loop.*
<!-- murano-review -->
## Review, round <N+1>

VERDICT: CHANGES_REQUESTED | READY_TO_MERGE

<the problem count, or "no problems found">

### Must fix
1. `src/murano/steps/x.py:120`: <problem>. Fix: <exact change>.
2. …

### AGENTS.md items failing
- §8.6 `gen_api_docs.py` MODULES is missing `murano.steps.x`
- …

### Maintainer comments not addressed
- @<login>: <what they asked>

### Maintainer action before merge
- <e.g. execute notebooks/applications/x.ipynb (the agent environment had no network)>
```

Leave out any heading that has nothing under it, except VERDICT. Each
finding is at most about 40 words: where, what is wrong, the exact fix. Give
the consequence only when it isn't obvious from the problem. The line under
VERDICT is the problem count, plus one sentence in later rounds on whether
the previous findings are fixed. No summary of the PR, no praise, no closing
remarks.

## Return

The verdict, the number of findings, and the comment URL.
