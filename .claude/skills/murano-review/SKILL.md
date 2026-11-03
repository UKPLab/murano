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
     per-head, GPT-2 tuple outputs, dtype and device)?
   - **Tests:** check against `murano-test`. Does each oracle have a test? Can
     each mechanism test fail? Are there shape-only "mechanism" tests, missing
     error paths, or Hub downloads?
   - **Conventions and reuse:** check `AGENTS.md` §2–8. Grep for every new
     private helper and see whether an existing one does the same job. Check
     backend-only model access, key handling, docstring shape, every
     registration-checklist item, scope creep, and changes to public
     signatures.
   - **Docs, prose and attribution:** check `AGENTS.md` §10–11. Look for
     humanizer tells, notebook template compliance, whether the notebook was
     executed, the guide length, the Attribution block, NOTICE, the license
     verdict, and the CHANGELOG bullet.
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
   - Anything else is `CHANGES_REQUESTED`.
   - An unexecuted notebook doesn't block the verdict when it's the only
     issue. List it under "Maintainer action before merge" instead.
7. **Post** the comment below and set the labels: remove `agent:needs-review`,
   add `agent:changes-requested` or `agent:ready-to-merge`. Use a comment and
   labels, not `gh pr review`, because GitHub won't let an author approve or
   request changes on their own PR.

## Comment format

```markdown
> *Generated by the murano agent loop.*
<!-- murano-review -->
## Review, round <N+1>

VERDICT: CHANGES_REQUESTED | READY_TO_MERGE

<one sentence on what the PR does; then the problem count, or "no problems found">

### Must fix
1. `src/murano/steps/x.py:120`: <problem and consequence>. Fix: <exact change; name the helper/key/file>.
2. …

### AGENTS.md items failing
- §8.6 `gen_api_docs.py` MODULES is missing `murano.steps.x`
- …

### Maintainer comments not addressed
- @<login>: <what they asked>

### Maintainer action before merge
- <e.g. execute notebooks/applications/x.ipynb (the agent environment had no network)>
```

Leave out any heading that has nothing under it, except VERDICT.

## Return

The verdict, the number of findings, and the comment URL.
