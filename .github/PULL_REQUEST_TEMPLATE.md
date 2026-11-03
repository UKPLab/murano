Closes #

<!-- One to three sentences: what this adds or fixes and why. A reviewer who reads only this decides whether the PR is theirs. Give the reviewer what the diff cannot; never list files or tests. -->

### Open questions

<!-- Decisions a maintainer may want to overturn, already taken provisionally so the loop never waits. One line each: "<question> Provisional: <what the PR does>, because <why>." Add one line of context only when the question needs it. Also list what a human must check: something you could not run, a change outside the PR's own new files. A maintainer answers in a PR comment. "None." if empty. -->

### Differences from the reference code

<!-- New methods only; delete otherwise. One line each: what Murano does differently and why, in under 25 words. The detail lives in the "# Differs from" comment at that line. "None." if none. -->

### Paper and attribution

<!-- New methods only; delete otherwise. -->

- Paper:
- Reference code (repo@sha, license, adapted / oracle only):
- Size (AGENTS.md section 1): chain / option on an existing step / preset / new step

<details><summary>Registration checklist (AGENTS.md section 8)</summary>

<!-- New methods only. Tick each true item. An item that does not apply stays unticked with "n/a: <reason>" on the same line. No other commentary. -->

- [ ] `steps/__init__.py` export (+ `murano/__init__.py` lazy export for headline names)
- [ ] `keys.py` constant + `tests/test_keys.py`
- [ ] `io.py` save/load + serializer registry
- [ ] `plotting/` function (+ `Plot` branch if there is a default view)
- [ ] `docs/scripts/gen_api_docs.py` MODULES
- [ ] README and `guides/pipeline.md` step tables
- [ ] `_CAUSAL_STEPS` (causal steps only)
- [ ] Notebook in `ORDER` and `notebooks/README.md`, executed with outputs
- [ ] Guide section, CHANGELOG `[Unreleased]` entry
- [ ] Attribution block in the module docstring + `NOTICE.txt`

</details>

### Checks run

- [ ] `uv run ruff format --check . && uv run ruff check .`
- [ ] `uv run pyright src`
- [ ] `uv run pytest`
