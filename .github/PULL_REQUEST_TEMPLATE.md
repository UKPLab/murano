Closes #

<!-- One to three plain sentences: what a user can now do and why it matters. Name the method, not its internals. Never list files or tests. -->

### Open questions

<!-- At most three. Only what a human must decide or check. One line each, a full plain sentence: "<question> Provisional: <what the PR does>, because <why>." "None." if empty. -->

<details><summary>Details for the reviewer</summary>

### Differences from the reference code

<!-- New methods only; delete otherwise. One line each, under 25 words: what Murano does differently and why. The detail lives in the "# Differs from" comment at that line. "None." if none. -->

### Paper and attribution

<!-- New methods only; delete otherwise. -->

- Paper:
- Reference code (repo@sha, license, adapted / oracle only):
- Size (AGENTS.md section 1): chain / option on an existing step / preset / new step

### Registration checklist (AGENTS.md section 8)

<!-- New methods only. Tick each true item. An item that does not apply stays unticked with "n/a: <reason>" on the same line. -->

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

### Checks run

- [ ] `uv run ruff format --check . && uv run ruff check .`
- [ ] `uv run pyright src`
- [ ] `uv run pytest`

</details>
