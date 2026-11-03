Closes #

### Summary

<!-- What the PR adds or fixes, and what someone can now do that they couldn't before. Two or three sentences. -->

### Paper and attribution

<!-- New methods only; delete otherwise. -->

- Paper:
- Reference code (repo@sha, license, adapted / oracle only):
- Size (AGENTS.md section 1): chain / option on an existing step / preset / new step, and why

### Differences from the reference code

<!-- New methods only. One bullet per difference: what Murano does, what the reference does (file at the pinned SHA), and why. Each one also has a "# Differs from ..." comment at its line. Write "none" if there are none. -->

-

### Registration checklist

<!-- New methods only (AGENTS.md section 8). Tick each item, or write "n/a" and why. -->

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
