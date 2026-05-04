# Contributing to Murano

Thank you for your interest in contributing. This document defines our development workflow, branching model, and review conventions.

## Quick Start

1. Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the repository (or your fork)
3. Install dependencies:
   ```bash
   uv sync --frozen --all-extras --dev
   ```
4. Verify the baseline runs:
   ```bash
   uv run pytest
   uv run pyright
   ```

### Optional: local pre-commit hooks

If you want fast local feedback before pushing (instead of discovering issues in CI), install the git hooks:

```bash
uv run pre-commit install
```

This is **optional**. The same hooks (ruff, pyright, uv-lock-check, prettier) run on every PR in CI — that's the hard enforcement gate. If you prefer format-on-save in your IDE, or just letting CI catch issues, skip this step. Commits should be for saving work; validation lives at the CI gate, not mid-commit.

## Branching Model

Murano follows **GitHub Flow**.

- `main` is always the integrated, releasable state. Tagged commits on `main` are releases (e.g. `v0.1.0-alpha.1`).
- All changes go through **short-lived branches** off `main`. Aim for branches that live **days, not weeks**.
- Every change lands on `main` via Pull Request. No direct pushes.
- Release branches (`release/x.y`) are created only if we start supporting multiple maintained versions.

### Branch Naming

| Purpose      | Pattern                         | Example                        |
|--------------|---------------------------------|--------------------------------|
| Feature      | `feat/<issue>-short-desc`       | `feat/44-docs-site`            |
| Bug fix      | `fix/<issue>-short-desc`        | `fix/38-interleaved-recording` |
| Refactor     | `refactor/<issue>-short-desc`   | `refactor/37-location-list`    |
| Docs         | `docs/<issue>-short-desc`       | `docs/44-api-reference`        |
| CI / tooling | `ci/<short-desc>`               | `ci/add-pr-triggers`           |
| Chore / deps | `chore/<short-desc>`            | `chore/bump-transformers`      |

Every branch should reference an issue — in the branch name, the PR body, or both.

## Development Workflow

1. Open or claim an issue describing the change.
2. Branch off `main` using the naming convention above.
3. Develop. Local commits are free-form — they get squashed on merge.
4. Before pushing, run:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   uv run pyright
   ```
5. Push and open a Pull Request against `main`. Open as **draft** if not ready, especially for cross-area work.
6. Address review feedback. Keep the branch fresh by rebasing on `main` if it goes stale.
7. A maintainer squash-merges once CI is green and required reviews are in.

## Commit Conventions

**PR titles** follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <short description>
```

Examples:
- `feat(lenses): add activation patching cache`
- `fix(model): handle None token_pos in Location`
- `refactor(pipeline): extract Results registry`
- `docs: document the steps API`
- `ci: run checks on pull requests`

Types: `feat`, `fix`, `refactor`, `docs`, `ci`, `chore`, `build`, `test`, `style`, `perf`.

Because we squash-merge, the **PR title becomes the commit message on `main`** — that's the surface worth keeping consistent (scanning, future changelog automation).

**Local commit messages on feature branches are free-form.** They are discarded on merge; optimize for your own clarity during development.

Rule of thumb: useful, not fussy. `feat(lenses): add activation patching cache` is great. Arguing 10 minutes about whether something is `refactor` vs `chore` is not.

## Pull Request Process

1. Fill in the PR template.
2. Link the issue with `Closes #<number>` or `Refs #<number>`.
3. Keep PRs focused and reviewable. Prefer a stack of small PRs over one mega-PR.
4. Before requesting review, confirm:
   - [ ] CI is green
   - [ ] Tests added or updated for new behavior
   - [ ] Docs updated if public API changed
   - [ ] `uv.lock` regenerated if dependencies changed
5. Request review from the relevant [CODEOWNERS](.github/CODEOWNERS) — GitHub will route this automatically.
6. Respond to feedback; re-request review after updates.
7. A maintainer squash-merges once approvals and checks are in.

## Code Style

- `ruff` for linting and formatting (pre-commit + CI)
- `pyright` for type checking (pre-commit + CI)
- PEP 8; type hints on all function signatures; docstrings on public APIs
- `# type: ignore` and `# noqa` require a comment explaining why

## Testing

- Tests live in `tests/`
- New behavior requires a test (unit or integration)
- CI runs the matrix across Python 3.10–3.13
- Mark Python-version-specific tests with `@pytest.mark.skipif`

## Reproducibility

When a commit freezes a shared result — a paper figure, a reproduction target, a demo baseline — tag it:

```bash
git tag -a experiment/<name>-<date> -m "what this pins"
git push origin experiment/<name>-<date>
```

Release tags (`v*.*.*`) trigger the release workflow and publish wheels. Experiment tags are reproducibility anchors only.

## Code of Conduct

Project participation is subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
