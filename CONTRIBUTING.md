# Contributing

Thank you for your interest in contributing to the murano framework! This document provides guidelines and instructions for contributing.

## Development Setup

1. Make sure you have Python 3.10+ installed
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
1. Fork the repository
1. Clone your fork: `git clone https://github.com/YOUR-USERNAME/murano.git`
1. Install dependencies:

   ```bash
   uv sync --frozen --all-extras --dev
   ```

Local pre-commit hooks are optional. If you want fast local feedback, run `uv run pre-commit install` after the sync above. CI runs the same checks (ruff, pyright, prettier, uv-lock-check) on every PR, so installing locally only saves you a round-trip.

## Branch and Commit Conventions

Pre-1.0, the workflow is intentionally lightweight:

- **Branch from `main`, PR back to `main`.** Release branches (`v1.1.x`) will exist starting at 1.0.
- **Direct push to `main`** is fine for low-risk changes (docs, config, small fixes by a maintainer).
- **Open a PR** for substantive changes: anything touching the `Step` protocol, public API, architecture, or anything you'd want a second pair of eyes on. Outside contributors should always PR.
- **Conventional-commit prefixes** are encouraged: `feat:`, `fix:`, `docs:`, `chore:`, `ci:`, `refactor:`, `test:`. Not enforced by tooling.

### Common gotcha: `uv.lock`

Any change to `pyproject.toml` that affects dependency resolution (i.e., adding/removing/bumping a dep, changing extras, or changing `requires-python`) must regenerate `uv.lock` in the same commit:

```bash
unset UV_RESOLUTION
uv lock
```

The `uv-lock-check` pre-commit hook fails CI if `uv.lock` and `pyproject.toml` are out of sync.

## Development Workflow

1. Choose the correct branch for your changes:
   - For bug fixes to a released version: use the latest release branch (e.g. v1.1.x for 1.1.3)
   - For new features: use the main branch (which will become the next minor/major version)
   - If unsure, ask in an issue first

1. Create a new branch from your chosen base branch

1. Make your changes

1. Ensure tests pass:

   ```bash
   uv run pytest
   ```

1. Run type checking:

   ```bash
   uv run pyright
   ```

1. Run linting:

   ```bash
   uv run ruff check .
   uv run ruff format .
   ```

1. Submit a pull request to the same branch you branched from

## Code Style

- We use `ruff` for linting and formatting
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for public APIs

## Adding a feature with its own dependency

The base install stays lean: it carries only what every workflow needs. Anything
feature-specific ships as an extra so users install per use case. To add a new
application that needs a heavyweight library:

1. Add an extra in `pyproject.toml` under `[project.optional-dependencies]`, and
   include it in the `all` extra.
2. Register the extra's top-level import names in `murano._optional.EXTRA_IMPORTS`
   (the key must match the extra name).
3. Import the library lazily at its point of use, calling
   `require_optional("<extra>")` immediately before the import so a missing
   dependency raises a clear, actionable error.
4. Run `uv lock` and commit the updated `uv.lock`.

No bespoke try/except guards: every optional import goes through
`require_optional` so error messages stay uniform.

## Pull Request Process

1. Update documentation as needed
1. Add tests for new functionality
1. Ensure CI passes
1. Maintainers will review your code
1. Address review feedback

## Code of Conduct

ToDo: adopt and link a Code of Conduct. We ask contributors to be respectful and constructive in the meantime.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0, the license this project is released under.
