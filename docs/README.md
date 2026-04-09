# Murano Docs

Documentation site for Murano, built with [Starlight](https://starlight.astro.build/).

## Setup

```bash
# From the repo root — create the Python venv once
python -m venv env
env/bin/pip install -e ".[plot]" griffe

# Install JS dependencies
cd docs
npm install
```

## Development

```bash
# 1. Regenerate API reference from docstrings (run from repo root)
env/bin/python docs/scripts/gen_api_docs.py

# 2. Start the dev server (from docs/)
cd docs && npm run dev
```

The dev server hot-reloads Markdown changes. Re-run step 1 whenever you change a Python docstring.

## Writing content

| What | Where |
|---|---|
| Landing page | `src/content/docs/index.mdx` |
| Getting started | `src/content/docs/docs/getting-started/` |
| Guides / tutorials | `src/content/docs/docs/guides/` |
| Reproductions guide | `src/content/docs/docs/reproductions/` |
| API reference | auto-generated — edit Python docstrings instead |

New pages are picked up automatically. To add a page to the sidebar, add an entry in `astro.config.mjs`.

## API reference generation

`scripts/gen_api_docs.py` uses [griffe](https://mkdocstrings.github.io/griffe/) to parse Python docstrings and writes Markdown files to `src/content/docs/docs/reference/`. That folder is gitignored — it must be regenerated before each build.

To add a new module to the reference, append it to the `MODULES` list in the script:

```python
("murano.steps.my_new_step", "steps/my-new-step"),
```

## Build

```bash
# From docs/
npm run build   # output goes to dist/
npm run preview # preview the built site locally
```