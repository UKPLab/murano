"""Render the executed notebooks into Starlight pages.

Run from the repo root:
    python docs/scripts/gen_notebook_docs.py

Outputs Markdown to docs/src/content/docs/docs/notebooks/ and figure images to
docs/public/notebook-figures/. Both are generated, gitignored, and rebuilt on
every docs deploy, exactly like the API reference.

The notebooks are committed with their outputs, so nothing here runs a model.
Figures are stored only as Plotly JSON, which no Markdown renderer understands,
so each figure is rebuilt into a static PNG with kaleido (the same headless
exporter the `plot` extra pins for Slurm nodes).
"""

from __future__ import annotations

import base64
import json
import re
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
NOTEBOOKS = REPO_ROOT / "notebooks"
OUT = REPO_ROOT / "docs" / "src" / "content" / "docs" / "docs" / "notebooks"
FIGURES = REPO_ROOT / "docs" / "public" / "notebook-figures"
FIGURE_URL = "/murano/notebook-figures"
GITHUB = "https://github.com/UKPLab/murano/blob/main/notebooks"

# Ordered so the sidebar reads as a curriculum rather than an alphabet.
ORDER = [
    "getting_started.ipynb",
    "applications/steering.ipynb",
    "applications/probing.ipynb",
    "applications/logit_lens.ipynb",
    "applications/logit_attribution.ipynb",
    "applications/attention.ipynb",
    "applications/ablation.ipynb",
    "applications/activation_patching.ipynb",
    "applications/circuit_discovery.ipynb",
    "applications/metrics.ipynb",
    "applications/custom_pipeline.ipynb",
    "applications/weight_ablation.ipynb",
    "applications/sae_features.ipynb",
    "applications/sae_steering.ipynb",
    "applications/sae_enrichment.ipynb",
]

_MAX_OUTPUT_LINES = 40
_MAX_DESCRIPTION = 150


def _title_and_description(cells: list[dict]) -> tuple[str, str]:
    """Take the page title from the H1 and the description from the paragraph under it."""
    header = "".join(cells[0]["source"])
    title = header.splitlines()[0].lstrip("# ").strip()
    body = header.split("\n", 1)[1].strip()
    first_paragraph = body.split("\n\n", 1)[0].replace("\n", " ").strip()
    description = re.sub(r"[*`]", "", first_paragraph)
    if len(description) > _MAX_DESCRIPTION:
        # Cut at a word boundary; a description sliced mid-word reads as a bug.
        description = description[:_MAX_DESCRIPTION].rsplit(" ", 1)[0] + "..."
    return title, description


def _stream_text(cell: dict) -> str:
    text = "".join(
        "".join(out.get("text", ""))
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream" and out.get("name") == "stdout"
    )
    for out in cell.get("outputs", []):
        if out.get("output_type") == "execute_result":
            text += "".join(out.get("data", {}).get("text/plain", ""))
    return text.rstrip()


def _write_figures(cell: dict, slug: str, counter: list[int]) -> list[str]:
    """Write each figure in a cell as a PNG; return the image paths.

    A Plotly figure lives in the notebook only as its JSON spec, which no Markdown
    renderer understands, so it is rebuilt through kaleido. A notebook that already
    rasterized its figures carries them as base64 ``image/png`` instead, and those
    are written straight out: dropping them would leave the page silently figureless.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    paths = []
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        spec = data.get("application/vnd.plotly.v1+json")
        png = data.get("image/png")
        if not spec and not png:
            continue

        counter[0] += 1
        name = f"{slug}-{counter[0]}.png"
        if spec:
            # The mime bundle carries a "config" key that go.Figure does not accept,
            # so rebuild from data and layout only.
            figure = go.Figure(data=spec.get("data", []), layout=spec.get("layout", {}))
            with warnings.catch_warnings():
                # kaleido<1 ships a self-contained Chromium and is the only exporter
                # that works headless; its deprecation notice is not actionable here.
                warnings.simplefilter("ignore", DeprecationWarning)
                pio.write_image(
                    figure, FIGURES / name, format="png", width=900, scale=2
                )
        else:
            payload = png if isinstance(png, str) else "".join(png)
            (FIGURES / name).write_bytes(base64.b64decode(payload))
        paths.append(f"{FIGURE_URL}/{name}")
    return paths


def _render(path: Path, order: int) -> str:
    notebook = json.loads(path.read_text())
    cells = notebook["cells"]
    title, description = _title_and_description(cells)
    relative = path.relative_to(NOTEBOOKS).as_posix()
    slug = path.stem

    # The header cell becomes the frontmatter plus the intro block: title, then
    # everything under it (overview, questions, outline, model, extras).
    intro = "".join(cells[0]["source"]).split("\n", 1)[1].strip()

    lines = [
        "---",
        # json.dumps yields a double-quoted scalar, which YAML accepts and which
        # survives the colons these titles contain.
        f"title: {json.dumps(title)}",
        f"description: {json.dumps(description)}",
        "sidebar:",
        f"  order: {order}",
        "---",
        "",
        ":::note",
        f"Generated from [`notebooks/{relative}`]({GITHUB}/{relative}), which you "
        "can run yourself. The outputs below are the ones it produced.",
        ":::",
        "",
        intro,
        "",
    ]

    counter = [0]
    for cell in cells[1:]:
        if cell["cell_type"] == "markdown":
            lines += ["".join(cell["source"]), ""]
            continue

        source = "".join(cell["source"]).strip()
        if source:
            lines += ["```python", source, "```", ""]

        text = _stream_text(cell)
        if text:
            shown = text.splitlines()
            if len(shown) > _MAX_OUTPUT_LINES:
                dropped = len(shown) - _MAX_OUTPUT_LINES
                shown = shown[:_MAX_OUTPUT_LINES] + [f"... ({dropped} more lines)"]
            lines += ["```text", *shown, "```", ""]

        for image in _write_figures(cell, slug, counter):
            lines += [f"![{title} figure {counter[0]}]({image})", ""]

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError as exc:  # pragma: no cover - deploy-time guard
        raise SystemExit(
            "gen_notebook_docs needs the plot extra: uv sync --all-extras"
        ) from exc

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    missing = [name for name in ORDER if not (NOTEBOOKS / name).exists()]
    if missing:
        raise SystemExit(f"listed in ORDER but not on disk: {missing}")

    found = {
        p.relative_to(NOTEBOOKS).as_posix()
        for p in [NOTEBOOKS / "getting_started.ipynb"]
        + sorted((NOTEBOOKS / "applications").glob("*.ipynb"))
    }
    unlisted = found - set(ORDER)
    if unlisted:
        raise SystemExit(f"notebooks missing from ORDER: {sorted(unlisted)}")

    for position, name in enumerate(ORDER, 1):
        page = _render(NOTEBOOKS / name, order=position)
        (OUT / f"{Path(name).stem}.md").write_text(page)
        print(f"  {name} -> {OUT.relative_to(REPO_ROOT)}/{Path(name).stem}.md")

    figures = len(list(FIGURES.glob("*.png")))
    print(f"\n{len(ORDER)} notebook pages, {figures} figures")


if __name__ == "__main__":
    main()
