"""Matplotlib recreation of the Murano dot-grid logo."""
import matplotlib.pyplot as plt
import numpy as np

GRID = 25
CENTER = (GRID - 1) / 2
FOCUS = (GRID * 0.58, GRID * 0.42)  # slight offset, like the reference
SIGMA = 3.2                          # size of the focused "lens"

PALETTE = {
    "ink":   "#191F1F",
    "blue":  "#9BA7B6",
    "steel": "#DCE6F2",
    "faint": "#CFCFCF",
}

rng = np.random.default_rng(7)

fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
fig.patch.set_facecolor("#F9F9F6")
ax.set_facecolor("#F9F9F6")

xs, ys, sizes, colors, alphas = [], [], [], [], []
for i in range(GRID):
    for j in range(GRID):
        dx, dy = i - FOCUS[0], j - FOCUS[1]
        d2 = dx * dx + dy * dy
        w = np.exp(-d2 / (2 * SIGMA ** 2))  # gaussian focus

        if w > 0.55:
            color = PALETTE["ink"]
            size = 28 + 55 * w
            alpha = 0.9
        elif w > 0.25:
            color = rng.choice([PALETTE["ink"], PALETTE["blue"]], p=[0.6, 0.4])
            size = 14 + 30 * w
            alpha = 0.75
        elif w > 0.08:
            color = rng.choice([PALETTE["blue"], PALETTE["steel"], PALETTE["faint"]],
                               p=[0.25, 0.35, 0.4])
            size = 6 + 14 * w
            alpha = 0.55
        else:
            color = PALETTE["faint"]
            size = 2.5
            alpha = 0.35

        xs.append(i); ys.append(j)
        sizes.append(size); colors.append(color); alphas.append(alpha)

ax.scatter(xs, ys, s=sizes, c=colors, alpha=alphas,
           edgecolors="none", marker="o")

ax.set_xlim(-1, GRID)
ax.set_ylim(-1, GRID)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("logo_out.png", dpi=300, facecolor=fig.get_facecolor())


# --- clean hand-built SVG (small, sharp, works as favicon + logo) ---
def write_svg(path, viewbox=100, margin=6, bg=None):
    step = (viewbox - 2 * margin) / (GRID - 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {viewbox} {viewbox}" '
        f'width="{viewbox}" height="{viewbox}">'
    ]
    if bg:
        lines.append(f'<rect width="{viewbox}" height="{viewbox}" fill="{bg}"/>')
    # scale dot radii: matplotlib "size" is area in pt^2 ≈ π r². Map to viewbox units.
    scale = 0.085  # tuned so focal dots fill the grid cell nicely
    for x, y, s, c, a in zip(xs, ys, sizes, colors, alphas):
        cx = margin + x * step
        cy = margin + (GRID - 1 - y) * step  # flip y so output matches matplotlib
        r = max(0.35, np.sqrt(s) * scale)
        lines.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
            f'fill="{c}" fill-opacity="{a:.2f}"/>'
        )
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


write_svg("logo_out.svg")                   # transparent bg, for dark/light themes
write_svg("favicon_out.svg", bg="#F9F9F6")  # opaque bg for browser tabs
plt.show()