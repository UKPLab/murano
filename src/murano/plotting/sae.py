"""Plotly visualizations for single SAE features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from math import log1p
from typing import TYPE_CHECKING, Any

from torch import (  # pyright: ignore[reportPrivateImportUsage]
    Tensor,
    argsort,  # pyright: ignore[reportPrivateImportUsage]
    as_tensor,  # pyright: ignore[reportPrivateImportUsage]
    bool as torch_bool,  # pyright: ignore[reportPrivateImportUsage]
    isfinite,  # pyright: ignore[reportPrivateImportUsage]
    ones_like,  # pyright: ignore[reportPrivateImportUsage]
    topk,  # pyright: ignore[reportPrivateImportUsage]
)

from murano._optional import require_optional

if TYPE_CHECKING:
    import plotly.graph_objects as go

_INK = "#17345b"
_PAPER = "#ffffff"
_GRID = "#d9e2ef"
_HEADER = "#eef3f8"
_NEG = "#ff6b6b"
_NEG_LIGHT = "#ffd6d6"
_POS = "#10b981"
_POS_LIGHT = "#bdf4df"


def _as_2d_tensor(name: str, value: Any) -> Tensor:
    """Return ``value`` as a detached 2-D float tensor."""
    # Normalize common tensor-like inputs without keeping autograd state.
    tensor = value.detach() if isinstance(value, Tensor) else as_tensor(value)
    tensor = tensor.float().cpu()
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(tensor.shape)}")
    return tensor


def _feature_decoder(
    decoder: Any, feature_id: int, unembedding_dims: tuple[int, int]
) -> Tensor:
    """Select one decoder vector whose width matches the unembedding."""
    # Match decoder orientation against the unembedding before selecting.
    matrix = _as_2d_tensor("decoder", decoder)
    if feature_id < 0:
        raise ValueError(f"feature_id must be >= 0, got {feature_id}")
    if feature_id < matrix.shape[0] and matrix.shape[1] in unembedding_dims:
        return matrix[feature_id]
    if feature_id < matrix.shape[1] and matrix.shape[0] in unembedding_dims:
        return matrix[:, feature_id]
    raise ValueError(
        "feature_id or decoder orientation is incompatible with unembedding; "
        f"got feature_id {feature_id}, decoder {tuple(matrix.shape)}, "
        f"unembedding {unembedding_dims}"
    )


def _logit_effects(decoder: Any, unembedding: Any, feature_id: int) -> Tensor:
    """Compute raw vocabulary effects for one feature with ``d_f @ W_U`` semantics."""
    # Match whichever unembedding orientation the caller supplied.
    w_u = _as_2d_tensor("unembedding", unembedding)
    unembedding_dims = (int(w_u.shape[0]), int(w_u.shape[1]))
    d_feature = _feature_decoder(decoder, feature_id, unembedding_dims)
    if d_feature.numel() == w_u.shape[0]:
        return d_feature @ w_u
    if d_feature.numel() == w_u.shape[1]:
        return w_u @ d_feature
    raise ValueError(
        "decoder feature width must match one unembedding dimension; "
        f"got feature width {d_feature.numel()} and unembedding {tuple(w_u.shape)}"
    )


def _token_labels(
    vocab_size: int,
    token_labels: Sequence[str] | None,
    token_ids: Sequence[int] | None,
) -> list[str]:
    """Build display labels aligned to the vocabulary dimension."""
    # Validate caller-provided display metadata at the public boundary.
    if token_labels is not None and len(token_labels) != vocab_size:
        raise ValueError(
            f"token_labels must have length {vocab_size}, got {len(token_labels)}"
        )
    if token_ids is not None and len(token_ids) != vocab_size:
        raise ValueError(
            f"token_ids must have length {vocab_size}, got {len(token_ids)}"
        )
    if token_labels is not None:
        return [str(label) for label in token_labels]
    if token_ids is not None:
        return [str(token_id) for token_id in token_ids]
    return [str(i) for i in range(vocab_size)]


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from either a mapping row or an object row."""
    # Keep activation rows agnostic without inventing an adapter type.
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _decode_token(
    token_id: int,
    token_labels: Sequence[str] | None,
    decode_token: Any,
) -> str:
    """Return a display label for one token id."""
    # Prefer the caller's decoder, then aligned labels, then the id itself.
    if decode_token is not None:
        return str(decode_token(int(token_id)))
    if token_labels is not None:
        return str(token_labels[int(token_id)])
    return str(int(token_id))


def _activation_examples(
    activations: Any,
    token_ids: Any,
    feature_id: int | None,
    attention_mask: Any,
    token_labels: Sequence[str] | None,
    decode_token: Any,
    bos_token_id: int | None,
    num_examples: int,
) -> list[dict[str, Any]]:
    """Prepare highest-activating token rows from raw activation arrays."""
    # Normalize token and activation arrays without importing Murano artifacts.
    acts = as_tensor(activations)
    tokens = as_tensor(token_ids).cpu()
    if acts.dim() == 3:
        if feature_id is None:
            raise ValueError(
                "feature_id is required when activations are [N, seq, features]"
            )
        if feature_id < 0 or feature_id >= acts.shape[-1]:
            raise ValueError(
                f"feature_id {feature_id} is out of range for activations {tuple(acts.shape)}"
            )
        acts = acts[..., feature_id]
    acts = acts.float().cpu()
    if acts.dim() != 2:
        raise ValueError(
            f"activations must be [N, seq] or [N, seq, features], got {tuple(acts.shape)}"
        )
    if tokens.shape != acts.shape:
        raise ValueError(
            f"token_ids must match activation shape {tuple(acts.shape)}, got {tuple(tokens.shape)}"
        )

    # Build the validity mask from optional padding and BOS metadata.
    if attention_mask is None:
        mask = ones_like(acts, dtype=torch_bool)
    else:
        mask = as_tensor(attention_mask).bool().cpu()
        if mask.shape != acts.shape:
            raise ValueError(
                f"attention_mask must match activation shape {tuple(acts.shape)}, got {tuple(mask.shape)}"
            )
    if bos_token_id is not None:
        mask = mask & (tokens != int(bos_token_id))

    # Sort examples by their strongest valid token activation.
    masked = acts.masked_fill(~mask, float("-inf"))
    max_values, max_indices = masked.max(dim=1)
    valid_rows = isfinite(max_values)
    if not bool(valid_rows.any()):
        raise ValueError("no valid token positions remain after masking")
    order = argsort(max_values.masked_fill(~valid_rows, float("-inf")), descending=True)

    # Decode each selected row into the same prepared-row contract.
    rows: list[dict[str, Any]] = []
    for n in order[:num_examples].tolist():
        if not bool(valid_rows[n]):
            continue
        keep = mask[n].tolist()
        row_tokens = tokens[n].tolist()
        labels = [
            _decode_token(int(token_id), token_labels, decode_token)
            for token_id, ok in zip(row_tokens, keep, strict=True)
            if ok
        ]
        values = [
            float(value) for value, ok in zip(acts[n].tolist(), keep, strict=True) if ok
        ]
        local_max = max(range(len(values)), key=values.__getitem__)
        rows.append(
            {
                "tokens": labels,
                "activations": values,
                "max_activation": float(max_values[n]),
                "max_token": _decode_token(
                    int(tokens[n, int(max_indices[n])]), token_labels, decode_token
                ),
                "max_activation_token_index": local_max,
            }
        )
    return rows


def _mint(value: float, max_value: float) -> str:
    """Return a mint background color scaled by log activation intensity."""
    # Clamp negative activations to the neutral color.
    if max_value <= 0 or value <= 0:
        return _PAPER
    alpha = min(1.0, max(0.0, log1p(float(value)) / log1p(max_value)))
    red = int(255 - 202 * alpha)
    green = int(255 - 44 * alpha)
    blue = int(255 - 122 * alpha)
    return f"rgb({red},{green},{blue})"


def _token_text(token: str) -> str:
    """Return the visible token label before HTML escaping."""
    # Strip tokenizer spacing while keeping explicit whitespace tokens visible.
    text = str(token).replace("\n", "\\n").replace("\t", "\\t")
    return text.strip() or repr(text)


def _display_token(token: str) -> str:
    """Return an escaped token label that is readable inside a Plotly annotation."""
    # Escape only at render time; layout widths use visible text, not HTML entities.
    return escape(_token_text(token))


def _token_width(token: str) -> float:
    """Estimate a token box width from its visible text length."""
    # Use a monospace approximation because Plotly does not expose text measurement.
    if not str(token).strip():
        return 2.4
    return max(1.8, min(18.0, 0.62 * len(_token_text(token)) + 1.2))


def plot_sae_feature_logit_effects(
    feature_id: int,
    decoder: Any,
    unembedding: Any,
    token_labels: Sequence[str] | None = None,
    token_ids: Sequence[int] | None = None,
    num_tokens: int = 10,
    bins: int = 100,
    title: str | None = None,
) -> go.Figure:
    """Plot the strongest vocabulary logit effects and the full effect histogram.

    Args:
        feature_id: SAE feature index to visualize.
        decoder: SAE decoder matrix, normally ``[n_features, d_model]``.
        unembedding: Raw model unembedding, either ``[d_model, vocab]`` or
            ``[vocab, d_model]``.
        token_labels: Optional vocabulary labels aligned to the logit effects.
        token_ids: Optional vocabulary token ids aligned to the logit effects.
        num_tokens: Number of positive and negative tokens to show.
        bins: Number of histogram bins.
        title: Optional figure title.

    Returns:
        A Plotly figure containing a positive/negative token table and a
        histogram over every vocabulary logit effect.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Validate simple numeric controls before computing the figure.
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    effects = _logit_effects(decoder, unembedding, feature_id)
    labels = _token_labels(effects.numel(), token_labels, token_ids)
    k = min(num_tokens, effects.numel())
    pos_vals, pos_idx = topk(effects, k=k)
    neg_vals, neg_idx = topk(-effects, k=k)

    # Render the shared effect vector as signed token columns and split histogram bars.
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "table"}, {"type": "histogram"}]],
        column_widths=[0.46, 0.54],
        subplot_titles=(
            "Negative / positive token effects",
            "Vocabulary effect distribution",
        ),
    )
    fig.add_trace(
        go.Table(
            header={
                "values": ["Negative token", "Effect", "Positive token", "Effect"],
                "fill_color": _HEADER,
                "align": "left",
                "font": {"color": _INK, "size": 12},
            },
            cells={
                "values": [
                    [labels[int(i)] for i in neg_idx],
                    [f"{-float(v):.4g}" for v in neg_vals],
                    [labels[int(i)] for i in pos_idx],
                    [f"{float(v):.4g}" for v in pos_vals],
                ],
                "fill_color": [_NEG_LIGHT, _PAPER, _POS_LIGHT, _PAPER],
                "align": "left",
                "font": {"color": _INK, "size": 12},
            },
        ),
        row=1,
        col=1,
    )
    x_min, x_max = float(effects.min()), float(effects.max())
    bin_size = max((x_max - x_min) / bins, 1e-9)
    fig.add_trace(
        go.Histogram(
            x=effects[effects < 0].tolist(),
            xbins={"start": x_min, "end": 0, "size": bin_size},
            marker_color=_NEG,
            opacity=0.9,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=effects[effects >= 0].tolist(),
            xbins={"start": 0, "end": x_max, "size": bin_size},
            marker_color=_POS,
            opacity=0.9,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=title or f"SAE feature {feature_id} logit effects",
        bargap=0.02,
        showlegend=False,
        barmode="overlay",
        template="plotly_white",
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        font={"color": _INK},
        margin={"l": 40, "r": 24, "t": 72, "b": 48},
    )
    fig.update_xaxes(title_text="raw logit effect", row=1, col=2)
    fig.update_yaxes(title_text="token count", row=1, col=2)
    return fig


def plot_sae_token_activations(
    examples: Sequence[Any] | None = None,
    *,
    activations: Any | None = None,
    token_ids: Any | None = None,
    feature_id: int | None = None,
    attention_mask: Any | None = None,
    token_labels: Sequence[str] | None = None,
    decode_token: Any | None = None,
    bos_token_id: int | None = None,
    num_examples: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Plot token activations for one selected SAE feature.

    Args:
        examples: Optional prepared rows with ``tokens`` and ``activations`` plus optional
            ``max_activation``, ``max_token``, and
            ``max_activation_token_index`` fields.
        activations: Optional raw activation array shaped ``[N, seq]`` or
            ``[N, seq, features]``.
        token_ids: Token id array aligned with raw activations.
        feature_id: Feature index required for ``[N, seq, features]`` activations.
        attention_mask: Optional mask aligned with raw activations.
        token_labels: Optional vocabulary labels for decoding token ids.
        decode_token: Optional callable taking one token id and returning a label.
        bos_token_id: Optional BOS token id to hide from raw activation rows.
        num_examples: Number of highest-activating examples to render.
        title: Optional figure title.

    Returns:
        A Plotly figure whose token backgrounds encode activation strength.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go

    # Validate the public row contract once, then keep rendering simple.
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")
    if examples is None:
        if activations is None or token_ids is None:
            raise ValueError(
                "provide either prepared examples or activations with token_ids"
            )
        examples = _activation_examples(
            activations=activations,
            token_ids=token_ids,
            feature_id=feature_id,
            attention_mask=attention_mask,
            token_labels=token_labels,
            decode_token=decode_token,
            bos_token_id=bos_token_id,
            num_examples=num_examples,
        )
    if not examples:
        raise ValueError("examples must contain at least one row")

    # Normalize rows and compute per-example maxima for sorting and labels.
    rows = []
    for row in examples:
        tokens = [str(token) for token in _row_get(row, "tokens", [])]
        row_activations = [float(value) for value in _row_get(row, "activations", [])]
        if not tokens:
            raise ValueError("each example must include at least one token")
        if len(tokens) != len(row_activations):
            raise ValueError("each example must have aligned tokens and activations")
        max_activation = _row_get(row, "max_activation")
        max_value = float(
            max(row_activations) if max_activation is None else max_activation
        )
        max_index = _row_get(row, "max_activation_token_index")
        if max_index is None:
            max_index = max(
                range(len(row_activations)), key=row_activations.__getitem__
            )
        max_token = _row_get(row, "max_token", tokens[int(max_index)])
        rows.append(
            {
                "tokens": tokens,
                "activations": row_activations,
                "max_activation": max_value,
                "max_index": int(max_index),
                "max_token": str(max_token),
            }
        )

    # Sort by strongest activation and crop each row around its strongest token.
    rows = sorted(rows, key=lambda item: item["max_activation"], reverse=True)[
        :num_examples
    ]
    global_max = max(row["max_activation"] for row in rows)

    # Draw row bands plus dynamic token boxes; fixed Plotly table cells cannot do this.
    fig = go.Figure()
    x_max, token_x0, token_x1, row_gap = 112.0, 10.5, 111.0, 1.48
    center_x, gap = (token_x0 + token_x1) / 2, 0.65
    header_y = len(rows) * row_gap + 0.65
    mono = {
        "color": _INK,
        "family": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        "size": 12,
    }
    fig.add_shape(
        type="rect",
        x0=0,
        x1=x_max,
        y0=header_y - 0.38,
        y1=header_y + 0.38,
        fillcolor=_HEADER,
        line={"width": 0},
    )
    fig.add_annotation(
        x=0.55,
        y=header_y,
        text="<b>TOP</b>",
        showarrow=False,
        xanchor="left",
        font={"color": _INK, "size": 11},
    )
    fig.add_annotation(
        x=3.2,
        y=header_y,
        text="<b>ACTIVATIONS</b>",
        showarrow=False,
        xanchor="left",
        bgcolor=_POS,
        borderpad=4,
        font={"color": _INK, "size": 11},
    )

    for row_index, row in enumerate(rows):
        y = header_y - (row_index + 1) * row_gap
        # Grow the visible window outward from the max-activation token by
        # budgeting box widths, since Plotly exposes no text-measurement API.
        all_tokens = row["tokens"]
        all_values = row["activations"]
        all_widths = [_token_width(token) for token in all_tokens]
        start = row["max_index"]
        end = start + 1
        left_budget = center_x - token_x0
        right_budget = token_x1 - center_x
        left_used = all_widths[start] / 2
        right_used = all_widths[start] / 2
        while True:
            options = []
            if start > 0 and left_used + gap + all_widths[start - 1] <= left_budget:
                options.append("left")
            if (
                end < len(all_tokens)
                and right_used + gap + all_widths[end] <= right_budget
            ):
                options.append("right")
            if not options:
                break
            if "left" in options and (
                "right" not in options or left_used <= right_used
            ):
                start -= 1
                left_used += gap + all_widths[start]
            else:
                right_used += gap + all_widths[end]
                end += 1

        lane_budget = token_x1 - token_x0
        span_used = left_used + right_used
        if start == 0:
            while (
                end < len(all_tokens)
                and span_used + gap + all_widths[end] <= lane_budget
            ):
                span_used += gap + all_widths[end]
                end += 1
        if end == len(all_tokens):
            while start > 0 and span_used + gap + all_widths[start - 1] <= lane_budget:
                start -= 1
                span_used += gap + all_widths[start]

        tokens = list(all_tokens[start:end])
        values = list(all_values[start:end])
        token_widths = list(all_widths[start:end])
        max_i = row["max_index"] - start
        ellipsis_width = _token_width("...")
        if start > 0 and span_used + gap + ellipsis_width <= lane_budget:
            tokens.insert(0, "...")
            values.insert(0, 0.0)
            token_widths.insert(0, ellipsis_width)
            max_i += 1
            span_used += gap + ellipsis_width
        if end < len(all_tokens) and span_used + gap + ellipsis_width <= lane_budget:
            tokens.append("...")
            values.append(0.0)
            token_widths.append(ellipsis_width)

        fig.add_shape(
            type="line",
            x0=0,
            x1=x_max,
            y0=y - 0.74,
            y1=y - 0.74,
            line={"color": _GRID, "width": 1},
        )
        max_label = _token_text(row["max_token"])
        if len(max_label) > 12:
            max_label = max_label[:9] + "..."
        fig.add_annotation(
            x=5.2,
            y=y + 0.23,
            text=f"<b>{escape(max_label)}</b>",
            showarrow=False,
            xanchor="center",
            align="center",
            bgcolor=_HEADER,
            borderpad=5,
            font=mono,
        )
        fig.add_annotation(
            x=5.2,
            y=y - 0.33,
            text=f"<b>{row['max_activation']:.4g}</b>",
            showarrow=False,
            xanchor="center",
            align="center",
            font={"color": _POS, "size": 12},
        )

        left_edges = [0.0] * len(tokens)
        left_edges[0] = token_x0
        for i in range(1, len(tokens)):
            left_edges[i] = left_edges[i - 1] + token_widths[i - 1] + gap

        for token, value, width, left in zip(
            tokens, values, token_widths, left_edges, strict=True
        ):
            if value > 0:
                fig.add_shape(
                    type="rect",
                    x0=left,
                    x1=left + width,
                    y0=y - 0.27,
                    y1=y + 0.31,
                    fillcolor=_mint(value, global_max),
                    line={"width": 0},
                )
            fig.add_annotation(
                x=left + 0.35,
                y=y,
                text=_display_token(token),
                showarrow=False,
                xanchor="left",
                align="left",
                font=mono,
            )

    fig.update_layout(
        title=title or "SAE token activations",
        template="plotly_white",
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        margin={"l": 8, "r": 8, "t": 56, "b": 8},
        width=1300,
        height=max(240, 70 + 52 * len(rows)),
    )
    fig.update_xaxes(visible=False, range=[0, x_max], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.2, header_y + 0.6], fixedrange=True)
    return fig
