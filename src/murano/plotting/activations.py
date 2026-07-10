"""Plotly visualization for recorded activations.

Projects a single component's high-dimensional activations down to two
dimensions and scatters them by class, so the linear structure a probe or a
steering vector exploits becomes visible.

The dimensionality reducer is supplied by the caller rather than imported here,
which keeps scikit-learn out of this module's dependencies and leaves the choice
of method (PCA, LDA, t-SNE, UMAP, ...) open.

Requires the ``plot`` extra (install with ``pip install murano-interp[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from murano._optional import require_optional
from murano.nodes import Node
from murano.plotting.plotly_utils import CATEGORY_COLORS

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from murano.nodes import AddressLike
    from murano.steps.record import ActivationStore, LabeledActivationStore


class Reducer(Protocol):
    """Any scikit-learn-style dimensionality reducer.

    Supervised reducers such as LDA consume ``y``; unsupervised ones such as PCA
    accept and ignore it, so a single call site serves both.
    """

    def fit_transform(self, X: Any, y: Any = None, /) -> Any: ...


def _labeled_points(
    store: ActivationStore | LabeledActivationStore,
    node: Node,
    label_names: list[str] | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return one component's activations, a class label per row, and the classes.

    Args:
        store: A contrastive :class:`~murano.steps.record.ActivationStore` or a
            :class:`~murano.steps.record.LabeledActivationStore`.
        node: Canonical address of the component to read.
        label_names: Names for the integer labels of a labeled store, indexed by
            label value. Ignored for a contrastive store, which names its own
            two classes.

    Returns:
        A ``(X, y, classes)`` triple: activations ``[N, d_model]`` as float32,
        ``N`` class-name strings, and the distinct class names in a fixed order
        that does not depend on the order of the rows.

    Raises:
        ValueError: If ``label_names`` does not cover every label in the store.
    """
    from murano.steps.record import ActivationStore as _ActivationStore

    if isinstance(store, _ActivationStore):
        positive = store.positive[node]
        negative = store.negative[node]
        # .float() first: a bf16 tensor has no numpy equivalent.
        rows = np.concatenate(
            [positive.float().cpu().numpy(), negative.float().cpu().numpy()]
        )
        names = ["positive"] * len(positive) + ["negative"] * len(negative)
        return rows, names, ["positive", "negative"]

    rows = store.activations[node].float().cpu().numpy()
    labels = [int(label) for label in store.labels.cpu().numpy()]
    # Ordered by label value, never by row order: otherwise shuffling the dataset
    # would silently swap which class gets which color.
    distinct = sorted(set(labels))

    if label_names is None:
        name_of = {value: str(value) for value in distinct}
    else:
        unnamed = [v for v in distinct if not 0 <= v < len(label_names)]
        if unnamed:
            raise ValueError(
                f"label_names has {len(label_names)} entries but the store holds "
                f"label(s) {unnamed}; pass one name per label value, indexed by "
                f"that value."
            )
        name_of = {value: label_names[value] for value in distinct}

    return rows, [name_of[label] for label in labels], [name_of[v] for v in distinct]


def plot_activation_projection(
    store: ActivationStore | LabeledActivationStore,
    layer: AddressLike,
    reducer: Reducer,
    *,
    normalize: bool = True,
    label_names: list[str] | None = None,
    title: str = "Activation projection",
) -> go.Figure:
    """Scatter one component's activations in a two-dimensional projection.

    Accepts either activation store: a contrastive
    :class:`~murano.steps.record.ActivationStore` is colored positive against
    negative, while a :class:`~murano.steps.record.LabeledActivationStore` is
    colored by its integer labels. That lets a single ``Record`` feed both this
    plot and the :class:`~murano.steps.probe.Probe` step.

    A reducer yielding one component is scattered along x with the classes
    separated vertically; two or more components use the first two.

    Args:
        store: Activation store recorded at a reduced token position.
        layer: Component address to project.
        reducer: Fitted-on-call reducer, e.g. ``PCA(n_components=2)`` or
            ``LinearDiscriminantAnalysis(n_components=1)``. Class labels are
            passed to ``fit_transform`` so supervised reducers work unchanged.
        normalize: Scale each activation vector to unit L2 norm before
            reducing, so the projection reflects direction rather than
            magnitude. Zero vectors are left untouched.
        label_names: Names for the integer labels of a labeled store, indexed
            by label value. Ignored for a contrastive store.
        title: Plot title.

    Returns:
        A ``plotly.graph_objects.Figure`` with one scatter trace per class.
        Classes are colored by label value (or positive-then-negative), not by
        the order they appear in the data, so the same class keeps its color
        across runs on reordered data.

    Raises:
        ValueError: If ``store`` was recorded with ``position="none"`` or
            ``per_head=True``, which carry extra dimensions this projection
            cannot interpret, or if ``label_names`` does not cover every label
            in the store.
    """
    require_optional("plot", "plotly")
    import plotly.graph_objects as go

    # Imported here, not at module scope: murano.plotting must stay importable
    # without pulling in murano.steps (and with it the whole nnsight model
    # stack) just to draw a scatter. This is the one shared definition of what
    # counts as a reduced store.
    from murano.steps.record import _require_reduced_store

    _require_reduced_store(store, "plot_activation_projection")

    # Coerce the address once, at the public boundary, so the shorthand forms
    # stop here and everything below works with a canonical Node.
    node = Node.coerce(layer)
    rows, names, classes = _labeled_points(store, node, label_names)

    if normalize:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        # A zero row has no direction to preserve; dividing by 1 leaves it at
        # the origin instead of producing NaNs that poison the reducer.
        norms[norms == 0] = 1.0
        rows = rows / norms

    reduced = np.asarray(reducer.fit_transform(rows, names))
    if reduced.ndim == 1:
        # Some reducers hand back a flat array for a single component.
        reduced = reduced[:, None]
    one_dimensional = reduced.shape[1] == 1

    fig = go.Figure()
    for index, label in enumerate(classes):
        mask = [name == label for name in names]
        selected = reduced[mask]
        indices = [i for i, keep in enumerate(mask) if keep]
        if one_dimensional:
            # Nothing to plot against, so fan the classes out on y purely to
            # keep their points from overlapping into a single line.
            x_values, y_values = selected[:, 0], np.full(len(selected), float(index))
        else:
            x_values, y_values = selected[:, 0], selected[:, 1]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=label,
                marker={
                    "size": 8,
                    "opacity": 0.75,
                    "color": CATEGORY_COLORS[index % len(CATEGORY_COLORS)],
                },
                customdata=indices,
                hovertemplate=f"{label}<br>example %{{customdata}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{title} ({node})",
        xaxis_title="Component 1",
        yaxis_title="" if one_dimensional else "Component 2",
        template="plotly_white",
        legend_title="Class",
    )
    if one_dimensional:
        fig.update_yaxes(showticklabels=False)
    return fig
