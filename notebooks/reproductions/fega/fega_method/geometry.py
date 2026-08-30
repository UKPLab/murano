"""Dense CPU geometry metrics used by FEGA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GeometryMetrics:
    """Complete point-estimate geometry record for one FEGA feature."""

    n_total: int
    n_valid: int
    skipped_nonfinite: int
    skipped_zero_norm: int
    c_ray: float | None
    r2: float | None
    eigenvalues: NDArray[np.float64]
    s_span: dict[int, float | None]
    u_span: dict[int, float | None]
    d_span: dict[int, float | None]
    b_axis: float | None
    r_span_ent: float | None
    r_span_pr: float | None
    centered_eigenvalues: NDArray[np.float64]
    e_res: float | None
    s_res: dict[int, float | None]
    r_ctr_ent: float | None
    r_ctr_pr: float | None


def compute_geometry_metrics(
    vectors: ArrayLike,
    gram: ArrayLike,
    *,
    k_values: Sequence[int] = (1, 2, 3, 4, 8),
    residual_k_values: Sequence[int] = (1, 2, 3, 4),
    eps: float = 1.0e-12,
) -> GeometryMetrics:
    """Compute source-equivalent FEGA metrics from residual-space rows.

    Rows are filtered using their norm under ``gram``. The retained dual kernel
    is then evaluated in float64, which is the native dense CPU analysis path.
    """
    # Validate shapes before constructing the quadratic dual kernel.
    rows = np.asarray(vectors)
    metric = np.asarray(gram)
    span_ks, residual_ks = _validate_inputs(
        rows, metric, k_values, residual_k_values, eps
    )

    # Preserve source row order while retaining explicit invalid-row counts.
    rows64 = rows.astype(np.float64, copy=False)
    gram64 = metric.astype(np.float64, copy=False)
    valid: list[NDArray[np.float64]] = []
    skipped_nonfinite = 0
    skipped_zero_norm = 0
    for row in rows64:
        if not np.isfinite(row).all():
            skipped_nonfinite += 1
            continue
        norm_sq = float(row @ gram64 @ row)
        norm_sq = 0.0 if -1.0e-7 < norm_sq < 0.0 else norm_sq
        norm = math.sqrt(norm_sq) if norm_sq >= 0.0 else math.nan
        if not math.isfinite(norm):
            skipped_nonfinite += 1
        elif norm <= eps:
            skipped_zero_norm += 1
        else:
            valid.append(row)

    # Pass the exact retained kernel and counts to the shared spectral formulas.
    retained = (
        np.stack(valid) if valid else np.empty((0, rows64.shape[1]), dtype=np.float64)
    )
    kernel = retained @ gram64 @ retained.T
    return geometry_metrics_from_kernel(
        kernel,
        ambient_dim=rows64.shape[1],
        k_values=span_ks,
        residual_k_values=residual_ks,
        eps=eps,
        n_total=rows64.shape[0],
        skipped_nonfinite=skipped_nonfinite,
        skipped_zero_norm=skipped_zero_norm,
    )


def geometry_metrics_from_kernel(
    kernel: ArrayLike,
    *,
    ambient_dim: int | None = None,
    k_values: Sequence[int] = (1, 2, 3, 4, 8),
    residual_k_values: Sequence[int] = (1, 2, 3, 4),
    eps: float = 1.0e-12,
    n_total: int | None = None,
    skipped_nonfinite: int = 0,
    skipped_zero_norm: int = 0,
) -> GeometryMetrics:
    """Compute FEGA metrics from a compact pre-normalized dual kernel.

    The fixture path assumes the kernel diagonal already represents unit rows,
    matching the source effect-direction artifact used by FEGA reporting.
    """
    # Validate the compact representation and requested spectral cutoffs.
    kernel64 = np.asarray(kernel, dtype=np.float64)
    span_ks, residual_ks = _validate_kernel(
        kernel64, ambient_dim, k_values, residual_k_values, eps
    )
    n_valid = int(kernel64.shape[0])
    ambient = n_valid if ambient_dim is None else int(ambient_dim)
    total_rows = n_valid if n_total is None else int(n_total)
    counts = {
        "n_total": total_rows,
        "n_valid": n_valid,
        "skipped_nonfinite": int(skipped_nonfinite),
        "skipped_zero_norm": int(skipped_zero_norm),
    }
    if n_valid == 0:
        return _empty_metrics(counts, span_ks, residual_ks)

    # Symmetrize only roundoff and retain the complete descending dual spectrum.
    symmetric = (kernel64 + kernel64.T) / 2.0
    eigenvalues, eigenvectors = _descending_eigh(symmetric, "span")
    total = float(eigenvalues.sum())
    denominator = total + eps

    # Compute ray concentration and every configured uncentered span diagnostic.
    summed_norm_sq = float(symmetric.sum())
    r2 = summed_norm_sq / float(n_valid * n_valid)
    c_ray = (
        (summed_norm_sq - float(n_valid)) / float(n_valid * (n_valid - 1))
        if n_valid >= 2
        else None
    )
    s_span: dict[int, float | None] = {}
    u_span: dict[int, float | None] = {}
    d_span: dict[int, float | None] = {}
    for k in span_ks:
        s_span[k] = float(eigenvalues[: min(k, n_valid)].sum() / denominator)
        index = k - 1
        if index < 0 or index >= ambient:
            u_span[k] = None
            d_span[k] = None
            continue
        lambda_k = float(eigenvalues[index]) if index < n_valid else 0.0
        u_span[k] = lambda_k / denominator
        next_index = index + 1
        d_span[k] = (
            None
            if next_index >= ambient
            else (float(eigenvalues[next_index]) if next_index < n_valid else 0.0)
            / (lambda_k + eps)
        )

    # Measure sign balance on the first source sample-space principal coordinate.
    b_axis: float | None = None
    if eigenvalues.size and float(eigenvalues[0]) > eps:
        coordinates = (
            symmetric @ eigenvectors[:, 0] / math.sqrt(float(eigenvalues[0]) + eps)
        )
        b_axis = min(float(np.mean(coordinates < 0)), float(np.mean(coordinates > 0)))

    # Center the same dual kernel for residual energy and concentration metrics.
    centering = np.eye(n_valid) - np.full((n_valid, n_valid), 1.0 / n_valid)
    centered = centering @ symmetric @ centering
    centered = (centered + centered.T) / 2.0
    centered_eigenvalues, _ = _descending_eigh(centered, "centered residual")
    centered_total = float(centered_eigenvalues.sum())
    centered_denominator = centered_total + eps
    s_res: dict[int, float | None] = {
        k: float(centered_eigenvalues[: min(k, n_valid)].sum() / centered_denominator)
        for k in residual_ks
    }
    e_res = float(np.trace(centered) / (np.trace(symmetric) + eps))
    span_rank = effective_rank(eigenvalues, eps=eps)
    centered_rank = effective_rank(centered_eigenvalues, eps=eps)
    return GeometryMetrics(
        **counts,
        c_ray=None if c_ray is None else float(c_ray),
        r2=float(r2),
        eigenvalues=eigenvalues,
        s_span=s_span,
        u_span=u_span,
        d_span=d_span,
        b_axis=b_axis,
        r_span_ent=span_rank[0],
        r_span_pr=span_rank[1],
        centered_eigenvalues=centered_eigenvalues,
        e_res=e_res,
        s_res=s_res,
        r_ctr_ent=centered_rank[0],
        r_ctr_pr=centered_rank[1],
    )


def effective_rank(
    spectrum: ArrayLike, *, eps: float = 1.0e-12
) -> tuple[float | None, float | None]:
    """Return entropy and participation effective ranks for a spectrum."""
    # Apply FEGA's visible-epsilon normalization to finite nonnegative values.
    values = np.asarray(spectrum, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("spectrum must be a finite nonnegative vector")
    if values.size == 0 or float(values.max(initial=0.0)) == 0.0:
        return None, None
    probabilities = values / (float(values.sum()) + eps)
    entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
    squared_sum = float(np.sum(probabilities * probabilities))
    return float(math.exp(entropy)), (
        math.inf if squared_sum == 0 else 1.0 / squared_sum
    )


def _empty_metrics(
    counts: dict[str, int], span_ks: tuple[int, ...], residual_ks: tuple[int, ...]
) -> GeometryMetrics:
    """Return explicit unavailable metrics for an empty retained row set."""
    # Keep configured field inventories stable even when no row is usable.
    return GeometryMetrics(
        **counts,
        c_ray=None,
        r2=None,
        eigenvalues=np.empty(0, dtype=np.float64),
        s_span={k: None for k in span_ks},
        u_span={k: None for k in span_ks},
        d_span={k: None for k in span_ks},
        b_axis=None,
        r_span_ent=None,
        r_span_pr=None,
        centered_eigenvalues=np.empty(0, dtype=np.float64),
        e_res=None,
        s_res={k: None for k in residual_ks},
        r_ctr_ent=None,
        r_ctr_pr=None,
    )


def _validate_inputs(
    rows: NDArray[np.generic],
    gram: NDArray[np.generic],
    k_values: Sequence[int],
    residual_k_values: Sequence[int],
    eps: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate residual rows, their metric, and requested dimensions."""
    # Reject only malformed public inputs before scientific row filtering.
    if rows.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square matrix")
    if rows.shape[1] != gram.shape[0]:
        raise ValueError("vectors and gram must share the residual dimension")
    if not np.isfinite(gram).all():
        raise ValueError("gram must contain only finite values")
    return _validate_controls(k_values, residual_k_values, eps)


def _validate_kernel(
    kernel: NDArray[np.float64],
    ambient_dim: int | None,
    k_values: Sequence[int],
    residual_k_values: Sequence[int],
    eps: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate a compact FEGA kernel and its spectral controls."""
    # Require an exact square finite kernel; symmetrization handles only roundoff.
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix")
    if not np.isfinite(kernel).all():
        raise ValueError("kernel must contain only finite values")
    if ambient_dim is not None and int(ambient_dim) < 1:
        raise ValueError("ambient_dim must be positive")
    return _validate_controls(k_values, residual_k_values, eps)


def _validate_controls(
    k_values: Sequence[int], residual_k_values: Sequence[int], eps: float
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Canonicalize source-supported spectral dimensions and epsilon."""
    # Preserve caller order while rejecting dimensions the metric cannot define.
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("eps must be finite and positive")
    span_ks = tuple(int(k) for k in k_values)
    residual_ks = tuple(int(k) for k in residual_k_values)
    if any(k <= 0 for k in span_ks):
        raise ValueError("k_values must be positive")
    invalid = [k for k in residual_ks if k not in {1, 2, 3, 4}]
    if invalid:
        raise ValueError(f"unsupported residual k values: {invalid}")
    return span_ks, residual_ks


def _descending_eigh(
    matrix: NDArray[np.float64], name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a descending nonnegative symmetric eigendecomposition."""
    # Fail on a materially indefinite kernel and clamp numerical roundoff only.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.size and float(eigenvalues.min()) < -1.0e-5:
        raise ValueError(f"{name} kernel has a materially negative eigenvalue")
    order = np.argsort(eigenvalues)[::-1]
    return np.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]
