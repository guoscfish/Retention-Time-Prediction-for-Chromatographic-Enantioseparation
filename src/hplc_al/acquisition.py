"""Pure LCMD/MaxDet kernels adapted from companion e28daef7.

LCMD excludes empty clusters during zero-mass ties; no CC target/model logic.
Actual upstream file hashes are recorded in the preflight provenance JSON.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import solve_triangular


@dataclass(frozen=True)
class LCMDResult:
    """Selected pool positions and an auditable selection trace."""

    selected_pool_positions: np.ndarray
    initial_nearest_center: np.ndarray
    initial_nearest_sq_distance: np.ndarray
    trace: tuple[dict[str, float | int], ...]


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a stable Euclidean squared-distance matrix."""

    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    values = (
        np.square(left64).sum(axis=1, keepdims=True)
        + np.square(right64).sum(axis=1)[None, :]
        - 2.0 * left64 @ right64.T
    )
    return np.maximum(values, 0.0)


def lcmd_tp_select(
    pool_features: np.ndarray,
    train_features: np.ndarray,
    batch_size: int,
) -> LCMDResult:
    """Select a batch with corrected LCMD-TP semantics.

    Existing training rows are installed as centers before the first pool
    selection.  At each step the cluster with the largest sum of squared
    nearest-center distances is chosen, then its farthest point becomes the
    next center.  Ties are resolved by the stable input order, matching
    ``argmax`` behavior in the reference implementation.
    """

    pool = np.asarray(pool_features, dtype=np.float64)
    train = np.asarray(train_features, dtype=np.float64)
    if pool.ndim != 2 or train.ndim != 2 or pool.shape[1] != train.shape[1]:
        raise ValueError("pool/train features must be aligned two-dimensional arrays")
    if not len(train):
        raise ValueError("TP mode requires at least one existing training center")
    if not 0 < int(batch_size) <= len(pool):
        raise ValueError("batch_size must be in [1, len(pool)]")
    if not np.isfinite(pool).all() or not np.isfinite(train).all():
        raise ValueError("LCMD features must be finite")

    initial = _squared_distances(pool, train)
    closest = np.argmin(initial, axis=1).astype(np.int64)
    minimum = initial[np.arange(len(pool)), closest]
    initial_closest = closest.copy()
    initial_minimum = minimum.copy()
    available = np.ones(len(pool), dtype=bool)
    selected: list[int] = []
    trace: list[dict[str, float | int]] = []
    center_count = len(train)

    for selection_order in range(int(batch_size)):
        cluster_scores = np.bincount(
            closest[available], weights=minimum[available], minlength=center_count
        )
        occupied = np.bincount(closest[available], minlength=center_count) > 0
        cluster_scores[~occupied] = -np.inf
        cluster = int(np.argmax(cluster_scores))
        candidates = np.flatnonzero(available & (closest == cluster))
        if not len(candidates):
            raise RuntimeError("LCMD selected an empty largest cluster")
        point = int(candidates[np.argmax(minimum[candidates])])
        selected.append(point)
        trace.append(
            {
                "selection_order": selection_order,
                "pool_position": point,
                "selected_from_center": cluster,
                "largest_cluster_score": float(cluster_scores[cluster]),
                "nearest_center_sq_distance_before_selection": float(minimum[point]),
            }
        )
        available[point] = False

        distance_to_new = _squared_distances(pool, pool[point : point + 1])[:, 0]
        improved = available & (distance_to_new < minimum)
        closest[improved] = center_count
        minimum[improved] = distance_to_new[improved]
        closest[point] = center_count
        minimum[point] = 0.0
        center_count += 1

    result = np.asarray(selected, dtype=np.int64)
    if len(np.unique(result)) != int(batch_size):
        raise RuntimeError("LCMD returned duplicate pool positions")
    return LCMDResult(result, initial_closest, initial_minimum, tuple(trace))


@dataclass(frozen=True)
class MaxDetResult:
    """A deterministic greedy batch and its label-free numerical audit."""

    selected_positions: np.ndarray
    selected_candidate_positions: np.ndarray
    normalization_scale: float
    initial_marginal_gains: np.ndarray
    trace: tuple[dict[str, float | int], ...]
    audit: dict[str, float | int | bool | str]


def _positions(values: Sequence[int], *, name: str, row_count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} positions must be a nonempty one-dimensional array")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} positions must be unique")
    if np.any(result < 0) or np.any(result >= row_count):
        raise ValueError(f"{name} positions escaped the feature matrix")
    return result


def conditional_gradient_maxdet(
    features: np.ndarray,
    labeled_positions: Sequence[int],
    candidate_positions: Sequence[int],
    batch_size: int,
) -> MaxDetResult:
    """Greedily maximize conditional log determinant in stable float64.

    The supplied candidate order is canonical and resolves exact score ties.
    Feature scaling and conditioning use only the current labeled feature rows.
    """

    started = time.perf_counter()
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("features must be a finite, nonempty two-dimensional matrix")
    labeled = _positions(labeled_positions, name="labeled", row_count=len(matrix))
    candidates = _positions(candidate_positions, name="candidate", row_count=len(matrix))
    if np.intersect1d(labeled, candidates).size:
        raise ValueError("labeled and candidate positions must be disjoint")
    size = int(batch_size)
    if not 0 < size <= len(candidates):
        raise ValueError("batch_size must be in [1, candidate_count]")

    labeled_norm_sq = np.einsum("ij,ij->i", matrix[labeled], matrix[labeled])
    scale = float(np.sqrt(labeled_norm_sq.mean()))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("current labeled features must have positive finite mean squared norm")

    labeled_scaled = matrix[labeled] / scale
    candidate_scaled = matrix[candidates] / scale
    information = np.eye(matrix.shape[1], dtype=np.float64) + labeled_scaled.T @ labeled_scaled
    try:
        cholesky = np.linalg.cholesky(information)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "conditional MaxDet labeled information matrix is not positive definite"
        ) from error

    # In the A_L-whitened coordinates, subsequent greedy updates start from I.
    whitened = solve_triangular(
        cholesky,
        candidate_scaled.T,
        lower=True,
        check_finite=False,
    ).T
    precision = np.eye(matrix.shape[1], dtype=np.float64)
    leverages = np.einsum("ij,ij->i", whitened, whitened)
    if not np.isfinite(leverages).all() or np.any(leverages < -1e-12):
        raise RuntimeError("conditional MaxDet produced invalid initial leverages")
    leverages = np.maximum(leverages, 0.0)
    initial_gains = np.log1p(leverages)
    available = np.ones(len(candidates), dtype=bool)
    selected_local: list[int] = []
    trace: list[dict[str, float | int]] = []

    for selection_order in range(size):
        available_positions = np.flatnonzero(available)
        available_leverages = leverages[available_positions]
        if not np.isfinite(available_leverages).all():
            raise RuntimeError("conditional MaxDet encountered a non-finite marginal leverage")
        local = int(available_positions[np.argmax(available_leverages)])
        pivot = float(leverages[local])
        gain = float(np.log1p(pivot))
        gains = np.log1p(np.maximum(available_leverages, 0.0))
        trace.append(
            {
                "selection_order": selection_order,
                "candidate_position": local,
                "feature_position": int(candidates[local]),
                "conditional_leverage": pivot,
                "marginal_logdet_gain": gain,
                "available_count": int(len(available_positions)),
                "available_gain_min": float(gains.min()),
                "available_gain_q25": float(np.quantile(gains, 0.25)),
                "available_gain_median": float(np.quantile(gains, 0.5)),
                "available_gain_q75": float(np.quantile(gains, 0.75)),
                "available_gain_max": float(gains.max()),
            }
        )
        selected_local.append(local)
        available[local] = False

        direction = precision @ whitened[local]
        denominator = 1.0 + float(whitened[local] @ direction)
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("conditional MaxDet encountered a non-positive rank-one pivot")
        covariance = whitened @ direction
        leverages[available] -= np.square(covariance[available]) / denominator
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(leverages[available]), initial=0.0)))
        if np.any(leverages[available] < -tolerance):
            raise RuntimeError("conditional MaxDet rank-one update lost positive semidefiniteness")
        leverages = np.maximum(leverages, 0.0)
        leverages[~available] = 0.0
        precision -= np.outer(direction, direction) / denominator
        precision = 0.5 * (precision + precision.T)
        if not np.isfinite(precision).all():
            raise RuntimeError("conditional MaxDet precision update became non-finite")

    selected_candidate = np.asarray(selected_local, dtype=np.int64)
    selected = candidates[selected_candidate]
    if len(np.unique(selected)) != size or not set(selected.tolist()) <= set(candidates.tolist()):
        raise RuntimeError("conditional MaxDet returned an invalid batch")
    audit: dict[str, float | int | bool | str] = {
        "definition": "greedy conditional D-optimal logdet with current-L conditioning",
        "dtype": "float64",
        "feature_rows": int(len(matrix)),
        "feature_dimension": int(matrix.shape[1]),
        "labeled_count": int(len(labeled)),
        "candidate_count": int(len(candidates)),
        "batch_size": size,
        "normalization_scale": scale,
        "all_finite": True,
        "selected_unique": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return MaxDetResult(selected, selected_candidate, scale, initial_gains, tuple(trace), audit)
