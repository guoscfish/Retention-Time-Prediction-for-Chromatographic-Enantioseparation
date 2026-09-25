"""CC scalar row-kernel IVR, transferred to HPLC central-output gradients."""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata


def select(features, labeled_positions, candidate_positions, batch_size=32):
    """Greedy integrated variance reduction on the fixed L0 union U0 reference.

    The caller supplies only label-free 512D CountSketch rows. Candidate order
    supplies the deterministic tie break. Prior precision and noise are both 1.
    """
    started = time.perf_counter()
    raw = np.asarray(features, dtype=np.float64)
    labeled = np.asarray(labeled_positions, dtype=np.int64)
    candidate = np.asarray(candidate_positions, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 512 or not np.isfinite(raw).all():
        raise ValueError("finite 512D gradients required")
    if (labeled.ndim != 1 or candidate.ndim != 1 or
            len(set(labeled.tolist())) != len(labeled) or
            len(set(candidate.tolist())) != len(candidate) or
            set(labeled.tolist()) | set(candidate.tolist()) != set(range(len(raw))) or
            set(labeled.tolist()) & set(candidate.tolist()) or
            not 0 < batch_size <= len(candidate)):
        raise ValueError("L and U must partition the fixed reference")
    mean_squared_norm = float(np.mean(np.einsum("ij,ij->i", raw, raw)))
    if mean_squared_norm <= 0 or not np.isfinite(mean_squared_norm):
        raise ValueError("invalid global gradient RMS")
    x = raw / np.sqrt(mean_squared_norm)
    precision = np.eye(512) + x[labeled].T @ x[labeled]
    factor = cho_factor(precision, lower=True, check_finite=True)
    covariance = cho_solve(factor, np.eye(512))
    moment = x.T @ x / len(x)
    q = x @ covariance
    h = q @ moment
    available = np.ones(len(candidate), dtype=bool)
    selected, trace = [], []
    first_scores = None
    first_denominators = None
    for step in range(batch_size):
        variance = np.einsum("ij,ij->i", q, x)
        numerator = np.einsum("ij,ij->i", q, h)
        if (not np.isfinite(variance).all() or not np.isfinite(numerator).all() or
                variance.min() < -1e-9 or numerator.min() < -1e-9):
            raise RuntimeError("IVR posterior became numerically invalid")
        denominators = 1 + np.maximum(variance[candidate], 0)
        scores = np.maximum(numerator[candidate], 0) / denominators
        if first_scores is None:
            first_scores, first_denominators = scores.copy(), denominators.copy()
        scores[~available] = -np.inf
        winner = int(np.argmax(scores))
        position = int(candidate[winner])
        before = float(variance.mean())
        reduction = float(scores[winner])
        cross = q @ x[position]
        q -= np.outer(cross, q[position].copy()) / denominators[winner]
        h -= np.outer(cross, h[position].copy()) / denominators[winner]
        after = float(np.einsum("ij,ij->", q, x) / len(x))
        if not np.isclose(before - after, reduction, rtol=1e-7, atol=1e-11):
            raise RuntimeError("IVR score and covariance update disagree")
        trace.append(dict(step=step + 1, candidate_position=winner,
                          reference_position=position, score=reduction,
                          integrated_variance_before=before,
                          integrated_variance_after=after,
                          denominator=float(denominators[winner])))
        selected.append(winner)
        available[winner] = False
    percentiles = (rankdata(first_scores, method="average") - .5) / len(candidate) * 100
    return dict(selected_candidate_positions=selected, trace=trace,
                initial_scores=first_scores, initial_score_percentiles=percentiles,
                audit=dict(reference_rows=len(x), labeled_rows=len(labeled),
                           candidate_rows=len(candidate), dimension=512,
                           global_rms=float(np.sqrt(mean_squared_norm)),
                           prior_precision=1, noise_variance=1,
                           integrated_variance_before=trace[0]["integrated_variance_before"],
                           integrated_variance_after=trace[-1]["integrated_variance_after"],
                           predicted_total_reduction=float(sum(row["score"] for row in trace)),
                           relative_predicted_reduction=float(sum(row["score"] for row in trace) / trace[0]["integrated_variance_before"]),
                           initial_top1_score=float(first_scores.max()),
                           initial_median_score=float(np.median(first_scores)),
                           initial_q90_score=float(np.quantile(first_scores, .9)),
                           precision_condition=float(np.linalg.cond(precision)),
                           posterior_condition=float(np.linalg.cond(covariance)),
                           minimum_initial_denominator=float(first_denominators.min()),
                           minimum_selected_denominator=float(min(row["denominator"] for row in trace)),
                           elapsed_seconds=time.perf_counter() - started))
