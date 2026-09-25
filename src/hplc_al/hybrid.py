"""HPLC adaptation: CC ensemble Top-25% then unit h_graph k-center."""

import math

import numpy as np
from scipy.stats import rankdata

from .ensemble import uncertainty


def select(unit_latent, labeled_positions, candidate_positions, predictions,
           frozen_l333_sd, batch_size=32):
    z = np.asarray(unit_latent, dtype=np.float64)
    labeled = np.asarray(labeled_positions, dtype=np.int64)
    candidate = np.asarray(candidate_positions, dtype=np.int64)
    if z.ndim != 2 or z.shape[1] != 128 or not np.isfinite(z).all():
        raise ValueError("finite 128D unit h_graph required")
    if not np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-6):
        raise ValueError("h_graph must be unit normalized")
    if (len(set(labeled.tolist())) != len(labeled) or
            len(set(candidate.tolist())) != len(candidate) or
            set(labeled.tolist()) | set(candidate.tolist()) != set(range(len(z))) or
            set(labeled.tolist()) & set(candidate.tolist()) or
            not 0 < batch_size <= len(candidate)):
        raise ValueError("L and U must partition reference")
    raw, normalized, ranking = uncertainty(predictions, frozen_l333_sd)
    if len(raw) != len(candidate):
        raise ValueError("ensemble predictions must align with current U")
    shortlist_size = max(batch_size, math.ceil(.25 * len(candidate)))
    shortlist = ranking[:shortlist_size]
    positions = candidate[shortlist]
    nearest = np.clip(1 - z[positions] @ z[labeled].T, 0, 2).min(axis=1)
    available = np.ones(shortlist_size, dtype=bool)
    selected, trace = [], []
    for step in range(batch_size):
        # shortlist order resolves exact distance ties by uncertainty rank,
        # which itself resolves exact uncertainty ties by current U order.
        local = int(np.argmax(np.where(available, nearest, -np.inf)))
        pool_position = int(shortlist[local])
        selected.append(pool_position)
        trace.append(dict(step=step + 1, candidate_position=pool_position,
                          nearest_center_distance=float(nearest[local]),
                          uncertainty=float(raw[pool_position]),
                          uncertainty_rank=int(np.flatnonzero(ranking == pool_position)[0] + 1)))
        available[local] = False
        nearest = np.minimum(nearest, np.clip(1 - z[positions] @ z[positions[local]], 0, 2))
    if len(set(selected)) != batch_size or not set(selected) <= set(shortlist.tolist()):
        raise RuntimeError("Hybrid batch escaped shortlist or contains duplicates")
    percentiles = (rankdata(raw, method="average") - .5) / len(raw) * 100
    return dict(selected_candidate_positions=selected,
                shortlist_candidate_positions=shortlist.tolist(),
                uncertainty_ranking=ranking.tolist(), uncertainty=raw,
                normalized_uncertainty=normalized, uncertainty_percentiles=percentiles,
                trace=trace, audit=dict(shortlist_fraction=.25,
                                        shortlist_size=shortlist_size,
                                        shortlist_threshold=float(raw[shortlist[-1]]),
                                        pool_size=len(candidate), center_count=len(labeled),
                                        batch_size=batch_size))
