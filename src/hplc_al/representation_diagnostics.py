"""Target-free representation diagnostics and acquisition geometry."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr, pearsonr

from .coreset import cosine_distance


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("finite nonempty summary required")
    return {"mean": float(a.mean()), "min": float(a.min()),
            "q25": float(np.quantile(a, .25)), "median": float(np.median(a)),
            "q75": float(np.quantile(a, .75)), "max": float(a.max())}


def correlation(a, b, kind="spearman"):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = (spearmanr if kind == "spearman" else pearsonr)(a, b)[0]
    return float(value) if np.isfinite(value) else None


def sampled_pairs(n, count=50000, seed=525):
    rng = np.random.default_rng(seed)
    left = rng.integers(n, size=count)
    right = rng.integers(n - 1, size=count)
    right += right >= left
    return left, right


def representation_relation(raw, unit, chemistry, morgan, graph_atom_counts):
    left, right = sampled_pairs(len(raw))
    latent_dist = np.clip(1 - (unit[left] * unit[right]).sum(axis=1), 0, 2)
    norm = np.linalg.norm(raw.astype(np.float64), axis=1)
    audit = {"pair_count": len(left), "pair_seed": 525,
             "pairs_sampling": "ordered distinct endpoints with replacement",
             "latent_morgan_spearman": correlation(latent_dist, morgan[left, right]),
             "latent_norm": summary(norm), "norm_size": {}}
    for name, values in [("graph_atom_count", graph_atom_counts),
                         ("rdkit_atom_count", [r["atom_count"] for r in chemistry]),
                         ("heavy_atom_count", [r["heavy_atom_count"] for r in chemistry])]:
        audit["norm_size"][name] = {kind: correlation(norm, values, kind)
                                          for kind in ("spearman", "pearson")}
    return audit


def stability(first, second, geometry):
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or len(a) <= 50:
        raise ValueError("aligned stability bank with >50 rows required")
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ua = a / np.maximum(an, 1e-12)[:, None]
    ub = b / np.maximum(bn, 1e-12)[:, None]
    if geometry == "raw_gradient_euclidean":
        da, db = cdist(a, a), cdist(b, b)
    elif geometry == "unit_latent_cosine":
        da, db = cosine_distance(ua), cosine_distance(ub)
    else:
        raise ValueError("unknown stability geometry")
    upper = np.triu_indices(len(a), 1)
    result = dict(
        geometry=geometry, row_cosine=summary(np.sum(ua * ub, axis=1)),
        zero_rows_first=int((an == 0).sum()), zero_rows_second=int((bn == 0).sum()),
        pairwise_distance_rank_correlation=correlation(da[upper], db[upper]),
        pair_count=len(upper[0]),
    )
    if geometry == "raw_gradient_euclidean":
        result["norm_rank_spearman"] = correlation(an, bn)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    # Stable column ordering is ascending sample ID in the frozen candidate bank.
    na = np.argsort(da, axis=1, kind="stable")
    nb = np.argsort(db, axis=1, kind="stable")
    for k in (10, 50):
        result[f"knn_overlap_at_{k}"] = summary([
            len(set(x[:k]) & set(y[:k])) / k for x, y in zip(na, nb)])
    return result


def coverage(ids, labeled, unlabeled, selected, unit, morgan, chemistry, raw=None):
    """Percentiles always compare with the method's current pre-acquisition U."""
    index = {key: i for i, key in enumerate(ids)}
    lp, up, sp = ([index[i] for i in group] for group in (labeled, unlabeled, selected))
    selected_in_u = [unlabeled.index(i) for i in selected]
    nearest_latent = np.clip(1 - unit[up] @ unit[lp].T, 0, 2).min(axis=1)
    novelty = morgan[np.ix_(up, lp)].min(axis=1)
    scaffolds = {chemistry[i]["scaffold"] for i in lp}
    novel = np.array([chemistry[i]["scaffold"] not in scaffolds for i in up], dtype=float)
    values = dict(
        latent_nearest_L_cosine_distance=nearest_latent,
        morgan_nearest_L_max_tanimoto=1 - novelty, morgan_novelty=novelty,
        scaffold_novelty=novel,
        heavy_atom_count=np.array([chemistry[i]["heavy_atom_count"] for i in up]),
        molecular_weight=np.array([chemistry[i]["molecular_weight"] for i in up]),
    )
    if raw is not None:
        values["gradient_norm"] = np.linalg.norm(raw[up].astype(np.float64), axis=1)
    rows = [{"id": int(i)} for i in selected]
    result = {}
    for name, pool in values.items():
        chosen = pool[selected_in_u]
        # Midrank empirical percentile treats duplicate molecules symmetrically.
        percentiles = (rankdata(pool, method="average") - .5) / len(pool) * 100
        result[name] = {"selected": summary(chosen), "pool": summary(pool),
                        "selected_pool_percentile": summary(percentiles[selected_in_u])}
        for row, value, percentile in zip(rows, chosen, percentiles[selected_in_u]):
            row[name] = float(value)
            row[name + "_pool_percentile"] = float(percentile)
    result["novel_scaffold_fraction"] = float(novel[selected_in_u].mean())
    if raw is not None:
        norms = values["gradient_norm"]
        result["top_decile_gradient_fraction"] = float(np.mean(
            norms[selected_in_u] >= np.quantile(norms, .9)))
    upper = np.triu_indices(len(selected), 1)
    result["batch_pairwise_latent_cosine_distance"] = summary(
        np.clip(1 - unit[sp] @ unit[sp].T, 0, 2)[upper])
    result["batch_pairwise_morgan_tanimoto_distance"] = summary(morgan[np.ix_(sp, sp)][upper])
    singular = np.linalg.svd(unit[sp] - unit[sp].mean(axis=0), compute_uv=False)
    probabilities = singular[singular > 1e-12]
    probabilities /= max(probabilities.sum(), 1e-12)
    result["latent_effective_rank"] = float(np.exp(-(probabilities * np.log(probabilities)).sum())) if len(probabilities) else 0
    result["latent_effective_rank_definition"] = "exp entropy of singular values of centered selected unit latent"
    result["selected_rows"] = rows
    return result
