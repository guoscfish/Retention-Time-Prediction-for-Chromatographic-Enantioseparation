"""Label-free same-state geometry and validation-only error diagnostics."""

import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr

from hplc_al.acquisition import conditional_gradient_maxdet, lcmd_tp_select
from hplc_al.common import array_hash, atomic_json, stable_hash, state_hash
from hplc_al.data import assert_label_free, batches
from hplc_al.gradient import mapping

EPSILON = 1e-12


def unit_features(features, epsilon=EPSILON):
    features = np.asarray(features, dtype=np.float64)
    if not np.isfinite(features).all() or epsilon <= 0:
        raise ValueError("finite features and positive epsilon required")
    norms = np.linalg.norm(features, axis=1)
    return features / np.maximum(norms, epsilon)[:, None]


def gradients(model, graphs, ids):
    """Collect original central gradients and per-row norms without any targets."""
    assert_label_free(graphs)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("nonempty unique gradient IDs required")
    model.eval()
    before = state_hash(model)
    sections, buckets, signs, mapping_hash = mapping(model)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    features = np.empty((len(ids), 512), dtype=np.float32)
    rows = []
    started = time.perf_counter()
    with torch.enable_grad():
        for index, (graph, angle) in enumerate(batches(graphs, ids, 1)):
            output, _ = model(graph, angle)
            central = output[0, 1]
            values = torch.autograd.grad(central, parameters, allow_unused=True)
            full = np.zeros(len(buckets), dtype=np.float64)
            for value, section in zip(values, sections):
                if value is not None:
                    full[section["start"] : section["stop"]] = value.detach().numpy().reshape(-1)
            features[index] = np.bincount(buckets, weights=full * signs, minlength=512).astype(
                np.float32
            )
            if not np.isfinite(full).all() or not np.isfinite(features[index]).all():
                raise RuntimeError("nonfinite gradient")
            rows.append(
                dict(
                    sample_index=ids[index],
                    central_prediction=float(central.item()),
                    full_gradient_norm=float(np.linalg.norm(full)),
                    sketch_gradient_norm=float(np.linalg.norm(features[index].astype(np.float64))),
                )
            )
            if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(ids):
                print(
                    dict(
                        stage="gradient",
                        rows=index + 1,
                        total=len(ids),
                        seconds=round(time.perf_counter() - started, 2),
                    ),
                    flush=True,
                )
    if state_hash(model) != before:
        raise RuntimeError("gradient extraction mutated model or BN state")
    audit = dict(
        rows=len(ids),
        mapping_hash=mapping_hash,
        feature_hash=array_hash(features),
        ordered_ids_hash=stable_hash(ids),
        checkpoint_state_hash=before,
        state_unchanged=True,
        seconds=time.perf_counter() - started,
        parameters=sections,
        zero_full_norm_count=sum(row["full_gradient_norm"] == 0 for row in rows),
        zero_sketch_norm_count=sum(row["sketch_gradient_norm"] == 0 for row in rows),
    )
    return features, pd.DataFrame(rows), audit


def correlations(frame):
    result = []
    for norm in ["full_gradient_norm", "sketch_gradient_norm"]:
        for error in ["absolute_error", "squared_error"]:
            valid = len(frame) >= 3 and frame[norm].nunique() > 1 and frame[error].nunique() > 1
            result.append(
                dict(
                    norm=norm,
                    error=error,
                    rows=len(frame),
                    spearman=float(spearmanr(frame[norm], frame[error]).statistic)
                    if valid
                    else None,
                    pearson=float(pearsonr(frame[norm], frame[error]).statistic) if valid else None,
                    status="defined" if valid else "undefined_constant_or_insufficient_rows",
                )
            )
    return result


def selection_diagnostic(raw, ids, labeled, candidates, full_norms, directory):
    """No labels accepted; select B=32 at a fixed L0/U0 state in both geometries."""
    directory.mkdir(parents=True, exist_ok=True)
    index = {row: position for position, row in enumerate(ids)}
    train = np.array([index[row] for row in labeled])
    pool = np.array([index[row] for row in candidates])
    raw = np.asarray(raw, dtype=np.float64)
    unit = unit_features(raw)
    norms = np.linalg.norm(raw, axis=1)
    raw_percentile = (
        np.searchsorted(np.sort(norms[pool]), norms[pool], side="right") / len(pool) * 100
    )
    full_percentile = (
        np.searchsorted(np.sort(full_norms[pool]), full_norms[pool], side="right") / len(pool) * 100
    )
    rows, selections, traces = [], {}, {}
    for geometry, matrix in [("raw", raw), ("unit", unit)]:
        for method in ["lcmd", "maxdet"]:
            key = f"{geometry}_{method}"
            if method == "lcmd":
                selected = lcmd_tp_select(matrix[pool], matrix[train], 32)
                local = selected.selected_pool_positions
                trace = selected.trace
                gains = None
            else:
                selected = conditional_gradient_maxdet(matrix, train.tolist(), pool.tolist(), 32)
                local = selected.selected_candidate_positions
                trace = selected.trace
                gains = [r["marginal_logdet_gain"] for r in trace]
            positions = pool[local]
            chosen = [int(ids[position]) for position in positions]
            if len(set(chosen)) != 32 or not set(chosen) <= set(candidates):
                raise RuntimeError("invalid diagnostic batch")
            selections[key] = chosen
            traces[key] = list(trace)
            row = dict(
                geometry=geometry,
                method=method,
                selected_count=32,
                selected_sketch_norm_percentile_median=float(np.median(raw_percentile[local])),
                selected_full_norm_percentile_median=float(np.median(full_percentile[local])),
                selected_sketch_norm_median=float(np.median(norms[positions])),
                selected_top_decile_fraction=float(np.mean(raw_percentile[local] >= 90)),
                pool_zero_norm_count=int((norms[pool] == 0).sum()),
                pool_below_epsilon_count=int((norms[pool] < EPSILON).sum()),
                selected_zero_norm_count=int((norms[positions] == 0).sum()),
                maxdet_logdet_gain_sum=float(sum(gains)) if gains is not None else None,
                maxdet_gain_median=float(np.median(gains)) if gains is not None else None,
            )
            # Evaluate every batch in both common coordinate systems. Raw and unit
            # distances must not be compared as if they have the same physical scale.
            for name, coordinates in [("raw", raw), ("unit", unit)]:
                pairwise = pdist(coordinates[positions])
                nearest = cdist(coordinates[positions], coordinates[train]).min(axis=1)
                row.update(
                    {
                        f"{name}_pairwise_distance_mean": float(pairwise.mean()),
                        f"{name}_pairwise_distance_median": float(np.median(pairwise)),
                        f"{name}_nearest_L_distance_mean": float(nearest.mean()),
                        f"{name}_nearest_L_distance_median": float(np.median(nearest)),
                    }
                )
            rows.append(row)
    overlaps = {}
    for method in ["lcmd", "maxdet"]:
        left, right = set(selections[f"raw_{method}"]), set(selections[f"unit_{method}"])
        overlaps[method] = dict(
            intersection=len(left & right), jaccard=len(left & right) / len(left | right)
        )
        for row in rows:
            if row["method"] == method:
                row.update(
                    raw_unit_intersection=overlaps[method]["intersection"],
                    raw_unit_jaccard=overlaps[method]["jaccard"],
                )
    atomic_json(
        directory / "selections.json",
        dict(
            selections=selections,
            overlaps=overlaps,
            traces=traces,
            epsilon=EPSILON,
            label_reveals=0,
            L_hash=stable_hash(labeled),
            U_hash=stable_hash(candidates),
            raw_feature_hash=array_hash(raw),
            unit_feature_hash=array_hash(unit),
            percentile_definition="100 * count(U0 norm <= selected norm) / len(U0); evaluated on original raw norms",
            zero_handling="retain all rows; divide by max(norm, 1e-12); exact zeros stay zero",
            maxdet_note="gains are within-geometry diagnostics, not calibrated cross-geometry information gain",
        ),
    )
    return rows
