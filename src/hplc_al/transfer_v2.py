"""Preregistered HPLC gradient transfer-study contract (independent study)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deterministic_shuffle import assert_epoch_coverage, batch_sizes, epoch_order


STUDY_NAME = "odh_gradient_al_transfer_v2"
TRAINING = {
    "batch_size": 256,
    "maximum_epochs": 500,
    "patience": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "scratch": True,
    "fresh_adam_every_round": True,
    "shuffle": "deterministic_each_epoch",
    "scheduler": False,
    "checkpoint": "validation_only",
}
L0 = 333
BUDGETS = (333, 365, 397, 429)
ACQUISITIONS = (
    "random",
    "raw_gradient_lcmd",
    "unit_gradient_lcmd",
    "raw_gradient_maxdet",
    "unit_gradient_maxdet",
)
ACQUISITION_ROUNDS = 3
OUTER_TRAIN = 4447
CC_MATCHED_LABELS = 333
L0_SEED = 104_802


@dataclass(frozen=True)
class GradientSemantics:
    definition: str = "grad_theta f_central_RTv(x)"
    sketch_dimension: int = 512
    normalization: str = "row-wise L2 after CountSketch; zero rows retained"


GRADIENT = GradientSemantics()
FORBIDDEN_METHODS = (
    "center_width_lcmd",
    "q10_q90_width_selector",
    "kernel_ivr",
    "hybrid",
    "ensemble_uncertainty",
    "latent_fusion",
)


def training_config(initialization_seed: int = 525, training_seed: int = 525) -> dict:
    """Materialize the registered scratch-training contract for every round."""
    return {
        **TRAINING,
        "initialization_seed": int(initialization_seed),
        "training_seed": int(training_seed),
        "loss": "original_q10_pinball_central_MSE_q90_pinball_order_deadtime",
        "model": "original HPLC QGeoGNN",
        "scheduler": False,
        "warm_start": False,
    }


def label_firewall_keys():
    """All records that must be frozen before a single test-truth reveal."""
    return {f"{method}/{round_index}" for method in ACQUISITIONS for round_index in range(4)}


def verify_pre_test_freeze(entries, test_truth_access_count=0):
    """Require all 20 method/budget records before test truth can be unlocked."""
    if test_truth_access_count != 0:
        raise PermissionError("test truth was accessed before the global freeze")
    if set(entries) != label_firewall_keys():
        raise PermissionError("all five methods x four budgets must be frozen")
    for key, record in entries.items():
        method, round_text = key.split("/")
        round_index = int(round_text)
        if record.get("method") != method or record.get("round") != round_index:
            raise PermissionError("frozen trajectory identity mismatch")
        if record.get("budget") != BUDGETS[round_index]:
            raise PermissionError("frozen absolute budget mismatch")
        if record.get("test_truth_access_count", 0) != 0:
            raise PermissionError("round record accessed test truth")
    return True


def unit_gradient(phi_raw):
    """Normalize CountSketch rows while preserving exact zero rows."""
    phi = np.asarray(phi_raw, dtype=np.float64)
    if phi.ndim != 2 or not np.isfinite(phi).all():
        raise ValueError("phi_raw must be a finite matrix")
    norms = np.linalg.norm(phi, axis=1)
    return (phi / np.maximum(norms, 1e-12)[:, None]).astype(phi.dtype, copy=False)


def selection_audit(phi_raw, selected_positions, labeled_positions):
    """Compute label-free geometry fields required by the transfer report."""
    raw = np.asarray(phi_raw, dtype=np.float64)
    selected = np.asarray(selected_positions, dtype=np.int64)
    labeled = np.asarray(labeled_positions, dtype=np.int64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("phi_raw must be a finite matrix")
    if any(len(np.unique(values)) != len(values) for values in (selected, labeled)):
        raise ValueError("selected and labeled positions must be unique")
    if np.any(selected < 0) or np.any(selected >= len(raw)):
        raise ValueError("selected position outside feature matrix")
    if np.any(labeled < 0) or np.any(labeled >= len(raw)):
        raise ValueError("labeled position outside feature matrix")
    norms = np.linalg.norm(raw, axis=1)
    selected_norms = norms[selected]
    pool_norms = norms
    centered = raw[selected] - raw[selected].mean(axis=0, keepdims=True)
    effective_rank = float(np.linalg.matrix_rank(centered)) if len(selected) > 1 else 0.0
    nearest = np.linalg.norm(raw[selected, None, :] - raw[None, labeled, :], axis=2).min(axis=1)
    pairwise = np.linalg.norm(raw[selected, None, :] - raw[None, selected, :], axis=2)
    upper = pairwise[np.triu_indices(len(selected), 1)] if len(selected) > 1 else np.array([0.0])
    return {
        "raw_norm": {"min": float(norms.min()), "median": float(np.median(norms)), "max": float(norms.max())},
        "selected_norm_percentile": float(np.mean(pool_norms[:, None] <= selected_norms[None, :], axis=0).mean() * 100),
        "top_decile_selected_fraction": float(np.mean(selected_norms >= np.quantile(pool_norms, 0.9))),
        "effective_rank": effective_rank,
        "nearest_L_distance": float(np.mean(nearest)),
        "batch_pairwise_diversity": float(np.mean(upper)),
        "zero_gradient_rows": int(np.count_nonzero(norms == 0)),
    }


def validate_budget_protocol():
    if L0 != CC_MATCHED_LABELS or BUDGETS != (333, 365, 397, 429):
        raise AssertionError("absolute-label matched budget protocol drift")
    if len(ACQUISITIONS) != 5 or ACQUISITION_ROUNDS != 3:
        raise AssertionError("registered acquisition protocol drift")
    return True


def transfer_l0_ids(outer_train_ids) -> list[int]:
    """Make one frozen 333-row absolute-label-matched initial set."""
    outer = np.asarray(sorted(int(i) for i in outer_train_ids), dtype=np.int64)
    if len(outer) != OUTER_TRAIN or len(np.unique(outer)) != len(outer):
        raise ValueError("outer train IDs must be the 4,447 unique baseline rows")
    return sorted(np.random.default_rng(L0_SEED).choice(outer, L0, replace=False).tolist())


def transfer_partition(base_partition) -> dict:
    """Derive the new L333/U0 roles without changing historical study files."""
    rows = [dict(row) for row in base_partition["rows"]]
    outer = [r["sample_index"] for r in rows if r["role"] in {"l0", "u0"}]
    selected = set(transfer_l0_ids(outer))
    for row in rows:
        if row["role"] in {"l0", "u0"}:
            row["role"] = "l0" if row["sample_index"] in selected else "u0"
    return {**base_partition, "rows": rows, "l0_seed": L0_SEED, "transfer_study": STUDY_NAME}


def training_batch_sizes() -> list[int]:
    return batch_sizes(L0, TRAINING["batch_size"])


def validate_epoch_alignment(labeled_ids, truth, training_seed, epoch, graph_ids):
    """Validate graph IDs and separately stored truth after one permutation."""
    order = epoch_order(labeled_ids, truth, training_seed, epoch)
    graph_ids = np.asarray(graph_ids)
    if not np.array_equal(graph_ids, order.ids):
        raise AssertionError("graph IDs and truth permutation are misaligned")
    assert_epoch_coverage(labeled_ids, order.ids)
    return order
