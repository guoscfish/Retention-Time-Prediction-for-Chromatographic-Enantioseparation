"""L333 validation-only duration smoke; this module never performs acquisition."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import ROOT, atomic_json, read_json, stable_hash, write_once
from .data import load_graphs
from .protocol import RestrictedLabelStore, make_partition, role_ids
from .runner import assert_environment
from .training import fit
from .transfer_v2 import TRAINING, transfer_partition


STUDY = ROOT / "studies/active_learning/odh_gradient_al_transfer_v2"
SEEDS = (525, 1525)


def run_duration_smoke(study=STUDY) -> dict:
    """Train two registered L333 fits on training/validation labels only."""
    study = Path(study)
    environment = assert_environment()
    base = make_partition()
    partition = transfer_partition(base)
    l0 = role_ids(partition, "l0")
    valid = role_ids(partition, "validation")
    u0, test = role_ids(partition, "u0"), role_ids(partition, "test")
    if len(l0) != 333 or set(l0) & (set(u0) | set(valid) | set(test)):
        raise RuntimeError("transfer L333 role firewall failed")
    partition_path = study / "duration_smoke/partition.json"
    write_once(partition_path, partition)
    graphs = load_graphs(partition)
    store = RestrictedLabelStore(
        partition, study / "duration_smoke/label_access_audit.csv", "duration_smoke"
    )
    truth = store.reveal(l0, "fit")
    valid_truth = store.reveal(valid, "validation")
    l0_scale = float(np.std(truth.astype(np.float64), ddof=0))
    binding = dict(
        study="odh_gradient_al_transfer_v2",
        partition_hash=stable_hash(partition),
        code_contract="transfer_v2_training_and_deterministic_shuffle",
        test_ids_hash=stable_hash(test),
        U0_ids_hash=stable_hash(u0),
        test_truth_access_count=0,
    )
    records = []
    for seed in SEEDS:
        config = {
            **TRAINING,
            "maximum_epochs": 500,
            "patience": 100,
            "min_delta": 0.0,
            "initialization_seed": seed,
            "training_seed": seed,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "optimizer": "Adam",
            "loss": "original_complete",
            "checkpoint": "best_validation_central_MSE",
            "wall_time_limit_seconds": None,
        }
        tick = time.perf_counter()
        record = fit(
            graphs, l0, truth, valid, valid_truth, config,
            study / f"duration_smoke/seed_{seed}", binding,
        )
        # Bind the reporting scale and verify the expected two batches per epoch.
        if record["steps_per_epoch"] != 2 or record["observed_batch_sizes"] != [256, 77]:
            raise RuntimeError("L333 training batch geometry drift")
        if record["stopping_reason"] == "maximum_epochs" or record["best_epoch"] > 400:
            status = "INSUFFICIENT_DURATION_PROTOCOL"
        else:
            status = "PASS"
        enriched = {
            **record,
            "seed": seed,
            "best_epoch": record["best_epoch"],
            "stopped_epoch": record["stopped_epoch"],
            "best_optimizer_step": record["best_optimizer_step"],
            "best_validation_rmse": record["best_validation_metrics"]["rmse"],
            "best_validation_mae": record["best_validation_metrics"]["mae"],
            "best_validation_r2": record["best_validation_metrics"]["r2"],
            "best_validation_nrmse": record["best_validation_metrics"]["nrmse"],
            "final_validation_metrics": record["final_validation_metrics"],
            "L0_population_std": l0_scale,
            "steps_per_epoch": record["steps_per_epoch"],
            "actual_batch_sizes_per_epoch": record["observed_batch_sizes"],
            "wall_runtime_seconds": time.perf_counter() - tick,
            "duration_gate": status,
            "U0_labels_accessed": False,
            "test_labels_accessed": False,
        }
        atomic_json(study / f"duration_smoke/seed_{seed}/duration_record.json", enriched)
        records.append(enriched)
    passed = all(r["duration_gate"] == "PASS" for r in records)
    result = {
        "status": "PASS" if passed else "INSUFFICIENT_DURATION_PROTOCOL",
        "environment": environment,
        "seeds": list(SEEDS),
        "training_protocol": TRAINING,
        "L0": len(l0),
        "validation_count": len(valid),
        "steps_per_epoch": 2,
        "observed_batch_sizes": [256, 77],
        "duration_records": [f"seed_{seed}/duration_record.json" for seed in SEEDS],
        "test_truth_access_count": 0,
        "U0_truth_access_count": 0,
        "records": records,
        "long_al_started": False,
    }
    atomic_json(study / "duration_smoke/summary.json", result)
    if passed:
        atomic_json(study / "training_protocol_frozen.json", {
            "status": "FROZEN_AFTER_DURATION_SMOKE",
            "training": TRAINING,
            "seeds": list(SEEDS),
            "duration_summary_hash": stable_hash(result),
            "test_truth_access_count": 0,
        })
    return result

