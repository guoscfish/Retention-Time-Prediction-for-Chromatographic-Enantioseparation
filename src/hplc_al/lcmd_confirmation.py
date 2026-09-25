"""Paired two-seed confirmation of Random versus raw-gradient LCMD."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .acquisition import lcmd_tp_select
from .common import ROOT, atomic_json, code_hashes, metrics, read_json, sha, stable_hash
from .data import load_graphs, predict
from .gradient import extract
from .protocol import RestrictedLabelStore, make_partition, role_ids, transition
from .runner import assert_environment
from .training import fit, load_model
from .transfer_v2 import BUDGETS, L0, OUTER_TRAIN, transfer_partition

STUDY = ROOT / "studies/active_learning/odh_lcmd_confirmation_v2"
SEEDS = (1525, 2525)
METHODS = ("random", "raw_gradient_lcmd")
RANDOM_SEED_BASE = 525_900_001


def _training(seed):
    return {
        "maximum_epochs": 500, "patience": 100, "min_delta": 0.0,
        "initialization_seed": seed, "training_seed": seed,
        "learning_rate": 0.001, "weight_decay": 1e-5,
        "optimizer": "Adam", "batch_size": 256,
        "shuffle": "deterministic_each_epoch", "scheduler": False,
        "loss": "original_complete", "checkpoint": "best_validation_central_MSE",
        "scratch": True, "wall_time_limit_seconds": None,
    }


def _ids_hash(ids):
    return stable_hash(sorted(map(int, ids)))


def _write_predictions(path, ids, values):
    with path.open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray(ids), predictions=np.asarray(values))


def _train(graphs, labeled, truth, valid, valid_truth, seed, directory, binding):
    return fit(graphs, labeled, truth, valid, valid_truth, _training(seed), directory, binding)


def _fit_artifacts(directory, record):
    return {str((directory / name).relative_to(STUDY)): sha(directory / name) for name in record["files"]} | {
        str((directory / "fit.json").relative_to(STUDY)): sha(directory / "fit.json")
    }


def prepare():
    environment = assert_environment()
    if (STUDY / "protocol.json").exists():
        protocol = read_json(STUDY / "protocol.json")
        if protocol["source_hashes"] != code_hashes() or protocol["environment"] != environment:
            raise RuntimeError("confirmation source or environment changed; refusing a mixed run")
        partition = read_json(STUDY / "splits/partition.json")
        return partition, protocol
    if STUDY.exists() and any(STUDY.iterdir()):
        raise RuntimeError(f"confirmation study directory has files but no frozen protocol: {STUDY}")
    partition = transfer_partition(make_partition())
    l0, u0 = role_ids(partition, "l0"), role_ids(partition, "u0")
    if len(l0) != L0 or len(l0) + len(u0) != OUTER_TRAIN:
        raise RuntimeError("fixed L333/U0 partition count drift")
    protocol = {
        "status": "FROZEN_BEFORE_TRAINING",
        "question": "Does raw-gradient LCMD improve label efficiency over Random across two paired new initialization seeds?",
        "seeds": list(SEEDS), "methods": list(METHODS), "budgets": list(BUDGETS),
        "initial_labeled_ids": l0, "candidate_ids": u0,
        "initialization_and_training_seed": "matched per seed across methods; distinct across seeds",
        "random_order": "one fixed permutation of sorted U0 per seed; three disjoint consecutive B=32 batches",
        "acquisition": "512D CountSketch of full-network central RTv gradient; raw feature geometry; B=32",
        "training": {str(seed): _training(seed) for seed in SEEDS},
        "primary_metric": "test normalized RMSE AULC over 333-429, trapezoidal area divided by 96; lower is better",
        "secondary_metrics": ["validation normalized RMSE AULC", "per-budget RMSE/MAE/R2/NRMSE", "runtime"],
        "nrmse_scale": "population standard deviation of authorized L333 RTv labels, shared by all rounds and methods within seed",
        "checkpoint": "best fixed-validation central MSE; no test-based selection",
        "test_truth": "one reveal only after all trajectories, selections, checkpoints, predictions and audits are frozen",
        "interpretation": "two-seed development confirmation; existing test cohort has historical exposure and is not independent external validation",
        "environment": environment,
        "source_hashes": code_hashes(),
        "input_hashes": {"partition_source_sha256": sha(ROOT / "artifacts/reproduction/odh_baseline_20260922/split_manifest.json"),
                         "dataset_sha256": sha(ROOT / "dataset/ODH_charity_0616.csv")},
    }
    STUDY.mkdir(parents=True, exist_ok=True)
    atomic_json(STUDY / "splits/partition.json", partition)
    atomic_json(STUDY / "protocol.json", protocol)
    atomic_json(STUDY / "runtime/environment.json", environment)
    return partition, protocol


def run():
    partition, protocol = prepare()
    valid, test = role_ids(partition, "validation"), role_ids(partition, "test")
    l0, u0 = role_ids(partition, "l0"), role_ids(partition, "u0")
    graphs = load_graphs(partition)
    entries = {}
    for seed in SEEDS:
        for method in METHODS:
            RestrictedLabelStore(partition, STUDY / "label_access_audit.csv", f"{seed}/{method}")
        validation_store = RestrictedLabelStore(partition, STUDY / "label_access_audit.csv", f"{seed}/shared")
        valid_truth = validation_store.reveal(valid, "validation")
        initial_store = RestrictedLabelStore(partition, STUDY / "label_access_audit.csv", f"{seed}/shared")
        initial_truth = initial_store.reveal(l0, "fit")
        initial_dir = STUDY / f"shared/seed_{seed}/fit"
        initial_fit = _train(graphs, l0, initial_truth, valid, valid_truth, seed, initial_dir,
                             {"study": STUDY.name, "seed": seed, "round": 0, "shared_methods": list(METHODS)})
        initial_model = load_model(initial_dir / initial_fit["checkpoint_path"])
        initial_predictions = predict(initial_model, graphs, valid + test)
        scale = float(np.std(initial_truth.astype(np.float64), ddof=0))
        order = np.random.default_rng(RANDOM_SEED_BASE + seed).permutation(u0).tolist()
        random_order_path = STUDY / f"selections/seed_{seed}/random_order.json"
        if random_order_path.exists():
            saved_order = read_json(random_order_path)
            if saved_order != {"seed": seed, "ids": order, "sha256": stable_hash(order)}:
                raise RuntimeError("frozen random order drift")
        else:
            atomic_json(random_order_path, {"seed": seed, "ids": order, "sha256": stable_hash(order)})
        raw, gradient_audit = extract(initial_model, graphs, sorted(l0 + u0))
        seed_initial = {"fit": initial_fit, "directory": initial_dir, "predictions": initial_predictions,
                        "scale": scale, "truth": initial_truth, "raw": raw, "gradient_audit": gradient_audit}
        for method in METHODS:
            labeled, unlabeled = list(l0), list(u0)
            store = RestrictedLabelStore(partition, STUDY / "label_access_audit.csv", method)
            for round_index, budget in enumerate(BUDGETS):
                key = f"{seed}/{method}/{round_index}"
                directory = STUDY / f"runtime/seed_{seed}/{method}/round_{round_index}"
                directory.mkdir(parents=True, exist_ok=True)
                round_path = directory / "round.json"
                if round_path.exists():
                    record = read_json(round_path)
                    if record["labeled_ids"] != labeled or record["unlabeled_ids"] != unlabeled:
                        raise RuntimeError("resumed trajectory state differs from frozen round")
                    for name, digest in record["files"].items():
                        if sha(STUDY / name) != digest:
                            raise RuntimeError(f"resumed artifact hash mismatch: {name}")
                    entries[f"{seed}/{method}/{round_index}"] = {
                        "path": str(round_path.relative_to(STUDY)), "sha256": sha(round_path)}
                    if round_index < len(BUDGETS) - 1:
                        store.commit_selection(directory / "selection.json")
                        labeled, unlabeled = transition(labeled, unlabeled, record["selected"])
                    continue
                started = time.perf_counter()
                if round_index == 0:
                    fit_record, fit_dir = seed_initial["fit"], seed_initial["directory"]
                    predictions, phi = seed_initial["predictions"], seed_initial["raw"]
                    gradient_record = seed_initial["gradient_audit"]
                else:
                    truth = store.reveal(labeled, "fit")
                    fit_dir = directory / "fit"
                    fit_record = _train(graphs, labeled, truth, valid, valid_truth, seed, fit_dir,
                                        {"study": STUDY.name, "seed": seed, "method": method, "round": round_index})
                    model = load_model(fit_dir / fit_record["checkpoint_path"])
                    predictions = predict(model, graphs, valid + test)
                    phi = None
                    gradient_record = None
                    if round_index < len(BUDGETS) - 1:
                        phi, gradient_record = extract(model, graphs, sorted(l0 + u0))
                prediction_path = directory / "predictions.npz"
                _write_predictions(prediction_path, valid + test, predictions)
                files = _fit_artifacts(fit_dir, fit_record) | {str(prediction_path): sha(prediction_path)}
                selected = []
                if round_index < len(BUDGETS) - 1:
                    if method == "random":
                        selected = order[round_index * 32:(round_index + 1) * 32]
                        trace, audit = [], {"selector": "fixed_random_permutation"}
                    else:
                        outer = sorted(l0 + u0)
                        positions = {sample_id: i for i, sample_id in enumerate(outer)}
                        result = lcmd_tp_select(phi[[positions[i] for i in unlabeled]],
                                                phi[[positions[i] for i in labeled]], 32)
                        selected = [unlabeled[int(i)] for i in result.selected_pool_positions]
                        trace, audit = list(result.trace), {"selector": "raw_gradient_lcmd",
                                                          "gradient": gradient_record}
                    selection = {"method": method, "seed": seed, "round": round_index,
                                 "L_hash": _ids_hash(labeled), "U_hash": _ids_hash(unlabeled),
                                 "selected": selected, "selected_hash": stable_hash(selected),
                                 "trace": trace, "audit": audit}
                    selection_path = directory / "selection.json"
                    atomic_json(selection_path, selection)
                    files[str(selection_path)] = sha(selection_path)
                record = {"method": method, "seed": seed, "round": round_index, "budget": budget,
                          "labeled_ids": labeled, "unlabeled_ids": unlabeled,
                          "L_hash": _ids_hash(labeled), "U_hash": _ids_hash(unlabeled),
                          "selected": selected, "initialization_seed": seed, "training_seed": seed,
                          "checkpoint_hash": fit_record["checkpoint_hash"],
                          "checkpoint_state_hash": fit_record["checkpoint_state_hash"],
                          "prediction_hash": sha(prediction_path), "validation_metrics": metrics(
                              predictions[:len(valid), 1], valid_truth, scale),
                          "normalization_scale": scale,
                          "training_seconds": fit_record["seconds"], "round_seconds": time.perf_counter() - started,
                          "test_truth_access_count": 0, "files": files,
                          "protocol_hash": stable_hash(protocol)}
                atomic_json(round_path, record)
                entries[key] = {"path": str(round_path.relative_to(STUDY)), "sha256": sha(round_path)}
                if selected:
                    store.commit_selection(directory / "selection.json")
                    labeled, unlabeled = transition(labeled, unlabeled, selected)
                print({"seed": seed, "method": method, "budget": budget,
                       "validation_rmse": record["validation_metrics"]["rmse"],
                       "round_seconds": record["round_seconds"]}, flush=True)
    pre_test_audit = STUDY / "pre_test_label_access_audit.csv"
    if not pre_test_audit.exists():
        shutil.copyfile(STUDY / "label_access_audit.csv", pre_test_audit)
    freeze_files = {"protocol.json": sha(STUDY / "protocol.json"),
                    "splits/partition.json": sha(STUDY / "splits/partition.json"),
                    "runtime/environment.json": sha(STUDY / "runtime/environment.json"),
                    "pre_test_label_access_audit.csv": sha(pre_test_audit)}
    for entry in entries.values():
        round_path = STUDY / entry["path"]
        freeze_files[entry["path"]] = sha(round_path)
        record = read_json(round_path)
        freeze_files.update({name: digest for name, digest in record["files"].items()})
    freeze = {"status": "FROZEN_BEFORE_TEST_TRUTH", "entries": entries,
              "test_truth_access_count": 0, "files": freeze_files}
    freeze_path = STUDY / "global_pre_test_freeze.json"
    if freeze_path.exists():
        if read_json(freeze_path) != freeze:
            raise RuntimeError("existing global freeze differs from replayed trajectories")
    else:
        atomic_json(freeze_path, freeze)
    # Verify every bound artifact before the one and only test-label reveal.
    for name, digest in freeze_files.items():
        if sha(STUDY / name) != digest:
            raise RuntimeError(f"pre-test freeze hash mismatch: {name}")
    if not (STUDY / "test_reveal.json").exists():
        final_store = RestrictedLabelStore(partition, STUDY / "label_access_audit.csv", "final_test")
        final_store.test_frozen = True
        truth = final_store.reveal(test, "final_test")
        np.savez_compressed(STUDY / "test_truth.npz", ids=np.asarray(test), truth=truth)
        atomic_json(STUDY / "test_reveal.json", {"status": "REVEALED_AFTER_GLOBAL_FREEZE",
                                                  "test_truth_access_count": 1, "rows": len(test)})
    report()


def report():
    partition = read_json(STUDY / "splits/partition.json")
    valid, test = role_ids(partition, "validation"), role_ids(partition, "test")
    with np.load(STUDY / "test_truth.npz") as saved:
        if saved["ids"].tolist() != test:
            raise RuntimeError("revealed test ID alignment drift")
        test_truth = saved["truth"].astype(np.float64)
    rows = []
    for seed in SEEDS:
        scale = read_json(STUDY / f"runtime/seed_{seed}/random/round_0/round.json")["normalization_scale"]
        valid_truth = RestrictedLabelStore(partition, STUDY / "label_access_audit.csv",
                                           f"{seed}/report_validation").reveal(valid, "validation")
        for method in METHODS:
            for round_index, budget in enumerate(BUDGETS):
                directory = STUDY / f"runtime/seed_{seed}/{method}/round_{round_index}"
                record = read_json(directory / "round.json")
                with np.load(directory / "predictions.npz") as saved:
                    if saved["ids"].tolist() != valid + test:
                        raise RuntimeError("prediction ID alignment drift")
                    predictions = saved["predictions"]
                for split, truth, offset in (("validation", valid_truth, 0), ("test", test_truth, len(valid))):
                    metric = metrics(predictions[offset:offset + len(truth), 1], truth, scale)
                    rows.append({"seed": seed, "method": method, "budget": budget,
                                 "split": split, **metric, "training_seconds": record["training_seconds"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(STUDY / "results/metrics.csv", index=False)
    aulc = []
    for (seed, method, split), group in frame.groupby(["seed", "method", "split"]):
        group = group.sort_values("budget")
        value = float(np.trapz(group.nrmse, group.budget) / (BUDGETS[-1] - BUDGETS[0]))
        aulc.append({"seed": seed, "method": method, "split": split, "normalized_nrmse_aulc": value})
    summary = pd.DataFrame(aulc)
    summary.to_csv(STUDY / "results/aulc_by_seed.csv", index=False)
    paired = summary.pivot(index=["seed", "split"], columns="method", values="normalized_nrmse_aulc").reset_index()
    paired["relative_lcmd_improvement"] = 1 - paired["raw_gradient_lcmd"] / paired["random"]
    paired.to_csv(STUDY / "results/paired_aulc.csv", index=False)
    print({"status": "COMPLETE", "results": str(STUDY / "results/paired_aulc.csv"),
           "paired_test_improvement": paired[paired.split == "test"].relative_lcmd_improvement.tolist()}, flush=True)
