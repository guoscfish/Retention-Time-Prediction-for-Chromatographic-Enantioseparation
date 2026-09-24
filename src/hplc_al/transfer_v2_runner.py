"""Bounded 1-seed x 5-method x 3-acquisition transfer study runner."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .acquisition import conditional_gradient_maxdet, lcmd_tp_select
from .common import atomic_json, ids_hash, metrics, read_json, sha, stable_hash, write_once, verify_files, code_hashes
from .data import load_graphs, predict
from .gradient import extract
from .protocol import RestrictedLabelStore, make_partition, role_ids, transition
from .training import fit, load_model
from .runner import assert_environment
from .transfer_v2 import (
    ACQUISITIONS,
    ACQUISITION_ROUNDS,
    BUDGETS,
    L0,
    OUTER_TRAIN,
    STUDY_NAME,
    TRAINING,
    selection_audit,
    transfer_partition,
    unit_gradient,
    verify_pre_test_freeze,
)


STUDY = Path(__file__).resolve().parents[2] / "studies/active_learning" / STUDY_NAME
SEED = 525
RANDOM_SEED = 525_900_001


def _fit_config():
    return {
        **TRAINING,
        "maximum_epochs": 500,
        "patience": 100,
        "min_delta": 0.0,
        "initialization_seed": SEED,
        "training_seed": SEED,
        "optimizer": "Adam",
        "loss": "original_complete",
        "checkpoint": "best_validation_central_MSE",
        "wall_time_limit_seconds": None,
    }


def _save_predictions(path, ids, predictions):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray(ids), predictions=np.asarray(predictions))
    tmp.replace(path)


def _fit_files(study, fit_dir, fit_record):
    files = {str((fit_dir / name).relative_to(study)): sha(fit_dir / name) for name in fit_record["files"]}
    files[str((fit_dir / "fit.json").relative_to(study))] = sha(fit_dir / "fit.json")
    return files


def _gradient_bank(study, model, graphs, outer, fit_record, directory):
    directory.mkdir(parents=True, exist_ok=True)
    input_record = {
        "checkpoint_hash": fit_record["checkpoint_hash"],
        "checkpoint_state_hash": fit_record["checkpoint_state_hash"],
        "ordered_ids": outer,
        "definition": "512D CountSketch(full-network central gradient)",
    }
    write_once(directory / "input.json", input_record)
    if (directory / "complete.json").exists():
        complete = read_json(directory / "complete.json")
        verify_files(directory, complete["files"])
        with np.load(directory / "features.npz") as saved:
            if saved["ids"].tolist() != outer:
                raise RuntimeError("gradient bank ID alignment drift")
            features = saved["features"].copy()
        return features, read_json(directory / "audit.json"), True
    features, audit = extract(model, graphs, outer, progress=lambda row: print({"stage": "gradient", "directory": str(directory), **row}, flush=True))
    with (directory / "features.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray(outer), features=features)
    (directory / "features.npz.tmp").replace(directory / "features.npz")
    atomic_json(directory / "audit.json", audit)
    atomic_json(directory / "complete.json", {
        "files": {name: sha(directory / name) for name in ["input.json", "features.npz", "audit.json"]}
    })
    return features, audit, False


def _selection(method, raw, labeled, unlabeled, outer, round_index):
    if method == "random":
        started = time.perf_counter()
        order = np.random.default_rng(RANDOM_SEED).permutation(unlabeled)
        selected = order[round_index * 32 : (round_index + 1) * 32].tolist()
        audit = {"selector_seconds": time.perf_counter() - started}
        if raw is not None:
            index = {int(row): pos for pos, row in enumerate(outer)}
            audit["geometry"] = selection_audit(raw, [index[i] for i in selected], [index[i] for i in labeled])
            audit["gradient_use"] = "diagnostic only; random selection ignores gradients"
        return selected, [], audit, None
    index = {int(row): pos for pos, row in enumerate(outer)}
    lp = [index[i] for i in labeled]
    up = [index[i] for i in unlabeled]
    matrix = raw if method.startswith("raw_") else unit_gradient(raw)
    started = time.perf_counter()
    if method.endswith("lcmd"):
        result = lcmd_tp_select(matrix[up], matrix[lp], 32)
        selected = [unlabeled[int(i)] for i in result.selected_pool_positions]
        trace = list(result.trace)
        audit = {}
    else:
        result = conditional_gradient_maxdet(matrix, lp, up, 32)
        selected = [outer[int(i)] for i in result.selected_positions]
        trace = list(result.trace)
        audit = {k: v for k, v in result.audit.items() if k != "elapsed_seconds"}
    audit["selector_seconds"] = time.perf_counter() - started
    audit["geometry"] = selection_audit(raw, [index[i] for i in selected], lp)
    audit["geometry_name"] = "raw" if method.startswith("raw_") else "unit-normalized sketched gradient"
    return selected, trace, audit, matrix


def _record_files(study, paths):
    return {str(path.relative_to(study)): sha(path) for path in paths}


def run(study=STUDY):
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    if (study / "global_pre_test_freeze.json").exists() or (study / "test_reveal.json").exists():
        raise RuntimeError("study already frozen; use read-only reporting")
    environment = assert_environment()
    duration = read_json(study / "duration_smoke/summary.json")
    frozen = read_json(study / "training_protocol_frozen.json")
    runtime_gate = read_json(study / "runtime_gate/runtime_preflight.json")
    implementation = read_json(study / "implementation_audit.json")
    if (duration["status"] != "PASS" or frozen["duration_summary_hash"] != stable_hash(duration)
            or frozen["training"] != TRAINING or runtime_gate["status"] != "PASS"
            or implementation["status"] != "PASS_PRE_AL"):
        raise RuntimeError("runtime/duration/implementation gates must pass before AL")
    if environment != duration["environment"]:
        raise RuntimeError("duration and AL runtime environments differ")
    if runtime_gate["source_hashes"] != code_hashes():
        raise RuntimeError("source changed after runtime gate")
    write_once(study / "runtime/environment.json", environment)
    partition = transfer_partition(make_partition())
    partition_path = study / "splits/partition.json"
    write_once(partition_path, partition)
    graphs = load_graphs(partition)
    valid = role_ids(partition, "validation")
    test = role_ids(partition, "test")
    l0 = role_ids(partition, "l0")
    u0 = role_ids(partition, "u0")
    outer = sorted(l0 + u0)
    if len(l0) != L0 or len(outer) != OUTER_TRAIN:
        raise RuntimeError("transfer partition count drift")
    records = {}
    # The validated duration fit is already the exact seed-525 L333 predictor.
    # Reuse it for all five methods; never train another round-0 predictor.
    shared_fit_dir = study / f"duration_smoke/seed_{SEED}"
    shared_gradient_dir = study / "runtime/shared_round_0/raw_gradient_bank"
    valid_truth = RestrictedLabelStore(partition, study / "label_access_audit.csv", "shared").reveal(valid, "validation")
    shared_store = RestrictedLabelStore(partition, study / "label_access_audit.csv", "shared")
    shared_truth = shared_store.reveal(l0, "fit")
    shared_binding = read_json(shared_fit_dir / "input.json")["binding"]
    if shared_binding["partition_hash"] != stable_hash(partition):
        raise RuntimeError("duration and AL partitions differ")
    shared_fit = fit(graphs, l0, shared_truth, valid, valid_truth, _fit_config(), shared_fit_dir, shared_binding)
    shared_model = load_model(shared_fit_dir / shared_fit["checkpoint_path"])
    shared_predictions = predict(shared_model, graphs, valid + test)
    shared_raw, shared_gradient_audit, _ = _gradient_bank(study, shared_model, graphs, outer, shared_fit, shared_gradient_dir)
    shared_unit = unit_gradient(shared_raw)
    if not (shared_gradient_dir / "unit_features.npz").exists():
        np.savez_compressed(shared_gradient_dir / "unit_features.npz", ids=np.asarray(outer), features=shared_unit)
    else:
        with np.load(shared_gradient_dir / "unit_features.npz") as saved:
            if saved["ids"].tolist() != outer or not np.array_equal(saved["features"], shared_unit):
                raise RuntimeError("unit bank must be row normalization of the shared raw bank")
    shared_unit_hash = stable_hash(shared_unit.tolist())
    initial_metric = metrics(shared_predictions[: len(valid), 1], valid_truth, float(np.std(shared_truth.astype(np.float64), ddof=0)))
    for method in ACQUISITIONS:
        store = RestrictedLabelStore(partition, study / "label_access_audit.csv", method)
        labeled, unlabeled = l0[:], u0[:]
        for round_index, budget in enumerate(BUDGETS):
            directory = study / f"runtime/{method}/round_{round_index}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "round.json").exists():
                record = read_json(directory / "round.json")
                verify_files(study, record["files"])
                if record["labeled_ids"] != labeled or record["unlabeled_ids"] != unlabeled:
                    raise RuntimeError("resumed trajectory state differs")
                records[f"{method}/{round_index}"] = {"path": str((directory / "round.json").relative_to(study))}
                if round_index < ACQUISITION_ROUNDS:
                    store.commit_selection(directory / "selection.json")
                    labeled, unlabeled = transition(labeled, unlabeled, record["selected"])
                continue
            if round_index == 0:
                fit_record, model, predictions = shared_fit, shared_model, shared_predictions
                raw, gradient_audit = shared_raw, shared_gradient_audit
                fit_dir, gradient_dir = shared_fit_dir, shared_gradient_dir
                fit_reused = True
            else:
                truth = store.reveal(labeled, "fit")
                fit_dir = directory / "fit"
                fit_record = fit(graphs, labeled, truth, valid, valid_truth, _fit_config(), fit_dir, {"study": STUDY_NAME, "round": round_index, "method": method})
                model = load_model(fit_dir / fit_record["checkpoint_path"])
                predictions = predict(model, graphs, valid + test)
                if round_index == ACQUISITION_ROUNDS:
                    gradient_dir = None
                    raw, gradient_audit = None, None
                else:
                    gradient_dir = directory / "raw_gradient_bank"
                    raw, gradient_audit, _ = _gradient_bank(study, model, graphs, outer, fit_record, gradient_dir)
                fit_reused = False
            prediction_path = directory / "predictions.npz"
            _save_predictions(prediction_path, valid + test, predictions)
            selected, trace, selection_audit_record, _ = _selection(method, raw, labeled, unlabeled, outer, round_index) if round_index < ACQUISITION_ROUNDS else ([], [], {}, None)
            if round_index < ACQUISITION_ROUNDS:
                selection = {
                    "method": method, "round": round_index,
                    "L_hash": ids_hash(labeled), "U_hash": ids_hash(unlabeled),
                    "selected": selected, "selected_hash": stable_hash(selected),
                    "trace": trace, "selection_audit": selection_audit_record,
                    "geometry": "raw" if method.startswith("raw_") else ("unit-normalized sketched gradient" if method != "random" else "none"),
                    "raw_gradient_bank_hash": gradient_audit["feature_hash"] if gradient_audit else None,
                    "unit_gradient_bank_hash": shared_unit_hash if round_index == 0 and method.startswith("unit_") else None,
                }
                atomic_json(directory / "selection.json", selection)
            val_metrics = metrics(predictions[: len(valid), 1], valid_truth, float(np.std(shared_truth.astype(np.float64), ddof=0)))
            files = _fit_files(study, fit_dir, fit_record)
            files.update(_record_files(study, [prediction_path]))
            if gradient_dir is not None:
                files.update(
                    _record_files(
                        study,
                        [gradient_dir / name for name in ["input.json", "features.npz", "audit.json", "complete.json"]]
                        + ([gradient_dir / "unit_features.npz"] if round_index == 0 else []),
                    )
                )
            if round_index < ACQUISITION_ROUNDS:
                files.update(_record_files(study, [directory / "selection.json"]))
            record = {
                "method": method, "round": round_index, "budget": budget,
                "labeled_ids": labeled, "unlabeled_ids": unlabeled,
                "L_hash": ids_hash(labeled), "U_hash": ids_hash(unlabeled),
                "selected": selected, "protocol_hash": stable_hash(read_json(study / "protocol.json")) if (study / "protocol.json").exists() else stable_hash(STUDY_NAME),
                "checkpoint_hash": fit_record["checkpoint_hash"],
                "prediction_hash": sha(prediction_path),
                "validation_metrics": val_metrics,
                "gradient_feature_hash": gradient_audit["feature_hash"] if gradient_audit else None,
                "countsketch_hash": gradient_audit["mapping_hash"] if gradient_audit else None,
                "raw_gradient_bank_shared_round_0": round_index == 0,
                "unit_normalized_after_sketch": method.startswith("unit_"),
                "fit_reused": fit_reused,
                "initialization_seed": SEED,
                "training_seed": SEED,
                "training_seconds": fit_record["seconds"],
                "gradient_seconds": gradient_audit["seconds"] if gradient_audit else 0,
                "files": files,
                "test_truth_access_count": 0,
            }
            write_once(directory / "round.json", record)
            records[f"{method}/{round_index}"] = {"path": str((directory / "round.json").relative_to(study))}
            if round_index < ACQUISITION_ROUNDS:
                store.commit_selection(directory / "selection.json")
                labeled, unlabeled = transition(labeled, unlabeled, selected)
    verify_pre_test_freeze({key: read_json(study / value["path"]) for key, value in records.items()})
    freeze = {
        "status": "FROZEN_BEFORE_TEST_TRUTH", "entries": records,
        "test_truth_access_count": 0,
        "files": {
            "protocol.json": sha(study / "protocol.json"),
            "splits/partition.json": sha(partition_path),
            **{name: sha(study / name) for name in [
                "runtime/environment.json", "runtime_gate/runtime_preflight.json",
                "training_protocol_frozen.json", "duration_smoke/summary.json",
                "implementation_audit.json", "frozen_source.zip",
            ]},
            **{value["path"]: sha(study / value["path"]) for value in records.values()},
        },
    }
    atomic_json(study / "global_pre_test_freeze.json", freeze)
    verify_runtime_freeze(study)
    # The firewall is now closed and verified; this is the only test-truth reveal.
    final_store = RestrictedLabelStore(partition, study / "label_access_audit.csv", "final_test")
    final_store.test_frozen = True
    test_truth = final_store.reveal(test, "final_test")
    np.savez_compressed(study / "test_truth.npz", ids=np.asarray(test), truth=test_truth)
    atomic_json(study / "test_reveal.json", {"status": "REVEALED_AFTER_GLOBAL_FREEZE", "test_truth_access_count": 1, "records": len(records)})
    return freeze


def verify_runtime_freeze(study=STUDY):
    """Verify every checkpoint/prediction/selection/state hash before reveal."""
    study = Path(study)
    freeze = read_json(study / "global_pre_test_freeze.json")
    if freeze["status"] != "FROZEN_BEFORE_TEST_TRUTH" or freeze["test_truth_access_count"] != 0:
        raise PermissionError("invalid global pre-test freeze")
    verify_files(study, freeze["files"])
    records = {key: read_json(study / value["path"]) for key, value in freeze["entries"].items()}
    verify_pre_test_freeze(records)
    for record in records.values():
        verify_files(study, record["files"])
    initial = [record for record in records.values() if record["round"] == 0]
    for key in ["checkpoint_hash", "gradient_feature_hash", "prediction_hash", "L_hash"]:
        if len({record[key] for record in initial}) != 1:
            raise PermissionError(f"round-0 shared artifact drift: {key}")
    return records
