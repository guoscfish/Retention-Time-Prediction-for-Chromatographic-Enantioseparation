"""Gated ODH development trajectory; test truth is a final separate operation."""

import csv
import importlib.metadata
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .acquisition import conditional_gradient_maxdet, lcmd_tp_select
from .common import (
    BUDGETS,
    INIT_SEED,
    METHODS,
    RANDOM_SEED,
    ROOT,
    SEED,
    SKETCH_SEED,
    STUDY,
    atomic_json,
    code_hashes,
    ids_hash,
    metrics,
    read_json,
    sha,
    stable_hash,
    verify_files,
    verify_frozen_source,
    write_once,
)
from .data import load_graphs, predict
from .gradient import extract
from .protocol import (
    RestrictedLabelStore,
    make_partition,
    role_ids,
    transition,
    validate_partition,
    verify_global_freeze,
)
from .training import fit, load_model


def _require_open_study(study):
    """Completed evidence is immutable, including its original pytest gate."""
    if (Path(study) / "completion_manifest.json").exists():
        raise RuntimeError("study is complete; use verify or run pytest directly")


def verify_completed_study(study=STUDY):
    """Read-only artifact/source verification; no fitting or label-store reads."""
    study = Path(study)
    completion = read_json(study / "completion_manifest.json")
    verify_files(study, completion["files"])
    freeze = verify_global_freeze(study)
    protocol = read_json(study / "protocol.json")
    if protocol["partition_hash"] != stable_hash(checked_partition(study)):
        raise RuntimeError("frozen partition drift")
    source = verify_frozen_source(study, protocol["code"])
    return dict(
        status="VERIFIED",
        frozen_source=source,
        budget_records=len(freeze["entries"]),
        artifacts=len(completion["files"]),
    )


def training_config(maximum_epochs, patience, wall_time=None):
    """Shared scratch-training settings; only the prospective stopping caps vary."""
    return dict(
        maximum_epochs=maximum_epochs,
        patience=patience,
        min_delta=0.0,
        initialization_seed=INIT_SEED,
        learning_rate=0.001,
        weight_decay=1e-5,
        optimizer="Adam",
        batch_size=2048,
        shuffle=False,
        scheduler=False,
        loss="original_q10_pinball_central_MSE_q90_pinball_order_deadtime",
        checkpoint="strict_best_eval_central_validation_MSE_earlier_ties",
        scratch=True,
        wall_time_limit_seconds=wall_time,
    )


def prepare(study=STUDY):
    """Write immutable split identities and prospective duration-audit rules."""
    _require_open_study(study)
    study = Path(study)
    partition = make_partition()
    write_once(study / "splits/partition.json", partition)
    split_csv = study / "splits/roles.csv"
    if not split_csv.exists():
        pd.DataFrame(partition["rows"]).to_csv(split_csv, index=False)
    duration = dict(
        seed=SEED,
        initial_labels=356,
        validation_labels=247,
        stages=[training_config(300, 300, 1200), training_config(600, 600, 1800)],
        stage2_rule="only_if_stage1_best_epoch_above_200_and_no_wall_limit",
        freeze_rule="require_at_least_100_epochs_after_best; maximum_epochs=audited_stage_cap; patience=100",
        gradient_gate=dict(
            maximum_full_pool_seconds=1200,
            maximum_zero_fraction=0.05,
            maximum_duplicate_fraction=0.50,
        ),
        test_access=False,
    )
    write_once(study / "duration_audit/protocol.json", duration)
    if not (study / "decision.json").exists():
        write_once(
            study / "decision.json",
            dict(status="PENDING_PHASE_2", evidence="ENGINEERING / DEVELOPMENT EVIDENCE ONLY"),
        )
    return partition


def assert_environment():
    if not (Path(sys.prefix) / "conda-meta").is_dir():
        raise RuntimeError("this study requires its isolated conda environment")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").lower() in ("true", "1", "yes"):
        raise RuntimeError("unsafe OpenMP override is prohibited")
    torch.set_num_threads(4)
    return dict(
        python=sys.version,
        executable=sys.executable,
        prefix=sys.prefix,
        platform=platform.platform(),
        cpu_threads=4,
        packages={
            p: importlib.metadata.version(p)
            for p in ["torch", "numpy", "scipy", "torch-geometric", "rdkit", "pandas", "pytest"]
        },
    )


def tests_gate(study=STUDY):
    """Run tests before study freeze and bind the evidence to current source."""
    _require_open_study(study)
    study = Path(study)
    environment = assert_environment()
    study.mkdir(parents=True, exist_ok=True)
    # Forking a subprocess after loading Torch aborts on this macOS runtime.
    # Run pytest in this disposable CLI process and keep the complete transcript.
    import contextlib
    import io

    import pytest

    transcript = io.StringIO()
    with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
        returncode = pytest.main(
            ["-q", str(ROOT / "tests/hplc_al"), f"--junitxml={study / 'phase2_tests.xml'}"]
        )
    (study / "phase2_tests.log").write_text(transcript.getvalue())
    print(transcript.getvalue(), flush=True)
    if returncode:
        raise RuntimeError("Phase 2 test gate failed")
    import xml.etree.ElementTree as ET

    suites = list(ET.parse(study / "phase2_tests.xml").getroot().iter("testsuite"))
    totals = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites)
        for k in ["tests", "failures", "errors", "skipped"]
    }
    if totals["tests"] < 15 or any(totals[k] for k in ["failures", "errors", "skipped"]):
        raise RuntimeError("incomplete Phase 2 tests")
    atomic_json(
        study / "phase2_test_gate.json",
        dict(
            status="PASS",
            code=code_hashes(),
            totals=totals,
            environment=environment,
            report_hash=sha(study / "phase2_tests.xml"),
        ),
    )


def checked_partition(study):
    partition = read_json(Path(study) / "splits/partition.json")
    validate_partition(partition)
    if partition != make_partition():
        raise RuntimeError("partition/source drift")
    return partition


def assert_tests(study):
    gate = read_json(Path(study) / "phase2_test_gate.json")
    if gate["status"] != "PASS" or gate["code"] != code_hashes():
        raise RuntimeError("passing current-code test gate required")
    if gate["report_hash"] != sha(Path(study) / "phase2_tests.xml"):
        raise RuntimeError("test evidence drift")


def duration_audit(study=STUDY):
    """Choose training caps from L0/validation only, then check gradient feasibility."""
    _require_open_study(study)
    study = Path(study)
    assert_environment()
    assert_tests(study)
    partition = checked_partition(study)
    protocol = read_json(study / "duration_audit/protocol.json")
    graphs = load_graphs(partition)
    l0, valid = role_ids(partition, "l0"), role_ids(partition, "validation")
    store = RestrictedLabelStore(
        partition, study / "duration_audit/label_access_audit.csv", "duration_audit"
    )
    truth, valid_truth = store.reveal(l0, "fit"), store.reveal(valid, "validation")
    binding = dict(
        code=code_hashes(),
        partition=stable_hash(partition),
        duration_protocol=stable_hash(protocol),
    )
    for stage, config in enumerate(protocol["stages"]):
        directory = study / f"duration_audit/stage_{stage + 1}"
        fitted = fit(graphs, l0, truth, valid, valid_truth, config, directory, binding)
        if fitted["stopping_reason"] == "wall_time_limit":
            raise RuntimeError("duration audit resource cap reached; no smoke protocol frozen")
        if fitted["epochs_run"] - fitted["best_epoch"] >= 100:
            break
    else:
        raise RuntimeError("no stable L0 duration within preregistered audit budget")
    model = load_model(directory / fitted["checkpoint_path"])
    outer = sorted(l0 + role_ids(partition, "u0"))
    small, a = extract(model, graphs, outer[:16])
    repeat, b = extract(model, graphs, outer[:16])
    if not np.array_equal(small, repeat) or a["mapping_hash"] != b["mapping_hash"]:
        raise RuntimeError("real checkpoint gradient repeatability failed")
    features, audit = extract(
        model, graphs, outer, progress=lambda x: print(dict(stage="gradient_gate", **x), flush=True)
    )
    thresholds = protocol["gradient_gate"]
    if (
        audit["seconds"] > thresholds["maximum_full_pool_seconds"]
        or audit["sketch_norm"]["zero_count"] / len(outer) > thresholds["maximum_zero_fraction"]
        or audit["duplicate_feature_rows"] / len(outer) > thresholds["maximum_duplicate_fraction"]
    ):
        atomic_json(study / "duration_audit/gradient_gate_failed.json", audit)
        raise RuntimeError("gradient feasibility/geometry gate failed")
    atomic_json(study / "duration_audit/gradient_audit.json", audit)
    summary = dict(
        status="PASS",
        stage=stage + 1,
        fit=fitted,
        fit_path=str(directory.relative_to(study)),
        chosen_training=training_config(config["maximum_epochs"], 100),
        L0_population_std=float(np.std(truth.astype(np.float64), ddof=0)),
        gradient_seconds=audit["seconds"],
        gradient_repeatability_exact=True,
        binding=binding,
        test_truth_access_count=0,
    )
    atomic_json(study / "duration_audit/summary.json", summary)
    return summary


def freeze_protocol(study=STUDY):
    """Seal the tested code, measured training choice and data roles before AL."""
    _require_open_study(study)
    study = Path(study)
    environment = assert_environment()
    assert_tests(study)
    partition = checked_partition(study)
    audit = read_json(study / "duration_audit/summary.json")
    if audit["status"] != "PASS" or audit["binding"]["code"] != code_hashes():
        raise RuntimeError("current-code duration audit required")
    record = dict(
        seed=SEED,
        methods=list(METHODS),
        acquisition_rounds=3,
        batch_size=32,
        budgets=list(BUDGETS),
        initial_labels=356,
        outer_train=4447,
        validation=247,
        test=247,
        eligible=4942,
        experimental_roles=4941,
        unused_rounding=[4209],
        training=audit["chosen_training"],
        initialization_seed=INIT_SEED,
        random_trajectory_seed=RANDOM_SEED,
        sketch_seed=SKETCH_SEED,
        gradient="single-output eval-clamped central full-network CountSketch512; float64 accumulation, float32 storage",
        gradient_gate=read_json(study / "duration_audit/protocol.json")["gradient_gate"],
        metric_nrmse="RMSE / original_L0_population_std_ddof0",
        L0_population_std=audit["L0_population_std"],
        partial_aulc="trapezoidal error over 356..452; mean=raw/96; lower is better",
        test_barrier="all three methods x all four budgets, checkpoints, selections and predictions",
        evidence="ENGINEERING / DEVELOPMENT EVIDENCE ONLY",
        partition_hash=stable_hash(partition),
        code=code_hashes(),
        environment=environment,
        duration_audit_hash=sha(study / "duration_audit/summary.json"),
        phase2_test_hash=sha(study / "phase2_test_gate.json"),
    )
    write_once(study / "protocol.json", record)
    atomic_json(
        study / "decision.json",
        dict(status="PHASE_2_COMPLETE_SMOKE_PENDING", evidence=record["evidence"]),
    )
    return record


def assert_frozen_protocol(study):
    protocol = read_json(Path(study) / "protocol.json")
    if protocol["code"] != code_hashes() or protocol["environment"] != assert_environment():
        raise RuntimeError("frozen code/environment drift")
    if protocol["partition_hash"] != stable_hash(checked_partition(study)):
        raise RuntimeError("frozen partition drift")
    if protocol["duration_audit_hash"] != sha(Path(study) / "duration_audit/summary.json"):
        raise RuntimeError("duration evidence drift")
    if protocol["phase2_test_hash"] != sha(Path(study) / "phase2_test_gate.json"):
        raise RuntimeError("test gate drift")
    return protocol


def _gradient_cache(model, graphs, outer, directory, protocol, fit_record):
    directory.mkdir(parents=True, exist_ok=True)
    contract = dict(
        checkpoint=fit_record["checkpoint_hash"],
        state=fit_record["checkpoint_state_hash"],
        protocol=stable_hash(protocol),
        ordered_ids=outer,
    )
    write_once(directory / "input.json", contract)
    if (directory / "complete.json").exists():
        complete = read_json(directory / "complete.json")
        verify_files(directory, complete["files"])
        with np.load(directory / "features.npz") as saved:
            if saved["ids"].tolist() != outer:
                raise RuntimeError("gradient cache ID order changed")
            values = saved["features"].copy()
        return values, read_json(directory / "audit.json"), True
    values, audit = extract(
        model, graphs, outer, progress=lambda x: print(dict(stage="gradient", **x), flush=True)
    )
    gate = protocol["gradient_gate"]
    if (
        audit["sketch_norm"]["zero_count"] / len(outer) > gate["maximum_zero_fraction"]
        or audit["duplicate_feature_rows"] / len(outer) > gate["maximum_duplicate_fraction"]
    ):
        atomic_json(directory / "failed_geometry.json", audit)
        raise RuntimeError("current-model representation collapsed")
    with (directory / "features.npz.tmp").open("wb") as f:
        np.savez_compressed(f, ids=np.array(outer), features=values)
    (directory / "features.npz.tmp").replace(directory / "features.npz")
    atomic_json(directory / "audit.json", audit)
    atomic_json(
        directory / "complete.json",
        dict(files={n: sha(directory / n) for n in ["input.json", "features.npz", "audit.json"]}),
    )
    return values, audit, False


def run_trajectories(study=STUDY):
    """Run the authorized trajectories, or verify an already completed study."""
    study = Path(study)
    if (study / "completion_manifest.json").exists():
        return verify_completed_study(study)
    protocol = assert_frozen_protocol(study)
    if (study / "global_pre_test_freeze.json").exists():
        return verify_global_freeze(study)
    partition = checked_partition(study)
    graphs = load_graphs(partition)
    valid, test = role_ids(partition, "validation"), role_ids(partition, "test")
    l0, u0 = role_ids(partition, "l0"), role_ids(partition, "u0")
    outer = sorted(l0 + u0)
    random_order = np.random.default_rng(RANDOM_SEED).permutation(u0).tolist()
    records = {}
    for method in METHODS:
        store = RestrictedLabelStore(partition, study / "label_access_audit.csv", method)
        valid_truth = store.reveal(valid, "validation")
        labeled, unlabeled = l0[:], u0[:]
        for round_index, budget in enumerate(BUDGETS):
            directory = study / f"runtime/{method}/round_{round_index}"
            directory.mkdir(parents=True, exist_ok=True)
            record_path = directory / "round.json"
            if record_path.exists():
                record = read_json(record_path)
                verify_files(study, record["files"])
                if (record["L_hash"], record["U_hash"], record["protocol_hash"]) != (
                    ids_hash(labeled),
                    ids_hash(unlabeled),
                    stable_hash(protocol),
                ):
                    raise RuntimeError("completed round state drift")
                if round_index < 3:
                    store.commit_selection(directory / "selection.json")
                    labeled, unlabeled = transition(labeled, unlabeled, record["selected"])
                records[f"{method}/{round_index}"] = dict(path=str(record_path.relative_to(study)))
                continue
            truth = store.reveal(labeled, "fit")
            if len(labeled) != budget:
                raise RuntimeError("budget schedule violation")
            fit_dir = study / "runtime/shared_round_0" if round_index == 0 else directory / "fit"
            reused_fit = (fit_dir / "fit.json").exists()
            fit_record = fit(
                graphs,
                labeled,
                truth,
                valid,
                valid_truth,
                protocol["training"],
                fit_dir,
                dict(protocol=stable_hash(protocol), round=round_index),
            )
            model = load_model(fit_dir / fit_record["checkpoint_path"])
            predictions = predict(model, graphs, valid + test)
            with (directory / "predictions.npz.tmp").open("wb") as f:
                np.savez_compressed(f, ids=np.array(valid + test), predictions=predictions)
            (directory / "predictions.npz.tmp").replace(directory / "predictions.npz")
            val_metrics = metrics(
                predictions[: len(valid), 1], valid_truth, protocol["L0_population_std"]
            )
            gradient_audit, gradient_path, gradient_reused = None, None, False
            selected, trace, acquisition_seconds = [], [], 0.0
            if round_index < 3:
                if method == "random":
                    tick = time.perf_counter()
                    selected = random_order[round_index * 32 : (round_index + 1) * 32]
                    acquisition_seconds = time.perf_counter() - tick
                else:
                    gradient_path = (
                        study / "runtime/shared_round_0/gradient"
                        if round_index == 0
                        else directory / "gradient"
                    )
                    features, gradient_audit, gradient_reused = _gradient_cache(
                        model, graphs, outer, gradient_path, protocol, fit_record
                    )
                    lookup = {i: j for j, i in enumerate(outer)}
                    lp, up = [lookup[i] for i in labeled], [lookup[i] for i in unlabeled]
                    tick = time.perf_counter()
                    if method == "lcmd":
                        result = lcmd_tp_select(features[up], features[lp], 32)
                        selected = [unlabeled[int(i)] for i in result.selected_pool_positions]
                    else:
                        result = conditional_gradient_maxdet(features, lp, up, 32)
                        selected = [outer[int(i)] for i in result.selected_positions]
                    acquisition_seconds = time.perf_counter() - tick
                    trace = list(result.trace)
                transition(labeled, unlabeled, selected)
                selection = dict(
                    method=method,
                    round=round_index,
                    L_hash=ids_hash(labeled),
                    U_hash=ids_hash(unlabeled),
                    selected=selected,
                    selected_hash=stable_hash(selected),
                    trace=trace,
                )
                if method == "maxdet":
                    # Omit elapsed time from the immutable numerical contract:
                    # a deterministic interrupted acquisition must be replayable.
                    selection["numerical_audit"] = {
                        k: v for k, v in result.audit.items() if k != "elapsed_seconds"
                    }
                # Commit selection before the only operation that can authorize new labels.
                write_once(directory / "selection.json", selection)
            files = {
                str((fit_dir / name).relative_to(study)): value
                for name, value in fit_record["files"].items()
            }
            files[str((fit_dir / "fit.json").relative_to(study))] = sha(fit_dir / "fit.json")
            files[str((directory / "predictions.npz").relative_to(study))] = sha(
                directory / "predictions.npz"
            )
            if round_index < 3:
                files[str((directory / "selection.json").relative_to(study))] = sha(
                    directory / "selection.json"
                )
            if gradient_path:
                for name in ["input.json", "features.npz", "audit.json", "complete.json"]:
                    files[str((gradient_path / name).relative_to(study))] = sha(
                        gradient_path / name
                    )
            record = dict(
                method=method,
                round=round_index,
                budget=budget,
                labeled_ids=labeled,
                L_hash=ids_hash(labeled),
                U_hash=ids_hash(unlabeled),
                selected=selected,
                selected_sample_ids=[f"ODH_charity_0616:row:{i}" for i in selected],
                protocol_hash=stable_hash(protocol),
                checkpoint_hash=fit_record["checkpoint_hash"],
                prediction_hash=sha(directory / "predictions.npz"),
                gradient_feature_hash=gradient_audit["feature_hash"] if gradient_audit else None,
                countsketch_hash=gradient_audit["mapping_hash"] if gradient_audit else None,
                gradient_not_required=("random" if method == "random" else "terminal_round")
                if not gradient_audit
                else None,
                initialization_hash=fit_record["initialization_hash"],
                training_seed=INIT_SEED,
                training_seconds=fit_record["seconds"],
                training_actual_seconds=0 if reused_fit else fit_record["seconds"],
                fit_reused=reused_fit,
                epochs_run=fit_record["epochs_run"],
                best_epoch=fit_record["best_epoch"],
                gradient_seconds=gradient_audit["seconds"] if gradient_audit else 0,
                gradient_actual_seconds=0
                if gradient_reused or not gradient_audit
                else gradient_audit["seconds"],
                gradient_reused=gradient_reused,
                acquisition_seconds=acquisition_seconds,
                validation_metrics=val_metrics,
                files=files,
            )
            write_once(record_path, record)
            if round_index < 3:
                store.commit_selection(directory / "selection.json")
                labeled, unlabeled = transition(labeled, unlabeled, selected)
            records[f"{method}/{round_index}"] = dict(path=str(record_path.relative_to(study)))
            print(
                dict(
                    stage="round_frozen",
                    method=method,
                    round=round_index,
                    budget=budget,
                    validation_rmse=val_metrics["rmse"],
                ),
                flush=True,
            )
        write_once(
            study / f"runtime/{method}/trajectory_freeze.json",
            dict(
                method=method,
                rounds={
                    str(r): sha(study / f"runtime/{method}/round_{r}/round.json") for r in range(4)
                },
                final_active_count=len(labeled),
                total_selected=96,
                test_truth_access_count=0,
            ),
        )
    files = {
        "protocol.json": sha(study / "protocol.json"),
        "splits/partition.json": sha(study / "splits/partition.json"),
    }
    initial_hashes = set()
    for entry in records.values():
        record = read_json(study / entry["path"])
        verify_files(study, record["files"])
        files.update(record["files"])
        files[entry["path"]] = sha(study / entry["path"])
        initial_hashes.add(record["initialization_hash"])
    if len(initial_hashes) != 1:
        raise RuntimeError("scratch initialization differs across methods/rounds")
    for method in METHODS:
        name = f"runtime/{method}/trajectory_freeze.json"
        files[name] = sha(study / name)
    # The pre-test audit is sealed separately; final evaluation appends to the main audit.
    with (study / "label_access_audit.csv").open() as f:
        accesses = list(csv.DictReader(f))
    if any(r["purpose"] == "final_test" for r in accesses):
        raise RuntimeError("test request before global freeze")
    atomic_json(study / "pre_test_label_access_audit.json", accesses)
    files["pre_test_label_access_audit.json"] = sha(study / "pre_test_label_access_audit.json")
    freeze = dict(
        status="FROZEN_BEFORE_TEST_TRUTH",
        entries=records,
        files=files,
        test_truth_access_count=0,
        utc=datetime.now(timezone.utc).isoformat(),
    )
    write_once(study / "global_pre_test_freeze.json", freeze)
    verify_global_freeze(study)
    return freeze
