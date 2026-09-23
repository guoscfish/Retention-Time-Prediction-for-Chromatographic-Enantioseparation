"""Preregistered nine-fit diagnosis; no AL trajectory or candidate/test reveal."""

import csv
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from hplc_al.common import ROOT, atomic_json, read_json, sha, stable_hash, verify_files, write_once
from hplc_al.common import STUDY as ORIGINAL
from hplc_al.data import load_graphs, predict
from hplc_al.protocol import RestrictedLabelStore, role_ids, validate_partition
from hplc_al.runner import assert_environment
from hplc_al.training import load_model

from .geometry import EPSILON, correlations, gradients, selection_diagnostic
from .training import ARMS, choose_challenger, exposure, fixed_fit

STUDY = ROOT / "studies/active_learning/odh_training_protocol_diagnosis"


class DiagnosisLabels(RestrictedLabelStore):
    """Deny query commits and test access even if caller tries to unlock them."""

    def commit_selection(self, path):
        raise PermissionError("diagnosis never reveals selected candidate labels")

    def unlock_test(self, study):
        raise PermissionError("diagnosis never unlocks test labels")

    def reveal(self, ids, purpose):
        if purpose not in {"fit", "validation"}:
            self._audit(ids, purpose, False)
            raise PermissionError("diagnosis allows original L0/validation labels only")
        return super().reveal(ids, purpose)


def source_hashes():
    paths = list((ROOT / "src/hplc_diagnosis").glob("*.py"))
    paths += list((ROOT / "src/hplc_al").glob("*.py"))
    paths += list((ROOT / "tests/hplc_al").glob("*.py"))
    paths += [
        ROOT / p
        for p in [
            "src/reproduction/official.py",
            "code/Single_column_prediction.py",
            "code/compound_tools.py",
            "scripts/run_hplc_training_diagnosis.py",
            "requirements-hplc-al.txt",
            "requirements-reproduction.txt",
        ]
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def prepare():
    """Freeze settings and actual input/code evidence before any diagnostic fit."""
    import xml.etree.ElementTree as ET

    environment = assert_environment()
    partition_path = ORIGINAL / "splits/partition.json"
    partition = read_json(partition_path)
    validate_partition(partition)
    original = read_json(ORIGINAL / "protocol.json")
    if original["partition_hash"] != stable_hash(partition):
        raise RuntimeError("original partition hash drift")
    # Hash all original files, including any label-bearing files, without parsing
    # their contents. No historical test metrics enter the diagnosis process.
    completion = read_json(ORIGINAL / "completion_manifest.json")
    verify_files(ORIGINAL, completion["files"])
    original_files = dict(completion["files"])
    original_files["completion_manifest.json"] = sha(ORIGINAL / "completion_manifest.json")
    original_files["frozen_source.zip"] = sha(ORIGINAL / "frozen_source.zip")
    report = STUDY / "tests.xml"
    suites = list(ET.parse(report).getroot().iter("testsuite"))
    totals = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ["tests", "failures", "errors", "skipped"]
    }
    if totals["tests"] < 60 or any(totals[key] for key in ["failures", "errors", "skipped"]):
        raise RuntimeError("passing diagnosis and original tests required before registration")
    write_once(STUDY / "partition.json", partition)
    protocol = dict(
        status="PREREGISTERED_VALIDATION_ONLY_DIAGNOSIS",
        arms=ARMS,
        part_a_seed=525,
        part_b_seeds=[1525, 2525],
        total_successful_fits=9,
        practical_arm_selection="minimum seed525 best validation RMSE among B-E; ties: fewer updates, fewer epochs, arm ID; select even if worse than A",
        no_early_stopping=True,
        no_wall_time_cap=True,
        no_shuffle=True,
        drop_last=False,
        optimizer="Adam",
        learning_rate=0.001,
        weight_decay=1e-5,
        scheduler=False,
        architecture="original HPLC QGeoGNN 5 layers/128/sum/dropout0/q10-central-q90",
        loss="original complete loss",
        scratch=True,
        checkpoint="strict best validation central MSE over all registered epochs; earlier ties",
        NRMSE_denominator=original["L0_population_std"],
        NRMSE_definition="RMSE / original L0 population std ddof0",
        correlation="Spearman primary, Pearson descriptive; both full/sketch norms vs absolute/squared residual; no p-value-based selection",
        geometry_seed=525,
        geometry_arms="A and selected practical arm only",
        acquisition_batch=32,
        geometry_epsilon=EPSILON,
        gradient_definition=original["gradient"],
        sketch_seed=original["sketch_seed"],
        geometry_comparison="raw/unit LCMD/MaxDet; exact same L0/U0/checkpoint; no query commits or reveal",
        forbidden=[
            "U0 truth",
            "test truth or metrics",
            "new AL trajectory",
            "test-based tuning",
            "lr/architecture/target changes",
        ],
        partition_hash=stable_hash(partition),
        original_partition_file_sha256=sha(partition_path),
        original_protocol_sha256=sha(ORIGINAL / "protocol.json"),
        original_files=original_files,
        L0=role_ids(partition, "l0"),
        validation=role_ids(partition, "validation"),
        test_identity_hash=stable_hash(role_ids(partition, "test")),
        environment=environment,
        code=source_hashes(),
        tests=totals,
        preregistration_files={
            name: sha(STUDY / name) for name in ["PROTOCOL.md", "tests.xml", "tests.log"]
        },
        std_definition="sample std ddof1 across initialization seeds; seeds share fixed validation cohort",
        interpretation="descriptive development comparison; batch, BN, sample presentations and checkpoint opportunity confound pure causal attribution",
    )
    write_once(STUDY / "protocol.json", protocol)
    archive = STUDY / "frozen_source.zip"
    if not archive.exists():
        with ZipFile(archive, "w", ZIP_DEFLATED) as snapshot:
            for path in protocol["code"]:
                snapshot.write(ROOT / path, path)
    print(dict(status="PREREGISTERED", arms=ARMS, tests=totals), flush=True)


def check_protocol():
    protocol = read_json(STUDY / "protocol.json")
    verify_files(ROOT, protocol["code"])
    verify_files(STUDY, protocol["preregistration_files"])
    if assert_environment() != protocol["environment"]:
        raise RuntimeError("environment changed after registration")
    partition = read_json(STUDY / "partition.json")
    if stable_hash(partition) != protocol["partition_hash"]:
        raise RuntimeError("partition drift")
    verify_files(ORIGINAL, protocol["original_files"])
    return protocol, partition


def label_audit(partition):
    path = STUDY / "label_access_audit.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    allowed_ids = {
        purpose: set(role_ids(partition, role))
        for purpose, role in [("fit", "l0"), ("validation", "validation")]
    }
    for row in rows:
        if row["allowed"] == "True":
            ids = {int(value) for value in row["ids"].split(";")}
            if row["purpose"] not in allowed_ids or not ids <= allowed_ids[row["purpose"]]:
                raise RuntimeError("forbidden label access in diagnosis audit")
    return dict(
        successful_requests=sum(row["allowed"] == "True" for row in rows),
        U0_label_requests=0,
        test_label_requests=0,
        selection_commits=0,
        label_audit_sha256=sha(path) if path.exists() else None,
    )


def write_tables(records):
    rows = []
    for record in records:
        row = {
            key: record[key]
            for key in [
                "arm",
                "initialization_seed",
                "batch_size",
                "epochs_run",
                "steps_per_epoch",
                "total_optimizer_steps",
                "total_sample_presentations",
                "best_epoch",
                "best_optimizer_step",
                "seconds",
                "updates_per_second",
                "initialization_hash",
                "checkpoint_hash",
            ]
        }
        row.update(
            {f"best_validation_{key}": value for key, value in record["best_metrics"].items()}
        )
        row.update(
            {f"final_validation_{key}": value for key, value in record["final_metrics"].items()}
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(STUDY / "results/training_protocol_comparison.csv", index=False)
    pd.DataFrame(
        [
            dict(arm=arm, **config, **exposure(356, config["batch_size"], config["maximum_epochs"]))
            for arm, config in ARMS.items()
        ]
    ).to_csv(STUDY / "results/optimizer_step_comparison.csv", index=False)


def train():
    protocol, partition = check_protocol()
    if (STUDY / "completion_manifest.json").exists():
        raise RuntimeError("diagnosis completed; use verify")
    graphs = load_graphs(partition)
    # The fit receives only L0 and validation graphs, never test or U0 graphs.
    labeled, validation = protocol["L0"], protocol["validation"]
    training_graphs = {row: graphs[row] for row in labeled + validation}
    store = DiagnosisLabels(partition, STUDY / "label_access_audit.csv", "training_diagnosis")
    truth, valid_truth = store.reveal(labeled, "fit"), store.reveal(validation, "validation")
    scale = protocol["NRMSE_denominator"]
    if not np.isclose(np.std(truth.astype(np.float64)), scale, rtol=0, atol=1e-10):
        raise RuntimeError("L0 normalization drift")
    binding = dict(
        protocol_sha256=sha(STUDY / "protocol.json"), partition_hash=protocol["partition_hash"]
    )
    records = []
    for arm in ARMS:
        record = fixed_fit(
            training_graphs,
            labeled,
            truth,
            validation,
            valid_truth,
            arm,
            525,
            scale,
            STUDY / f"runtime/seed_525/{arm}",
            binding,
        )
        records.append(record)
        write_tables(records)
    chosen = choose_challenger(records)
    selection = dict(
        arm=chosen,
        rule=protocol["practical_arm_selection"],
        evidence={r["arm"]: sha(STUDY / f"runtime/seed_525/{r['arm']}/fit.json") for r in records},
        part_a_rmse={r["arm"]: r["best_metrics"]["rmse"] for r in records},
    )
    write_once(STUDY / "practical_arm_selection.json", selection)
    print(dict(stage="part_b_selection", **selection), flush=True)
    for seed in protocol["part_b_seeds"]:
        for arm in ["A", chosen]:
            record = fixed_fit(
                training_graphs,
                labeled,
                truth,
                validation,
                valid_truth,
                arm,
                seed,
                scale,
                STUDY / f"runtime/seed_{seed}/{arm}",
                binding,
            )
            records.append(record)
            write_tables(records)
    for seed in [525, *protocol["part_b_seeds"]]:
        if (
            len({r["initialization_hash"] for r in records if r["initialization_seed"] == seed})
            != 1
        ):
            raise RuntimeError("initialization differs within seed")
    if len(records) != 9:
        raise RuntimeError("exactly nine registered fits required")
    atomic_json(
        STUDY / "training_complete.json",
        dict(
            status="COMPLETE_9_FITS",
            selected_arm=chosen,
            fit_files={
                f"runtime/seed_{r['initialization_seed']}/{r['arm']}/fit.json": sha(
                    STUDY / f"runtime/seed_{r['initialization_seed']}/{r['arm']}/fit.json"
                )
                for r in records
            },
            label_audit=label_audit(partition),
        ),
    )


def geometry():
    protocol, partition = check_protocol()
    if (STUDY / "completion_manifest.json").exists():
        raise RuntimeError("diagnosis completed; use verify")
    complete = read_json(STUDY / "training_complete.json")
    verify_files(STUDY, complete["fit_files"])
    chosen = read_json(STUDY / "practical_arm_selection.json")["arm"]
    graphs = load_graphs(partition)
    labeled, validation = protocol["L0"], protocol["validation"]
    candidates = role_ids(partition, "u0")
    outer = sorted(labeled + candidates)
    store = DiagnosisLabels(partition, STUDY / "label_access_audit.csv", "validation_geometry")
    valid_truth = store.reveal(validation, "validation")
    # No test graphs are passed to diagnostics; no candidate labels are requested.
    graphs = {row: graphs[row] for row in outer + validation}
    relevance, selection_rows, correlation_rows = [], [], []
    for arm in ["A", chosen]:
        directory = STUDY / f"runtime/seed_525/{arm}"
        record = read_json(directory / "fit.json")
        verify_files(directory, record["files"])
        destination = STUDY / f"runtime/geometry_seed_525/{arm}"
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / "complete.json"
        if manifest.exists():
            cached = read_json(manifest)
            verify_files(destination, cached["files"])
            if cached["checkpoint_hash"] != record["checkpoint_hash"]:
                raise RuntimeError("geometry checkpoint drift")
        else:
            model = load_model(directory / record["checkpoint_path"])
            features, frame, audit = gradients(model, graphs, validation)
            batch_predictions = predict(model, graphs, validation)[:, 1]
            if not np.allclose(frame.central_prediction, batch_predictions, rtol=1e-5, atol=1e-5):
                raise RuntimeError(
                    "single-row gradient prediction differs from validation prediction"
                )
            # Residuals use the same per-row eval central scalar whose derivative is measured.
            frame["truth"] = valid_truth
            frame["absolute_error"] = np.abs(frame.central_prediction - valid_truth)
            frame["squared_error"] = np.square(frame.central_prediction - valid_truth)
            frame["arm"] = arm
            frame["initialization_seed"] = 525
            frame.to_csv(destination / "validation_relevance.csv", index=False)
            audit["maximum_batch_prediction_difference"] = float(
                np.max(np.abs(frame.central_prediction - batch_predictions))
            )
            atomic_json(destination / "validation_gradient_audit.json", audit)
            np.savez(
                destination / "validation_features.npz", ids=np.array(validation), features=features
            )
            pool_features, pool_frame, pool_audit = gradients(model, graphs, outer)
            np.savez(
                destination / "outer_features.npz",
                ids=np.array(outer),
                features=pool_features,
                full_gradient_norm=pool_frame.full_gradient_norm.to_numpy(),
            )
            pool_frame.to_csv(destination / "outer_norms_label_free.csv", index=False)
            atomic_json(destination / "outer_gradient_audit.json", pool_audit)
            selections = selection_diagnostic(
                pool_features,
                outer,
                labeled,
                candidates,
                pool_frame.full_gradient_norm.to_numpy(),
                destination,
            )
            pd.DataFrame(selections).to_csv(destination / "selection_comparison.csv", index=False)
            atomic_json(
                manifest,
                dict(
                    checkpoint_hash=record["checkpoint_hash"],
                    files={
                        p.name: sha(p)
                        for p in destination.iterdir()
                        if p.is_file() and p != manifest
                    },
                ),
            )
        frame = pd.read_csv(destination / "validation_relevance.csv")
        relevance.append(frame)
        correlation_rows.extend(
            dict(arm=arm, initialization_seed=525, **row) for row in correlations(frame)
        )
        rows = pd.read_csv(destination / "selection_comparison.csv")
        rows.insert(0, "arm", arm)
        rows.insert(1, "initialization_seed", 525)
        selection_rows.append(rows)
    pd.concat(relevance).to_csv(STUDY / "results/gradient_error_relevance.csv", index=False)
    atomic_json(
        STUDY / "results/gradient_error_relevance.json",
        dict(
            correlations=correlation_rows,
            primary="Spearman",
            interpretation="same fixed validation used for checkpoint selection; descriptive, no independent validation",
            squared_error_note="Spearman(abs_error) and Spearman(squared_error) must coincide for nonnegative absolute errors",
        ),
    )
    pd.DataFrame(correlation_rows).to_csv(
        STUDY / "results/gradient_error_correlations.csv", index=False
    )
    pd.concat(selection_rows).to_csv(
        STUDY / "results/same_state_selection_comparison.csv", index=False
    )
    atomic_json(
        STUDY / "geometry_complete.json",
        dict(
            status="COMPLETE",
            label_audit=label_audit(partition),
            files={
                str(p.relative_to(STUDY)): sha(p)
                for p in sorted((STUDY / "runtime/geometry_seed_525").rglob("*"))
                if p.is_file()
            },
        ),
    )
    print(dict(status="GEOMETRY_COMPLETE", **label_audit(partition)), flush=True)
