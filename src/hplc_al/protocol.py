"""Immutable baseline roles and a method-local audited label boundary."""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    BASELINE,
    BUDGETS,
    METHODS,
    SEED,
    SOURCE,
    ids_hash,
    read_json,
    sha,
    stable_hash,
    verify_files,
)


def make_partition():
    manifest = read_json(BASELINE / "split_manifest.json")
    if sha(SOURCE) != manifest["csv_sha256"]:
        raise RuntimeError("ODH source hash changed")
    outer = sorted(manifest["split_row_indices"]["train"])
    l0 = set(np.random.default_rng(SEED + 104_729).choice(outer, 356, replace=False).tolist())
    # Explicit allowlist: no RT, RTv or target columns enter this frame.
    features = pd.read_csv(SOURCE, usecols=["Unnamed: 0", "SMILES", "Speed", "i-PrOH_proportion"])
    role = {i: "excluded" for i in range(len(features))}
    for i in manifest["unused_rounding_rows"]:
        role[i] = "unused_rounding"
    for i in outer:
        role[i] = "l0" if i in l0 else "u0"
    for split, name in [("valid", "validation"), ("test", "test")]:
        for i in manifest["split_row_indices"][split]:
            role[i] = name
    rows = [
        {
            "sample_index": i,
            "sample_id": f"ODH_charity_0616:row:{i}",
            "source_index": int(features.loc[i, "Unnamed: 0"]),
            "role": role[i],
        }
        for i in range(len(features))
    ]
    partition = {
        "seed": SEED,
        "l0_seed": SEED + 104_729,
        "baseline_manifest_sha256": sha(BASELINE / "split_manifest.json"),
        "source_sha256": sha(SOURCE),
        "rows": rows,
    }
    validate_partition(partition)
    return partition


def role_ids(partition, role):
    return [int(r["sample_index"]) for r in partition["rows"] if r["role"] == role]


def validate_partition(partition):
    m = read_json(BASELINE / "split_manifest.json")
    rows = partition["rows"]
    if [r["sample_index"] for r in rows] != list(range(4972)):
        raise ValueError("complete canonical source identity required")
    if any(r["sample_id"] != f"ODH_charity_0616:row:{r['sample_index']}" for r in rows):
        raise ValueError("stable sample_id mismatch")
    expected = dict(l0=356, u0=4091, validation=247, test=247, unused_rounding=1, excluded=30)
    if {r["role"] for r in rows} != set(expected):
        raise ValueError("unknown role")
    if any(len(role_ids(partition, role)) != count for role, count in expected.items()):
        raise ValueError("role count drift")
    for roles, original in [
        (("l0", "u0"), "train"),
        (("validation",), "valid"),
        (("test",), "test"),
    ]:
        if set(sum([role_ids(partition, role) for role in roles], [])) != set(
            m["split_row_indices"][original]
        ):
            raise ValueError("baseline membership drift")
    if role_ids(partition, "unused_rounding") != m["unused_rounding_rows"]:
        raise ValueError("unused rounding membership drift")
    if set(role_ids(partition, "excluded")) != {r["sample_index"] for r in m["excluded_rows"]}:
        raise ValueError("exclusion membership drift")


def transition(labeled, unlabeled, selected, batch=32):
    """Move one unique acquisition batch from the candidate pool to labeled rows."""
    labeled_set, candidate_set, selected_set = set(labeled), set(unlabeled), set(selected)
    if (
        len(labeled_set) != len(labeled)
        or len(candidate_set) != len(unlabeled)
        or labeled_set & candidate_set
    ):
        raise ValueError("invalid prior state")
    if len(selected) != batch or len(selected_set) != batch or not selected_set <= candidate_set:
        raise ValueError("selection must be exactly B unique previous-U IDs")
    return sorted(labeled_set | selected_set), sorted(candidate_set - selected_set)


class RestrictedLabelStore:
    """Logical data boundary, not an OS security sandbox.

    Only authorized requested rows have target strings converted to numbers.
    Acquisition never receives this object. All requests, including denials,
    are persisted immediately; method state is reconstructed from sealed batches.
    """

    def __init__(self, partition, audit_path, method, source=SOURCE):
        self.roles = {int(r["sample_index"]): r["role"] for r in partition["rows"]}
        self.method = method
        self.source = Path(source)
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.active = {i for i, role in self.roles.items() if role == "l0"}
        self.round = 0
        self.test_frozen = False
        self.test_revealed = False

    def commit_selection(self, path):
        selection = read_json(path)
        if selection["method"] != self.method or selection["round"] != self.round:
            raise ValueError("selection method/round mismatch")
        u = sorted(i for i, role in self.roles.items() if role == "u0" and i not in self.active)
        if selection["L_hash"] != ids_hash(self.active) or selection["U_hash"] != ids_hash(u):
            raise ValueError("selection prior-state hash mismatch")
        if selection["selected_hash"] != stable_hash(selection["selected"]):
            raise ValueError("selection content hash mismatch")
        new_l, _ = transition(sorted(self.active), u, selection["selected"])
        self.active = set(new_l)
        self.round += 1

    def unlock_test(self, study):
        verify_global_freeze(study)
        self.test_frozen = True

    def _audit(self, ids, purpose, allowed):
        fields = [
            "utc",
            "method",
            "round",
            "purpose",
            "allowed",
            "requested_rows",
            "ids_hash",
            "ids",
            "test_frozen",
        ]
        exists = self.audit_path.exists()
        with self.audit_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                dict(
                    utc=datetime.now(timezone.utc).isoformat(),
                    method=self.method,
                    round=self.round,
                    purpose=purpose,
                    allowed=allowed,
                    requested_rows=len(ids),
                    ids_hash=ids_hash(ids),
                    ids=";".join(str(i) for i in ids),
                    test_frozen=self.test_frozen,
                )
            )
            f.flush()
            os.fsync(f.fileno())

    def reveal(self, ids, purpose):
        ids = [int(i) for i in ids]
        valid = bool(ids) and len(set(ids)) == len(ids) and all(i in self.roles for i in ids)
        if purpose == "fit":
            allowed = valid and all(i in self.active for i in ids)
        elif purpose == "validation":
            allowed = valid and all(self.roles[i] == "validation" for i in ids)
        elif purpose == "final_test":
            allowed = (
                valid
                and self.test_frozen
                and not self.test_revealed
                and set(ids) == {i for i, role in self.roles.items() if role == "test"}
            )
        else:
            allowed = False
        self._audit(ids, purpose, allowed)
        if not allowed:
            raise PermissionError(f"unauthorized labels: {self.method}/{purpose}")
        wanted, requested = sorted(ids), set(ids)
        # Skip unauthorized records before constructing any target-bearing frame.
        # Disk parsing is trusted I/O; no U/test target strings reach AL objects.
        frame = pd.read_csv(
            self.source,
            usecols=["RT", "Speed"],
            skiprows=lambda line: line > 0 and line - 1 not in requested,
        )
        if len(frame) != len(wanted):
            raise RuntimeError("authorized source rows missing")
        found = dict(
            zip(
                wanted,
                (
                    frame.RT.to_numpy(dtype=np.float64) * frame.Speed.to_numpy(dtype=np.float64)
                ).astype(np.float32),
            )
        )
        values = np.array([found[i] for i in ids], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite authorized labels")
        if purpose == "final_test":
            self.test_revealed = True
        return values


def verify_global_freeze(study):
    study = Path(study)
    freeze = read_json(study / "global_pre_test_freeze.json")
    expected = {f"{m}/{r}" for m in METHODS for r in range(len(BUDGETS))}
    if (
        freeze.get("status") != "FROZEN_BEFORE_TEST_TRUTH"
        or set(freeze.get("entries", {})) != expected
    ):
        raise PermissionError("complete three-method/four-budget freeze required")
    if freeze.get("test_truth_access_count") != 0:
        raise PermissionError("pre-freeze test access detected")
    verify_files(study, freeze["files"])
    required = {"protocol.json", "splits/partition.json", "pre_test_label_access_audit.json"}
    if not required <= set(freeze["files"]):
        raise PermissionError("global freeze lacks protocol, partition or access evidence")
    partition = read_json(study / "splits/partition.json")
    validate_partition(partition)
    protocol_hash = stable_hash(read_json(study / "protocol.json"))
    states = {m: (role_ids(partition, "l0"), role_ids(partition, "u0")) for m in METHODS}
    initial = set()
    for key in sorted(expected):
        entry = freeze["entries"][key]
        method, number = key.split("/")
        record = read_json(study / entry["path"])
        if (
            record["method"] != method
            or record["round"] != int(number)
            or record["budget"] != BUDGETS[int(number)]
        ):
            raise PermissionError("global freeze round identity mismatch")
        if freeze["files"].get(entry["path"]) != sha(study / entry["path"]):
            raise PermissionError("unbound round entry")
        labeled, unlabeled = states[method]
        if (record["L_hash"], record["U_hash"], record["labeled_ids"], record["protocol_hash"]) != (
            ids_hash(labeled),
            ids_hash(unlabeled),
            labeled,
            protocol_hash,
        ):
            raise PermissionError("invalid frozen trajectory state")
        if not record.get("files") or any(
            freeze["files"].get(p) != h for p, h in record["files"].items()
        ):
            raise PermissionError("round artifacts not bound into global freeze")
        directory = Path(entry["path"]).parent
        prediction = str(directory / "predictions.npz")
        if record["files"].get(prediction) != record["prediction_hash"]:
            raise PermissionError("prediction not bound")
        if record["checkpoint_hash"] not in record["files"].values():
            raise PermissionError("checkpoint not bound")
        initial.add(record["initialization_hash"])
        if int(number) < 3:
            selection_path = str(directory / "selection.json")
            if selection_path not in record["files"]:
                raise PermissionError("selection not bound")
            selection = read_json(study / selection_path)
            if (
                selection["method"],
                selection["round"],
                selection["L_hash"],
                selection["U_hash"],
                selection["selected"],
                selection["selected_hash"],
            ) != (
                method,
                int(number),
                ids_hash(labeled),
                ids_hash(unlabeled),
                record["selected"],
                stable_hash(record["selected"]),
            ):
                raise PermissionError("selection contract drift")
            states[method] = transition(labeled, unlabeled, record["selected"])
        elif record["selected"]:
            raise PermissionError("terminal budget may not acquire")
    if len(initial) != 1:
        raise PermissionError("scratch initialization mismatch")
    for access in read_json(study / "pre_test_label_access_audit.json"):
        if access["purpose"] == "final_test":
            raise PermissionError("test requested before freeze")
        if access["allowed"] == "True":
            ids = [int(i) for i in access["ids"].split(";")]
            if access["purpose"] == "validation":
                allowed = set(role_ids(partition, "validation"))
            elif access["purpose"] == "fit" and access["method"] in METHODS:
                r = int(access["round"])
                if r not in range(4):
                    raise PermissionError("invalid access round")
                rec = read_json(study / freeze["entries"][f"{access['method']}/{r}"]["path"])
                allowed = set(rec["labeled_ids"])
            else:
                raise PermissionError("invalid pre-freeze access purpose")
            if not set(ids) <= allowed:
                raise PermissionError("pre-freeze access escaped authorized labels")
    return freeze
