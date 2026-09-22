import copy
import csv
from pathlib import Path

import numpy as np
import pytest

from hplc_al.common import (
    BASELINE,
    BUDGETS,
    METHODS,
    atomic_json,
    ids_hash,
    metrics,
    partial_aulc,
    read_json,
    sha,
    stable_hash,
    verify_files,
)
from hplc_al.data import assert_label_free
from hplc_al.protocol import (
    RestrictedLabelStore,
    role_ids,
    transition,
    validate_partition,
    verify_global_freeze,
)


def test_split_disjoint_and_union(partition):
    roles = [set(role_ids(partition, r)) for r in ["l0", "u0", "validation", "test"]]
    assert len(set.union(*roles)) == 4941
    assert sum(map(len, roles)) == 4941
    assert role_ids(partition, "unused_rounding") == [4209]
    assert len(set.union(*roles, set(role_ids(partition, "unused_rounding")))) == 4942


def test_fixed_baseline_membership(partition):
    m = read_json(BASELINE / "split_manifest.json")["split_row_indices"]
    assert set(role_ids(partition, "test")) == set(m["test"])
    assert set(role_ids(partition, "validation")) == set(m["valid"])
    assert set(role_ids(partition, "l0") + role_ids(partition, "u0")) == set(m["train"])


def test_membership_drift_rejected(partition):
    p = copy.deepcopy(partition)
    a, b = role_ids(p, "test")[0], role_ids(p, "u0")[0]
    p["rows"][a]["role"], p["rows"][b]["role"] = "u0", "test"
    with pytest.raises(ValueError):
        validate_partition(p)


@pytest.mark.parametrize(
    "role,purpose",
    [
        ("u0", "fit"),
        ("test", "fit"),
        ("test", "final_test"),
        ("u0", "validation"),
        ("unused_rounding", "fit"),
    ],
)
def test_truth_denied_before_source_open(partition, tmp_path, role, purpose):
    store = RestrictedLabelStore(
        partition, tmp_path / "audit.csv", "lcmd", source=tmp_path / "does_not_exist.csv"
    )
    with pytest.raises(PermissionError):
        store.reveal(role_ids(partition, role), purpose)
    with (tmp_path / "audit.csv").open() as f:
        row = list(csv.DictReader(f))[0]
    assert row["allowed"] == "False"


def selection(partition, method):
    labeled, unlabeled = role_ids(partition, "l0"), role_ids(partition, "u0")
    return dict(
        method=method,
        round=0,
        L_hash=ids_hash(labeled),
        U_hash=ids_hash(unlabeled),
        selected=unlabeled[:32],
        selected_hash=stable_hash(unlabeled[:32]),
    )


def test_reveal_after_committed_selection_and_method_isolation(partition, tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_text("RT,Speed\n" + "1,2\n" * 4972)
    a = RestrictedLabelStore(partition, tmp_path / "audit.csv", "lcmd", source)
    b = RestrictedLabelStore(partition, tmp_path / "audit.csv", "maxdet", source)
    selected = selection(partition, "lcmd")
    atomic_json(tmp_path / "selected.json", selected)
    a.commit_selection(tmp_path / "selected.json")
    assert np.all(a.reveal(selected["selected"], "fit") == 2)
    with pytest.raises(PermissionError):
        b.reveal(selected["selected"], "fit")
    with pytest.raises(ValueError):
        b.commit_selection(tmp_path / "selected.json")
    with pytest.raises(ValueError):
        a.commit_selection(tmp_path / "selected.json")


def test_selection_integrity(partition, tmp_path):
    selected = selection(partition, "lcmd")
    selected["selected"][0] = role_ids(partition, "test")[0]
    atomic_json(tmp_path / "selected.json", selected)
    s = RestrictedLabelStore(partition, tmp_path / "audit.csv", "lcmd")
    with pytest.raises(ValueError):
        s.commit_selection(tmp_path / "selected.json")


def test_trajectory_transition(partition):
    labeled, unlabeled = role_ids(partition, "l0"), role_ids(partition, "u0")
    union = set(labeled + unlabeled)
    for size in [388, 420, 452]:
        labeled, unlabeled = transition(labeled, unlabeled, unlabeled[:32])
        assert len(labeled) == size
        assert not set(labeled) & set(unlabeled)
        assert set(labeled + unlabeled) == union
    assert len(unlabeled) == 3995


@pytest.mark.parametrize("selected", [list(range(31)), [1] * 32, list(range(100, 132))])
def test_invalid_transition(selected):
    with pytest.raises(ValueError):
        transition([99], list(range(40)), selected)


def test_real_graphs_have_only_allowlisted_fields(graphs):
    assert len(graphs) == 4941
    assert_label_free(graphs)


@pytest.mark.parametrize("field", ["y", "RT", "true_RTv", "truth"])
def test_hidden_truth_attribute_rejected(graphs, field):
    key = next(iter(graphs))
    g, h = graphs[key]
    bad = g.clone()
    bad[field] = 123.0
    with pytest.raises(PermissionError):
        assert_label_free({key: (bad, h)})


def test_test_barrier_incomplete(partition, tmp_path):
    atomic_json(
        tmp_path / "global_pre_test_freeze.json",
        dict(status="FROZEN_BEFORE_TEST_TRUTH", entries={}),
    )
    store = RestrictedLabelStore(partition, tmp_path / "audit.csv", "evaluation")
    with pytest.raises(PermissionError):
        store.unlock_test(tmp_path)
    with pytest.raises(PermissionError):
        store.reveal(role_ids(partition, "test"), "final_test")


def frozen_fixture(tmp_path, partition):
    entries, files = {}, {}
    atomic_json(tmp_path / "protocol.json", {"test": "synthetic"})
    atomic_json(tmp_path / "splits/partition.json", partition)
    atomic_json(tmp_path / "pre_test_label_access_audit.json", [])
    for name in ["protocol.json", "splits/partition.json", "pre_test_label_access_audit.json"]:
        files[name] = sha(tmp_path / name)
    for m in METHODS:
        labeled, unlabeled = role_ids(partition, "l0"), role_ids(partition, "u0")
        for r, b in enumerate(BUDGETS):
            path = f"{m}/round_{r}/round.json"
            base = Path(path).parent
            predictions = str(base / "predictions.npz")
            checkpoint = str(base / "best.pt")
            atomic_json(tmp_path / predictions, {"synthetic": r})
            atomic_json(tmp_path / checkpoint, {"synthetic": r})
            bound = {n: sha(tmp_path / n) for n in [predictions, checkpoint]}
            selected = unlabeled[:32] if r < 3 else []
            if r < 3:
                selection_path = str(base / "selection.json")
                atomic_json(
                    tmp_path / selection_path,
                    dict(
                        method=m,
                        round=r,
                        L_hash=ids_hash(labeled),
                        U_hash=ids_hash(unlabeled),
                        selected=selected,
                        selected_hash=stable_hash(selected),
                    ),
                )
                bound[selection_path] = sha(tmp_path / selection_path)
            atomic_json(
                tmp_path / path,
                dict(
                    method=m,
                    round=r,
                    budget=b,
                    labeled_ids=labeled,
                    L_hash=ids_hash(labeled),
                    U_hash=ids_hash(unlabeled),
                    selected=selected,
                    protocol_hash=stable_hash({"test": "synthetic"}),
                    initialization_hash="same",
                    prediction_hash=bound[predictions],
                    checkpoint_hash=bound[checkpoint],
                    files=bound,
                ),
            )
            entries[f"{m}/{r}"] = dict(path=path)
            files[path] = sha(tmp_path / path)
            files.update(bound)
            if r < 3:
                labeled, unlabeled = transition(labeled, unlabeled, selected)
    atomic_json(
        tmp_path / "global_pre_test_freeze.json",
        dict(
            status="FROZEN_BEFORE_TEST_TRUTH",
            entries=entries,
            files=files,
            test_truth_access_count=0,
        ),
    )


def test_test_barrier_hash_tamper(tmp_path, partition):
    frozen_fixture(tmp_path, partition)
    verify_global_freeze(tmp_path)
    atomic_json(tmp_path / "random/round_0/round.json", dict(method="random", round=0, budget=999))
    with pytest.raises(RuntimeError):
        verify_global_freeze(tmp_path)


def test_test_barrier_nested_artifact_omission(tmp_path, partition):
    frozen_fixture(tmp_path, partition)
    freeze = read_json(tmp_path / "global_pre_test_freeze.json")
    del freeze["files"]["random/round_0/best.pt"]
    atomic_json(tmp_path / "global_pre_test_freeze.json", freeze)
    with pytest.raises(PermissionError):
        verify_global_freeze(tmp_path)


def test_test_barrier_replays_access_scope(tmp_path, partition):
    frozen_fixture(tmp_path, partition)
    audit = "pre_test_label_access_audit.json"
    atomic_json(
        tmp_path / audit,
        [
            dict(
                method="random",
                round="0",
                purpose="fit",
                allowed="True",
                ids=str(role_ids(partition, "u0")[0]),
            )
        ],
    )
    freeze = read_json(tmp_path / "global_pre_test_freeze.json")
    freeze["files"][audit] = sha(tmp_path / audit)
    atomic_json(tmp_path / "global_pre_test_freeze.json", freeze)
    with pytest.raises(PermissionError):
        verify_global_freeze(tmp_path)


def test_source_loader_never_materializes_forbidden_labels(partition, tmp_path):
    target = role_ids(partition, "l0")[0]
    source = tmp_path / "poison.csv"
    rows = ["RT,Speed"] + ["FORBIDDEN,FORBIDDEN"] * 4972
    rows[target + 1] = "2,3"
    source.write_text("\n".join(rows) + "\n")
    store = RestrictedLabelStore(partition, tmp_path / "audit.csv", "random", source)
    assert store.reveal([target], "fit").tolist() == [6.0]


def test_cache_drift_and_incomplete_file(tmp_path):
    p = tmp_path / "checkpoint"
    p.write_text("original")
    files = {"checkpoint": sha(p)}
    verify_files(tmp_path, files)
    p.write_text("partial")
    with pytest.raises(RuntimeError):
        verify_files(tmp_path, files)
    with pytest.raises(RuntimeError):
        verify_files(tmp_path, {"missing": "x"})


def test_aulc_known_linear_curve_and_missing_budget():
    area = partial_aulc(list(BUDGETS), [4, 3, 2, 1])
    assert area == dict(raw=240.0, mean=2.5)
    with pytest.raises(ValueError):
        partial_aulc([356, 388, 452], [4, 3, 1])


def test_metric_definition():
    result = metrics(np.array([2.0, 4.0]), np.array([1.0, 3.0]), 2.0)
    assert result == dict(rmse=1.0, mae=1.0, r2=0.0, nrmse=0.5)
