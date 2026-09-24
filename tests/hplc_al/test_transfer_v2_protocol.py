import numpy as np
import pytest

from hplc_al.deterministic_shuffle import (
    assert_epoch_coverage, batch_sizes, epoch_batches, epoch_order,
)
from hplc_al.preflight_v2 import duration_decision
from hplc_al.transfer_v2 import (
    ACQUISITIONS,
    BUDGETS,
    L0,
    OUTER_TRAIN,
    label_firewall_keys,
    transfer_partition,
    training_config,
    unit_gradient,
    selection_audit,
    validate_budget_protocol,
    verify_pre_test_freeze,
)


def test_same_seed_same_epoch_is_byte_identical():
    ids = np.arange(17)
    truth = ids.astype(np.float32) + 0.25
    a = epoch_order(ids, truth, 525, 3)
    b = epoch_order(ids, truth, 525, 3)
    assert np.array_equal(a.permutation, b.permutation)
    assert np.array_equal(a.ids, b.ids)
    assert np.array_equal(a.truth, b.truth)


def test_epoch_changes_and_covers_each_labeled_id_once():
    ids = np.arange(200)
    truth = ids.astype(np.float32)
    first = epoch_order(ids, truth, 525, 0)
    second = epoch_order(ids, truth, 525, 1)
    assert not np.array_equal(first.permutation, second.permutation)
    assert_epoch_coverage(ids, first.ids)
    assert np.array_equal(first.truth, first.ids.astype(np.float32))


def test_graph_ids_and_separate_truth_stay_paired_in_every_batch():
    ids = np.arange(513)
    truth = ids.astype(np.float64) * 10.0
    seen = []
    for batch_ids, batch_truth in epoch_batches(ids, truth, 73, 8, batch_size=256):
        assert np.array_equal(batch_truth, batch_ids * 10.0)
        seen.extend(batch_ids.tolist())
    assert_epoch_coverage(ids, seen)


def test_validation_test_and_u0_are_outside_shuffle_api():
    ids = np.arange(5)
    truth = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        epoch_order(ids, truth[:-1], 1, 0)
    # The API accepts only the explicitly supplied labeled arrays; no hidden U0
    # or validation collection can be consumed by the permutation.
    assert len(epoch_order(ids, truth, 1, 0).ids) == len(ids)


def test_repeated_runs_are_deterministic_and_no_duplicates():
    ids = np.arange(333)
    truth = np.sin(ids)
    runs = [epoch_order(ids, truth, 991, 4) for _ in range(3)]
    assert all(np.array_equal(runs[0].ids, run.ids) for run in runs[1:])
    assert len(np.unique(runs[0].ids)) == len(ids)


def test_transfer_budget_methods_and_training_contract():
    assert validate_budget_protocol()
    assert L0 == 333 and OUTER_TRAIN == 4447
    assert BUDGETS == (333, 365, 397, 429)
    assert len(ACQUISITIONS) == 5
    config = training_config()
    assert config["batch_size"] == 256
    assert config["maximum_epochs"] == 500
    assert config["patience"] == 100
    assert config["shuffle"] == "deterministic_each_epoch"
    assert config["fresh_adam_every_round"] and config["scratch"]
    assert batch_sizes(333, 256) == [256, 77]
    assert sum(batch_sizes(333, 256)) == 333


def test_transfer_training_permutation_excludes_nonlabeled_roles(partition):
    transfer = transfer_partition(partition)
    l0 = np.array([r["sample_index"] for r in transfer["rows"] if r["role"] == "l0"])
    protected = {
        r["sample_index"] for r in transfer["rows"]
        if r["role"] in {"u0", "validation", "test"}
    }
    truth = l0.astype(np.float64) * 3.0 + 1
    order = epoch_order(l0, truth, 525, 2)
    assert len(l0) == 333
    assert not (set(order.ids.tolist()) & protected)
    assert np.array_equal(order.truth, order.ids.astype(np.float64) * 3.0 + 1)


def test_unit_gradient_normalizes_after_sketch_and_retains_zeros():
    raw = np.array([[0.0, 0.0], [3.0, 4.0], [-5.0, 0.0]])
    unit = unit_gradient(raw)
    assert np.array_equal(unit[0], raw[0])
    assert np.allclose(np.linalg.norm(unit[1:], axis=1), 1.0)


def test_selection_audit_records_required_label_free_geometry():
    raw = np.arange(40, dtype=float).reshape(10, 4)
    raw[0] = 0
    audit = selection_audit(raw, [2, 4, 7], [0, 1, 3])
    assert {
        "selected_norm_percentile",
        "top_decile_selected_fraction",
        "effective_rank",
        "nearest_L_distance",
        "batch_pairwise_diversity",
        "zero_gradient_rows",
    } <= set(audit)
    assert audit["zero_gradient_rows"] == 1


def test_global_freeze_requires_all_five_methods_and_four_budgets():
    with pytest.raises(PermissionError):
        verify_pre_test_freeze({})
    entries = {}
    for key in sorted(label_firewall_keys()):
        method, round_text = key.split("/")
        round_index = int(round_text)
        entries[key] = {"method": method, "round": round_index, "budget": BUDGETS[round_index]}
    assert verify_pre_test_freeze(entries)
    with pytest.raises(PermissionError):
        verify_pre_test_freeze(entries, test_truth_access_count=1)


def test_duration_rule_reports_insufficient_without_retuning():
    assert duration_decision(450).status == "INSUFFICIENT_DURATION_PROTOCOL"
    assert duration_decision(300).status == "PASS"


def test_random_post_l0_needs_no_gradient_audit():
    from hplc_al.transfer_v2_runner import _selection
    selected, _, audit, _ = _selection('random', None, list(range(333)), list(range(333,1000)), list(range(1000)), 1)
    assert len(selected) == len(set(selected)) == 32
    assert set(selected) <= set(range(333,1000))
    assert audit['selector_seconds'] >= 0


def test_gradient_cache_rejects_tampered_features(tmp_path):
    from hplc_al.common import atomic_json, sha
    from hplc_al.transfer_v2_runner import _gradient_bank
    record = {'checkpoint_hash':'checkpoint', 'checkpoint_state_hash':'state'}
    atomic_json(tmp_path/'input.json', {'checkpoint_hash':'checkpoint', 'checkpoint_state_hash':'state', 'ordered_ids':[1], 'definition':'512D CountSketch(full-network central gradient)'})
    np.savez(tmp_path/'features.npz',ids=[1],features=np.ones((1,512)))
    atomic_json(tmp_path/'audit.json',{})
    atomic_json(tmp_path/'complete.json',{'files':{name:sha(tmp_path/name) for name in ['input.json','features.npz','audit.json']}})
    (tmp_path/'features.npz').write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='artifact hash mismatch'):
        _gradient_bank(tmp_path,None,None,[1],record,tmp_path)
