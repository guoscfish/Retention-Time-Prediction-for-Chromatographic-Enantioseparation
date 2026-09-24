import numpy as np
import pytest

from hplc_al.common import read_json
from hplc_al.protocol import role_ids, transition
from hplc_al.representation_extension import (MAX_EXTRA_ROUNDS, METHODS, PARENT,
                                               SIGNAL_THRESHOLD, _commit_round, _state, positive_signal)


def _row(gradient, latent, morgan, random=0.9):
    return dict(zip(METHODS, (random, gradient, latent, morgan)))


def test_validation_signal_at_either_new_budget_and_exact_threshold():
    positive, details = positive_signal([(461, _row(.88, .87, .91)),
                                         (493, _row(.86, .90, .89))])
    assert positive
    assert details[0]['positive_methods'] == ['latent_coreset']
    assert details[1]['positive_methods'] == []
    assert details[0]['gains_vs_lcmd']['latent_coreset'] == pytest.approx(SIGNAL_THRESHOLD)
    positive, details = positive_signal([(461, _row(.88, .875, .91)),
                                         (493, _row(.86, .90, .84))])
    assert positive and details[1]['positive_methods'] == ['morgan_coreset']


def test_no_signal_stops_pair_and_incomplete_metric_rejected():
    rows = [(461, _row(.88, .875, .91)), (493, _row(.86, .858, .88))]
    assert not positive_signal(rows)[0]
    with pytest.raises(ValueError):
        positive_signal(rows[:1])
    with pytest.raises(ValueError):
        positive_signal([(461, dict(raw_gradient_lcmd=np.nan)), rows[1]])


def test_parent_terminal_lineage_and_nested_random_continuation():
    partition = read_json(PARENT / 'splits/partition.json')
    order = read_json(PARENT / 'random_order.json')['ids']
    assert MAX_EXTRA_ROUNDS == 8
    assert len(order) == len(role_ids(partition, 'u0'))
    for method in METHODS:
        labeled, pool = _state(PARENT, partition, method, 3)
        terminal = read_json(PARENT / f'runtime/{method}/round_3/round.json')
        assert (labeled, pool) == (terminal['labeled_ids'], terminal['unlabeled_ids'])
    random_l, random_u = _state(PARENT, partition, 'random', 3)
    batch = order[96:128]
    assert len(batch) == len(set(batch)) == 32
    assert set(batch) <= set(random_u)
    next_l, next_u = transition(random_l, random_u, batch)
    assert len(next_l) == 461 and len(next_u) == len(random_u) - 32


def test_round_commit_merges_fit_and_selection_artifacts(tmp_path):
    directory = tmp_path / 'runtime/random/round_3'
    directory.mkdir(parents=True)
    (directory / 'selection.json').write_text('{}')
    stage = dict(method='random', round=3, files={'fit.bin': 'hash'}, selected=[])
    selection = dict(selected=list(range(32)), files={'bank.bin': 'hash'})
    record = _commit_round(tmp_path, 'random', 3, stage, selection)
    assert record['selected'] == list(range(32))
    assert set(record['files']) == {'fit.bin', 'bank.bin', 'runtime/random/round_3/selection.json'}
    assert record == _commit_round(tmp_path, 'random', 3, stage, selection)
