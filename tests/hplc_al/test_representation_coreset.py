from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hplc_al import training
from hplc_al.common import ids_hash, state_hash
from hplc_al.coreset import cosine_distance, kcenter, morgan_bank, nested_random_order
from hplc_al.latent import extract_h_graph
from hplc_al.protocol import role_ids, transition
from hplc_al.representation_runner import METHODS, verify_records
from hplc_al.transfer_v2_runner import _fit_config


def test_nested_random_is_one_fixed_order_with_disjoint_batches():
    ids = list(range(100))
    order = nested_random_order(ids, 123)
    assert order == nested_random_order(ids[::-1], 123)
    assert order == np.random.default_rng(123).permutation(ids).tolist()
    labeled, pool = [101], ids
    for r in range(3):
        labeled, pool = transition(labeled, pool, order[r * 32:(r + 1) * 32])
    assert len(labeled) == 97 and len(pool) == 4
    assert nested_random_order(ids, 124) != order


def test_kcenter_existing_centers_and_stable_ties():
    ids = [30, 20, 10, 40]
    positions = np.array([0, 2, -2, 3])
    distance = np.abs(positions[:, None] - positions)
    selected, _ = kcenter(distance, ids, [30], [40, 20, 10], 3)
    assert selected == [40, 10, 20]
    selected, _ = kcenter(distance, ids, [30, 40], [20, 10], 1)
    assert selected == [10]
    with pytest.raises(ValueError):
        kcenter(distance, ids, [30], [30, 20], 1)
    with pytest.raises(ValueError):
        cosine_distance(np.zeros((2, 128)))


def test_latent_exact_order_and_state_preservation(graphs, partition):
    model = training.fresh_model()
    ids = role_ids(partition, 'l0')[:3][::-1]
    before = state_hash(model)
    bank, audit = extract_h_graph(model, graphs, ids)
    assert bank['ids'].tolist() == ids
    assert bank['raw_latent'].shape == (3, 128)
    assert audit['state_before'] == audit['state_after'] == before
    assert np.allclose(np.linalg.norm(bank['unit_latent'], axis=1), 1)
    again, _ = extract_h_graph(model, graphs, ids[::-1])
    assert np.allclose(bank['raw_latent'], again['raw_latent'][::-1], atol=1e-5)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError):
        extract_h_graph(model, graphs, ids + ids)
    g, h = graphs[ids[0]]
    bad = g.clone()
    bad.y = torch.tensor([1.0])
    with pytest.raises(PermissionError):
        extract_h_graph(model, {ids[0]: (bad, h)}, [ids[0]])


def test_morgan_chirality_and_label_free_input(tmp_path):
    path = tmp_path / 'smiles.csv'
    path.write_text('SMILES\nN[C@@H](C)C(=O)O\nN[C@H](C)C(=O)O\n')
    bits, d, _, audit = morgan_bank([0, 1], path)
    assert bits.shape == (2, 2048)
    assert 0 < d[0, 1] < 1
    assert audit['observed_distinguished_chiral'] == 1
    assert audit['observed_distinguished_achiral'] == 0


def test_fit_seed_metadata_different_seeds(tmp_path, monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, g, h):
            return self.weight.expand(g.num_graphs, 3), None

    monkeypatch.setattr(training, 'fresh_model', lambda seed: Tiny())
    monkeypatch.setattr(training, 'load_model', lambda path: Tiny())
    monkeypatch.setattr(training, 'batches', lambda graphs, ids, size:
                        iter([(SimpleNamespace(num_graphs=len(ids)), None)]))
    monkeypatch.setattr(training, 'predict', lambda *args: np.zeros((2, 3)))
    monkeypatch.setattr(training, 'original_loss', lambda p, y: (p[:, 1] - y).square().mean())
    config = {**_fit_config(), 'maximum_epochs': 1, 'initialization_seed': 525, 'training_seed': 123}
    rec = training.fit({}, [0, 1], np.array([1., 2.]), [2, 3], np.array([2., 3.]), config, tmp_path, {})
    assert rec['initialization_seed'] == 525
    assert rec['training_seed'] == 123


def test_complete_firewall_and_round_zero_fairness():
    partition = {'rows': [dict(sample_index=i, role='l0' if i < 333 else 'u0') for i in range(500)]}
    records = {}
    for method in METHODS:
        labeled, pool = list(range(333)), list(range(333, 500))
        for r in range(4):
            selected = pool[:32] if r < 3 else []
            records[f'{method}/{r}'] = dict(method=method, round=r, budget=333 + r * 32,
                labeled_ids=labeled, unlabeled_ids=pool, selected=selected,
                checkpoint_hash='shared', prediction_hash='shared', L_hash=ids_hash(labeled),
                initialization_hash='same', test_truth_access_count=0)
            if r < 3:
                labeled, pool = transition(labeled, pool, selected)
    verify_records(records, partition)
    missing = dict(records)
    missing.pop('morgan_coreset/3')
    with pytest.raises(PermissionError):
        verify_records(missing, partition)
    records['morgan_coreset/0']['prediction_hash'] = 'different'
    with pytest.raises(PermissionError):
        verify_records(records, partition)
