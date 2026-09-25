import inspect
import math

import numpy as np
import pytest

from hplc_al.common import read_json, sha, verify_files
from hplc_al.ensemble import member_seed, uncertainty
from hplc_al.hybrid import select as hybrid_select
from hplc_al.ivr_hybrid_screen import PARENT, EXTENSION
from hplc_al.kernel_ivr import select as ivr_select
from hplc_al.protocol import RestrictedLabelStore, role_ids


def test_ivr_matches_direct_recomputed_covariance_and_is_deterministic():
    small = np.random.default_rng(41).normal(size=(18, 7))
    features = np.pad(small, ((0, 0), (0, 505)))
    labeled, candidates = np.arange(4), np.arange(4, 18)
    first = ivr_select(features, labeled, candidates, 5)
    second = ivr_select(features, labeled, candidates, 5)
    assert first['selected_candidate_positions'] == second['selected_candidate_positions']
    x = features[:, :7] / np.sqrt(np.mean(np.sum(features**2, axis=1)))
    current = labeled.tolist()
    available = candidates.tolist()
    direct = []
    for trace in first['trace']:
        c = np.linalg.inv(np.eye(7) + x[current].T @ x[current])
        m = x.T @ x / len(x)
        scores = [float(x[i] @ c @ m @ c @ x[i] / (1 + x[i] @ c @ x[i])) for i in available]
        chosen = available.pop(int(np.argmax(scores)))
        direct.append(chosen)
        assert trace['score'] == pytest.approx(max(scores), rel=1e-9, abs=1e-12)
        current.append(chosen)
    assert candidates[first['selected_candidate_positions']].tolist() == direct
    assert 'truth' not in inspect.signature(ivr_select).parameters
    assert 'labels' not in inspect.signature(ivr_select).parameters


def test_ensemble_scalar_std_shape_order_and_seed_formula():
    predictions = np.array([[1., 2., 3.], [1., 4., 5.], [1., 6., 7.]])
    raw, normalized, order = uncertainty(predictions, 2.)
    assert raw.shape == normalized.shape == (3,)
    np.testing.assert_allclose(normalized, raw / 2)
    assert order.tolist() == [1, 2, 0]
    assert member_seed(0, 1) == 525000001
    assert member_seed(4, 2) == 525040002


def test_hybrid_exact_top_quarter_membership_and_ties():
    rng = np.random.default_rng(17)
    latent = rng.normal(size=(10 + 160, 128))
    latent /= np.linalg.norm(latent, axis=1)[:, None]
    predictions = np.vstack([np.zeros(160), np.arange(160), 2*np.arange(160)])
    result = hybrid_select(latent, np.arange(10), np.arange(10, 170), predictions, 8.8792405476)
    shortlist = result['shortlist_candidate_positions']
    assert len(shortlist) == max(32, math.ceil(.25 * 160)) == 40
    assert shortlist == list(range(159, 119, -1))
    assert len(result['selected_candidate_positions']) == 32
    assert set(result['selected_candidate_positions']) <= set(shortlist)
    tied = np.zeros_like(predictions)
    first = hybrid_select(latent, np.arange(10), np.arange(10, 170), tied, 8.8792405476)
    second = hybrid_select(latent, np.arange(10), np.arange(10, 170), tied, 8.8792405476)
    assert first['shortlist_candidate_positions'] == list(range(40))
    assert first['selected_candidate_positions'] == second['selected_candidate_positions']
    equal_latent = np.ones((42, 128)) / np.sqrt(128)
    equal = hybrid_select(equal_latent, np.arange(2), np.arange(2, 42),
                          np.zeros((3, 40)), 1., batch_size=32)
    assert equal['selected_candidate_positions'] == list(range(32))


def test_historical_l333_and_comparator_seals_and_test_gate(tmp_path):
    protocol = read_json(PARENT / 'protocol.json')
    shared = read_json(PARENT / 'shared/fit/fit.json')
    assert shared['checkpoint_hash'] == protocol['shared_checkpoint_hash']
    for method in ('random', 'raw_gradient_lcmd'):
        record = read_json(PARENT / f'runtime/{method}/round_0/round.json')
        assert record['checkpoint_hash'] == shared['checkpoint_hash']
        assert record['prediction_hash'] == read_json(PARENT / 'runtime/random/round_0/round.json')['prediction_hash']
    verify_files(PARENT, read_json(PARENT / 'completion_manifest.json')['files'])
    verify_files(EXTENSION, read_json(EXTENSION / 'completion_manifest.json')['files'])
    partition = read_json(PARENT / 'splits/partition.json')
    test_ids = role_ids(partition, 'test')
    gate = RestrictedLabelStore(partition, tmp_path / 'access.csv', 'gate')
    with pytest.raises(PermissionError):
        gate.reveal(test_ids, 'final_test')
    assert sha(PARENT / 'shared/fit/attempt_000/best.pt') == shared['checkpoint_hash']
