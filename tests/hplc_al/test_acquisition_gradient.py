import numpy as np
import pytest
import torch

from hplc_al.acquisition import conditional_gradient_maxdet, lcmd_tp_select
from hplc_al.common import INIT_SEED, RANDOM_SEED, state_hash
from hplc_al.data import batches, predict
from hplc_al.gradient import extract, mapping
from hplc_al.protocol import role_ids
from hplc_al.training import fresh_model


@pytest.mark.parametrize("batch", [1, 32, 40])
def test_lcmd_zero_mass_empty_cluster_fixed(batch):
    result = lcmd_tp_select(np.ones((40, 1)), np.array([[0.0], [1.0]]), batch)
    assert result.selected_pool_positions.tolist() == list(range(batch))


@pytest.mark.parametrize("method", ["lcmd", "maxdet"])
def test_exact_batch_unique_deterministic(method):
    x = np.random.default_rng(22).normal(size=(100, 12))
    if method == "lcmd":
        first = lcmd_tp_select(x[10:], x[:10], 32).selected_pool_positions
        second = lcmd_tp_select(x[10:], x[:10], 32).selected_pool_positions
        assert set(first) <= set(range(90))
    else:
        first = conditional_gradient_maxdet(x, range(10), range(10, 100), 32).selected_positions
        second = conditional_gradient_maxdet(x, range(10), range(10, 100), 32).selected_positions
        assert set(first) <= set(range(10, 100))
    assert len(first) == len(set(first)) == 32
    assert np.array_equal(first, second)


def test_lcmd_reference_geometry():
    # A literal independent slow implementation compares recomputed distances each step.
    x = np.random.default_rng(99).normal(size=(90, 7))
    pool, centers = x[10:], list(x[:10])
    available = list(range(len(pool)))
    selected = []
    for _ in range(32):
        d = ((pool[available, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2)
        near = d.argmin(axis=1)
        minimum = d[np.arange(len(available)), near]
        mass = np.bincount(near, weights=minimum, minlength=len(centers))
        mass[np.bincount(near, minlength=len(centers)) == 0] = -np.inf
        cluster = mass.argmax()
        members = np.flatnonzero(near == cluster)
        idx = int(members[minimum[members].argmax()])
        point = available.pop(idx)
        selected.append(point)
        centers.append(pool[point])
    assert lcmd_tp_select(x[10:], x[:10], 32).selected_pool_positions.tolist() == selected


def test_maxdet_direct_inverse_reference():
    x = np.random.default_rng(19).normal(size=(80, 6))
    psi = x / np.sqrt(np.mean(np.sum(x[:12] ** 2, axis=1)))
    A = np.eye(6) + psi[:12].T @ psi[:12]
    remaining = list(range(12, 80))
    selected = []
    gains = []
    for _ in range(32):
        p = psi[remaining]
        scores = np.einsum("ij,ji->i", p, np.linalg.solve(A, p.T))
        index = int(scores.argmax())
        point = remaining.pop(index)
        selected.append(point)
        gains.append(np.log1p(scores[index]))
        A += np.outer(psi[point], psi[point])
    result = conditional_gradient_maxdet(x, range(12), range(12, 80), 32)
    assert result.selected_positions.tolist() == selected
    assert np.allclose(
        [r["marginal_logdet_gain"] for r in result.trace], gains, rtol=1e-12, atol=1e-12
    )


def test_maxdet_deterministic_ties():
    x = np.ones((50, 3))
    assert conditional_gradient_maxdet(
        x, [0, 1], list(range(2, 50)), 32
    ).selected_positions.tolist() == list(range(2, 34))


def test_random_nested_independent_repeatable(partition):
    u = role_ids(partition, "u0")
    a = np.random.default_rng(RANDOM_SEED).permutation(u)
    b = np.random.default_rng(RANDOM_SEED).permutation(u)
    assert np.array_equal(a, b) and len(set(a[:96])) == 96
    assert not set(a[:32]) & set(a[32:64])
    assert not np.array_equal(a, np.random.default_rng(INIT_SEED).permutation(u))


def test_scratch_initial_state_and_bn():
    models = [fresh_model() for _ in range(3)]
    assert len({state_hash(m) for m in models}) == 1
    with torch.no_grad():
        next(models[0].parameters()).add_(1)
    assert state_hash(models[0]) != state_hash(fresh_model())


def test_single_output_sketch_dimension_and_repeatability(graphs, partition):
    model = fresh_model()
    with torch.no_grad():
        model.graph_pred_linear.bias[1].fill_(100)
    ids = role_ids(partition, "l0")[:3]
    before = state_hash(model)
    a, aa = extract(model, graphs, ids)
    b, bb = extract(model, graphs, ids)
    assert a.shape == (3, 512) and np.isfinite(a).all()
    assert np.array_equal(a, b) and aa["mapping_hash"] == bb["mapping_hash"]
    assert state_hash(model) == before
    assert aa["parameter_count"] == 818091
    assert aa["tensors_receiving_gradient"] > 100
    assert aa["sketch_norm"]["zero_count"] == 0
    slices, buckets, signs, _ = mapping(model)
    assert len(buckets) == len(signs) == sum(p.numel() for p in model.parameters())
    assert slices[-1]["stop"] == 818091


def test_clamped_negative_central_zero_gradient(graphs, partition):
    model = fresh_model()
    with torch.no_grad():
        model.graph_pred_linear.weight[1].zero_()
        model.graph_pred_linear.bias[1].fill_(-100)
    ids = role_ids(partition, "l0")[:1]
    features, audit = extract(model, graphs, ids)
    assert np.count_nonzero(features) == 0 and audit["clamped_central_rows"] == 1


def test_label_free_inference_batch_consistency(graphs, partition):
    model = fresh_model()
    ids = role_ids(partition, "l0")[:3]
    together = predict(model, graphs, ids)
    separate = np.concatenate([predict(model, graphs, [i]) for i in ids])
    assert np.allclose(together, separate, atol=1e-5, rtol=1e-5)


def test_gradient_batch_organization_consistent(graphs, partition):
    model = fresh_model()
    with torch.no_grad():
        model.graph_pred_linear.bias[1].fill_(100)
    ids = role_ids(partition, "l0")[:2]
    individual, _ = extract(model, graphs, ids)
    slices, buckets, signs, _ = mapping(model)
    g, h = next(batches(graphs, ids, 2))
    output, _ = model(g, h)
    params = tuple(model.parameters())
    for i in range(2):
        gradients = torch.autograd.grad(output[i, 1], params, allow_unused=True, retain_graph=True)
        flat = np.concatenate(
            [
                np.zeros(p.numel()) if grad is None else grad.detach().numpy().reshape(-1)
                for p, grad in zip(params, gradients)
            ]
        )
        feature = np.bincount(buckets, weights=flat * signs, minlength=512)
        assert np.allclose(individual[i], feature, atol=1e-3, rtol=1e-3)
