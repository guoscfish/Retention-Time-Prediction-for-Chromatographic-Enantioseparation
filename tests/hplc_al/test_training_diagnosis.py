"""Diagnosis exposure, firewall and gradient equivalence without real extra fits."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from hplc_al.gradient import extract
from hplc_al.protocol import role_ids
from hplc_al.training import fresh_model
from hplc_diagnosis import training
from hplc_diagnosis.geometry import correlations, gradients, unit_features
from hplc_diagnosis.runner import DiagnosisLabels


@pytest.mark.parametrize(
    "arm,steps,updates,presentations",
    [
        ("A", 1, 300, 106800),
        ("B", 2, 300, 53400),
        ("C", 3, 300, 35600),
        ("D", 2, 600, 106800),
        ("E", 3, 900, 106800),
    ],
)
def test_registered_exact_exposure(arm, steps, updates, presentations):
    config = training.ARMS[arm]
    assert training.exposure(356, config["batch_size"], config["maximum_epochs"]) == {
        "steps_per_epoch": steps,
        "total_optimizer_steps": updates,
        "total_sample_presentations": presentations,
    }


def test_practical_selection_excludes_reference_and_uses_registered_ties():
    records = [
        dict(
            arm=arm,
            initialization_seed=525,
            best_metrics={"rmse": 1 if arm == "A" else 2},
            epochs_run=config["maximum_epochs"],
            **training.exposure(356, config["batch_size"], config["maximum_epochs"]),
        )
        for arm, config in training.ARMS.items()
    ]
    assert training.choose_challenger(records) == "C"
    records[-1]["best_metrics"]["rmse"] = 0.5
    assert training.choose_challenger(records) == "E"
    with pytest.raises(ValueError):
        training.choose_challenger(records[:-1])


@pytest.mark.parametrize(
    "role,purpose", [("u0", "fit"), ("test", "validation"), ("test", "final_test")]
)
def test_diagnosis_firewall_before_source_open(partition, tmp_path, role, purpose):
    store = DiagnosisLabels(
        partition, tmp_path / "audit.csv", "diagnosis", source=tmp_path / "absent.csv"
    )
    store.test_frozen = True
    with pytest.raises(PermissionError):
        store.reveal(role_ids(partition, role), purpose)
    with pytest.raises(PermissionError):
        store.commit_selection(tmp_path / "absent.json")
    with pytest.raises(PermissionError):
        store.unlock_test(tmp_path)


def test_unit_direction_preserves_zero_rows():
    features = np.array([[0.0, 0.0], [3.0, 4.0], [-5.0, 0.0]])
    unit = unit_features(features)
    assert np.array_equal(unit[0], features[0])
    assert np.allclose(np.linalg.norm(unit[1:], axis=1), 1)
    assert np.allclose(unit[1], [0.6, 0.8])
    with pytest.raises(ValueError):
        unit_features(features, 0)


def test_correlation_square_rank_invariance_and_constant_handling():
    frame = pd.DataFrame(
        dict(
            full_gradient_norm=[1.0, 4.0, 2.0, 3.0],
            sketch_gradient_norm=[2.0, 8.0, 4.0, 6.0],
            absolute_error=[0.0, 3.0, 1.0, 2.0],
            squared_error=[0.0, 9.0, 1.0, 4.0],
        )
    )
    rows = correlations(frame)
    assert rows[0]["spearman"] == rows[1]["spearman"] == 1
    frame["full_gradient_norm"] = 0
    assert correlations(frame)[0]["spearman"] is None


def test_diagnosis_gradients_match_original_engine(graphs, partition):
    model = fresh_model()
    ids = role_ids(partition, "l0")[:2]
    reference, reference_audit = extract(model, graphs, ids)
    features, frame, audit = gradients(model, graphs, ids)
    assert np.array_equal(features, reference)
    assert audit["mapping_hash"] == reference_audit["mapping_hash"]
    assert np.allclose(
        frame.sketch_gradient_norm, np.linalg.norm(features.astype(np.float64), axis=1)
    )
    assert frame.full_gradient_norm.mean() == pytest.approx(
        reference_audit["full_gradient_norm"]["mean"]
    )


def test_fixed_fit_does_not_stop_on_validation_plateau(tmp_path, monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, graph, angle):
            return self.weight.expand(graph.num_graphs, 3), None

    monkeypatch.setitem(training.ARMS, "A", dict(batch_size=2, maximum_epochs=3))
    monkeypatch.setattr(training, "fresh_model", lambda seed: TinyModel())
    monkeypatch.setattr(
        training,
        "batches",
        lambda graphs, ids, size: [
            (SimpleNamespace(num_graphs=min(size, len(ids) - start)), None)
            for start in range(0, len(ids), size)
        ],
    )
    monkeypatch.setattr(
        training, "original_loss", lambda pred, target: (pred[:, 1] - target).square().mean()
    )
    monkeypatch.setattr(training, "predict", lambda *args: np.zeros((2, 3), dtype=np.float32))
    record = training.fixed_fit(
        {},
        list(range(5)),
        np.ones(5) * 2,
        [5, 6],
        np.array([1.0, 2.0]),
        "A",
        525,
        1.0,
        tmp_path,
        {"test": "synthetic"},
    )
    assert record["epochs_run"] == 3
    assert record["best_epoch"] == 1
    assert record["total_optimizer_steps"] == 9
    assert record["total_sample_presentations"] == 15
    assert record["final_metrics"] == record["best_metrics"]
    assert record["checkpoint_hash"] != record["final_checkpoint_hash"]
    assert record == training.fixed_fit(
        {},
        list(range(5)),
        np.ones(5) * 2,
        [5, 6],
        np.array([1.0, 2.0]),
        "A",
        525,
        1.0,
        tmp_path,
        {"test": "synthetic"},
    )
