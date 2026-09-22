"""Exercise interrupted real-model fitting without reading any dataset labels."""

import numpy as np
import pytest

from hplc_al import training
from hplc_al.protocol import role_ids
from hplc_al.runner import training_config


def test_interrupted_fit_scratch_restart_and_cache_binding(
    graphs, partition, tmp_path, monkeypatch
):
    ids = role_ids(partition, "l0")[:3]
    valid = role_ids(partition, "validation")[:3]
    args = (
        graphs,
        ids,
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
        valid,
        np.array([4.0, 5.0, 6.0], dtype=np.float32),
        training_config(2, 2),
    )
    loss = training.original_loss
    calls = 0

    def interrupt(pred, truth):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return loss(pred, truth)

    monkeypatch.setattr(training, "original_loss", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.fit(*args, tmp_path / "resumed", {"test": "recovery"})
    monkeypatch.setattr(training, "original_loss", loss)
    resumed = training.fit(*args, tmp_path / "resumed", {"test": "recovery"})
    reference = training.fit(*args, tmp_path / "reference", {"test": "recovery"})
    assert resumed["attempted_fits"] == 2 and resumed["abandoned_attempts"] == ["attempt_000"]
    assert resumed["initialization_hash"] == reference["initialization_hash"]
    assert resumed["checkpoint_state_hash"] == reference["checkpoint_state_hash"]
    assert training.fit(*args, tmp_path / "resumed", {"test": "recovery"}) == resumed
    with pytest.raises(RuntimeError, match="immutable artifact differs"):
        training.fit(*args, tmp_path / "resumed", {"test": "different"})
    checkpoint = tmp_path / "resumed" / resumed["checkpoint_path"]
    checkpoint.write_bytes(b"partial corrupt write")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        training.fit(*args, tmp_path / "resumed", {"test": "recovery"})
