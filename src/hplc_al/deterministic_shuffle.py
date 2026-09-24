"""Shared deterministic epoch permutations for HPLC graph/target training.

The permutation is created once per epoch and applied to both graph identities
and the separately stored target array.  This module intentionally contains no
DataLoader(shuffle=True) path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EpochOrder:
    epoch: int
    permutation: np.ndarray
    ids: np.ndarray
    truth: np.ndarray


def epoch_order(
    labeled_ids: Sequence[int],
    truth: Sequence[float],
    training_seed: int,
    epoch: int,
) -> EpochOrder:
    """Return one shared, reproducible graph/target order for ``epoch``.

    ``epoch`` is zero based.  Validation, test and U0 arrays are not accepted
    here, so they cannot accidentally enter the shuffle path.
    """
    ids = np.asarray(labeled_ids)
    values = np.asarray(truth)
    if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values):
        raise ValueError("labeled IDs and truth must be aligned one-dimensional arrays")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("labeled IDs must be unique")
    if int(epoch) < 0:
        raise ValueError("epoch must be nonnegative")
    rng = np.random.default_rng(int(training_seed) * 10000 + int(epoch))
    perm = rng.permutation(len(ids))
    ordered_ids = ids[perm].copy()
    ordered_truth = values[perm].copy()
    if not np.array_equal(ordered_ids, ids[perm]) or not np.array_equal(ordered_truth, values[perm]):
        raise RuntimeError("epoch order construction is not deterministic")
    return EpochOrder(int(epoch), perm, ordered_ids, ordered_truth)


def epoch_batches(
    labeled_ids: Sequence[int],
    truth: Sequence[float],
    training_seed: int,
    epoch: int,
    batch_size: int = 256,
):
    """Yield ``(ids, truth)`` batches from the one shared permutation."""
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    order = epoch_order(labeled_ids, truth, training_seed, epoch)
    for start in range(0, len(order.ids), int(batch_size)):
        stop = min(start + int(batch_size), len(order.ids))
        yield order.ids[start:stop], order.truth[start:stop]


def batch_sizes(count: int, batch_size: int = 256) -> list[int]:
    """Return the exact non-dropping batch sizes for a labeled set."""
    if int(count) < 0 or int(batch_size) < 1:
        raise ValueError("count must be nonnegative and batch_size positive")
    return [min(int(batch_size), int(count) - start) for start in range(0, int(count), int(batch_size))]


def assert_epoch_coverage(labeled_ids: Sequence[int], epoch_ids: Sequence[int]) -> None:
    """Prove that an epoch contains every labeled ID exactly once."""
    expected = np.asarray(labeled_ids)
    observed = np.asarray(epoch_ids)
    if len(observed) != len(expected) or len(np.unique(observed)) != len(observed):
        raise AssertionError("epoch has duplicate or missing labeled IDs")
    if set(observed.tolist()) != set(expected.tolist()):
        raise AssertionError("epoch labeled-ID coverage drift")
