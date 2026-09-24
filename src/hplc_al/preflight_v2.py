"""Short, label-free preflight for the v2 transfer protocol."""

from __future__ import annotations

from dataclasses import dataclass

from .deterministic_shuffle import epoch_order
from .transfer_v2 import BUDGETS, L0, TRAINING, label_firewall_keys


@dataclass(frozen=True)
class DurationDecision:
    status: str
    best_epoch: int
    maximum_epochs: int
    patience: int


def duration_decision(best_epoch: int, maximum_epochs: int = 500, patience: int = 100) -> DurationDecision:
    """Apply the preregistered insufficient-duration rule without retuning."""
    if maximum_epochs < 1 or patience < 1 or not 1 <= best_epoch <= maximum_epochs:
        raise ValueError("invalid duration audit values")
    status = "INSUFFICIENT_DURATION_PROTOCOL" if maximum_epochs - best_epoch < patience else "PASS"
    return DurationDecision(status, best_epoch, maximum_epochs, patience)


def run_preflight() -> dict:
    """Check protocol constants, repeated permutations and coverage only."""
    ids = list(range(L0))
    truth = [float(i) for i in ids]
    first = epoch_order(ids, truth, 525, 0)
    repeat = epoch_order(ids, truth, 525, 0)
    second = epoch_order(ids, truth, 525, 1)
    return {
        "status": "PASS_SHUFFLE_PROTOCOL_ONLY",
        "training": TRAINING,
        "budgets": list(BUDGETS),
        "method_budget_records": len(label_firewall_keys()),
        "same_seed_same_epoch": bool((first.ids == repeat.ids).all()),
        "different_epoch_changes": bool(not (first.ids == second.ids).all()),
        "epoch_coverage": bool(len(set(first.ids.tolist())) == L0),
        "duration_rule": "report insufficient if maximum_epochs - best_epoch < patience",
        "duration_training_smoke": "NOT_RUN_TORCH_UNAVAILABLE",
        "long_al_started": False,
    }
