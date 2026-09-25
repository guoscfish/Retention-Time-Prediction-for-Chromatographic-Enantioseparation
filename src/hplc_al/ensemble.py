"""Three scratch-trained scalar central-output predictors for HPLC Hybrid."""

import numpy as np


def member_seed(round_index, member):
    if round_index not in range(5) or member not in (1, 2):
        raise ValueError("only acquisition rounds 0..4 and extra members 1/2")
    return 525_000_000 + 10_000 * round_index + member


def uncertainty(predictions, frozen_l333_sd):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError("three finite member-by-candidate central predictions required")
    if not np.isfinite(frozen_l333_sd) or frozen_l333_sd <= 0:
        raise ValueError("positive frozen L333 scale required")
    raw = values.std(axis=0, ddof=0)
    return raw, raw / frozen_l333_sd, np.argsort(-raw, kind="stable")
