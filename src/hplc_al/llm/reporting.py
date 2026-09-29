"""Selection diagnostics and validation-only fixed-denominator metrics."""

import numpy as np

from ..common import metrics
from .full_pool import distribution, pair_counts


def selection_diagnostics(cards, plan):
    lookup = {c["id"]: c for c in cards}
    selected = [lookup[i] for i in plan["selected_ids"]]
    usages = [r.get("usage") for r in plan["receipts"]]
    return {
        "full_pool_screen_coverage": plan["screening_audit"][
            "full_pool_screen_coverage"
        ],
        "screened_candidate_count": len(plan["screening_audit"]["all_screened_ids"]),
        "stage1_nominee_rate": len(plan["nominee_ids"]) / len(cards),
        "selected_from_nominees_rate": len(
            set(plan["selected_ids"]) & set(plan["nominee_ids"])
        )
        / len(selected),
        "selected_coverage": distribution([c["coverage"] for c in selected]),
        "selected_q_width": distribution([c["q_width"] for c in selected]),
        "scaffold_diversity": len({c["scaffold"] for c in selected}),
        "empty_scaffold_count": sum(not c["scaffold"] for c in selected),
        **pair_counts(selected),
        "llm_total_calls": len(plan["receipts"]),
        "usage_complete": all(u is not None for u in usages),
        "input_tokens": sum(u["input_tokens"] for u in usages)
        if all(u is not None for u in usages)
        else None,
        "output_tokens": sum(u["output_tokens"] for u in usages)
        if all(u is not None for u in usages)
        else None,
    }


def evaluate(prediction, truth, scale, budget):
    return {
        "budget": budget,
        **metrics(prediction, truth, scale),
        "fixed_denominator": scale,
    }


def label_aulc(records):
    budgets = [r["budget"] for r in records]
    if len(budgets) < 2 or any(b <= a for a, b in zip(budgets, budgets[1:])):
        raise ValueError("AULC requires at least two increasing budgets")
    if len({r["fixed_denominator"] for r in records}) != 1:
        raise ValueError("NRMSE denominator drift")
    errors = np.asarray([r["nrmse"] for r in records])
    if not np.isfinite(errors).all():
        raise ValueError("nonfinite AULC")
    raw = float(np.sum(np.diff(budgets) * (errors[1:] + errors[:-1]) / 2))
    return {
        "interval": [budgets[0], budgets[-1]],
        "raw_nrmse_area": raw,
        "mean_nrmse": raw / (budgets[-1] - budgets[0]),
    }
