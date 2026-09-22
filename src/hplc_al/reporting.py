"""One post-freeze test reveal; descriptive development reporting only."""

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    BUDGETS,
    METHODS,
    STUDY,
    atomic_json,
    metrics,
    partial_aulc,
    read_json,
    sha,
)
from .protocol import RestrictedLabelStore, role_ids, verify_global_freeze
from .runner import assert_frozen_protocol, checked_partition, verify_completed_study


def _test_truth(study, partition):
    path, receipt = study / "results/test_truth.npz", study / "results/test_reveal.json"
    test = role_ids(partition, "test")
    if receipt.exists():
        r = read_json(receipt)
        if sha(path) != r["sha256"] or r["freeze_hash"] != sha(
            study / "global_pre_test_freeze.json"
        ):
            raise RuntimeError("post-freeze truth cache drift")
        with np.load(path) as stored:
            if stored["ids"].tolist() != test:
                raise RuntimeError("test truth ID order drift")
            return stored["truth"].copy()
    with (study / "label_access_audit.csv").open() as f:
        previous = [
            r for r in csv.DictReader(f) if r["purpose"] == "final_test" and r["allowed"] == "True"
        ]
    if previous:
        raise RuntimeError("interrupted test reveal: refuse a second source-label read")
    store = RestrictedLabelStore(partition, study / "label_access_audit.csv", "global_evaluation")
    store.unlock_test(study)
    truth = store.reveal(test, "final_test")
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez(f, ids=np.array(test), truth=truth)
    path.with_suffix(".tmp").replace(path)
    atomic_json(
        receipt,
        dict(
            sha256=sha(path),
            freeze_hash=sha(study / "global_pre_test_freeze.json"),
            rows=len(test),
            source_reveal_count=1,
        ),
    )
    return truth


def report(study=STUDY):
    """Generate post-freeze diagnostics once; preserve completed reviewed reports."""
    study = Path(study)
    if (study / "completion_manifest.json").exists():
        verified = verify_completed_study(study)
        print(json.dumps(dict(**verified, report=str(study / "DEVELOPMENT_REPORT.md"))))
        return verified
    protocol = assert_frozen_protocol(study)
    verify_global_freeze(study)
    partition = checked_partition(study)
    out = study / "results"
    out.mkdir(exist_ok=True)
    truth = _test_truth(study, partition)
    valid, test = role_ids(partition, "validation"), role_ids(partition, "test")
    rows, runtimes, selections, overlap = [], [], {}, []
    for method in METHODS:
        selections[method] = []
        for r, b in enumerate(BUDGETS):
            directory = study / f"runtime/{method}/round_{r}"
            record = read_json(directory / "round.json")
            with np.load(directory / "predictions.npz") as saved:
                if saved["ids"].tolist() != valid + test:
                    raise RuntimeError("frozen prediction ID alignment failed")
                test_metrics = metrics(
                    saved["predictions"][len(valid) :, 1], truth, protocol["L0_population_std"]
                )
            for split, values in [
                ("validation", record["validation_metrics"]),
                ("test", test_metrics),
            ]:
                rows.append(
                    dict(
                        method=method,
                        round=r,
                        budget=b,
                        split=split,
                        validation_label_count=247,
                        total_observed_non_test=b + 247,
                        **values,
                    )
                )
            runtimes.append(
                dict(
                    method=method,
                    round=r,
                    budget=b,
                    **{
                        k: record[k]
                        for k in [
                            "training_seconds",
                            "training_actual_seconds",
                            "epochs_run",
                            "best_epoch",
                            "gradient_seconds",
                            "gradient_actual_seconds",
                            "acquisition_seconds",
                            "fit_reused",
                            "gradient_reused",
                        ]
                    },
                )
            )
            if r < 3:
                selections[method].append(set(record["selected"]))
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "learning_curves.csv", index=False)
    pd.DataFrame(runtimes).to_csv(out / "runtime_audit.csv", index=False)
    aulc = []
    for (method, split), group in curves.groupby(["method", "split"]):
        group = group.sort_values("budget")
        for metric in ["rmse", "nrmse"]:
            area = partial_aulc(group.budget.tolist(), group[metric].tolist())
            aulc.append(
                dict(
                    method=method,
                    split=split,
                    metric=metric,
                    partial_aulc_raw=area["raw"],
                    partial_aulc_mean=area["mean"],
                )
            )
    pd.DataFrame(aulc).to_csv(out / "partial_aulc.csv", index=False)
    for first, second in combinations(METHODS, 2):
        for r in range(4):
            a, b = (
                (selections[first][r], selections[second][r])
                if r < 3
                else (set.union(*selections[first]), set.union(*selections[second]))
            )
            overlap.append(
                dict(
                    first=first,
                    second=second,
                    scope=f"batch_{r + 1}" if r < 3 else "cumulative_96",
                    intersection=len(a & b),
                    union=len(a | b),
                    jaccard=len(a & b) / len(a | b),
                )
            )
    pd.DataFrame(overlap).to_csv(out / "selection_overlap.csv", index=False)
    signals = {}
    for split in ["validation", "test"]:
        a = {
            r["method"]: r["partial_aulc_mean"]
            for r in aulc
            if r["split"] == split and r["metric"] == "nrmse"
        }
        endpoint = (
            curves[(curves.split == split) & (curves.budget == 452)]
            .set_index("method")
            .nrmse.to_dict()
        )
        signals[split] = {
            m: dict(
                aulc_relative_improvement=(a["random"] - a[m]) / a["random"],
                endpoint_relative_improvement=(endpoint["random"] - endpoint[m])
                / endpoint["random"],
                descriptive_direction_both_lower=a[m] < a["random"]
                and endpoint[m] < endpoint["random"],
            )
            for m in ["lcmd", "maxdet"]
        }
    recommend = any(r["descriptive_direction_both_lower"] for r in signals["validation"].values())
    decision = dict(
        status="COMPLETE_TINY_DEVELOPMENT_SMOKE",
        phase2="COMPLETE",
        phase3="tiny development smoke",
        evidence="ENGINEERING / DEVELOPMENT EVIDENCE ONLY",
        seeds=1,
        acquisition_rounds=3,
        signals=signals,
        winner_claim=False,
        external_validation=False,
        recommend_next_screen=bool(recommend),
        next_step=(
            "consider explicitly authorized 2-seed 6–10-round screen with frozen protocol"
            if recommend
            else "inspect gradient geometry, training stability, overlap and batch diversity before expanding"
        ),
        automatic_expansion=False,
    )
    atomic_json(study / "decision.json", decision)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ["validation", "test"]):
        for method in METHODS:
            group = curves[(curves.split == split) & (curves.method == method)].sort_values(
                "budget"
            )
            ax.plot(group.budget, group.rmse, marker="o", label=method)
        ax.set(
            xlabel="Active labels (247 validation labels additional)",
            ylabel="Central RTv RMSE",
            title=split,
        )
        ax.set_xticks(BUDGETS)
        ax.legend()
    figure.suptitle("ENGINEERING / DEVELOPMENT EVIDENCE ONLY — 1 seed")
    figure.tight_layout()
    figure.savefig(out / "learning_curves.png", dpi=160)
    plt.close(figure)
    lines = [
        "# ODH gradient AL tiny development smoke",
        "",
        "**ENGINEERING / DEVELOPMENT EVIDENCE ONLY.** One seed; no winner or independent external-validation claim.",
        "",
        "## Frozen training protocol",
        "",
        json.dumps(protocol["training"], indent=2),
        "",
        f"NRMSE = RMSE / L0 population standard deviation ({protocol['L0_population_std']:.9g}).",
        "Validation 247; active budgets 356/388/420/452; total observed non-test labels 603/635/667/699.",
        "",
        "## Diagnostic learning curves",
        "",
        "| Split | Method | Budget | RMSE | MAE | R2 | NRMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['method']} | {row['budget']} | {row['rmse']:.5f} | {row['mae']:.5f} | {row['r2']:.5f} | {row['nrmse']:.5f} |"
        )
    lines += [
        "",
        "## Partial AULC (mean = trapezoidal area / 96; lower is better)",
        "",
        "| Split | Method | Mean NRMSE AULC |",
        "| --- | --- | ---: |",
    ]
    for row in aulc:
        if row["metric"] == "nrmse":
            lines.append(f"| {row['split']} | {row['method']} | {row['partial_aulc_mean']:.6f} |")
    lines += [
        "",
        "## Runtime",
        "",
        "| Method | Budget | Epochs / best | Training seconds | Gradient seconds | Acquisition seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in runtimes:
        lines.append(
            f"| {row['method']} | {row['budget']} | {row['epochs_run']} / {row['best_epoch']} | {row['training_seconds']:.2f} | {row['gradient_seconds']:.2f} | {row['acquisition_seconds']:.3f} |"
        )
    lines += [
        "",
        "Round-0 training is shared by all methods; round-0 features by LCMD/MaxDet. Actual incremental costs are separate CSV fields.",
        "",
        "## Selection overlap",
        "",
        "| Pair | Scope | Intersection | Jaccard |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in overlap:
        lines.append(
            f"| {row['first']} / {row['second']} | {row['scope']} | {row['intersection']} | {row['jaccard']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation and stopping decision",
        "",
        json.dumps(signals, indent=2),
        "",
        decision["next_step"] + ".",
        "No next-stage execution is authorized by this report. A single seed and 96 added labels cannot establish a winner.",
        "The historical fixed test was already exposed during baseline reproduction. This study enforces a new global pre-test barrier but does not create a new independent test cohort.",
        "Later-round overlap compares different current pools. All methods retain identical architecture/loss and scratch initial state.",
        "If performance is flat, inspect extraction audits, zero norms/duplicates, gradient spread, acquisition traces and training curves before adding methods.",
        "",
        "![Learning curves](results/learning_curves.png)",
        "",
    ]
    (study / "DEVELOPMENT_REPORT.md").write_text("\n".join(lines))
    atomic_json(
        out / "report_manifest.json",
        {
            p.name: sha(p)
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "report_manifest.json"
        },
    )
    print(json.dumps(decision, indent=2), flush=True)
