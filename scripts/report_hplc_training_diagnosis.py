"""Summarize completed diagnosis evidence; never train or materialize new labels."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hplc_diagnosis_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hplc_al.common import atomic_json, read_json, sha, verify_files
from hplc_diagnosis.runner import STUDY, check_protocol, label_audit
from hplc_diagnosis.training import ARMS, choose_challenger

COLORS = {"A": "#52677e", "B": "#007f86", "C": "#d89428", "D": "#8160a4", "E": "#cb5948"}


def verified_records():
    protocol, partition = check_protocol()
    complete = read_json(STUDY / "training_complete.json")
    verify_files(STUDY, complete["fit_files"])
    records = []
    curves = []
    for name in complete["fit_files"]:
        directory = (STUDY / name).parent
        record = read_json(STUDY / name)
        verify_files(directory, record["files"])
        frame = pd.read_csv(directory / record["training_curve"])
        frame["arm"] = record["arm"]
        frame["initialization_seed"] = record["initialization_seed"]
        steps = record["steps_per_epoch"]
        assert len(frame) == ARMS[record["arm"]]["maximum_epochs"]
        assert np.array_equal(frame.epoch, np.arange(1, len(frame) + 1))
        assert np.array_equal(frame.optimizer_steps, frame.epoch * steps)
        assert np.array_equal(frame.sample_presentations, frame.epoch * 356)
        assert int(frame.validation_mse.idxmin()) + 1 == record["best_epoch"]
        assert np.isclose(frame.validation_mse.min() ** 0.5, record["best_metrics"]["rmse"])
        curves.append(frame)
        records.append(record)
    assert len(records) == 9
    chosen = choose_challenger([r for r in records if r["initialization_seed"] == 525])
    assert chosen == read_json(STUDY / "practical_arm_selection.json")["arm"]
    selection = read_json(STUDY / "practical_arm_selection.json")
    for arm, digest in selection["evidence"].items():
        assert digest == sha(STUDY / f"runtime/seed_525/{arm}/fit.json")
    for seed in [525, 1525, 2525]:
        assert (
            len({r["initialization_hash"] for r in records if r["initialization_seed"] == seed})
            == 1
        )
    combined = pd.concat(curves, ignore_index=True)
    # Longer arms must reproduce the shorter arm prefix exactly: same seed,
    # minibatches and optimizer, differing only in registered duration.
    prefix_checks = {}
    for short, long in [("B", "D"), ("C", "E")]:
        left = combined[(combined.arm == short) & (combined.initialization_seed == 525)]
        right = combined[(combined.arm == long) & (combined.initialization_seed == 525)].iloc[
            : len(left)
        ]
        columns = ["train_loss", "validation_mse", "optimizer_steps", "sample_presentations"]
        assert np.array_equal(left[columns].to_numpy(), right[columns].to_numpy())
        prefix_checks[f"{short}_{long}"] = dict(identical=True, epochs=len(left))
    return protocol, partition, records, combined, chosen


def figure_save(figure, name):
    figure.savefig(
        STUDY / f"results/figures/{name}.png", dpi=180, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def summarize():
    if (STUDY / "completion_manifest.json").exists():
        raise RuntimeError("completed diagnosis is immutable; use verify")
    protocol, partition, records, curves, chosen = verified_records()
    verify_files(STUDY, read_json(STUDY / "geometry_complete.json")["files"])
    curves.to_csv(STUDY / "results/all_training_curves.csv", index=False)
    table = pd.read_csv(STUDY / "results/training_protocol_comparison.csv")
    aggregate, pairs = [], []
    for arm in ["A", chosen]:
        rows = table[table.arm == arm]
        for metric in ["rmse", "mae", "r2", "nrmse"]:
            values = rows[f"best_validation_{metric}"].to_numpy()
            assert len(values) == 3
            aggregate.append(
                dict(
                    arm=arm,
                    metric=metric,
                    mean=float(values.mean()),
                    std=float(values.std(ddof=1)),
                    initialization_seeds=3,
                )
            )
    for seed in [525, 1525, 2525]:
        rows = table[table.initialization_seed == seed].set_index("arm")
        reference = float(rows.loc["A", "best_validation_rmse"])
        practical = float(rows.loc[chosen, "best_validation_rmse"])
        pairs.append(
            dict(
                initialization_seed=seed,
                reference_rmse=reference,
                practical_arm=chosen,
                practical_rmse=practical,
                absolute_improvement=reference - practical,
                relative_improvement_percent=(reference - practical) / reference * 100,
            )
        )
    pd.DataFrame(aggregate).to_csv(STUDY / "results/seed_metric_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(STUDY / "results/paired_seed_comparison.csv", index=False)
    mean_ref = float(np.mean([p["reference_rmse"] for p in pairs]))
    mean_practical = float(np.mean([p["practical_rmse"] for p in pairs]))
    summary = dict(
        selected_practical_arm=chosen,
        total_fits=9,
        pairs=pairs,
        aggregate=aggregate,
        mean_reference_rmse=mean_ref,
        mean_practical_rmse=mean_practical,
        mean_absolute_rmse_improvement=mean_ref - mean_practical,
        relative_mean_rmse_improvement_percent=(mean_ref - mean_practical) / mean_ref * 100,
        improved_seed_count=sum(p["absolute_improvement"] > 0 for p in pairs),
        total_fit_seconds=sum(r["seconds"] for r in records),
        total_optimizer_updates=sum(r["total_optimizer_steps"] for r in records),
        total_sample_presentations=sum(r["total_sample_presentations"] for r in records),
        label_audit=label_audit(partition),
        limitations=[
            "same validation selects checkpoints and Part B arm",
            "seed525 is selection seed, two seeds are repeats",
            "batch size changes BN/noise/sample exposure and checkpoint opportunities",
            "three initializations, one fixed L0/validation cohort",
        ],
    )
    summary["deterministic_prefix_checks"] = {
        "B_D_first_150_epochs": True,
        "C_E_first_100_epochs": True,
    }
    atomic_json(STUDY / "results/training_summary.json", summary)
    initial = table[table.initialization_seed == 525].set_index("arm")
    contrasts = []
    for left, right, interpretation in [
        ("A", "B", "matched_300_updates"),
        ("A", "C", "matched_300_updates"),
        ("B", "D", "batch256_duration_extension"),
        ("C", "E", "batch128_duration_extension"),
        ("A", "D", "matched_300_epochs"),
        ("A", "E", "matched_300_epochs"),
    ]:
        start = float(initial.loc[left, "best_validation_rmse"])
        end = float(initial.loc[right, "best_validation_rmse"])
        contrasts.append(
            dict(
                reference=left,
                comparison=right,
                contrast=interpretation,
                reference_best_rmse=start,
                comparison_best_rmse=end,
                best_rmse_improvement_percent=100 * (start - end) / start,
                reference_final_rmse=float(initial.loc[left, "final_validation_rmse"]),
                comparison_final_rmse=float(initial.loc[right, "final_validation_rmse"]),
            )
        )
    pd.DataFrame(contrasts).to_csv(STUDY / "results/exposure_contrasts.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
        }
    )
    for column, xlabel, filename in [
        ("optimizer_steps", "Cumulative optimizer updates", "validation_rmse_vs_optimizer_steps"),
        ("epoch", "Epoch", "validation_rmse_vs_epoch"),
        (
            "sample_presentations",
            "Training sample presentations",
            "validation_rmse_vs_sample_presentations",
        ),
    ]:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for axis, seed in zip(axes, [525, 1525, 2525]):
            for arm in ARMS:
                frame = curves[(curves.arm == arm) & (curves.initialization_seed == seed)]
                if frame.empty:
                    continue
                axis.plot(frame[column], frame.rmse, color=COLORS[arm], alpha=0.23, linewidth=0.8)
                axis.plot(
                    frame[column],
                    frame.rmse.cummin(),
                    color=COLORS[arm],
                    linewidth=1.6,
                    label=f"{arm}: batch {ARMS[arm]['batch_size']}, {ARMS[arm]['maximum_epochs']} epochs",
                )
            axis.set(title=f"Initialization seed {seed}", xlabel=xlabel)
            axis.legend(fontsize=8)
        axes[0].set_ylabel("Validation RMSE (RTv)")
        figure.suptitle("Fixed L0 (356 rows): full registered training duration", fontsize=14)
        figure.text(
            0.5,
            0.01,
            "Faint: each epoch. Solid: best validation RMSE so far. No early stopping; no test data.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.95))
        figure_save(figure, filename)

    relevance = pd.read_csv(STUDY / "results/gradient_error_relevance.csv")
    corr = pd.read_csv(STUDY / "results/gradient_error_correlations.csv")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for row, arm in enumerate(["A", chosen]):
        frame = relevance[relevance.arm == arm]
        for col, norm in enumerate(["full_gradient_norm", "sketch_gradient_norm"]):
            axis = axes[row, col]
            selected = corr[
                (corr.arm == arm) & (corr.norm == norm) & (corr.error == "absolute_error")
            ].iloc[0]
            axis.scatter(
                frame[norm],
                frame.absolute_error,
                color=COLORS[arm],
                alpha=0.5,
                s=14,
                edgecolors="none",
            )
            axis.set(
                xlabel=norm.replace("_", " ").capitalize(),
                ylabel="Absolute validation error (RTv)",
                title=f"{arm}, seed 525 | Spearman {selected.spearman:.3f}, Pearson {selected.pearson:.3f}",
            )
    figure.suptitle("Gradient magnitude versus prediction error: 247 validation rows", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure_save(figure, "gradient_norm_vs_absolute_error")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for axis, metric in zip(axes, ["rmse", "r2"]):
        for i, seed in enumerate([525, 1525, 2525]):
            frame = table[
                (table.initialization_seed == seed) & table.arm.isin(["A", chosen])
            ].set_index("arm")
            values = [frame.loc[arm, f"best_validation_{metric}"] for arm in ["A", chosen]]
            axis.plot([0, 1], values, marker="o", label=str(seed), alpha=0.8)
        axis.set_xticks([0, 1], ["Reference A", f"Practical {chosen}"])
        axis.set_ylabel(f"Best validation {metric.upper()}")
        axis.legend(title="Initialization seed")
    figure.suptitle("Paired initialization comparison; fixed L0 and validation")
    figure.tight_layout()
    figure_save(figure, "validation_seed_comparison")

    geometry = pd.read_csv(STUDY / "results/same_state_selection_comparison.csv")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [f"{r.arm}\n{r.geometry} {r.method}" for r in geometry.itertuples()]
    for axis, metric, title in zip(
        axes,
        [
            "selected_sketch_norm_percentile_median",
            "unit_pairwise_distance_mean",
            "unit_nearest_L_distance_mean",
        ],
        [
            "Selected raw sketch-norm percentile",
            "Pairwise distance in unit coordinates",
            "Nearest-L distance in unit coordinates",
        ],
    ):
        axis.bar(
            np.arange(len(geometry)),
            geometry[metric],
            color=["#52677e" if r.geometry == "raw" else "#007f86" for r in geometry.itertuples()],
        )
        axis.set_xticks(np.arange(len(geometry)), labels, rotation=55, ha="right", fontsize=8)
        axis.set_title(title, fontsize=10)
    axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 105)
    figure.suptitle("Same-state selection only: no new labels, no AL trajectories", fontsize=14)
    figure.tight_layout()
    figure_save(figure, "same_state_selection_geometry")
    print(json.dumps(summary, indent=2))


def verify():
    protocol, partition = check_protocol()
    manifest = read_json(STUDY / "completion_manifest.json")
    verify_files(STUDY, manifest["files"])
    verify_files(ROOT, manifest["external_files"])
    print(
        json.dumps(
            dict(
                status="VERIFIED",
                files=len(manifest["files"]),
                external_files=len(manifest["external_files"]),
                **label_audit(partition),
            ),
            indent=2,
        )
    )


def seal():
    if (STUDY / "completion_manifest.json").exists():
        return verify()
    _, partition, records, _, chosen = verified_records()
    verify_files(STUDY, read_json(STUDY / "geometry_complete.json")["files"])
    decision = read_json(STUDY / "decision.json")
    assert decision["status"] == "COMPLETE_STOPPED"
    assert decision["total_fits"] == len(records) == 9
    assert decision["selected_practical_arm"] == chosen
    assert (ROOT / "docs/research/HPLC_AL_TRAINING_DIAGNOSIS.md").is_file()
    manifest = dict(
        status="COMPLETE_STOPPED",
        label_audit=label_audit(partition),
        files={str(p.relative_to(STUDY)): sha(p) for p in sorted(STUDY.rglob("*")) if p.is_file()},
        external_files={
            name: sha(ROOT / name)
            for name in [
                "docs/research/HPLC_AL_TRAINING_DIAGNOSIS.md",
                "scripts/report_hplc_training_diagnosis.py",
            ]
        },
    )
    atomic_json(STUDY / "completion_manifest.json", manifest)
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["summarize", "seal", "verify"])
    args = parser.parse_args()
    {"summarize": summarize, "seal": seal, "verify": verify}[args.action]()
