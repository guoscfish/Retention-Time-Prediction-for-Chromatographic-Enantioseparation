"""Official ODH graph/cache adapter, smoke test, and fixed-budget reproduction."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/odh_matplotlib")
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from reproduction.official import ATOM_NAMES, BOND_NAMES, definitions, original_loss


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command_or_unknown(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def command_int_or_zero(command):
    value = command_or_unknown(command)
    try:
        return int(value)
    except ValueError:
        return 0


def dataset(cache):
    csv_path = ROOT / "dataset/ODH_charity_0616.csv"
    if digest(csv_path) != digest(cache / csv_path.name):
        raise RuntimeError("Official cached CSV differs from repository ODH rows")
    frame = pd.read_csv(csv_path)
    frame["sample_index"] = np.arange(len(frame))
    frame["sample_id"] = [f"ODH_charity_0616:row:{i}" for i in frame.index]
    frame["true_RTv"] = (frame.RT * frame.Speed).astype(np.float32)
    frame["split"] = "pending"
    frame["exclusion_reason"] = ""
    frame.loc[4231, "exclusion_reason"] = "official_bad_conformer"
    frame.loc[frame.true_RTv > 60, "exclusion_reason"] = "official_RTv_gt_60"
    source = frame.drop(index=4231)
    graphs = np.load(cache / "dataset_ODH.npy", allow_pickle=True).tolist()
    descriptors = np.load(cache / "dataset_ODH_morder.npy")
    assert len(graphs) == len(source) == len(descriptors)
    assert np.isfinite(descriptors[:, [820, 821, 822, 1568, 457]]).all()
    pairs, row_ids = [], []
    for position, (_, row) in enumerate(source.iterrows()):
        if row.true_RTv > 60:
            continue
        graph = graphs[position]
        x = torch.tensor(np.stack([graph[k] for k in ATOM_NAMES], axis=1), dtype=torch.long)
        bonds = torch.tensor(np.stack([graph[k] for k in BOND_NAMES], axis=1), dtype=torch.long)
        length = torch.tensor(graph["bond_length"], dtype=torch.float32).reshape(-1, 1)
        prop = torch.full_like(length, row["i-PrOH_proportion"])
        edge_attr = torch.cat([bonds, length, prop], dim=1)
        angles = torch.tensor(graph["bond_angle"], dtype=torch.float32).reshape(-1, 1)
        descriptor = descriptors[position, [820, 821, 822, 1568, 457]].copy()
        descriptor[0] /= 100
        angle_attr = torch.cat([angles, torch.tensor(descriptor, dtype=torch.float32).repeat(len(angles), 1)], dim=1)
        g = Data(x=x, edge_index=torch.tensor(graph["edges"].T, dtype=torch.long),
                 edge_attr=edge_attr, y=torch.tensor([row.true_RTv]),
                 sample_key=torch.tensor([row.sample_index]),
                 data_index=torch.tensor([int(row["Unnamed: 0"])]))
        h = Data(edge_index=torch.tensor(graph["BondAngleGraph_edges"].T, dtype=torch.long),
                 edge_attr=angle_attr, num_nodes=len(bonds))
        assert len(x) > 0 and g.num_edges > 0 and h.num_edges > 0
        assert int(h.edge_index.max()) + 1 == len(bonds), "Legacy H batching would differ"
        assert torch.isfinite(g.edge_attr).all() and torch.isfinite(h.edge_attr).all()
        assert torch.isfinite(g.y).all()
        assert int(g.edge_index.max()) < len(x)
        pairs.append((g, h))
        row_ids.append(int(row.sample_index))
    n = len(pairs)
    order = np.random.RandomState(388).permutation(n)
    split = {"train": order[:int(n * .90)].tolist(),
             "valid": order[int(n * .90):int(n * .90) + int(n * .05)].tolist(),
             "test": order[n - int(n * .05):].tolist()}
    assigned = set(sum(split.values(), []))
    unused = sorted(set(range(n)) - assigned)
    frame.loc[frame.exclusion_reason != "", "split"] = "excluded"
    for name, indices in split.items():
        frame.loc[[row_ids[i] for i in indices], "split"] = name
    frame.loc[[row_ids[i] for i in unused], "split"] = "unused_rounding"
    assert set(frame.split) <= {"train", "valid", "test", "excluded", "unused_rounding"}
    manifest = {"numpy_seed": 388, "torch_seed": 525, "mode": "fixed",
                "ratios": [.9, .05, .05], "eligible_rows": n,
                "split_row_indices": {k: [row_ids[i] for i in v] for k, v in split.items()},
                "unused_rounding_rows": [row_ids[i] for i in unused],
                "excluded_rows": frame.loc[frame.split == "excluded", ["sample_index", "exclusion_reason"]].to_dict("records"),
                "csv_sha256": digest(csv_path),
                "graph_sha256": digest(cache / "dataset_ODH.npy"),
                "descriptor_sha256": digest(cache / "dataset_ODH_morder.npy")}
    return pairs, frame, split, manifest


def loaders(pairs, indices, batch_size):
    return (DataLoader([pairs[i][0] for i in indices], batch_size=batch_size, shuffle=False),
            DataLoader([pairs[i][1] for i in indices], batch_size=batch_size, shuffle=False))


def evaluate(model, device, pair_loaders):
    model.eval()
    predictions, targets, rows = [], [], []
    with torch.no_grad():
        for g, h in zip(*pair_loaders):
            g, h = g.to(device), h.to(device)
            pred, latent = model(g, h)
            assert pred.shape == (g.num_graphs, 3) and latent.shape == (g.num_graphs, 128)
            assert torch.isfinite(pred).all() and torch.isfinite(latent).all()
            predictions.append(pred.cpu().numpy())
            targets.append(g.y.cpu().numpy())
            rows.extend(g.sample_key.cpu().tolist())
    return np.concatenate(predictions), np.concatenate(targets), rows


def metrics(pred, true):
    error = pred[:, 1] - true
    return {"n": len(true), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "r2": float(1 - np.sum(error**2) / np.sum((true - np.mean(true))**2)),
            "mean_relative_error": float(np.mean(np.abs(error) / true)),
            "median_relative_error": float(np.median(np.abs(error) / true)),
            "relative_l2_nrmse": float(np.sqrt(np.sum(error**2) / np.sum(true**2))),
            "range_nrmse": float(np.sqrt(np.mean(error**2)) / np.ptp(true)),
            "q10_q90_coverage": float(np.mean((pred[:, 0] <= true) & (true <= pred[:, 2]))),
            "quantile_crossing_any": float(np.mean((pred[:, 0] > pred[:, 1]) | (pred[:, 1] > pred[:, 2]))),
            "q10_gt_q90": float(np.mean(pred[:, 0] > pred[:, 2])),
            "mean_interval_width": float(np.mean(pred[:, 2] - pred[:, 0]))}


def figures(out, predictions, curve):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    target = predictions[predictions.split == "test"].sort_values("true_RTv")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(target.true_RTv, target.pred_central, s=12, alpha=.6)
    ax.plot([0, 60], [0, 60], color="black", linestyle="--")
    ax.set(xlabel="True RTv", ylabel="Predicted central RTv", title=out.name)
    fig.tight_layout()
    fig.savefig(out / "figures/predicted_vs_true.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(len(target)), target.true_RTv, label="True RTv", color="black")
    ax.fill_between(np.arange(len(target)), target.pred_q10, target.pred_q90, alpha=.3, label="q10-q90")
    ax.plot(np.arange(len(target)), target.pred_central, lw=.7, label="Central")
    ax.set(xlabel="Test samples sorted by true RTv", ylabel="RTv", title=out.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures/quantile_interval_diagnostics.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3))
    if curve:
        ax.plot([r["epoch"] for r in curve], [r["train_loss"] for r in curve], label="Training objective")
        ax.plot([r["epoch"] for r in curve], [r["validation_loss"] for r in curve], label="Validation MSE")
        ax.legend()
    else:
        ax.text(.5, .5, "Author checkpoint: training history unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set(xlabel="Epoch", ylabel="Loss", title=out.name)
    fig.tight_layout()
    fig.savefig(out / "figures/training_curve.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train", "official-checkpoint"], required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/reproduction/odh_cache")
    parser.add_argument("--resume", action="store_true",
                        help="resume from output/checkpoint_final.pt and append the existing curve")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise RuntimeError("Choose a new output directory; existing experiments are immutable")
    (out / "figures").mkdir(parents=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    start = time.perf_counter()
    pairs, frame, split, manifest = dataset(args.cache)
    save_json(out / "split_manifest.json", manifest)
    frame.to_csv(out / "sample_manifest.csv", index=False)
    epochs = 1500 if args.mode == "train" else args.smoke_epochs if args.mode == "smoke" else 0
    config = {"mode": args.mode, "epochs": epochs, "formal_epochs": 1500,
              "batch_size": 2048, "layers": 5, "hidden": 128, "pooling": "sum", "dropout": 0,
              "target": "RT * Speed", "outputs": ["q10", "MSE central", "q90"],
              "optimizer": "Adam", "lr": .001, "weight_decay": 1e-5,
              "split_seed": 388, "initialization_seed": 525, "shuffle": False,
              "scheduler_steps": False, "early_stopping": False,
              "checkpoint_rule": "final epoch (official-code-compatible; paper early-stop rule unspecified)",
              "device": args.device, "threads": args.threads,
              "command": shlex.join([sys.executable, *sys.argv]),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": digest(ROOT / "code/Single_column_prediction.py"),
              "adapter_sha256": digest(ROOT / "src/reproduction/official.py"),
              "runner_sha256": digest(Path(__file__)),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    save_json(out / "config.json", config)
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "processor": command_or_unknown(["sysctl", "-n", "machdep.cpu.brand_string"]),
                   "memory_bytes": command_int_or_zero(["sysctl", "-n", "hw.memsize"]),
                   "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
                   "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
                   "torch_threads": torch.get_num_threads()}
    save_json(out / "environment.json", environment)
    model_module = definitions(device)
    def fresh():
        torch.manual_seed(525)
        np.random.seed(388)
        random.seed(525)
        return model_module.GINGraphPooling(num_tasks=3, num_layers=5, emb_dim=128,
                    drop_ratio=0, graph_pooling="sum", descriptor_dim=1827).to(device)
    model = fresh()
    print("Prepared", {k: len(v) for k, v in split.items()}, "parameters", sum(p.numel() for p in model.parameters()), flush=True)
    smoke = {"all_graphs_finite_nonempty": True, "legacy_H_batch_offsets_equal": True}
    g, h = next(zip(*loaders(pairs, split["train"][:8], 8)))
    g, h = g.to(device), h.to(device)
    model.train()
    pred, latent = model(g, h)
    loss = original_loss(pred, g.y)
    assert torch.isfinite(loss) and pred.shape == (8, 3) and latent.shape == (8, 128)
    gradients = torch.autograd.grad(pred[:, 1].sum(), tuple(model.parameters()), allow_unused=True, retain_graph=True)
    assert any(x is not None and torch.count_nonzero(x) > 0 for x in gradients)
    assert all(x is None or torch.isfinite(x).all() for x in gradients)
    smoke.update({"output_shape": list(pred.shape), "h_graph_shape": list(latent.shape),
                  "initial_loss": loss.item(), "central_gradient_finite": True,
                  "parameters_with_gradient": sum(x is not None for x in gradients),
                  "parameters_without_gradient": sum(x is None for x in gradients)})
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    before = model.graph_pred_linear.weight.detach().clone()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.graph_pred_linear.weight)
    model.eval()
    unlabelled_g = g.clone()
    del unlabelled_g.y
    with torch.no_grad():
        assert torch.equal(model(unlabelled_g, h)[0], model(g, h)[0])
    smoke.update({"backward_and_optimizer_step": True, "label_free_forward_equal": True})
    save_json(out / "smoke_checks.json", smoke)
    del model, optimizer, gradients, loss, pred, latent, g, h, unlabelled_g
    model = fresh()  # The smoke probe must not change training initialization or BN state.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    curve = []
    if args.mode == "official-checkpoint":
        checkpoint = args.cache / "model_save_1500.pth"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        config["checkpoint_sha256"] = digest(checkpoint)
        save_json(out / "config.json", config)
    curve_fields = ["epoch", "train_loss", "validation_loss", "train_seconds", "validation_seconds"]
    best_validation = (float("inf"), None)
    checkpoint_path = out / "checkpoint_final.pt"
    resume_epoch = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise RuntimeError(f"Cannot resume without {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        resume_epoch = int(saved["epoch"])
        if resume_epoch >= epochs:
            raise RuntimeError(f"Checkpoint already reaches epoch {resume_epoch}; no work remains")
        existing_curve = out / "training_curve.csv"
        curve = []
        if existing_curve.exists():
            with existing_curve.open(newline="") as existing:
                curve = [
                    {key: (int(value) if key == "epoch" else float(value)) for key, value in row.items()}
                    for row in csv.DictReader(existing)
                ]
        print(f"Resuming from checkpoint epoch {resume_epoch}; appending to {existing_curve}", flush=True)
    curve_fields = ["epoch", "train_loss", "validation_loss", "train_seconds", "validation_seconds"]
    curve_mode = "a" if args.resume and (out / "training_curve.csv").exists() else "w"
    with (out / "training_curve.csv").open(curve_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        if curve_mode == "w":
            writer.writeheader()
        for epoch in range(resume_epoch + 1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            losses = []
            for g, h in zip(*loaders(pairs, split["train"], 2048)):
                g, h = g.to(device), h.to(device)
                optimizer.zero_grad()
                pred, _ = model(g, h)
                loss = original_loss(pred, g.y)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                losses.append(loss.item())
            train_seconds = time.perf_counter() - epoch_start
            valid_start = time.perf_counter()
            pred_valid, true_valid, _ = evaluate(model, device, loaders(pairs, split["valid"], 2048))
            valid_mse = float(np.mean((pred_valid[:, 1] - true_valid)**2))
            if valid_mse < best_validation[0]:
                best_validation = valid_mse, epoch
            row = dict(zip(curve_fields, [epoch, float(np.mean(losses)), valid_mse,
                                         train_seconds, time.perf_counter() - valid_start]))
            curve.append(row)
            writer.writerow(row)
            f.flush()
            if epoch <= 5 or epoch % 100 == 0:
                print(row, flush=True)
            if epoch % 100 == 0 or epoch == epochs:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, checkpoint_path)
    # Test inference occurs only after the fixed training trajectory has ended.
    save_json(out / "trajectory_freeze.json", {"epochs_completed": epochs, "checkpoint_rule": config["checkpoint_rule"],
              "checkpoint_sha256": digest(out / "checkpoint_final.pt") if epochs else config["checkpoint_sha256"],
              "frozen_utc": datetime.now(timezone.utc).isoformat()})
    all_predictions, all_metrics = [], {}
    for name in ["train", "valid", "test"]:
        pred, true, rows = evaluate(model, device, loaders(pairs, split[name], 256))
        all_metrics[name] = metrics(pred, true)
        rows_frame = frame.loc[rows, ["sample_index", "sample_id", "Unnamed: 0", "SMILES", "Column", "RT", "Speed", "true_RTv", "split"]].copy()
        rows_frame = rows_frame.rename(columns={"RT": "true_RT", "Unnamed: 0": "source_index", "Column": "column"})
        rows_frame[["pred_q10", "pred_central", "pred_q90"]] = pred
        all_predictions.append(rows_frame)
    predictions = pd.concat(all_predictions)
    predictions.to_csv(out / "predictions.csv", index=False)
    elapsed = time.perf_counter() - start
    all_metrics["run"] = {"mode": args.mode, "epochs_completed": epochs, "runtime_seconds": elapsed,
                          "best_validation_epoch_diagnostic_only": best_validation[1],
                          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "formal_training_complete": args.mode == "train" and epochs == 1500}
    if curve:
        all_metrics["run"]["estimated_1500_epoch_hours"] = float(np.mean([r["train_seconds"] + r["validation_seconds"] for r in curve]) * 1500 / 3600)
    save_json(out / "metrics.json", all_metrics)
    figures(out, predictions, curve)
    print(json.dumps(all_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
