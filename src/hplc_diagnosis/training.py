"""Fixed-duration scratch fits with measured optimizer/sample exposure."""

import csv
import time
from pathlib import Path

import numpy as np
import torch

from hplc_al.common import (
    atomic_json,
    metrics,
    read_json,
    sha,
    stable_hash,
    state_hash,
    verify_files,
    write_once,
)
from hplc_al.data import batches, predict
from hplc_al.training import fresh_model, save_checkpoint
from reproduction.official import original_loss

ARMS = {
    "A": {"batch_size": 2048, "maximum_epochs": 300},
    "B": {"batch_size": 256, "maximum_epochs": 150},
    "C": {"batch_size": 128, "maximum_epochs": 100},
    "D": {"batch_size": 256, "maximum_epochs": 300},
    "E": {"batch_size": 128, "maximum_epochs": 300},
}


def exposure(count, batch_size, epochs):
    if min(count, batch_size, epochs) < 1:
        raise ValueError("positive count/batch/epochs required")
    steps = (count + batch_size - 1) // batch_size
    return dict(
        steps_per_epoch=steps,
        total_optimizer_steps=steps * epochs,
        total_sample_presentations=count * epochs,
    )


def choose_challenger(records):
    """Select B–E by seed-525 validation only, with a predetermined tie rule."""
    if len(records) != 5 or {r["arm"] for r in records} != set(ARMS):
        raise ValueError("all five Part A arms required")
    if any(r["initialization_seed"] != 525 for r in records):
        raise ValueError("selection uses seed 525 only")
    choices = [r for r in records if r["arm"] != "A"]
    if not all(np.isfinite(r["best_metrics"]["rmse"]) for r in choices):
        raise ValueError("finite validation scores required")
    return min(
        choices,
        key=lambda r: (
            r["best_metrics"]["rmse"],
            r["total_optimizer_steps"],
            r["epochs_run"],
            r["arm"],
        ),
    )["arm"]


def fixed_fit(
    graphs, labeled, truth, validation, valid_truth, arm, seed, scale, directory, binding
):
    """Run every registered epoch; validation selects a diagnostic checkpoint only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(
        ARMS[arm],
        initialization_seed=seed,
        lr=0.001,
        weight_decay=1e-5,
        shuffle=False,
        drop_last=False,
        early_stopping=False,
        scheduler=False,
        loss="original_complete",
        checkpoint="strict_best_validation_central_MSE",
    )
    contract = dict(
        arm=arm,
        labeled=labeled,
        validation=validation,
        config=config,
        binding=binding,
        truth_hash=stable_hash(np.asarray(truth).tolist()),
        validation_truth_hash=stable_hash(np.asarray(valid_truth).tolist()),
        scale=scale,
    )
    write_once(directory / "input.json", contract)
    if (directory / "fit.json").exists():
        record = read_json(directory / "fit.json")
        if record["input_hash"] != sha(directory / "input.json"):
            raise RuntimeError("fit input drift")
        verify_files(directory, record["files"])
        return record
    previous = sorted(directory.glob("attempt_*"))
    attempt = directory / f"attempt_{len(previous):03d}"
    attempt.mkdir()
    started = time.perf_counter()
    model = fresh_model(seed)
    initial = state_hash(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    y = torch.tensor(truth, dtype=torch.float32)
    train_batches = list(batches(graphs, labeled, config["batch_size"]))
    best, best_epoch, updates, presentations = float("inf"), 0, 0, 0
    fields = [
        "epoch",
        "optimizer_steps",
        "sample_presentations",
        "train_loss",
        "validation_mse",
        "rmse",
        "mae",
        "r2",
        "nrmse",
        "train_seconds",
        "validation_seconds",
        "elapsed_seconds",
    ]
    with (attempt / "training_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["maximum_epochs"] + 1):
            tick = time.perf_counter()
            model.train()
            weighted_loss, offset = 0.0, 0
            for graph, angle in train_batches:
                optimizer.zero_grad(set_to_none=True)
                output, _ = model(graph, angle)
                loss = original_loss(output, y[offset : offset + graph.num_graphs])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training objective")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError("nonfinite training gradient")
                optimizer.step()
                updates += 1
                presentations += graph.num_graphs
                offset += graph.num_graphs
                weighted_loss += loss.item() * graph.num_graphs
            if offset != len(labeled):
                raise RuntimeError("training did not present all L0 rows exactly once")
            train_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            predictions = predict(model, graphs, validation)
            current = metrics(predictions[:, 1], valid_truth, scale)
            mse = current["rmse"] ** 2
            validation_seconds = time.perf_counter() - tick
            if mse < best:
                best, best_epoch = mse, epoch
                best_metrics = current
                best_predictions = predictions.copy()
                save_checkpoint(
                    attempt / "best.pt",
                    dict(
                        model=model.state_dict(),
                        epoch=epoch,
                        validation_mse=mse,
                        initialization_hash=initial,
                    ),
                )
            writer.writerow(
                dict(
                    epoch=epoch,
                    optimizer_steps=updates,
                    sample_presentations=presentations,
                    train_loss=weighted_loss / len(labeled),
                    validation_mse=mse,
                    **current,
                    train_seconds=train_seconds,
                    validation_seconds=validation_seconds,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            stream.flush()
            if epoch == 1 or epoch % 25 == 0 or epoch == config["maximum_epochs"]:
                print(
                    dict(
                        stage="training",
                        arm=arm,
                        seed=seed,
                        epoch=epoch,
                        steps=updates,
                        best_epoch=best_epoch,
                        best_rmse=best_metrics["rmse"],
                        elapsed_seconds=round(time.perf_counter() - started, 2),
                    ),
                    flush=True,
                )
    save_checkpoint(
        attempt / "final.pt",
        dict(
            model=model.state_dict(), epoch=epoch, validation_mse=mse, initialization_hash=initial
        ),
    )
    np.savez(
        attempt / "validation_predictions.npz",
        ids=np.asarray(validation),
        best=best_predictions,
        final=predictions,
    )
    expected = exposure(len(labeled), config["batch_size"], config["maximum_epochs"])
    if (
        updates != expected["total_optimizer_steps"]
        or presentations != expected["total_sample_presentations"]
    ):
        raise RuntimeError("measured exposure differs from registered plan")
    elapsed = time.perf_counter() - started
    paths = [directory / "input.json", *attempt.iterdir()]
    record = dict(
        arm=arm,
        config=config,
        initialization_seed=seed,
        initialization_hash=initial,
        batch_size=config["batch_size"],
        epochs_run=epoch,
        best_epoch=best_epoch,
        best_optimizer_step=best_epoch * expected["steps_per_epoch"],
        best_metrics=best_metrics,
        final_metrics=current,
        **expected,
        seconds=elapsed,
        updates_per_second=updates / elapsed,
        checkpoint_path=str((attempt / "best.pt").relative_to(directory)),
        checkpoint_hash=sha(attempt / "best.pt"),
        final_checkpoint_hash=sha(attempt / "final.pt"),
        training_curve=str((attempt / "training_curve.csv").relative_to(directory)),
        predictions_path=str((attempt / "validation_predictions.npz").relative_to(directory)),
        stopping_reason="registered_maximum_epochs",
        input_hash=sha(directory / "input.json"),
        abandoned_attempts=[p.name for p in previous],
        files={str(p.relative_to(directory)): sha(p) for p in paths},
    )
    atomic_json(directory / "fit.json", record)
    return record
