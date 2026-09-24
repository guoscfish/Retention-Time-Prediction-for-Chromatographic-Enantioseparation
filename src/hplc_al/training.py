"""Scratch training, best-validation checkpoints and transaction-safe restart."""

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from reproduction.official import definitions, original_loss

from .common import (
    INIT_SEED,
    atomic_json,
    ids_hash,
    metrics,
    read_json,
    sha,
    stable_hash,
    state_hash,
    verify_files,
    write_once,
)
from .data import batches, predict
from .deterministic_shuffle import batch_sizes, epoch_batches


def fresh_model(seed=INIT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return definitions("cpu").GINGraphPooling(
        num_tasks=3,
        num_layers=5,
        emb_dim=128,
        drop_ratio=0,
        graph_pooling="sum",
        descriptor_dim=1827,
    )


def save_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def load_model(path):
    model = fresh_model()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model


def fit(graphs, labeled, truth, validation, valid_truth, config, directory, binding):
    """Incomplete fits restart deterministically from scratch; never warm start.

    Completed fits are reused only after validating their input contract and all
    artifacts. A partial fit is quarantined in a numbered attempt directory,
    preserving cost/history; no half-written optimizer state is trusted.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = dict(
        labeled=labeled,
        validation=validation,
        config=config,
        binding=binding,
        labeled_truth_hash=stable_hash(np.asarray(truth).tolist()),
        validation_truth_hash=stable_hash(np.asarray(valid_truth).tolist()),
    )
    write_once(directory / "input.json", contract)
    if (directory / "fit.json").exists():
        record = read_json(directory / "fit.json")
        verify_files(directory, record["files"])
        return record
    previous = sorted(directory.glob("attempt_*"))
    attempt = directory / f"attempt_{len(previous):03d}"
    attempt.mkdir()
    started = time.perf_counter()
    model = fresh_model(config["initialization_seed"])
    initial = state_hash(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    # Graphs remain label-free even for training; targets are separate authorized arrays.
    best, best_epoch, stale = float("inf"), 0, 0
    fields = [
        "epoch", "train_loss", "validation_mse", "train_seconds",
        "validation_seconds", "steps_per_epoch", "observed_batch_sizes",
    ]
    curve = []
    reason = "maximum_epochs"
    with (attempt / "training_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["maximum_epochs"] + 1):
            tick = time.perf_counter()
            model.train()
            losses = []
            offset = 0
            if config.get("shuffle") == "deterministic_each_epoch":
                paired_batches = epoch_batches(
                    labeled, truth, config.get("training_seed", config["initialization_seed"]), epoch - 1, config["batch_size"]
                )
            else:
                # Keep the historical fixed-order path available for immutable
                # development artifacts.  New transfer studies must opt into
                # deterministic_each_epoch explicitly.
                paired_batches = ((
                    labeled[start : start + config["batch_size"]],
                    truth[start : start + config["batch_size"]],
                ) for start in range(0, len(labeled), config["batch_size"]))
            for batch_ids, batch_truth in paired_batches:
                graph_batches = batches(graphs, batch_ids, len(batch_ids))
                g, h = next(graph_batches)
                if hasattr(g, "sample_key"):
                    graph_ids = g.sample_key.detach().cpu().numpy().reshape(-1)
                    if not np.array_equal(graph_ids, np.asarray(batch_ids)):
                        raise RuntimeError("graph IDs and truth batch are misaligned")
                optimizer.zero_grad(set_to_none=True)
                output, _ = model(g, h)
                target = torch.tensor(batch_truth, dtype=torch.float32)
                loss = original_loss(output, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training objective")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError("nonfinite training gradient")
                optimizer.step()
                losses.append((loss.item(), g.num_graphs))
                offset += g.num_graphs
            if offset != len(labeled):
                raise RuntimeError("epoch did not present every labeled ID exactly once")
            train_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            val = predict(model, graphs, validation)[:, 1]
            score = float(np.mean((val.astype(np.float64) - valid_truth.astype(np.float64)) ** 2))
            if not np.isfinite(score):
                raise RuntimeError("nonfinite validation criterion")
            row = dict(
                zip(
                    fields,
                    [
                        epoch,
                        sum(v * n for v, n in losses) / len(labeled),
                        score,
                        train_seconds,
                        time.perf_counter() - tick,
                        len(losses),
                        str([n for _, n in losses]),
                    ],
                )
            )
            curve.append(row)
            writer.writerow(row)
            f.flush()
            if score < best - config["min_delta"]:
                best, best_epoch, stale = score, epoch, 0
                save_checkpoint(
                    attempt / "best.pt",
                    dict(
                        model=model.state_dict(),
                        epoch=epoch,
                        validation_mse=score,
                        initialization_hash=initial,
                    ),
                )
            else:
                stale += 1
            if epoch == 1 or epoch % 25 == 0:
                print(
                    dict(
                        stage="fit",
                        directory=str(directory),
                        epoch=epoch,
                        best_epoch=best_epoch,
                        validation_mse=score,
                        seconds=time.perf_counter() - started,
                    ),
                    flush=True,
                )
            if stale >= config["patience"]:
                reason = "patience"
                break
            if (
                config.get("wall_time_limit_seconds")
                and time.perf_counter() - started >= config["wall_time_limit_seconds"]
            ):
                reason = "wall_time_limit"
                break
    final_epoch = epoch
    final_predictions = predict(model, graphs, validation)
    scale = float(np.std(np.asarray(truth, dtype=np.float64), ddof=0))
    final_metrics = metrics(final_predictions[:, 1], valid_truth, scale)
    save_checkpoint(
        attempt / "final.pt",
        dict(model=model.state_dict(), epoch=final_epoch, initialization_hash=initial),
    )
    np.savez(
        attempt / "final_validation_predictions.npz",
        ids=np.array(validation),
        predictions=final_predictions,
    )
    checkpoint = attempt / "best.pt"
    model = load_model(checkpoint)
    valid_predictions = predict(model, graphs, validation)
    best_metrics = metrics(valid_predictions[:, 1], valid_truth, scale)
    np.savez(
        attempt / "validation_predictions.npz",
        ids=np.array(validation),
        predictions=valid_predictions,
    )
    record = dict(
        initialization_seed=config["initialization_seed"],
        training_seed=config["training_seed"],
        initialization_hash=initial,
        checkpoint_state_hash=state_hash(model),
        checkpoint_path=str(checkpoint.relative_to(directory)),
        checkpoint_hash=sha(checkpoint),
        epochs_run=epoch,
        total_optimizer_steps=epoch * len(batch_sizes(len(labeled), config["batch_size"])),
        best_epoch=best_epoch,
        best_validation_mse=best,
        best_validation_metrics=best_metrics,
        best_optimizer_step=best_epoch * ((len(labeled) + config["batch_size"] - 1) // config["batch_size"]),
        final_validation_metrics=final_metrics,
        final_checkpoint_path=str((attempt / "final.pt").relative_to(directory)),
        final_checkpoint_hash=sha(attempt / "final.pt"),
        stopped_epoch=final_epoch,
        seconds=time.perf_counter() - started,
        stopping_reason=reason,
        attempted_fits=len(previous) + 1,
        abandoned_attempts=[p.name for p in previous],
        L_hash=ids_hash(labeled),
        budget=len(labeled),
        steps_per_epoch=len(batch_sizes(len(labeled), config["batch_size"])),
        observed_batch_sizes=batch_sizes(len(labeled), config["batch_size"]),
        input_hash=sha(directory / "input.json"),
        files={
            str(p.relative_to(directory)): sha(p)
            for p in [
                directory / "input.json",
                checkpoint,
                attempt / "training_curve.csv",
                attempt / "validation_predictions.npz",
                attempt / "final.pt",
                attempt / "final_validation_predictions.npz",
            ]
        },
    )
    atomic_json(directory / "fit.json", record)
    return record
