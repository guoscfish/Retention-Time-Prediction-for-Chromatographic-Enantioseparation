"""Single-output full-network CountSketch, with fixed eval/BN semantics."""

import time

import numpy as np
import torch

from .common import SKETCH_SEED, array_hash, stable_hash, state_hash
from .data import assert_label_free, batches


def mapping(model, dimension=512, seed=SKETCH_SEED):
    slices, offset = [], 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            slices.append(
                dict(
                    name=name,
                    shape=list(p.shape),
                    start=offset,
                    stop=offset + p.numel(),
                    dtype=str(p.dtype),
                )
            )
            offset += p.numel()
    rng = np.random.default_rng(seed)
    buckets = rng.integers(0, dimension, size=offset, dtype=np.int64)
    signs = (2 * rng.integers(0, 2, size=offset, dtype=np.int8) - 1).astype(np.float64)
    digest = stable_hash(
        dict(
            slices=slices,
            dimension=dimension,
            seed=seed,
            buckets=array_hash(buckets),
            signs=array_hash(signs),
            accumulation="float64",
            output="float32",
            output_column=1,
        )
    )
    return slices, buckets, signs, digest


def extract(model, graphs, ids, progress=None):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("unique nonempty gradient IDs required")
    assert_label_free(graphs)
    model.eval()
    before = state_hash(model)
    slices, buckets, signs, digest = mapping(model)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    observed = np.zeros(len(params), dtype=bool)
    nonzero = np.zeros(len(buckets), dtype=bool)
    features = np.empty((len(ids), 512), dtype=np.float32)
    full_norms = np.empty(len(ids), dtype=np.float64)
    saturation = 0
    started = time.perf_counter()
    with torch.enable_grad():
        for row, (g, h) in enumerate(batches(graphs, ids, 1)):
            output, _ = model(g, h)
            saturation += int(output[0, 1].item() <= 0 or output[0, 1].item() >= 1e8)
            grads = torch.autograd.grad(output[0, 1], params, allow_unused=True)
            flat = np.zeros(len(buckets), dtype=np.float64)
            for i, (gradient, section) in enumerate(zip(grads, slices)):
                if gradient is not None:
                    observed[i] = True
                    flat[section["start"] : section["stop"]] = gradient.detach().numpy().reshape(-1)
            if not np.isfinite(flat).all():
                raise RuntimeError("nonfinite full gradient")
            nonzero |= flat != 0
            full_norms[row] = np.linalg.norm(flat)
            features[row] = np.bincount(buckets, weights=flat * signs, minlength=512).astype(
                np.float32
            )
            if progress and (row == 0 or (row + 1) % 250 == 0 or row + 1 == len(ids)):
                progress(
                    dict(completed=row + 1, total=len(ids), seconds=time.perf_counter() - started)
                )
    if before != state_hash(model):
        raise RuntimeError("gradient extraction mutated model parameters/buffers")
    if not np.isfinite(features).all():
        raise RuntimeError("nonfinite sketch")
    norms = np.linalg.norm(features.astype(np.float64), axis=1)

    def stats(a):
        return dict(
            min=float(a.min()),
            q25=float(np.quantile(a, 0.25)),
            median=float(np.median(a)),
            q75=float(np.quantile(a, 0.75)),
            max=float(a.max()),
            mean=float(a.mean()),
            zero_count=int((a == 0).sum()),
        )

    parameter_audit = [
        dict(
            **s,
            gradient_received=bool(observed[i]),
            nonzero_elements=int(nonzero[s["start"] : s["stop"]].sum()),
        )
        for i, s in enumerate(slices)
    ]
    audit = dict(
        rows=len(ids),
        dimensions=512,
        parameter_count=len(buckets),
        trainable_parameter_tensors=len(params),
        tensors_receiving_gradient=int(observed.sum()),
        tensors_without_gradient=int((~observed).sum()),
        ever_nonzero_elements=int(nonzero.sum()),
        zero_gradient_tensors=sum(p["nonzero_elements"] == 0 for p in parameter_audit),
        full_gradient_norm=stats(full_norms),
        sketch_norm=stats(norms),
        duplicate_feature_rows=len(ids) - len(np.unique(features, axis=0)),
        clamped_central_rows=saturation,
        finite=True,
        eval_mode=True,
        state_unchanged=True,
        mapping_hash=digest,
        feature_hash=array_hash(features),
        ordered_ids_hash=stable_hash(ids),
        seconds=time.perf_counter() - started,
        parameters=parameter_audit,
    )
    return features, audit
