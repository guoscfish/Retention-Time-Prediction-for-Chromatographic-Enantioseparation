"""Label-free task-aware QGeoGNN structure/geometry/condition representations."""

import numpy as np
import torch

from .common import array_hash, stable_hash, state_hash
from .data import assert_label_free, batches


def extract_h_graph(model, graphs, ids):
    """Return ordered raw/unit banks and audits without exposing any targets."""
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("unique nonempty latent IDs required")
    assert_label_free(graphs)
    model.eval()
    before = state_hash(model)
    rows, observed = [], []
    with torch.no_grad():
        for g, h in batches(graphs, ids, 256):
            observed.extend(g.sample_key.cpu().numpy().reshape(-1).tolist())
            _, latent = model(g, h)
            if latent.shape != (g.num_graphs, 128) or not torch.isfinite(latent).all():
                raise RuntimeError("nonfinite or misaligned h_graph")
            rows.append(latent.detach().cpu().numpy())
    after = state_hash(model)
    if before != after or observed != ids:
        raise RuntimeError("latent extraction mutated state or reordered IDs")
    raw = np.concatenate(rows)
    norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    unit = raw.astype(np.float64) / np.maximum(norms, 1e-12)[:, None]
    audit = dict(
        shape=list(raw.shape), finite=True, eval_mode=True, no_grad=True,
        duplicate_rows=len(ids) - len(np.unique(raw, axis=0)),
        zero_norm_rows=int(np.count_nonzero(norms == 0)),
        state_before=before, state_after=after, ordered_ids_hash=stable_hash(ids),
        raw_hash=array_hash(raw), unit_hash=array_hash(unit),
    )
    return {"ids": np.asarray(ids), "raw_latent": raw, "unit_latent": unit}, audit
