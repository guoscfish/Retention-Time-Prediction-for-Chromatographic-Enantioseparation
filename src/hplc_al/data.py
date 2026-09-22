"""Label-free reproduction of the fixed ODH graph inputs."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from reproduction.official import ATOM_NAMES, BOND_NAMES

from .common import BASELINE, CACHE, SOURCE, read_json, sha


def load_graphs(partition):
    manifest = read_json(BASELINE / "split_manifest.json")
    for path, key in [
        (SOURCE, "csv_sha256"),
        (CACHE / "dataset_ODH.npy", "graph_sha256"),
        (CACHE / "dataset_ODH_morder.npy", "descriptor_sha256"),
    ]:
        if sha(path) != manifest[key]:
            raise RuntimeError(f"input provenance drift: {path}")
    features = pd.read_csv(SOURCE, usecols=["Unnamed: 0", "i-PrOH_proportion"])
    source_graphs = np.load(CACHE / "dataset_ODH.npy", allow_pickle=True)
    descriptors = np.load(CACHE / "dataset_ODH_morder.npy")
    if len(source_graphs) != 4971 or len(descriptors) != 4971:
        raise RuntimeError("author cache alignment changed")
    result = {}
    for record in partition["rows"]:
        if record["role"] in ("excluded", "unused_rounding"):
            continue
        row = int(record["sample_index"])
        position = row if row < 4231 else row - 1
        graph = source_graphs[position]
        x = torch.tensor(np.stack([graph[k] for k in ATOM_NAMES], axis=1), dtype=torch.long)
        bonds = torch.tensor(np.stack([graph[k] for k in BOND_NAMES], axis=1), dtype=torch.long)
        length = torch.tensor(graph["bond_length"], dtype=torch.float32).reshape(-1, 1)
        prop = torch.full_like(length, float(features.loc[row, "i-PrOH_proportion"]))
        angles = torch.tensor(graph["bond_angle"], dtype=torch.float32).reshape(-1, 1)
        descriptor = descriptors[position, [820, 821, 822, 1568, 457]].copy()
        descriptor[0] /= 100
        g = Data(
            x=x,
            edge_index=torch.tensor(graph["edges"].T, dtype=torch.long),
            edge_attr=torch.cat([bonds, length, prop], dim=1),
            sample_key=torch.tensor([row]),
            data_index=torch.tensor([record["source_index"]]),
        )
        h = Data(
            edge_index=torch.tensor(graph["BondAngleGraph_edges"].T, dtype=torch.long),
            edge_attr=torch.cat(
                [angles, torch.tensor(descriptor, dtype=torch.float32).repeat(len(angles), 1)],
                dim=1,
            ),
            num_nodes=len(bonds),
        )
        if not len(x) or not g.num_edges or not h.num_edges:
            raise ValueError("empty graph")
        if int(h.edge_index.max()) + 1 != len(bonds) or int(g.edge_index.max()) >= len(x):
            raise ValueError("graph indexing drift")
        if not torch.isfinite(g.edge_attr).all() or not torch.isfinite(h.edge_attr).all():
            raise ValueError("nonfinite graph inputs")
        result[row] = (g, h)
    assert_label_free(result)
    return result


def assert_label_free(graphs):
    g_allowed = {"x", "edge_index", "edge_attr", "sample_key", "data_index"}
    h_allowed = {"edge_index", "edge_attr", "num_nodes"}
    for g, h in graphs.values():
        if not set(g.keys()) <= g_allowed or not set(h.keys()) <= h_allowed:
            raise PermissionError("graph contains a forbidden attribute")


def batches(graphs, ids, batch_size=2048):
    for start in range(0, len(ids), batch_size):
        subset = ids[start : start + batch_size]
        yield (
            Batch.from_data_list([graphs[i][0] for i in subset]),
            Batch.from_data_list([graphs[i][1] for i in subset]),
        )


def predict(model, graphs, ids):
    model.eval()
    values = []
    with torch.no_grad():
        for g, h in batches(graphs, ids):
            pred, _ = model(g, h)
            if pred.shape != (g.num_graphs, 3) or not torch.isfinite(pred).all():
                raise RuntimeError("nonfinite/misaligned prediction")
            values.append(pred.numpy())
    return np.concatenate(values)
