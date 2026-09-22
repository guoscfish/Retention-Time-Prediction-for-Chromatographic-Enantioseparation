"""Load official definitions without executing hard-coded data/training globals."""

import ast
import importlib.util
from pathlib import Path
import types
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    MessagePassing, GlobalAttention, Set2Set,
    global_add_pool, global_mean_pool, global_max_pool,
)

ROOT = Path(__file__).resolve().parents[2]
ATOM_NAMES = ["atomic_num", "chiral_tag", "degree", "explicit_valence", "formal_charge",
              "hybridization", "implicit_valence", "is_aromatic", "total_numHs"]
BOND_NAMES = ["bond_dir", "bond_type", "is_in_ring"]


def compound_tools():
    spec = importlib.util.spec_from_file_location("official_compound_tools", ROOT / "code/compound_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortableDevice(ast.NodeTransformer):
    """The only model rewrite is the unconditional .cuda() constructor call."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            assert not node.args and not node.keywords
            node.func.attr = "to"
            node.args = [ast.Name(id="device", ctx=ast.Load())]
        return node


def definitions(device):
    source = ROOT / "code/Single_column_prediction.py"
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if
                (isinstance(n, ast.ClassDef) and n.name != "ANN") or
                (isinstance(n, ast.FunctionDef) and n.name == "q_loss")]
    module = types.ModuleType("reproduction_official_model")
    c = compound_tools()
    module.__dict__.update(globals())
    module.__name__ = "reproduction_official_model"
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    module.__dict__.update({
        "device": torch.device(device), "atom_id_names": ATOM_NAMES,
        "bond_id_names": BOND_NAMES, "bond_float_names": ["bond_length", "prop"],
        "bond_angle_float_names": ["bond_angle", "TPSA", "RASA", "RPSA", "MDEC", "MATS"],
        "condition_name": ["silica_surface", "replace_basis"],
        "condition_float_name": ["eluent", "grain_radian"],
        "full_atom_feature_dims": c.get_atom_feature_dims(ATOM_NAMES),
        "full_bond_feature_dims": c.get_bond_feature_dims(BOND_NAMES),
    })
    tree = ast.fix_missing_locations(PortableDevice().visit(ast.Module(body=selected, type_ignores=[])))
    exec(compile(tree, str(source), "exec"), module.__dict__)
    return module


def original_loss(pred, true):
    def pinball(q, estimate):
        error = true - estimate
        return torch.maximum(q * error, (q - 1) * error).mean()
    return (pinball(.1, pred[:, 0]) + ((true - pred[:, 1]) ** 2).mean()
            + pinball(.9, pred[:, 2]) + F.relu(pred[:, 0] - pred[:, 1]).mean()
            + F.relu(pred[:, 1] - pred[:, 2]).mean() + F.relu(2 - pred).mean())
