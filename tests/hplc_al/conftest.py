import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
torch.set_num_threads(4)


@pytest.fixture(scope="session")
def partition():
    from hplc_al.protocol import make_partition

    return make_partition()


@pytest.fixture(scope="session")
def graphs(partition):
    from hplc_al.data import load_graphs

    return load_graphs(partition)
