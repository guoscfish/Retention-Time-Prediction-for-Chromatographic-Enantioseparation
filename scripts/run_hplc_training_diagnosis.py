"""Run the fixed-L0 training diagnosis, never an active-learning trajectory."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hplc_diagnosis_matplotlib")

from hplc_diagnosis.runner import geometry, prepare, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "geometry"])
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "geometry": geometry}[args.action]()


if __name__ == "__main__":
    main()
