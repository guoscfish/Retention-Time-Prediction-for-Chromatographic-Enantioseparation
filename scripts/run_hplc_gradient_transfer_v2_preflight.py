"""Run the label-free v2 shuffle/protocol preflight; never starts AL."""

import json
from pathlib import Path

from hplc_al.preflight_v2 import run_preflight


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "studies/active_learning/odh_gradient_al_transfer_v2/preflight.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(run_preflight(), indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
