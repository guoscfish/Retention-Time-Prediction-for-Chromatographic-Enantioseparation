"""Command-line entry point for the gated ODH active-learning study."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hplc_al_matplotlib")

from hplc_al.reporting import report
from hplc_al.runner import (
    duration_audit,
    freeze_protocol,
    prepare,
    run_trajectories,
    tests_gate,
    verify_completed_study,
)

ACTIONS = {
    "prepare": prepare,
    "test": tests_gate,
    "audit": duration_audit,
    "freeze": freeze_protocol,
    "run": run_trajectories,
    "report": report,
    "verify": verify_completed_study,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=ACTIONS,
        help="Use verify for completed results; the other stages enforce their freeze gates.",
    )
    arguments = parser.parse_args()
    result = ACTIONS[arguments.action]()
    if arguments.action == "verify":
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
