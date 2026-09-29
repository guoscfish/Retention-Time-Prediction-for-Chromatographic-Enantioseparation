"""Explicit V2 stages; this script never automatically selects or advances."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hplc_al.llm import runner
from hplc_al.llm.dry_run import dry_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["prepare", "dry-run", "select", "advance", "report"]
    )
    parser.add_argument("--round", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.command in ("select", "advance") and args.round is None:
        parser.error("--round is required")
    if args.command == "prepare":
        with runner.exclusive():
            runner.prepare()
        print("V2_REGISTERED_NO_SCIENTIFIC_CALLS")
    elif args.command == "dry-run":
        result = dry_run()
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "kind",
                        "legal_candidate_count",
                        "real_llm_calls",
                        "new_acquisition_labels_revealed",
                        "max_estimated_input_tokens",
                    )
                }
            )
        )
    elif args.command == "select":
        runner.run_selection(args.round)
        print("SELECTION_FROZEN_COMMIT_BEFORE_REVEAL")
    elif args.command == "advance":
        runner.advance(args.round)
        print("ROUND_FIT_COMPLETE")
    else:
        runner.report()
        print("COMPLETE_PHASE1_L397")
