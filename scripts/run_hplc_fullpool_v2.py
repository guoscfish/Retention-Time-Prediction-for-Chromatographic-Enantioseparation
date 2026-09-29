"""Explicit V2 stages plus a user-started, resumable six-round run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hplc_al.llm import runner
from hplc_al.llm.credentials import inject_key
from hplc_al.llm.dry_run import dry_run
from hplc_al.llm.full_pool import BUDGETS, ROUNDS
from hplc_al.llm.responses_transport import TransportError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "dry-run",
            "context-stress",
            "select",
            "advance",
            "report",
            "run",
        ],
    )
    parser.add_argument("--round", type=int, choices=range(ROUNDS))
    parser.add_argument(
        "--key-file",
        type=Path,
        help="Read a key file into this process only; supported by run",
    )
    args = parser.parse_args()
    if args.command in ("select", "advance") and args.round is None:
        parser.error("--round is required")
    if args.key_file is not None and args.command != "run":
        parser.error("--key-file is supported only by run")
    if args.command == "run":
        from hplc_al.llm.continuous import run

        try:
            if not all(
                (runner.runtime() / f"round_{r}/complete.json").exists()
                for r in range(ROUNDS)
            ):
                inject_key(interactive=True, path=args.key_file)
            run()
        except TransportError as error:
            print(str(error), file=sys.stderr)
            print(
                "Execution stopped; no automatic retry or provider/model fallback.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
    elif args.command == "prepare":
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
    elif args.command == "context-stress":
        from hplc_al.common import atomic_json, sha
        from hplc_al.llm.context_stress import exercise

        with runner.exclusive():
            protocol, _, _, packet, _, _ = runner.make_round(0)
            result = exercise(packet, runner.EXPECTED_CONFIG)
            result["protocol_sha256"] = sha(runner.STUDY / "protocol.json")
            atomic_json(
                runner.ROOT / "docs/repository/verification/context_stress_review.json",
                result,
            )
        print("SIX_ROUND_CONTEXT_STRESS_PASS_NO_NETWORK_NO_TRAINING")
    elif args.command == "select":
        runner.run_selection(args.round)
        print("SELECTION_FROZEN_COMMIT_BEFORE_REVEAL")
    elif args.command == "advance":
        runner.advance(args.round)
        print("ROUND_FIT_COMPLETE")
    else:
        runner.report()
        print(f"COMPLETE_PHASE1_L{BUDGETS[-1]}")
