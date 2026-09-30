"""One complete pool, one global choice. Offline verification is the default test path."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hplc_al.common import ROOT, atomic_json, read_json, stable_hash
from hplc_al.llm.credentials import inject_key
from hplc_al.llm.full_pool import ROUNDS
from hplc_al.llm.responses_transport import TransportError
from hplc_global import runner
from hplc_global.planner import LIMITS, ContextCapacityError, require_capacity
from hplc_global.verification import (
    baseline_packet,
    context_check,
    exercise,
    verify_gate,
)


def verify():
    with runner.exclusive():
        sources = runner.source_hashes()
        paths = [
            "tests/hplc_al/test_global_v3.py",
            "tests/hplc_al/test_fullpool_v2.py",
            "tests/hplc_al/test_continuous_v2.py",
            "tests/hplc_al/test_responses_transport.py",
        ]
        evidence = runner.STUDY / "verification"
        evidence.mkdir(parents=True, exist_ok=True)
        junit = evidence / "pytest.xml"
        subprocess.run(
            [sys.executable, "-m", "pytest", *paths, "-q", f"--junitxml={junit}"],
            cwd=ROOT,
            check=True,
        )
        packet = baseline_packet()
        capacity = context_check(packet)
        atomic_json(evidence / "context_check.json", capacity)
        simulation = exercise(packet, runner.EXPECTED_CONFIG)
        atomic_json(evidence / "six_round_simulation.json", simulation)
        if sources != runner.source_hashes():
            raise RuntimeError("source changed during verification")
        gate = {
            "status": "PASS",
            "kind": "OFFLINE_ENGINEERING_ONLY",
            "source_hashes": sources,
            "limits": dict(LIMITS),
            "config_sha256": stable_hash(runner.EXPECTED_CONFIG),
            "evidence": runner.bind(
                [
                    junit,
                    evidence / "context_check.json",
                    evidence / "six_round_simulation.json",
                ]
            ),
            "test_hashes": runner.bind(
                [ROOT / p for p in paths]
                + [ROOT / "tests/hplc_al/conftest.py", ROOT / "pyproject.toml"]
            ),
            "production_capacity_status": capacity["status"],
            "real_llm_calls": 0,
            "new_labels_revealed": 0,
            "fits": 0,
        }
        atomic_json(runner.STUDY / "test_gate.json", gate)
        print(
            json.dumps(
                {
                    "engineering_tests": "PASS",
                    "production_capacity": capacity["status"],
                    "required_context_tokens": capacity["required_context_tokens"],
                    "configured_context_tokens": LIMITS["context_tokens"],
                    "real_llm_calls": 0,
                },
                indent=2,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "context-check",
            "verify",
            "prepare",
            "select",
            "advance",
            "run",
            "report",
        ],
    )
    parser.add_argument("--round", type=int, choices=range(ROUNDS))
    parser.add_argument(
        "--context-tokens",
        type=int,
        help="Explicit provider-supported capacity; never inferred from model name",
    )
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.command in ("select", "advance") and args.round is None:
        parser.error("--round is required")
    if args.key_file and args.command not in ("run", "select"):
        parser.error("--key-file is supported only for run/select")
    protocol_path = runner.STUDY / "protocol.json"
    if protocol_path.exists():
        registered = read_json(protocol_path)["limits"]["context_tokens"]
        if args.context_tokens is not None and args.context_tokens != registered:
            parser.error("context capacity is frozen in this V3 protocol")
        LIMITS["context_tokens"] = registered
    elif args.context_tokens is not None:
        if args.context_tokens <= LIMITS["max_output_tokens"]:
            parser.error("context must exceed output reserve")
        LIMITS["context_tokens"] = args.context_tokens
    if args.command == "context-check":
        result = context_check(baseline_packet())
        output = runner.STUDY / "verification/context_check.json"
        atomic_json(output, result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "all_candidate_ids"}, indent=2
            )
        )
        return 0 if result["status"] == "PASS" else 2
    if args.command == "verify":
        verify()
        return 0
    # Check the real baseline BEFORE creating a protocol, reading a key or calling a provider.
    # Later rounds are separately checked by the planner with their actual accumulated memory.
    if args.command in ("prepare", "select", "run") and not protocol_path.exists():
        require_capacity(context_check(baseline_packet()))
    if args.command == "prepare":
        with runner.exclusive():
            runner.prepare()
        print("V3_REGISTERED_NO_SCIENTIFIC_CALLS")
    elif args.command == "select":
        with runner.exclusive():
            protocol, _, directory, packet, _, _ = runner.make_round(args.round)
            require_capacity(context_check(packet, protocol["limits"]))
            verify_gate(runner.STUDY, protocol, runner.source_hashes())
        if (directory / "selection_seal.json").exists():
            runner.run_selection(args.round)
            print("SELECTION_REUSED_NO_API_CALL")
            return 0
        inject_key(interactive=True, path=args.key_file)
        from hplc_al.llm.execution import preflight_with_retries
        from hplc_al.llm.responses_transport import preflight

        preflight_with_retries(
            lambda: preflight(
                runner.STUDY / "transport_preflight.json", runner.EXPECTED_CONFIG
            )
        )
        runner.run_selection(args.round)
        print("SELECTION_FROZEN_COMMIT_BEFORE_REVEAL")
    elif args.command == "advance":
        runner.advance(args.round)
        print("ROUND_FIT_COMPLETE")
    elif args.command == "run":
        if not all(
            (runner.runtime() / f"round_{r}/complete.json").exists()
            for r in range(ROUNDS)
        ):
            inject_key(interactive=True, path=args.key_file)
        from hplc_global.continuous import run

        run()
    else:
        runner.report()
        print("COMPLETE_PHASE1_L525")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContextCapacityError, TransportError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("Interrupted; saved global request/response retained.", file=sys.stderr)
        raise SystemExit(130) from None
