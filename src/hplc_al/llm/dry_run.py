"""Explicitly simulated planning on all real L333 candidates; zero network/reveals."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ..common import atomic_json, sha, stable_hash
from .full_pool import token_bound
from .planner import ARBITRATE_PROMPT, plan
from .reporting import selection_diagnostics
from .responses_transport import encode, payload
from .runner import EXPECTED_CONFIG, STUDY, exclusive, make_round


def simulated_transport(messages, config, max_output_tokens):
    """Deterministic test fixture, NEVER valid scientific evidence."""
    first = json.loads(messages[1]["content"])
    if messages[0]["content"] != ARBITRATE_PROMPT:
        card_table = first["cards"]
        ids = [r[card_table["columns"].index("id")] for r in card_table["rows"]]
        value = {
            "nominees": [
                {
                    "candidate_id": i,
                    "screen_reason": "SIMULATED test nomination",
                    "scientific_role": "exploratory",
                    "linked_hypothesis": None,
                    "priority": 1,
                    "what_evidence_would_be_learned": "SIMULATED contract check",
                }
                for i in ids[: first["nomination_cap"]]
            ],
            "chunk_summary": "SIMULATED; no scientific inference",
            "rescue_candidate_ids": ids[-1:],
        }
    else:
        nominee_table = first["nominees"]
        ids = [r[nominee_table["columns"].index("id")] for r in nominee_table["rows"]]
        rescue = next(
            i
            for r in first["screening_results"]
            for i in r["rescue_candidate_ids"]
            if i not in ids
        )
        if len(messages) == 2:
            value = {"type": "expand", "expand_candidate_ids": [rescue]}
        else:
            value = {
                "type": "selection",
                "packet_hash": first["packet_hash"],
                "choices": [
                    {
                        "id": i,
                        "reason": "SIMULATED",
                        "scientific_role": "exploratory",
                        "hypothesis_id": None,
                        "evidence_ids": [],
                    }
                    for i in [rescue] + ids[:31]
                ],
                "hypotheses": [],
                "previous_hypothesis_updates": [],
                "feedback_interpretation": "",
                "batch_rationale": "SIMULATED; no scientific acquisition",
                "unresolved_questions": [],
            }
    answer = encode(value).decode()
    return answer, {
        "response_id": "SIMULATED",
        "served_model": "SIMULATED",
        "usage": None,
        "request_sha256": hashlib.sha256(
            encode(payload(messages, config, max_output_tokens))
        ).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "native_tool_calls": 0,
    }


def dry_run():
    with exclusive():
        protocol, _, directory, packet, labeled, unlabeled = make_round(0)
        with TemporaryDirectory(prefix="hplc-fullpool-dry-") as tmp:
            saved = plan(
                **packet,
                directory=Path(tmp),
                config=EXPECTED_CONFIG,
                limits=protocol["limits"],
                transport=simulated_transport,
            )
            requests = list(Path(tmp).glob("*.request.json"))
            from ..common import read_json

            estimates = [
                token_bound(read_json(p)["request"]["input"]) for p in requests
            ]
            diagnostics = selection_diagnostics(packet["cards"], saved)
            result = {
                "status": "PASS",
                "kind": "SIMULATED_ENGINEERING_ONLY",
                "protocol_sha256": sha(STUDY / "protocol.json"),
                "packet_sha256": sha(directory / "packet.json"),
                "labeled_count": len(labeled),
                "legal_candidate_count": len(unlabeled),
                "screening_audit": saved["screening_audit"],
                "simulation_diagnostics": diagnostics,
                "max_estimated_input_tokens": max(estimates),
                "context_budget": protocol["limits"]["context_tokens"],
                "real_llm_calls": 0,
                "new_acquisition_labels_revealed": 0,
                "test_labels_read": 0,
                "selected_non_nominee_rescue_tested": diagnostics[
                    "selected_from_nominees_rate"
                ]
                < 1,
                "source_hashes_sha256": stable_hash(protocol["source_hashes"]),
            }
        atomic_json(STUDY / "dry_run.json", result)
        return result
