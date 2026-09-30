"""Offline evidence only: real inputs, synthetic decisions/measurements, no API."""

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from hplc_al.common import ROOT, read_json, stable_hash, verify_files
from hplc_al.llm.full_pool import ROUNDS, distribution, pair_counts
from hplc_al.llm.memory import OBS_FIELDS, build_memory
from hplc_al.llm.responses_transport import encode, payload

from .codec import decode_table
from .planner import LIMITS, build_request, plan


def baseline_packet():
    directory = ROOT / (
        "studies/active_learning/odh_free_llm32_fullpool_v2/runtime/"
        "seed_1525/free_llm32_fullpool/round_0"
    )
    verify_files(ROOT, read_json(directory / "prepared.json")["files"])
    packet = read_json(directory / "packet.json")
    if packet["round_index"] != 0 or packet["memory"]["recent_batches"]:
        raise ValueError("only baseline inputs may be reused; no V2 selection memory")
    from .runner import STUDY_ID, TRAJECTORY

    packet["memory"] = build_memory([], 1525, TRAJECTORY, study=STUDY_ID)
    return packet


def read_request(messages):
    metadata = json.loads(messages[1]["content"])
    end = json.loads(messages[-1]["content"])
    if end["kind"] != "END_OF_POOL":
        raise AssertionError("missing end marker")
    cards = []
    pages = messages[2:-1]
    for index, message in enumerate(pages):
        page = json.loads(message["content"])
        if page["page_index"] != index or page["page_count"] != len(pages):
            raise AssertionError("missing or misordered pages")
        cards.extend(decode_table(page["candidates"], metadata["dictionaries"]))
    if (
        len(cards) != end["candidate_count"]
        or len(cards) != metadata["candidate_count"]
    ):
        raise AssertionError("incomplete pool")
    if len({r["id"] for r in cards}) != len(cards):
        raise AssertionError("duplicate candidate")
    return metadata, cards


def simulated_transport(messages, config, budget):
    """Choose across the whole pool, including the LAST page; never scientific output."""
    metadata, cards = read_request(messages)
    chosen = [cards[i * (len(cards) - 1) // 31]["id"] for i in range(32)]
    r = metadata["round"]
    value = {
        "type": "selection",
        "packet_hash": metadata["packet_hash"],
        "choices": [
            {
                "id": i,
                "reason": "SIMULATED global decision",
                "scientific_role": "SIMULATED",
                "hypothesis_id": None,
                "evidence_ids": [],
            }
            for i in chosen
        ],
        "hypotheses": [
            {
                "id": f"h_{r}_{i}",
                "claim": "SIMULATED hypothesis",
                "alternative": "SIMULATED alternative",
                "expected_sign": 0,
                "candidate_ids": chosen[i : i + 8],
                "evidence_ids": [],
                "learning_value": "SIMULATED",
            }
            for i in range(8)
        ],
        "previous_hypothesis_updates": [
            {
                "id": h["id"],
                "status": "unresolved",
                "reason": "SIMULATED update",
                "supporting_observations": [],
                "contradicting_observations": [],
            }
            for h in metadata["memory"]["previous_hypotheses"]
        ],
        "feedback_interpretation": "SIMULATED; no measured evidence",
        "batch_rationale": "SIMULATED; all pages present before this one response",
        "unresolved_questions": [],
    }
    answer = encode(value).decode()
    return answer, {
        "response_id": "SIMULATED_GLOBAL_ONLY",
        "served_model": "SIMULATED",
        "usage": None,
        "request_sha256": hashlib.sha256(
            encode(payload(messages, config, budget))
        ).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "native_tool_calls": 0,
    }


def exercise(packet, config, limits=None):
    """All real candidates for six rounds; simulated labels and frozen predictions.

    A larger LOCAL SIMULATION budget tests plumbing even when production is blocked.
    It is not used for real requests or registered as provider capacity.
    """
    packet = copy.deepcopy(packet)
    limits = dict(limits or LIMITS)
    simulation_limits = {
        **limits,
        "context_tokens": max(limits["context_tokens"], 2_000_000),
    }
    history, rounds = [], []
    for r in range(ROUNDS):
        packet["round_index"] = r
        packet["memory"] = build_memory(
            history,
            packet["memory"]["seed"],
            packet["memory"]["method"],
            study=packet["memory"]["study"],
        )
        print(
            f"SIMULATED global round {r + 1}/{ROUNDS}; {len(packet['cards'])} candidates",
            flush=True,
        )
        with TemporaryDirectory(prefix="hplc-global-simulation-") as tmp:
            saved = plan(
                **packet,
                directory=Path(tmp),
                config=config,
                limits=simulation_limits,
                transport=simulated_transport,
            )

            # Resume must replay exactly and must not call the provider again.
            def forbidden(*args, **kwargs):
                raise AssertionError("resume attempted a new request")

            assert (
                plan(
                    **packet,
                    directory=Path(tmp),
                    config=config,
                    limits=simulation_limits,
                    transport=forbidden,
                )
                == saved
            )
            rounds.append(
                {
                    "round": r,
                    "legal_candidates": len(packet["cards"]),
                    "observed_count": len(packet["observed"]),
                    "previous_hypotheses": len(packet["memory"]["previous_hypotheses"]),
                    "selected_count": len(saved["selected_ids"]),
                    "simulated_calls": len(saved["receipts"]),
                    "input_audit": saved["input_audit"],
                    "production_capacity_pass": saved["input_audit"][
                        "required_context_tokens"
                    ]
                    <= limits["context_tokens"],
                    "replay_verified": True,
                }
            )
        lookup = {c["id"]: c for c in packet["cards"]}
        observations = []
        for choice in saved["response"]["choices"]:
            card = lookup[choice["id"]]
            observations.append(
                {
                    **{k: v for k, v in card.items() if k in OBS_FIELDS},
                    "response": card["pred_center"] - 1,
                    "premeasurement_prediction": [
                        card[k] for k in ("pred_q10", "pred_center", "pred_q90")
                    ],
                    "premeasurement_center": card["pred_center"],
                    "signed_error": 1.0,
                    "abs_error": 1.0,
                    "selection_reason": choice["reason"],
                    "scientific_role": choice["scientific_role"],
                    "hypothesis_id": None,
                }
            )
        history.append(
            {k: packet["memory"][k] for k in ("study", "seed", "method")}
            | {
                "round": r,
                "response": saved["response"],
                "observations": observations,
            }
        )
        packet["observed"] += observations
        selected = set(saved["selected_ids"])
        packet["cards"] = [c for c in packet["cards"] if c["id"] not in selected]
        packet["legal_ids"] = [c["id"] for c in packet["cards"]]
    return {
        "status": "PASS",
        "kind": "SIMULATED_ENGINEERING_ONLY",
        "rounds": rounds,
        "production_context_tokens": limits["context_tokens"],
        "simulation_context_tokens": simulation_limits["context_tokens"],
        "real_llm_calls": 0,
        "new_labels_revealed": 0,
        "fits": 0,
        "note": "Real baseline metadata and predictions; synthetic decisions and labels. "
        "No claim of scientific quality or provider context support.",
    }


def selection_diagnostics(cards, saved):
    lookup = {c["id"]: c for c in cards}
    selected = [lookup[i] for i in saved["selected_ids"]]
    return {
        "mode": "global_all_candidates",
        "input_coverage": saved["input_audit"]["input_coverage"],
        "visible_candidate_count": len(saved["visible_candidates"]),
        "screening_calls": 0,
        "llm_total_calls": len(saved["receipts"]),
        "selected_coverage": distribution([c["coverage"] for c in selected]),
        "selected_q_width": distribution([c["q_width"] for c in selected]),
        "scaffold_diversity": len({c["scaffold"] for c in selected}),
        **pair_counts(selected),
    }


def verify_gate(study, protocol, sources):
    gate = read_json(study / "test_gate.json")
    if (
        gate["status"] != "PASS"
        or gate["source_hashes"] != sources
        or gate["limits"] != protocol["limits"]
        or gate["config_sha256"] != stable_hash(protocol["transport_registration"])
    ):
        raise RuntimeError(
            "V3 passing verification for exact sources/config/limits required"
        )
    verify_files(ROOT, gate["evidence"])
    verify_files(ROOT, gate["test_hashes"])


def context_check(packet, limits=None):
    return build_request(**packet, limits=limits or LIMITS)[3]
