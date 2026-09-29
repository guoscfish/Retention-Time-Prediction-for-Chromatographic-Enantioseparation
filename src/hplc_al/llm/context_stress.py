"""Offline six-round capacity exercise. Synthetic labels/predictions, no training."""

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ..common import read_json
from .dry_run import simulated_transport
from .full_pool import LIMITS, ROUNDS, token_bound, tokenizer
from .memory import OBS_FIELDS, build_memory
from .planner import ARBITRATE_PROMPT, plan, validate_screen
from .responses_transport import encode, payload
from .wire import unpack


def prose(prefix, maximum):
    text = prefix
    words = " stereochemical recognition solvent composition retention prediction comparison observed evidence unresolved mechanism control".split()
    i = 0
    while (
        len(
            tokenizer().encode(
                text + " " + words[i % len(words)], disallowed_special=()
            )
        )
        <= maximum
    ):
        text += " " + words[i % len(words)]
        i += 1
    return text


def rich_transport(messages, config, budget):
    first = unpack(json.loads(messages[1]["content"]))
    raw, _ = simulated_transport(messages, config, budget)
    value = json.loads(raw)
    if messages[0]["content"] != ARBITRATE_PROMPT:
        previous = [h["id"] for h in first["memory"]["previous_hypotheses"]]
        ids = [r[0] for r in first["cards"]["rows"]]
        largest = sorted(first["cards"]["rows"], key=token_bound, reverse=True)
        for nomination, row in zip(value["nominees"], largest):
            nomination["candidate_id"] = row[0]
        nominated = {n["candidate_id"] for n in value["nominees"]}
        value["rescue_candidate_ids"] = [next(i for i in ids if i not in nominated)]
        # Approach the registered total reply cap with distinct, verbose reasons.
        for length in range(16, 65):
            trial = copy.deepcopy(value)
            for n in trial["nominees"]:
                for field in ("screen_reason", "what_evidence_would_be_learned"):
                    n[field] = prose(n["candidate_id"] + " " + field, length)
            try:
                validate_screen(trial, ids, previous, first["nomination_cap"])
            except ValueError:
                break
            value = trial
    elif value["type"] == "selection":
        memory = first["memory"]
        r = len(memory["previous_hypotheses"]) // 8
        observed_ids = [
            row[group["columns"].index("id")]
            for group in first["observed"]
            for row in group["rows"]
        ]
        evidence = observed_ids[:4]
        for choice in value["choices"]:
            choice["reason"] = prose(f"Round {r} {choice['id']}", 32)
            choice["scientific_role"] = prose("SIMULATED role", 16)
            choice["evidence_ids"] = evidence
        value["hypotheses"] = [
            {
                "id": f"h_{r}_{i}",
                "claim": prose(f"Claim {r} {i}", 48),
                "alternative": prose(f"Alternative {r} {i}", 48),
                "expected_sign": 0,
                "candidate_ids": [c["id"] for c in value["choices"][i : i + 8]],
                "evidence_ids": evidence,
                "learning_value": prose(f"Learning {r} {i}", 32),
            }
            for i in range(8)
        ]
        value["previous_hypothesis_updates"] = [
            {
                "id": h["id"],
                "status": "unresolved",
                "reason": prose(f"Update {r} {h['id']}", 32),
                "supporting_observations": evidence,
                "contradicting_observations": observed_ids[4:8],
            }
            for h in memory["previous_hypotheses"]
        ]
        value["feedback_interpretation"] = prose(f"SIMULATED feedback {r}", 128)
        value["batch_rationale"] = prose(f"SIMULATED batch {r}", 128)
        value["unresolved_questions"] = [
            prose(f"Question {r} {i}", 48) for i in range(8)
        ]
    answer = encode(value).decode()
    if len(tokenizer().encode(answer, disallowed_special=())) > budget:
        raise AssertionError("synthetic JSON exceeds output budget")
    return answer, {
        "response_id": "SIMULATED_CONTEXT_STRESS",
        "served_model": "SIMULATED",
        "usage": None,
        "request_sha256": hashlib.sha256(
            encode(payload(messages, config, budget))
        ).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "native_tool_calls": 0,
    }


def exercise(packet, config):
    packet = copy.deepcopy(packet)
    history, rounds = [], []
    for r in range(ROUNDS):
        print(f"SIMULATED context stress round {r + 1}/{ROUNDS}", flush=True)
        packet["round_index"] = r
        packet["memory"] = build_memory(
            history, packet["memory"]["seed"], packet["memory"]["method"]
        )
        with TemporaryDirectory(prefix="hplc-context-stress-") as tmp:
            directory = Path(tmp)
            saved = plan(
                **packet,
                directory=directory,
                config=config,
                limits=LIMITS,
                transport=rich_transport,
            )
            rounds.append(
                {
                    "round": r,
                    "labeled_count": len(packet["observed"]),
                    "legal_candidates": len(packet["cards"]),
                    "coverage": saved["screening_audit"]["full_pool_screen_coverage"],
                    "hypotheses_seen": len(packet["memory"]["previous_hypotheses"]),
                    "admission": read_json(directory / "context_admission.json"),
                    "max_estimated_input_tokens": max(
                        token_bound(read_json(p)["request"]["input"])
                        for p in directory.glob("*.request.json")
                    ),
                }
            )
        lookup = {c["id"]: c for c in packet["cards"]}
        observations = []
        for choice in saved["response"]["choices"]:
            c = lookup[choice["id"]]
            observations.append(
                {
                    **{k: v for k, v in c.items() if k in OBS_FIELDS},
                    "response": c["pred_center"] - 1,
                    "premeasurement_prediction": [
                        c["pred_q10"],
                        c["pred_center"],
                        c["pred_q90"],
                    ],
                    "premeasurement_center": c["pred_center"],
                    "signed_error": 1.0,
                    "abs_error": 1.0,
                    "selection_reason": choice["reason"],
                    "scientific_role": choice["scientific_role"],
                    "hypothesis_id": None,
                }
            )
        history.append(
            {
                "study": packet["memory"]["study"],
                "seed": packet["memory"]["seed"],
                "method": packet["memory"]["method"],
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
        "kind": "SIMULATED_CONTEXT_STRESS_ONLY",
        "rounds": rounds,
        "note": "Real candidate metadata and initial predictions; synthetic acquired labels, near-cap screening prose, maximum hypothesis count/field lengths/evidence lists. Later predictions held fixed; no real retraining.",
        "real_llm_calls": 0,
        "new_labels_revealed": 0,
        "fits": 0,
    }
