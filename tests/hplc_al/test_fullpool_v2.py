import copy
import hashlib
import json

import pytest

from hplc_al.common import atomic_json, ids_hash, read_json, sha, stable_hash
from hplc_al.llm import runner
from hplc_al.llm.dry_run import simulated_transport
from hplc_al.llm.full_pool import (
    FIELDS,
    LIMITS,
    STUDY_ID,
    chunks,
    coverage_audit,
    opaque,
    summary,
    table,
    validate_cards,
)
from hplc_al.llm.memory import build_memory, feedback, validate_observations
from hplc_al.llm.planner import (
    ARBITRATE_PROMPT,
    SCREEN_PROMPT,
    plan,
    validate_screen,
    validate_selection,
)
from hplc_al.llm.reporting import evaluate, label_aulc, selection_diagnostics
from hplc_al.llm.responses_transport import encode, payload
from hplc_al.protocol import RestrictedLabelStore


def cards(n=90):
    return [
        {
            "id": opaque(i),
            "smiles": "C[C@H](O)C(=O)O" if i % 2 else "C[C@@H](O)C(=O)O",
            "stereocenters": [{"atom": 1, "CIP": "S" if i % 2 else "R"}],
            "scaffold": "",
            "ipa_fraction": 0.1,
            "flow": 1.0,
            "pred_center": float(i),
            "q_width": 2.0,
            "coverage": i / n,
            "MW": 90.0,
            "LogP": 1.0,
            "TPSA": 57.0,
            "HBD": 2,
            "HBA": 3,
            "identity": str(i % 2),
            "scaffold_group": "acyclic",
            "pred_q10": float(i - 1),
            "pred_q90": float(i + 1),
            "functional_groups": ["COO"],
        }
        for i in range(n)
    ]


def memory():
    return build_memory([], 1525, runner.TRAJECTORY)


def parameters(tmp_path):
    c = cards()
    return dict(
        cards=c,
        legal_ids=[r["id"] for r in c],
        observed=[],
        memory=memory(),
        round_index=0,
        directory=tmp_path,
        config=runner.EXPECTED_CONFIG,
        limits={**LIMITS, "target_chunk_size": 35},
    )


def test_full_coverage_independent_order_and_salt():
    c = cards(401)
    args = (SCREEN_PROMPT, {"memory": memory()}, LIMITS)
    first = chunks(c, "round0", *args)
    assert first == chunks(list(reversed(c)), "round0", *args)
    assert first != chunks(c, "round1", *args)
    audit = coverage_audit(
        [r["id"] for r in c], [[r["id"] for r in p] for p in first], []
    )
    assert audit["full_pool_screen_coverage"] == 1 and audit["missing_ids"] == []
    assert all(n <= 300 for n in audit["chunk_sizes"])


@pytest.mark.parametrize("fault", ["missing", "duplicate", "illegal"])
def test_no_freeze_without_exact_coverage(fault):
    ids = ["a", "b", "c"]
    parts = [["a", "b"], ["c"]]
    if fault == "missing":
        parts[-1] = []
    if fault == "duplicate":
        parts[-1].append("a")
    if fault == "illegal":
        parts[-1].append("x")
    with pytest.raises(ValueError, match="COVERAGE"):
        coverage_audit(ids, parts, [])


@pytest.mark.parametrize(
    "field",
    ["truth", "response", "validation_label", "test_label", "post_retrain_error"],
)
def test_candidate_allowlist_blocks_leakage(field):
    c = cards()
    c[0][field] = 3
    with pytest.raises(ValueError, match="allowlist"):
        validate_cards(c, [r["id"] for r in c])


def test_compact_cards_preserve_all_required_fields():
    c = cards()
    t = table(c)
    assert [dict(zip(t["columns"], row)) for row in t["rows"]] == c
    assert {
        "smiles",
        "stereocenters",
        "ipa_fraction",
        "flow",
        "pred_center",
        "q_width",
        "coverage",
    } <= set(FIELDS)


def test_token_budget_splits_and_single_card_failure():
    c = cards(40)
    small = {**LIMITS, "context_tokens": 6500, "max_output_tokens": 1000}
    assert len(chunks(c, "salt", "", {}, small)) > 1
    with pytest.raises(ValueError):
        chunks(c, "salt", "x" * 90000, {}, small)


def test_stage1_zero_nominees_allowed_no_quota():
    assert (
        validate_screen(
            {
                "nominees": [],
                "chunk_summary": "no candidates",
                "rescue_candidate_ids": [],
            },
            ["a"],
            [],
            20,
        )
        == []
    )


def test_end_to_end_rescue_replay_and_usage(tmp_path):
    args = parameters(tmp_path)
    calls = []

    def transport(*a):
        calls.append(a)
        return simulated_transport(*a)

    result = plan(**args, transport=transport)
    count = len(calls)
    assert (
        len(result["selected_ids"]) == 32
        and count == result["screening_audit"]["number_of_chunks"] + 2
    )
    assert len(set(result["selected_ids"]) - set(result["nominee_ids"])) == 1
    assert plan(**args, transport=transport) == result and len(calls) == count
    diag = selection_diagnostics(args["cards"], result)
    assert (
        diag["full_pool_screen_coverage"] == 1
        and diag["selected_from_nominees_rate"] == 31 / 32
    )
    assert diag["input_tokens"] is None and not diag["usage_complete"]
    assert diag["stereo_contrast_pairs"] > 0


def test_directory_can_rescue_any_legal_id_with_no_nominees(tmp_path):
    args = parameters(tmp_path)
    sent = []

    def transport(messages, config, budget):
        first = json.loads(messages[1]["content"])
        last = json.loads(messages[-1]["content"])
        if messages[0]["content"] != ARBITRATE_PROMPT:
            value = {
                "nominees": [],
                "chunk_summary": "no forced quota",
                "rescue_candidate_ids": [],
            }
        elif len(messages) == 2:
            value = {"type": "directory", "chunk_indices": [0]}
        elif "chunk_ids" in last:
            value = {
                "type": "expand",
                "expand_candidate_ids": last["chunk_ids"]["0"][:32],
            }
        else:
            t = last["expanded_cards"]
            ids = [r[t["columns"].index("id")] for r in t["rows"]]
            value = {
                "type": "selection",
                "packet_hash": first["packet_hash"],
                "choices": [
                    {
                        "id": i,
                        "reason": "test",
                        "scientific_role": "exploratory",
                        "hypothesis_id": None,
                        "evidence_ids": [],
                    }
                    for i in ids
                ],
                "hypotheses": [],
                "previous_hypothesis_updates": [],
                "feedback_interpretation": "",
                "batch_rationale": "all rescued",
                "unresolved_questions": [],
            }
        sent.append(messages)
        answer = encode(value).decode()
        return answer, {
            "request_sha256": hashlib.sha256(
                encode(payload(messages, config, budget))
            ).hexdigest(),
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "native_tool_calls": 0,
        }

    result = plan(**args, transport=transport)
    assert result["nominee_ids"] == [] and len(result["selected_ids"]) == 32


def test_ambiguous_request_and_tampered_receipt_fail_closed(tmp_path):
    args = parameters(tmp_path)

    def fail(*args):
        raise RuntimeError("network unavailable")

    with pytest.raises(RuntimeError):
        plan(**args, transport=fail)

    def forbidden(*args):
        raise AssertionError("must not retry")

    with pytest.raises(RuntimeError, match="ambiguous"):
        plan(**args, transport=forbidden)
    other = tmp_path / "other"
    args["directory"] = other
    plan(**args, transport=simulated_transport)
    path = next(other.glob("*.receipt.json"))
    saved = read_json(path)
    saved["answer"] = "{}"
    atomic_json(path, saved)
    with pytest.raises(RuntimeError, match="receipt"):
        plan(**args, transport=forbidden)


def test_protocol_drift_before_replay(tmp_path):
    args = parameters(tmp_path)
    plan(**args, transport=simulated_transport)
    args["cards"][0]["pred_center"] += 1
    with pytest.raises(RuntimeError, match="immutable"):
        plan(**args, transport=simulated_transport)


def test_final_choice_validation(tmp_path):
    args = parameters(tmp_path)
    result = plan(**args, transport=simulated_transport)
    value = result["response"]
    legal = args["legal_ids"]
    visible = result["visible_in_arbitration"]
    for fault in (
        "short",
        "duplicate",
        "illegal",
        "not_observed",
        "wrong_binding",
        "hypothesis",
    ):
        bad = copy.deepcopy(value)
        if fault == "short":
            bad["choices"].pop()
        if fault == "duplicate":
            bad["choices"][1] = bad["choices"][0]
        if fault == "illegal":
            bad["choices"][0]["id"] = "unknown"
        if fault == "not_observed":
            bad["choices"][0]["evidence_ids"] = [legal[0]]
        if fault == "wrong_binding":
            bad["packet_hash"] = "wrong"
        if fault == "hypothesis":
            bad["choices"][0]["hypothesis_id"] = "invented"
        with pytest.raises(ValueError):
            validate_selection(bad, result["packet_hash"], legal, visible, [], [])


def test_memory_local_and_round0_has_no_invented_errors():
    validate_observations([{"id": "a", "response": 1.0, "smiles": "C"}])
    for key, value in [
        ("study", "other"),
        ("seed", 2525),
        ("method", "random"),
        ("round", 1),
    ]:
        batch = {
            "study": STUDY_ID,
            "seed": 1525,
            "method": runner.TRAJECTORY,
            "round": 0,
            "observations": [],
            "response": {},
        }
        batch[key] = value
        with pytest.raises(ValueError):
            build_memory([batch], 1525, runner.TRAJECTORY)
    assert memory()["previous_hypotheses"] == []


def test_feedback_frozen_prediction_reason_and_hypothesis():
    selection = {
        "study": STUDY_ID,
        "seed": 1525,
        "method": runner.TRAJECTORY,
        "round": 0,
        "selected": [2],
        "prediction_before_measurement": {"2": [1.0, 8.0, 10.0]},
        "response": {
            "choices": [
                {
                    "id": opaque(2),
                    "reason": "test blindness",
                    "scientific_role": "hypothesis_test",
                    "hypothesis_id": "h1",
                }
            ]
        },
    }
    row = feedback(selection, [3.0], {2: {"smiles": "C", "flow": 1.0}})["observations"][
        0
    ]
    assert (
        row["signed_error"] == row["abs_error"] == 5
        and row["premeasurement_center"] == 8
    )
    assert row["selection_reason"] == "test blindness" and row["hypothesis_id"] == "h1"


def test_summary_uses_only_observed_errors():
    c = cards()
    result = summary(c, [{"id": "observed", "scaffold": "", "abs_error": 7}])
    assert result["highest_error_related_scaffold_coverage"][0][
        "candidate_count"
    ] == len(c)
    assert summary(c, [])["highest_error_related_scaffold_coverage"] == []
    assert sum(result["ipa_flow_counts"].values()) == len(c)


def test_label_barrier_and_test_denial(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("RT,Speed\n" + "1,1\n" * 35)
    partition = {
        "rows": [
            {"sample_index": i, "role": "l0" if i == 0 else "test" if i == 34 else "u0"}
            for i in range(35)
        ]
    }
    store = RestrictedLabelStore(
        partition, tmp_path / "audit.csv", runner.TRAJECTORY, source
    )
    selected = list(range(1, 33))
    with pytest.raises(PermissionError):
        store.reveal(selected, "fit")
    value = {
        "method": runner.TRAJECTORY,
        "round": 0,
        "L_hash": ids_hash([0]),
        "U_hash": ids_hash(range(1, 34)),
        "selected": selected,
        "selected_hash": stable_hash(selected),
    }
    atomic_json(tmp_path / "selection.json", value)
    with pytest.raises(PermissionError):
        store.reveal(selected, "fit")
    store.commit_selection(tmp_path / "selection.json")
    assert len(store.reveal(selected, "fit")) == 32
    with pytest.raises(PermissionError):
        store.reveal([34], "final_test")


def test_git_seal_required_before_host_reveal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    atomic_json(tmp_path / "protocol.json", {})
    atomic_json(
        tmp_path / "selection_seal.json",
        {"files": {}, "protocol_sha256": sha(tmp_path / "protocol.json")},
    )
    with pytest.raises((ValueError, RuntimeError)):
        runner.require_git_commit(tmp_path)


def test_completed_advance_reuses_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(runner, "runtime", lambda: tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda: ({}, {}))
    atomic_json(tmp_path / "round_0/complete.json", {"files": {}, "budget": 365})

    def forbidden(*args, **kwargs):
        raise AssertionError("no refit")

    monkeypatch.setattr(runner, "fit", forbidden)
    assert runner.advance(0)["budget"] == 365


def test_no_round2():
    for operation in (runner.make_round, runner.advance):
        with pytest.raises(ValueError):
            operation(2)


def test_fixed_metrics_and_aulc():
    a = evaluate([1, 4, 5], [2, 4, 8], 8.879240547556758, 333)
    b = evaluate([2, 4, 6], [2, 4, 8], a["fixed_denominator"], 365)
    assert a["nrmse"] == a["rmse"] / a["fixed_denominator"]
    assert label_aulc([a, b])["mean_nrmse"] == pytest.approx(
        (a["nrmse"] + b["nrmse"]) / 2
    )


def test_initial_observation_cannot_invent_partial_errors():
    with pytest.raises(ValueError, match="incomplete"):
        validate_observations([{"id": "a", "response": 1.0, "signed_error": 0.0}])


def test_partial_observation_compression_is_lossless():
    from hplc_al.llm.full_pool import compact_records

    observed = [
        {"id": "initial", "response": 1.0},
        {"id": "later", "response": 2.0, "premeasurement_center": 3.0},
    ]
    groups = compact_records(observed)
    restored = [dict(zip(g["columns"], row)) for g in groups for row in g["rows"]]
    assert restored == observed and "premeasurement_center" not in groups[0]["columns"]


def test_round1_requires_real_observed_evidence_for_updates(tmp_path):
    args = parameters(tmp_path)
    result = plan(**args, transport=simulated_transport)
    value = result["response"]
    previous = ["old"]
    with pytest.raises(ValueError, match="previous hypothesis"):
        validate_selection(
            value,
            result["packet_hash"],
            args["legal_ids"],
            result["visible_in_arbitration"],
            [],
            previous,
        )
    value["previous_hypothesis_updates"] = [
        {
            "id": "old",
            "status": "unresolved",
            "reason": "unknown",
            "supporting_observations": [],
            "contradicting_observations": [],
        }
    ]
    value["feedback_interpretation"] = "unresolved until observed"
    assert (
        len(
            validate_selection(
                value,
                result["packet_hash"],
                args["legal_ids"],
                result["visible_in_arbitration"],
                [],
                previous,
            )
        )
        == 32
    )
    value["previous_hypothesis_updates"][0]["status"] = "supported"
    with pytest.raises(ValueError, match="unsupported"):
        validate_selection(
            value,
            result["packet_hash"],
            args["legal_ids"],
            result["visible_in_arbitration"],
            [],
            previous,
        )


def test_illegal_non_nominee_expansion_never_freezes(tmp_path):
    args = parameters(tmp_path)

    def transport(messages, config, budget):
        if messages[0]["content"] != ARBITRATE_PROMPT:
            return simulated_transport(messages, config, budget)
        answer = '{"type":"expand","expand_candidate_ids":["unknown"]}'
        return answer, {
            "request_sha256": hashlib.sha256(
                encode(payload(messages, config, budget))
            ).hexdigest(),
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "native_tool_calls": 0,
        }

    with pytest.raises(ValueError, match="screened legal"):
        plan(**args, transport=transport)
    assert not (tmp_path / "selection.json").exists()


def test_host_advance_stops_before_label_store_without_git_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(runner, "runtime", lambda: tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda: ({}, {}))

    def forbidden(*args, **kwargs):
        raise AssertionError("must not construct or read label store")

    monkeypatch.setattr(runner, "state", forbidden)
    with pytest.raises(FileNotFoundError):
        runner.advance(0)


def test_exclusive_host_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    with runner.exclusive():
        with pytest.raises(RuntimeError, match="another V2"):
            with runner.exclusive():
                pass
