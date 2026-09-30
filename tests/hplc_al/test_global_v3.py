"""No provider calls: verify global visibility, complete inputs and label barriers."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_fullpool_v2 import cards

from hplc_al.common import atomic_json, read_json
from hplc_al.llm.full_pool import token_bound
from hplc_al.llm.memory import build_memory
from hplc_al.llm.responses_transport import encode
from hplc_al.llm.wire import display_numbers, replace
from hplc_global import runner
from hplc_global.codec import decode_table, pack, unpack
from hplc_global.planner import LIMITS, ContextCapacityError, build_request, plan
from hplc_global.verification import read_request, simulated_transport


def packet(n=90):
    c = cards(n)
    return dict(
        cards=c,
        legal_ids=[r["id"] for r in c],
        observed=[],
        memory=build_memory([], 1525, runner.TRAJECTORY, study=runner.STUDY_ID),
        round_index=0,
    )


def parameters(tmp_path):
    return dict(
        **packet(),
        directory=tmp_path,
        config=runner.EXPECTED_CONFIG,
        limits={**LIMITS, "page_size": 17},
    )


def forbidden(*args, **kwargs):
    raise AssertionError("must not call provider or access labels")


def test_column_codec_preserves_every_field_and_alias_like_literals():
    p = packet()
    p["cards"][0]["MW"] = 90.123456789
    p["observed"] = [
        {"id": "observed", "response": 1.23456789, "smiles": "C", "CIP": "R"}
    ]
    p["memory"]["unresolved_questions"] = ["c0", "o0", "g0", "@1", "c1"]
    value, mapping = pack(p["cards"], p["observed"], p["memory"])
    assert not {"c0", "o0", "g0", "c1"} & set(mapping.values())
    assert unpack(value, mapping) == display_numbers(
        {k: p[k] for k in ("cards", "observed", "memory")}
    )
    assert len(value["dictionaries"]) > 0
    assert all(
        set(a) == set(b) for a, b in zip(p["cards"], unpack(value, mapping)["cards"])
    )


@pytest.mark.parametrize("index", [-1, 2, "0", True])
def test_bad_dictionary_indices_rejected(index):
    with pytest.raises(ValueError, match="dictionary"):
        decode_table({"columns": ["field"], "rows": [[index]]}, {"field": ["a", "b"]})


def test_one_request_contains_every_page_and_every_field(tmp_path):
    args = parameters(tmp_path)
    p = {k: args[k] for k in packet()}
    messages, mapping, _, audit = build_request(**p, limits=args["limits"])
    metadata, supplied = read_request(messages)
    assert audit["input_coverage"] == 1 and audit["candidate_count"] == 90
    assert audit["page_sizes"] == [17, 17, 17, 17, 17, 5]
    assert audit["screening_calls"] == 0 and audit["nomination_cap"] is None
    restored = replace(supplied, {v: k for k, v in mapping.items()})
    assert restored == display_numbers(sorted(args["cards"], key=lambda r: r["id"]))
    assert metadata["candidate_count"] == 90
    assert all(m["role"] == "user" for m in messages[1:])
    assert json.loads(messages[-1]["content"])["kind"] == "END_OF_POOL"


def test_global_selection_last_page_and_exact_replay(tmp_path):
    args = parameters(tmp_path)
    calls = []

    def transport(messages, config, budget):
        calls.append(messages)
        return simulated_transport(messages, config, budget)

    saved = plan(**args, transport=transport)
    assert len(calls) == 1 and len(saved["selected_ids"]) == 32
    assert sorted(args["legal_ids"])[-1] in saved["selected_ids"]
    assert saved["visible_candidates"] == sorted(args["legal_ids"])
    assert not list(tmp_path.glob("screen*"))
    assert len(list(tmp_path.glob("*.request.json"))) == 1
    args["cards"].reverse()
    args["legal_ids"].reverse()
    assert plan(**args, transport=forbidden) == saved


def test_capacity_counts_whole_conversation_not_largest_page(tmp_path):
    args = parameters(tmp_path)
    p = {k: args[k] for k in packet()}
    messages, _, _, audit = build_request(**p, limits=args["limits"])
    total = audit["required_context_tokens"]
    assert token_bound(messages) > max(token_bound([m]) for m in messages)
    args["limits"]["context_tokens"] = total - 1
    with pytest.raises(ContextCapacityError, match="no API call sent"):
        plan(**args, transport=forbidden)
    assert (
        read_json(tmp_path / "context_admission.json")["status"]
        == "BLOCKED_CONTEXT_CAPACITY"
    )
    assert not list(tmp_path.glob("*.request.json"))
    assert not (tmp_path / "selection.json").exists()


def test_reply_over_old_screen_cap_is_valid(tmp_path):
    args = parameters(tmp_path)

    def transport(messages, config, budget):
        raw, receipt = simulated_transport(messages, config, budget)
        value = json.loads(raw)
        for choice in value["choices"]:
            choice["reason"] = (
                "Compare chemical controls across all pages and observed evidence. " * 2
            )
        answer = encode(value).decode()
        assert token_bound(answer) - 4096 > 2048
        receipt["answer_sha256"] = hashlib.sha256(answer.encode()).hexdigest()
        return answer, receipt

    assert len(plan(**args, transport=transport)["selected_ids"]) == 32


@pytest.mark.parametrize(
    "fault", ["duplicate", "unknown", "unobserved_evidence", "expand", "wrong_hash"]
)
def test_invalid_global_selection_never_saved(tmp_path, fault):
    args = parameters(tmp_path)

    def transport(messages, config, budget):
        raw, receipt = simulated_transport(messages, config, budget)
        value = json.loads(raw)
        if fault == "duplicate":
            value["choices"][1] = value["choices"][0]
        elif fault == "unknown":
            value["choices"][0]["id"] = "c999999"
        elif fault == "unobserved_evidence":
            value["choices"][0]["evidence_ids"] = [value["choices"][0]["id"]]
        elif fault == "expand":
            value = {"type": "expand", "expand_candidate_ids": ["c0"]}
        else:
            value["packet_hash"] = "wrong"
        answer = encode(value).decode()
        receipt["answer_sha256"] = hashlib.sha256(answer.encode()).hexdigest()
        return answer, receipt

    with pytest.raises(ValueError):
        plan(**args, transport=transport)
    assert not (tmp_path / "selection.json").exists()
    # Preserve failed response for audit; never silently accept it on resume.
    assert (tmp_path / "global_selection.receipt.json").exists()
    with pytest.raises(ValueError):
        plan(**args, transport=forbidden)


def test_receipt_tampering_and_input_drift_rejected(tmp_path):
    args = parameters(tmp_path)
    plan(**args, transport=simulated_transport)
    path = tmp_path / "global_selection.receipt.json"
    saved = read_json(path)
    saved["answer"] = "{}"
    atomic_json(path, saved)
    with pytest.raises(RuntimeError, match="receipt binding"):
        plan(**args, transport=forbidden)
    args["cards"][0]["pred_center"] += 1
    with pytest.raises(RuntimeError, match="immutable"):
        plan(**args, transport=forbidden)


def test_duplicate_and_missing_candidate_stop_before_request(tmp_path):
    args = parameters(tmp_path)
    args["cards"][1] = args["cards"][0]
    with pytest.raises(ValueError, match="legal U"):
        plan(**args, transport=forbidden)


def test_v3_isolated_from_v2():
    from hplc_al.llm import runner as v2

    assert runner.STUDY != v2.STUDY and runner.TRAJECTORY != v2.TRAJECTORY
    assert "global" in str(runner.runtime())
    assert not any("hplc_global" in p for p in v2.source_hashes())
    assert any("hplc_global/planner.py" in p for p in runner.source_hashes())


def test_v3_label_barrier_and_round_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(runner, "runtime", lambda: tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda: ({}, {}))
    monkeypatch.setattr(runner, "state", forbidden)
    with pytest.raises(FileNotFoundError):
        runner.advance(0)
    with pytest.raises(ValueError, match="hard stop"):
        runner.advance(6)


def test_context_checked_before_credentials_in_selection(tmp_path, monkeypatch):
    limits = {**LIMITS, "context_tokens": 1}
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(
        runner,
        "make_round",
        lambda r: ({"limits": limits}, {}, tmp_path, packet(), [], []),
    )
    monkeypatch.setattr(runner, "require_key", forbidden)
    with pytest.raises(ContextCapacityError):
        runner.run_selection(0)


def test_gate_detects_stale_sources_and_modified_evidence(tmp_path):
    from hplc_global.verification import verify_gate

    root = Path(__file__).resolve().parents[2]
    sources = {"code": "digest"}
    protocol = {"limits": LIMITS, "transport_registration": runner.EXPECTED_CONFIG}
    from hplc_al.common import stable_hash

    gate = {
        "status": "PASS",
        "source_hashes": sources,
        "limits": LIMITS,
        "config_sha256": stable_hash(runner.EXPECTED_CONFIG),
        "evidence": {},
        "test_hashes": {},
    }
    atomic_json(tmp_path / "test_gate.json", gate)
    verify_gate(tmp_path, protocol, sources)
    with pytest.raises(RuntimeError, match="verification"):
        verify_gate(tmp_path, protocol, {"code": "changed"})
    gate["evidence"] = {str(root / "pyproject.toml"): "tampered"}
    atomic_json(tmp_path / "test_gate.json", gate)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        verify_gate(tmp_path, protocol, sources)


def test_real_baseline_preparation_preserves_full_pool_without_screening(monkeypatch):
    from tempfile import TemporaryDirectory

    from hplc_al.common import ROOT
    from hplc_global.verification import baseline_packet

    expected = baseline_packet()
    with TemporaryDirectory(prefix=".global-v3-test-", dir=ROOT) as tmp:
        monkeypatch.setattr(runner, "STUDY", Path(tmp))
        protocol, _, directory, actual, labeled, unlabeled = runner.make_round(0)
        assert actual == expected
        assert len(labeled) == 333 and len(unlabeled) == 4114
        assert protocol["selection_mode"] == "global_all_candidates"
        assert "nominees_per_chunk" not in protocol["limits"]
        assert not (directory / "llm").exists()
        assert runner.make_round(0)[3] == actual


@pytest.mark.parametrize("already_completed", [0, 3])
def test_v3_pipeline_seal_before_reveal_and_resume(
    monkeypatch, tmp_path, already_completed
):
    from contextlib import nullcontext

    from hplc_global import continuous, verification

    events = []
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(runner, "runtime", lambda: tmp_path)
    monkeypatch.setattr(runner, "exclusive", nullcontext)
    monkeypatch.setattr(runner, "prepare", lambda: None)
    monkeypatch.setattr(continuous, "check_checkout", lambda: None)
    monkeypatch.setattr(continuous, "require_key", lambda c: events.append("key"))
    monkeypatch.setattr(continuous, "preflight", lambda *a: events.append("preflight"))
    monkeypatch.setattr(continuous, "commit_artifacts", lambda msg: events.append(msg))
    monkeypatch.setattr(
        runner,
        "make_round",
        lambda r: ({"limits": LIMITS}, {}, tmp_path, packet(), [], []),
    )
    monkeypatch.setattr(verification, "verify_gate", lambda *a: None)
    monkeypatch.setattr(runner, "source_hashes", lambda: {})

    def select(r):
        events.append(f"select {r}")
        atomic_json(tmp_path / f"round_{r}/selection.json", {})

    def advance(r):
        if not (tmp_path / f"round_{r}/complete.json").exists():
            assert (tmp_path / f"round_{r}/selection.json").exists()
            assert events[-1].startswith("Seal Global V3")
            atomic_json(tmp_path / f"round_{r}/complete.json", {})
        events.append(f"advance {r}")

    monkeypatch.setattr(runner, "run_selection", select)
    monkeypatch.setattr(runner, "advance", advance)
    monkeypatch.setattr(runner, "report", lambda: {"status": "COMPLETE_PHASE1_L525"})
    for r in range(already_completed):
        atomic_json(tmp_path / f"round_{r}/complete.json", {})
    continuous.run(progress=lambda *a, **k: None)
    assert [e for e in events if e.startswith("select ")] == [
        f"select {r}" for r in range(already_completed, 6)
    ]
    assert events.index("preflight") < events.index(f"select {already_completed}")
    events.clear()
    continuous.run(progress=lambda *a, **k: None)
    assert "key" not in events and "preflight" not in events
    assert not any(e.startswith("select ") for e in events)
    assert read_json(tmp_path / "execution_status.json")["budget"] == 525


def test_phase_transport_accepts_commentary_and_one_final(monkeypatch, tmp_path):
    from test_responses_transport import CONFIG, HTTP, KEY, response

    from hplc_global import transport

    monkeypatch.setenv(CONFIG["env_key"], KEY)
    value = response()
    value["output"][-1]["phase"] = "final_answer"
    value["output"].insert(
        1,
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Reviewing the whole pool."}],
        },
    )
    captured = []

    def opener(request, timeout):
        captured.append(request)
        return HTTP(value)

    answer, receipt = transport.call(
        [{"role": "user", "content": "{}"}],
        CONFIG,
        opener=opener,
        diagnostic_directory=tmp_path,
    )
    assert answer == '{"status":"ok"}' and receipt["output_message_count"] == 2
    assert receipt["message_extraction"] == "final_answer_phase"
    from hplc_al.llm.responses_transport import wire_payload

    assert json.loads(captured[0].data) == wire_payload(
        [{"role": "user", "content": "{}"}], CONFIG, 32000
    )
    snapshot = read_json(next(tmp_path.glob("*.json")))
    assert len(snapshot["messages"]) == 2 and "reasoning" not in json.dumps(snapshot)
    assert KEY not in json.dumps(snapshot)


@pytest.mark.parametrize(
    "fault",
    [
        "two_finals",
        "commentary_only",
        "two_json",
        "mixed_prose",
        "tool",
        "incomplete",
        "secret",
    ],
)
def test_phase_transport_rejects_ambiguous_or_unsafe_outputs(
    monkeypatch, tmp_path, fault
):
    from test_responses_transport import CONFIG, HTTP, KEY, response

    from hplc_global import transport

    monkeypatch.setenv(CONFIG["env_key"], KEY)
    value = response()
    message = value["output"][-1]
    if fault == "two_finals":
        message["phase"] = "final_answer"
        value["output"].append(copy.deepcopy(message))
    elif fault == "commentary_only":
        message["phase"] = "commentary"
    elif fault in ("two_json", "mixed_prose"):
        extra = copy.deepcopy(message)
        if fault == "mixed_prose":
            extra["content"][0]["text"] = "Unmarked commentary"
        value["output"].append(extra)
    elif fault == "tool":
        value["output"].append({"type": "function_call"})
    elif fault == "incomplete":
        value["status"] = "incomplete"
    else:
        message["content"][0]["text"] = KEY
    with pytest.raises(transport.TransportError):
        transport.call(
            [{"role": "user", "content": "{}"}],
            CONFIG,
            opener=lambda *a, **k: HTTP(value),
            diagnostic_directory=tmp_path,
        )
    if fault == "secret":
        assert not list(tmp_path.glob("*.json"))


def test_phase_transport_joins_fragmented_json():
    from hplc_global.transport import extract_answer

    pieces = [
        {"role": "assistant", "content": [{"type": "output_text", "text": t}]}
        for t in ['{"ok":', "true}"]
    ]
    assert extract_answer(pieces)[0] == '{"ok":true}'
