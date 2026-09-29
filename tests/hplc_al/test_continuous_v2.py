import json
import subprocess
from contextlib import nullcontext

import pytest

from hplc_al.common import atomic_json, read_json
from hplc_al.llm import continuous, runner
from hplc_al.llm.full_pool import BUDGETS, LIMITS, ROUNDS
from hplc_al.llm.wire import (
    aliases,
    content,
    memory_view,
    pack,
    replace,
    summary_view,
    unpack,
)


def test_six_round_budget_and_explicit_target():
    assert ROUNDS == 6 and BUDGETS == [333, 365, 397, 429, 461, 493, 525]
    assert runner.EXPECTED_CONFIG["model"] == "gpt-6-astra"
    assert runner.EXPECTED_CONFIG["provider_id"] == "token4research"
    assert runner.EXPECTED_CONFIG["reasoning_effort"] == "high"


def test_wire_round_trip_preserves_structures_numbers_and_literal_references():
    structure = "C[C@H](O)c1ccccc1" * 4
    value = {
        structure: [structure, "@0", "@literal", 1.2345678901234567, None],
        "nested": {"@0": structure},
    }
    assert unpack(pack(value)) == value
    records = [{"identity": "0123456789abcdef", "scaffold_group": "fedcba9876543210"}]
    mapping = aliases(records, [])
    value = {"cards": records, "identity_counts": {records[0]["identity"]: 1}}
    assert (
        replace(unpack(pack(value, mapping)), {v: k for k, v in mapping.items()})
        == value
    )
    assert aliases([{"identity": "0", "scaffold_group": "acyclic"}], []) == {}


def test_memory_references_are_exact_and_drift_fails():
    row = {"id": "a", "response": 1.2, "selection_reason": "test"}
    memory = {
        "recent_batches": [{"observations": [row]}],
        "high_error_observations": [row],
    }
    view = memory_view(memory, [row])
    assert view["recent_batches"][0]["observations"] == [{"observed_id": "a"}]
    with pytest.raises(ValueError, match="mismatch"):
        memory_view(memory, [{**row, "response": 1.3}])


def test_nested_record_groups_round_trip_and_display_precision():
    rows = [
        {
            "id": str(i),
            "a_long_field_name": 1.2345678901234567,
            "nested": [{"long_key": "value" * 30}] * 4,
        }
        for i in range(20)
    ]
    rows[0].pop("a_long_field_name")
    encoded = pack(rows)
    assert unpack(encoded) == rows
    displayed = unpack(json.loads(content(rows)))
    assert displayed[1]["a_long_field_name"] == 1.234568
    assert rows[1]["a_long_field_name"] == 1.2345678901234567
    assert "a_long_field_name" not in displayed[0]


def test_grouped_summary_preserves_every_count():
    original = {
        "identity_counts": {"a": 1, "b": 1, "c": 3},
        "scaffold_counts": {"CC": 2, "": 3},
    }
    compressed = summary_view(original)
    for field, counts in original.items():
        assert {
            key: count for count, keys in compressed[field]["rows"] for key in keys
        } == counts


def fake_pipeline(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(runner, "runtime", lambda: tmp_path)
    monkeypatch.setattr(runner, "exclusive", nullcontext)
    monkeypatch.setattr(runner, "prepare", lambda: events.append("prepare"))
    monkeypatch.setattr(continuous, "check_checkout", lambda: events.append("checkout"))
    monkeypatch.setattr(continuous, "require_key", lambda c: events.append("key"))
    monkeypatch.setattr(continuous, "preflight", lambda *a: events.append("preflight"))
    monkeypatch.setattr(
        continuous, "commit_artifacts", lambda message: events.append(message)
    )

    def select(r):
        events.append(f"select {r}")
        atomic_json(tmp_path / f"round_{r}/selection.json", {})

    def advance(r):
        events.append(f"advance {r}")
        if not (tmp_path / f"round_{r}/complete.json").exists():
            assert (tmp_path / f"round_{r}/selection.json").exists()
            assert events[-2].startswith("Seal")
            atomic_json(tmp_path / f"round_{r}/complete.json", {})

    monkeypatch.setattr(runner, "run_selection", select)
    monkeypatch.setattr(runner, "advance", advance)
    monkeypatch.setattr(runner, "report", lambda: {"status": "COMPLETE_PHASE1_L525"})
    return events


def test_continuous_six_round_order_and_completed_resume(monkeypatch, tmp_path):
    events = fake_pipeline(monkeypatch, tmp_path)
    continuous.run(progress=lambda *a, **k: None)
    assert [e for e in events if e.startswith("select ")] == [
        f"select {r}" for r in range(6)
    ]
    assert events.index("preflight") < events.index("select 0")
    events.clear()
    continuous.run(progress=lambda *a, **k: None)
    assert "preflight" not in events and "key" not in events
    assert not any(e.startswith("select ") for e in events)
    assert read_json(tmp_path / "execution_status.json")["budget"] == 525


def test_preflight_failure_prevents_any_selection(monkeypatch, tmp_path):
    events = fake_pipeline(monkeypatch, tmp_path)

    def fail(*args):
        raise RuntimeError("preflight failure")

    monkeypatch.setattr(continuous, "preflight", fail)
    with pytest.raises(RuntimeError, match="preflight failure"):
        continuous.run(progress=lambda *a, **k: None)
    assert not any(e.startswith(("select ", "advance ")) for e in events)


def test_partial_resume_skips_finished_rounds(monkeypatch, tmp_path):
    events = fake_pipeline(monkeypatch, tmp_path)
    for r in range(3):
        atomic_json(tmp_path / f"round_{r}/complete.json", {})
    continuous.run(progress=lambda *a, **k: None)
    assert [e for e in events if e.startswith("select ")] == [
        "select 3",
        "select 4",
        "select 5",
    ]


def test_git_commits_only_study_and_rejects_staged_changes(monkeypatch, tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, stderr=subprocess.PIPE
        )

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    study = tmp_path / "study"
    study.mkdir()
    (tmp_path / "unrelated.txt").write_text("preserve")
    git("add", "unrelated.txt")
    git("commit", "-qm", "initial")
    monkeypatch.setattr(continuous, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    monkeypatch.setattr(runner, "exclusive", nullcontext)
    atomic_json(study / "selection_seal.json", {"test": True})
    (tmp_path / "unrelated.txt").write_text("uncommitted")
    continuous.commit_artifacts("seal fixture")
    assert json.loads(git("show", "HEAD:study/selection_seal.json")) == {"test": True}
    assert git("show", "HEAD:unrelated.txt") == b"preserve"
    git("add", "unrelated.txt")
    with pytest.raises(RuntimeError, match="staged"):
        continuous.commit_artifacts("must not commit")


def test_admission_failure_sends_zero_requests(tmp_path):
    from test_fullpool_v2 import parameters

    from hplc_al.llm.planner import plan

    args = parameters(tmp_path)
    args["limits"] = {**LIMITS, "context_tokens": 45000, "max_output_tokens": 32000}
    calls = []
    with pytest.raises(ValueError, match="ADMISSION_FAILED"):
        plan(**args, transport=lambda *a: calls.append(a))
    assert calls == [] and not list(tmp_path.glob("*.request.json"))


def test_duplicate_pipeline_lock_blocks_before_work(monkeypatch, tmp_path):
    import fcntl

    events = fake_pipeline(monkeypatch, tmp_path)
    with (tmp_path / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another continuous"):
            continuous.run(progress=lambda *a, **k: None)
    assert events == []


def test_feedback_write_before_seal_is_recoverable(monkeypatch, tmp_path):
    import numpy as np
    from test_fullpool_v2 import cards

    from hplc_al.llm.full_pool import opaque
    from hplc_al.llm.memory import feedback

    selected = list(range(1, 33))
    features = {i: cards(33)[i] for i in selected}
    selection = {
        "study": runner.STUDY_ID,
        "seed": 1525,
        "method": runner.TRAJECTORY,
        "round": 0,
        "selected": selected,
        "prediction_before_measurement": {str(i): [1.0, 8.0, 10.0] for i in selected},
        "response": {
            "choices": [
                {
                    "id": opaque(i),
                    "reason": "test",
                    "scientific_role": "control",
                    "hypothesis_id": None,
                }
                for i in selected
            ]
        },
    }
    directory = tmp_path / "round_0"
    atomic_json(directory / "selection.json", selection)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    expected = feedback(selection, np.ones(32), features)
    atomic_json(directory / "feedback.json", expected)

    class Store:
        def commit_selection(self, path):
            pass

        def reveal(self, ids, purpose):
            return np.ones(len(ids))

    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(runner, "runtime", lambda: tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda: ({"training": {}}, {}))
    monkeypatch.setattr(runner, "require_git_commit", lambda p: "sealed-commit")
    monkeypatch.setattr(runner, "state", lambda *a: ([0], selected, [], Store()))
    monkeypatch.setattr(runner, "metadata", lambda ids: (features, None))
    monkeypatch.setattr(runner, "role_ids", lambda *a: [])
    monkeypatch.setattr(runner, "RestrictedLabelStore", lambda *a: Store())
    monkeypatch.setattr(runner, "load_graphs", lambda *a: {})

    def stop_before_fit(*args):
        raise RuntimeError("FIT_BOUNDARY_REACHED")

    monkeypatch.setattr(runner, "fit", stop_before_fit)
    with pytest.raises(RuntimeError, match="FIT_BOUNDARY_REACHED"):
        runner.advance(0)
    assert read_json(directory / "feedback.json") == expected
    assert (
        runner.verify_record(directory / "feedback_seal.json")["selection_commit"]
        == "sealed-commit"
    )


def test_oversized_screening_prose_is_rejected():
    from hplc_al.llm.planner import validate_screen

    with pytest.raises(ValueError, match="bound"):
        validate_screen(
            {
                "nominees": [],
                "chunk_summary": " chemistry" * 300,
                "rescue_candidate_ids": [],
            },
            [],
            [],
            16,
        )
