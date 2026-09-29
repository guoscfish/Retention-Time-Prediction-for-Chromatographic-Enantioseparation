import hashlib
import io
from contextlib import nullcontext

import pytest

from hplc_al.common import atomic_json, read_json, sha, stable_hash
from hplc_al.llm import execution, runner
from hplc_al.llm.full_pool import LIMITS
from hplc_al.llm.planner import Journal
from hplc_al.llm.responses_transport import TransportError, encode, payload

CONFIG = runner.EXPECTED_CONFIG
MESSAGES = [{"role": "user", "content": "fixture only"}]


@pytest.fixture(autouse=True)
def instant(monkeypatch):
    monkeypatch.setattr(execution.time, "sleep", lambda _: None)
    monkeypatch.setattr(execution, "heartbeat", lambda *a, **kw: nullcontext())


def success(messages, config, budget):
    answer = '{"ok":true}'
    return answer, {
        "request_sha256": hashlib.sha256(
            encode(payload(messages, config, budget))
        ).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "native_tool_calls": 0,
    }


def journal(tmp_path, transport):
    return Journal(tmp_path, CONFIG, LIMITS, transport, retry_requests=True)


def test_524_then_success_identical_payload_and_no_paid_replay(tmp_path):
    bodies = []

    def transport(*args):
        bodies.append(encode(payload(*args)))
        if len(bodies) < 3:
            raise TransportError("RESPONSES_HTTP_524")
        return success(*args)

    assert journal(tmp_path, transport).ask("screen_000", MESSAGES) == {"ok": True}
    assert len(bodies) == 3 and len(set(bodies)) == 1
    journal(tmp_path, lambda *a: pytest.fail("receipt must be reused")).ask(
        "screen_000", MESSAGES
    )
    assert len(list(tmp_path.glob("*.started.json"))) == 3
    assert len(list(tmp_path.glob("*.failed.json"))) == 2


def test_retry_limit_survives_relaunch(tmp_path):
    calls = []

    def fail(*args):
        calls.append(1)
        raise TransportError("RESPONSES_HTTP_524")

    with pytest.raises(TransportError, match="524"):
        journal(tmp_path, fail).ask("screen_000", MESSAGES)
    assert len(calls) == 5
    with pytest.raises(TransportError, match="LIMIT_EXHAUSTED"):
        journal(tmp_path, fail).ask("screen_000", MESSAGES)
    assert len(calls) == 5


@pytest.mark.parametrize(
    "code",
    [
        "RESPONSES_HTTP_401",
        "RESPONSES_HTTP_403",
        "RESPONSES_SCHEMA_FAILURE",
        "RESPONSES_INVALID_JSON_ANSWER",
        "RESPONSES_NATIVE_TOOL_OR_INVALID_OUTPUT",
    ],
)
def test_terminal_failures_never_resampled_even_on_restart(tmp_path, code):
    calls = []

    def fail(*args):
        calls.append(1)
        raise TransportError(code)

    with pytest.raises(TransportError):
        journal(tmp_path, fail).ask("screen_000", MESSAGES)
    with pytest.raises(RuntimeError, match="terminal"):
        journal(tmp_path, fail).ask("screen_000", MESSAGES)
    assert len(calls) == 1


def test_original_ambiguous_request_counts_as_attempt_one(tmp_path):
    request = payload(MESSAGES, CONFIG, LIMITS["max_output_tokens"])
    record = {
        "request": request,
        "request_sha256": hashlib.sha256(encode(request)).hexdigest(),
        "config_sha256": stable_hash(CONFIG),
    }
    path = tmp_path / "screen_004.request.json"
    atomic_json(path, record)
    before = sha(path)
    journal(tmp_path, success).ask("screen_004", MESSAGES)
    assert sha(path) == before
    assert len(list(tmp_path.glob("*.started.json"))) == 2
    assert (
        "pre-amendment"
        in read_json(tmp_path / "screen_004.attempt_01.started.json")["origin"]
    )


def test_interrupt_consumes_attempt_and_resume_uses_next(tmp_path):
    def interrupt(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        journal(tmp_path, interrupt).ask("screen_000", MESSAGES)
    journal(tmp_path, success).ask("screen_000", MESSAGES)
    assert len(list(tmp_path.glob("*.started.json"))) == 2


def test_unknown_exception_does_not_leak_or_retry(tmp_path, capsys):
    secret = "dummy-secret-do-not-emit"

    def fail(*args):
        raise RuntimeError(secret)

    with pytest.raises(RuntimeError):
        journal(tmp_path, fail).ask("screen_000", MESSAGES)
    assert secret not in capsys.readouterr().out
    assert all(secret not in p.read_text() for p in tmp_path.glob("*.json"))


def test_preflight_retries_network_but_not_bad_key():
    calls = []

    def operation():
        calls.append(1)
        if len(calls) == 1:
            raise TransportError("RESPONSES_NETWORK_FAILURE")
        return "pass"

    assert execution.preflight_with_retries(operation) == "pass"
    assert len(calls) == 2

    def denied():
        calls.append(1)
        raise TransportError("RESPONSES_HTTP_401")

    with pytest.raises(TransportError):
        execution.preflight_with_retries(denied)
    assert len(calls) == 3


def test_training_progress_tolerates_partial_row(tmp_path):
    path = tmp_path / "fit/attempt_000/training_curve.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        'epoch,train_loss,validation_mse,observed_batch_sizes\n1,5,7,"[2, 1]"\n2,4'
    )
    assert "epoch=1" in execution.training_detail(tmp_path)


def test_amendment_cannot_change_scientific_sources(monkeypatch, tmp_path):
    protocol = {"source_hashes": {"src/hplc_al/llm/full_pool.py": "old"}}
    atomic_json(tmp_path / "protocol.json", protocol)
    current = {"src/hplc_al/llm/full_pool.py": "new"}
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(runner, "source_hashes", lambda: current)
    atomic_json(
        tmp_path / "execution_amendment.json",
        {
            "protocol_sha256": sha(tmp_path / "protocol.json"),
            "original_source_hashes": protocol["source_hashes"],
            "source_hashes": current,
            "authorization": "允许 V2 仅修改日志和重试，复用已有结果",
            "max_attempts_per_scientific_request": 5,
            "scientific_request_payload_changed": False,
        },
    )
    with pytest.raises(RuntimeError, match="amendment/source"):
        runner.verify_execution_amendment(protocol)


@pytest.mark.parametrize("kind", ["timeout", "truncated", "invalid_json"])
def test_transport_distinguishes_network_from_schema(monkeypatch, kind):
    import http.client

    from hplc_al.llm import responses_transport as rt

    monkeypatch.setenv(CONFIG["env_key"], "fake-secret-only")

    class Response(io.BytesIO):
        status = 200

        def read(self):
            if kind == "timeout":
                raise TimeoutError
            if kind == "truncated":
                raise http.client.IncompleteRead(b"partial")
            return b"not-json"

    expected = "SCHEMA_FAILURE" if kind == "invalid_json" else "NETWORK_FAILURE"
    with pytest.raises(TransportError, match=expected):
        rt.call(MESSAGES, CONFIG, opener=lambda *a, **k: Response())


def test_failed_attempts_have_unknown_usage_in_report(tmp_path):
    directory = tmp_path / "round_0/llm"
    directory.mkdir(parents=True)
    calls = []

    def transport(*args):
        calls.append(1)
        if len(calls) == 1:
            raise TransportError("RESPONSES_HTTP_524")
        return success(*args)

    journal(directory, transport).ask("screen_000", MESSAGES)
    report = execution.attempt_summary(tmp_path)
    assert report["scientific_dispatch_intents"] == 2
    assert report["validated_response_receipts"] == 1
    assert report["attempts_without_usage_receipts"] == 1


def test_failure_status_written_before_lock_release(monkeypatch, tmp_path):
    from hplc_al.llm import continuous

    monkeypatch.setattr(runner, "STUDY", tmp_path)
    monkeypatch.setattr(runner, "runtime", lambda: tmp_path)
    with pytest.raises(TransportError):
        with continuous.failure_status():
            raise TransportError("RESPONSES_HTTP_524")
    status = read_json(tmp_path / "execution_status.json")
    assert status["status"] == "STOPPED_FAILURE" and status["budget"] == 333
    assert read_json(tmp_path / "last_failure.json")["error"] == "RESPONSES_HTTP_524"
