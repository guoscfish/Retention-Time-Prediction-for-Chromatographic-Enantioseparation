import copy
import hashlib
import io
import json
import urllib.error

import pytest

from hplc_al.llm import responses_transport as rt

CONFIG = {
    "provider_id": "token4research",
    "model": "gpt-5.5",
    "reasoning_effort": "high",
    "base_url": "https://token4research.cn",
    "wire_api": "responses",
    "env_key": "TOKEN4RESEARCH_API_KEY",
}
KEY = "test-only-secret-never-log-this-value"


def response():
    return {
        "id": "resp_test",
        "status": "completed",
        "model": "gpt-5.5",
        "output": [
            {"type": "reasoning"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": '{"status":"ok"}'}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class HTTP:
    status = 200

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.value).encode()


def test_config_exact_no_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'model_provider="token4research"\nmodel="gpt-5.5"\nmodel_reasoning_effort="high"\n'
        '[model_providers.token4research]\nbase_url="https://token4research.cn"\n'
        'wire_api="responses"\nenv_key="TOKEN4RESEARCH_API_KEY"\nrequires_openai_auth=false\n'
    )
    assert rt.settings(path) == CONFIG
    path.write_text('model="gpt-5.5"')
    with pytest.raises(rt.TransportError, match="CONFIG_INVALID"):
        rt.settings(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wire_api", "chat"),
        ("env_key", ""),
        ("base_url", "http://example.com"),
        ("base_url", "https://user:secret@example.com"),
        ("base_url", "https://example.com/?key=sensitive"),
    ],
)
def test_config_rejects_unsafe_or_missing(field, value):
    config = {**CONFIG, field: value}
    with pytest.raises(rt.TransportError):
        rt.validate_settings(config)


def test_missing_key_never_calls_network_or_auth_file(monkeypatch):
    monkeypatch.delenv(CONFIG["env_key"], raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-use-fallback")

    def forbidden(*args, **kwargs):
        raise AssertionError("network or file access")

    monkeypatch.setattr(rt.Path, "read_text", forbidden)
    with pytest.raises(
        rt.TransportError,
        match="RESPONSES_API_KEY_NOT_CONFIGURED: TOKEN4RESEARCH_API_KEY",
    ):
        rt.call([{"role": "user", "content": "{}"}], CONFIG, opener=forbidden)


def test_explicit_wire_contract_and_safe_receipt(monkeypatch):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    captured = []

    def opener(request, timeout):
        captured.append(request)
        return HTTP(response())

    answer, receipt = rt.call(
        [{"role": "user", "content": "content-free"}], CONFIG, 2048, opener=opener
    )
    req = captured[0]
    body = json.loads(req.data)
    assert req.full_url == "https://token4research.cn/responses"
    assert body == {
        "model": "gpt-5.5",
        "reasoning": {"effort": "high"},
        "input": [{"role": "user", "content": "content-free"}],
        "tools": [],
        "store": False,
        "max_output_tokens": 2048,
        "stream": True,
    }
    assert req.get_header("Authorization") == "Bearer " + KEY
    assert req.get_header("User-agent") == rt.USER_AGENT
    assert req.get_header("Accept") == "text/event-stream, application/json"
    assert receipt["client_user_agent"] == rt.USER_AGENT
    assert receipt["wire_request_sha256"] == hashlib.sha256(req.data).hexdigest()
    del body["stream"]
    assert receipt["request_sha256"] == hashlib.sha256(rt.encode(body)).hexdigest()
    assert receipt["answer_sha256"] == hashlib.sha256(answer.encode()).hexdigest()
    assert (
        receipt["provider_id"] == "token4research"
        and receipt["base_url_hostname"] == "token4research.cn"
    )
    assert KEY not in json.dumps(receipt) and "Authorization" not in json.dumps(receipt)


@pytest.mark.parametrize(
    "answer",
    [
        "{} {}",
        "{}\n{}",
        "```json\n{}\n```",
        "[]",
        '{"a":1,"a":2}',
        '{"a":NaN}',
        "{} extra",
    ],
)
def test_strict_unique_json(answer):
    with pytest.raises(rt.TransportError):
        rt.parse_answer(answer)


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "tool",
        "two_messages",
        "refusal",
        "no_id",
        "wrong_model",
        "secret_echo",
        "bad_usage",
        "http",
    ],
)
def test_response_rejected(monkeypatch, fault):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    value = response()
    if fault == "incomplete":
        value["status"] = "incomplete"
    if fault == "tool":
        value["output"].append({"type": "function_call"})
    if fault == "two_messages":
        value["output"].append(copy.deepcopy(value["output"][-1]))
    if fault == "refusal":
        value["output"][-1]["content"][0]["type"] = "refusal"
    if fault == "no_id":
        del value["id"]
    if fault == "wrong_model":
        value["model"] = "different"
    if fault == "secret_echo":
        value["id"] = KEY
    if fault == "bad_usage":
        value["usage"]["input_tokens"] = "not numeric"
    http = HTTP(value)
    if fault == "http":
        http.status = 500
    with pytest.raises(rt.TransportError) as exc:
        rt.call(
            [{"role": "user", "content": "{}"}], CONFIG, opener=lambda *a, **kw: http
        )
    assert KEY not in str(exc.value)


def test_provider_may_omit_served_model_and_usage(monkeypatch):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    value = response()
    del value["model"]
    del value["usage"]
    _, receipt = rt.call(
        [{"role": "user", "content": "{}"}], CONFIG, opener=lambda *a, **kw: HTTP(value)
    )
    assert receipt["served_model"] is None and receipt["usage"] is None


def test_redirect_rejected():
    with pytest.raises(rt.TransportError, match="REDIRECT"):
        rt._NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.com")


def test_content_free_preflight(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    calls = []

    def transport(messages, config, budget):
        calls.append((messages, config, budget))
        return '{"status":"ok"}', {"response_id": "test"}

    result = rt.preflight(tmp_path / "preflight.json", CONFIG, transport=transport)
    assert (
        result["status"] == "PASS"
        and not result["scientific_content"]
        and len(calls) == 1
    )
    assert calls[0][0] == [
        {"role": "user", "content": 'Return exactly {"status":"ok"}'}
    ]
    assert KEY not in (tmp_path / "preflight.json").read_text()


def test_preflight_failure_no_retry(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    calls = []

    def fail(*args):
        calls.append(1)
        raise rt.TransportError("RESPONSES_HTTP_401")

    with pytest.raises(rt.TransportError):
        rt.preflight(tmp_path / "check.json", CONFIG, transport=fail)
    assert (
        len(calls) == 1
        and json.loads((tmp_path / "check.json").read_text())["status"] == "FAILED"
    )


@pytest.mark.parametrize(
    "body,code",
    [
        (b"error code: 1010\n", "RESPONSES_HTTP_403_CLOUDFLARE_1010"),
        (b'{"error_code": 1010}', "RESPONSES_HTTP_403_CLOUDFLARE_1010"),
        (b'{"error":"permissions denied"}', "RESPONSES_HTTP_403"),
    ],
)
def test_http_diagnostic_never_echoes_body_or_retries(monkeypatch, body, code):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(
            request.full_url, 403, "Forbidden", {}, io.BytesIO(body + KEY.encode())
        )

    with pytest.raises(rt.TransportError) as error:
        rt.call([{"role": "user", "content": "{}"}], CONFIG, opener=opener)
    assert str(error.value).split(":")[0] == code
    assert KEY not in str(error.value) and len(calls) == 1


class SSE(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream; charset=utf-8"}


def sse_event(event):
    return b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\r\n\r\n"


def test_stream_keeps_legacy_identity_and_complete_validation(monkeypatch, capsys):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    value = response()
    value["output"][-1]["content"][0]["text"] = '{"status":"完成"}'
    events = (
        b": keepalive\r\n\r\n"
        + sse_event({"type": "response.created", "response": {"status": "in_progress"}})
        + sse_event({"type": "response.output_text.delta", "delta": KEY})
        + b'event: response.completed\r\ndata: {"type":"response.completed",\r\ndata: "response": '
        + json.dumps(value, ensure_ascii=False).encode()
        + b"}\r\n\r\n"
    )
    messages = [{"role": "user", "content": "fixture"}]
    answer, receipt = rt.call(messages, CONFIG, opener=lambda *a, **k: SSE(events))
    assert json.loads(answer) == {"status": "完成"}
    assert (
        receipt["request_sha256"]
        == hashlib.sha256(rt.encode(rt.payload(messages, CONFIG, 32000))).hexdigest()
    )
    assert receipt["response_format"] == "sse"
    assert receipt["stream"]["events_received"] == 3
    assert receipt["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert KEY not in capsys.readouterr().out + json.dumps(receipt)


@pytest.mark.parametrize(
    "ending", [b"", b"data: [DONE]\n\n", b'data: {"type":"response.completed"}']
)
def test_stream_disconnect_never_accepts_partial_output(monkeypatch, ending):
    from hplc_al.llm.execution import retryable

    monkeypatch.setenv(CONFIG["env_key"], KEY)
    body = (
        sse_event({"type": "response.output_text.delta", "delta": '{"status":"ok"}'})
        + ending
    )
    with pytest.raises(rt.TransportError, match="STREAM_INTERRUPTED") as exc:
        rt.call(
            [{"role": "user", "content": "fixture"}],
            CONFIG,
            opener=lambda *a, **k: SSE(body),
        )
    assert retryable(exc.value)


@pytest.mark.parametrize("kind", ["error", "response.failed", "response.incomplete"])
def test_stream_terminal_failure_is_safe_and_not_resampled(monkeypatch, kind, capsys):
    from hplc_al.llm.execution import retryable

    monkeypatch.setenv(CONFIG["env_key"], KEY)
    body = sse_event({"type": kind, "error": {"message": KEY}})
    with pytest.raises(rt.TransportError, match="STREAM_FAILED") as exc:
        rt.call(
            [{"role": "user", "content": "fixture"}],
            CONFIG,
            opener=lambda *a, **k: SSE(body),
        )
    assert not retryable(exc.value)
    assert KEY not in str(exc.value) + capsys.readouterr().out


@pytest.mark.parametrize("fault", ["incomplete", "tool", "wrong_model", "secret_echo"])
def test_completed_stream_uses_same_scientific_checks(monkeypatch, fault):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    value = response()
    if fault == "incomplete":
        value["status"] = "incomplete"
    elif fault == "tool":
        value["output"].append({"type": "function_call"})
    elif fault == "wrong_model":
        value["model"] = "different"
    else:
        value["id"] = KEY
    body = sse_event({"type": "response.completed", "response": value})
    with pytest.raises(rt.TransportError) as exc:
        rt.call(
            [{"role": "user", "content": "fixture"}],
            CONFIG,
            opener=lambda *a, **k: SSE(body),
        )
    assert KEY not in str(exc.value)


def test_stream_schema_and_read_failure_distinguished(monkeypatch):
    monkeypatch.setenv(CONFIG["env_key"], KEY)
    with pytest.raises(rt.TransportError, match="SCHEMA_FAILURE"):
        rt.call(
            [{"role": "user", "content": "fixture"}],
            CONFIG,
            opener=lambda *a, **k: SSE(b"data: not-json\n\n"),
        )

    class Broken(SSE):
        def readline(self, size):
            raise TimeoutError

    with pytest.raises(rt.TransportError, match="NETWORK_FAILURE"):
        rt.call(
            [{"role": "user", "content": "fixture"}],
            CONFIG,
            opener=lambda *a, **k: Broken(),
        )
