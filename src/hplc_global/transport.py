"""V3 Responses transport: phase-aware final text, unchanged request payload."""

import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from hplc_al.common import stable_hash, write_once
from hplc_al.llm.responses_transport import (
    MAX_OUTPUT_TOKENS,
    USER_AGENT,
    TransportError,
    _NoRedirect,
    _progress,
    _stream_result,
    encode,
    parse_answer,
    payload,
    require_key,
    wire_payload,
)


def extract_answer(messages):
    if not messages or any(m.get("role") != "assistant" for m in messages):
        raise TransportError("RESPONSES_ASSISTANT_MESSAGE_REQUIRED")
    for message in messages:
        parts = message.get("content")
        if (
            not isinstance(parts, list)
            or not parts
            or any(
                not isinstance(p, dict)
                or p.get("type") != "output_text"
                or not isinstance(p.get("text"), str)
                for p in parts
            )
        ):
            raise TransportError("RESPONSES_TEXT_ANSWER_REQUIRED")
    finals = [m for m in messages if m.get("phase") == "final_answer"]
    if finals:
        if (
            len(finals) != 1
            or messages[-1] is not finals[0]
            or any(m.get("phase") != "commentary" for m in messages[:-1])
        ):
            raise TransportError("RESPONSES_AMBIGUOUS_FINAL_ANSWER")
        selected, kind = finals, "final_answer_phase"
    elif all(m.get("phase") is None for m in messages):
        # SDK-compatible text fragmentation. Strict JSON parsing below rejects
        # commentary mixed with JSON, duplicate objects, fences and truncation.
        selected, kind = messages, "unphased_text_concatenation"
    else:
        raise TransportError("RESPONSES_FINAL_ANSWER_MISSING")
    answer = "".join(p["text"] for m in selected for p in m["content"])
    parse_answer(answer)
    return answer, kind


def call(
    messages,
    config,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    *,
    opener=None,
    diagnostic_directory=None,
):
    key = require_key(config)
    request_hash = hashlib.sha256(
        encode(payload(messages, config, max_output_tokens))
    ).hexdigest()
    body = encode(wire_payload(messages, config, max_output_tokens))
    request = urllib.request.Request(
        config["base_url"] + "/responses",
        data=body,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
            "User-Agent": USER_AGENT,
        },
    )
    started = time.monotonic()
    stats = {"events_received": 0, "bytes_received": 0, "first_byte_seconds": None}
    response_format = "json"
    try:
        open_request = opener or urllib.request.build_opener(_NoRedirect()).open
        with open_request(request, timeout=600) as response:
            if not 200 <= response.status < 300:
                raise TransportError("RESPONSES_HTTP_FAILURE")
            headers = getattr(response, "headers", {})
            content_type = (
                headers.get("Content-Type", "application/json")
                .split(";")[0]
                .strip()
                .lower()
            )
            _progress(f"HTTP response opened after {time.monotonic() - started:.1f}s")
            if content_type == "text/event-stream":
                response_format = "sse"
                result = _stream_result(response, stats, started)
            elif content_type == "application/json":
                # Some compatible providers return JSON despite stream=true.
                # Accept a complete response without sending a fallback request.
                result = json.loads(response.read())
            else:
                raise TransportError("RESPONSES_CONTENT_TYPE_INVALID")
    except urllib.error.HTTPError as error:
        # Classify a known gateway rejection, but never expose remote error text:
        # it can contain credentials or other untrusted fields.
        try:
            failure = error.read(8192).lower()
        except OSError:
            failure = b""
        if error.code == 403 and (
            b"error code: 1010" in failure
            or b'"error_code":1010' in failure.replace(b" ", b"")
        ):
            raise TransportError(
                "RESPONSES_HTTP_403_CLOUDFLARE_1010: client signature rejected; no automatic retry"
            ) from None
        raise TransportError(f"RESPONSES_HTTP_{error.code}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        raise TransportError("RESPONSES_NETWORK_FAILURE") from None
    except ValueError:
        raise TransportError("RESPONSES_SCHEMA_FAILURE") from None
    if (
        not isinstance(result, dict)
        or result.get("status") != "completed"
        or result.get("error")
    ):
        raise TransportError("RESPONSES_NOT_COMPLETED")
    outputs = result.get("output")
    if not isinstance(outputs, list) or any(
        not isinstance(i, dict) or i.get("type") not in ("message", "reasoning")
        for i in outputs
    ):
        raise TransportError("RESPONSES_NATIVE_TOOL_OR_INVALID_OUTPUT")
    # Redact/reject before preserving any provider-visible message text.
    if key in json.dumps(result):
        raise TransportError("RESPONSES_SENSITIVE_ECHO_REJECTED")
    messages_out = [i for i in outputs if i["type"] == "message"]
    if diagnostic_directory is not None:
        # Keep visible replies for offline repair, never reasoning items or headers.
        snapshot = {
            "request_sha256": request_hash,
            "response_id": result.get("id"),
            "status": result.get("status"),
            "model": result.get("model"),
            "messages": [
                {
                    k: message[k]
                    for k in ("type", "role", "phase", "content")
                    if k in message
                }
                for message in messages_out
            ],
        }
        write_once(
            Path(diagnostic_directory) / (stable_hash(snapshot) + ".json"), snapshot
        )
    answer, extraction = extract_answer(messages_out)
    parse_answer(answer)
    served = result.get("model")
    if served is not None and (
        not isinstance(served, str)
        or (served != config["model"] and not served.startswith(config["model"] + "-"))
    ):
        raise TransportError("RESPONSES_SERVED_MODEL_MISMATCH")
    if not isinstance(result.get("id"), str) or not result["id"]:
        raise TransportError("RESPONSES_MISSING_ID")
    usage = result.get("usage")
    if usage is not None and (
        not isinstance(usage, dict)
        or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("input_tokens", "output_tokens")
        )
    ):
        raise TransportError("RESPONSES_INVALID_USAGE")
    # Persist only usage counters, never arbitrary provider fields.
    usage = (
        None
        if usage is None
        else {k: usage[k] for k in ("input_tokens", "output_tokens")}
    )
    return answer, {
        "response_id": result["id"],
        "served_model": served,
        "usage": usage,
        "request_sha256": request_hash,
        "request_hash_scope": "scientific_payload_without_stream",
        "wire_request_sha256": hashlib.sha256(body).hexdigest(),
        "transport": "responses_sse",
        "response_format": response_format,
        "stream": stats,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "provider_id": config["provider_id"],
        "base_url_hostname": urlsplit(config["base_url"]).hostname,
        "client_user_agent": USER_AGENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "native_tool_calls": 0,
        "message_extraction": extraction,
        "output_message_count": len(messages_out),
    }
