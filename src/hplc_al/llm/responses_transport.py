"""Direct, stateless Responses-compatible HTTP. No credential or provider fallback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import tomllib

from ..common import atomic_json, stable_hash

MAX_OUTPUT_TOKENS = 32000
USER_AGENT = "QGeoGNN-Scientist/2.0"


class TransportError(RuntimeError):
    """Safe diagnostics: never include request headers or remote error bodies."""


def settings(path=None):
    try:
        config = tomllib.loads(
            Path(path or Path.home() / ".codex/config.toml").read_text()
        )
        provider_id = config["model_provider"]
        provider = config["model_providers"][provider_id]
        result = {
            "provider_id": provider_id,
            "model": config["model"],
            "reasoning_effort": config["model_reasoning_effort"],
            "base_url": provider["base_url"].rstrip("/"),
            "wire_api": provider["wire_api"],
            "env_key": provider["env_key"],
        }
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        raise TransportError(
            "RESPONSES_CONFIG_INVALID: explicit provider/model/effort/base_url/wire_api/env_key required"
        ) from None
    validate_settings(result)
    if provider.get("requires_openai_auth", False):
        raise TransportError(
            "RESPONSES_CONFIG_INVALID: environment-key authentication required"
        )
    return result


def validate_settings(config):
    if set(config) != {
        "provider_id",
        "model",
        "reasoning_effort",
        "base_url",
        "wire_api",
        "env_key",
    }:
        raise TransportError("RESPONSES_CONFIG_INVALID: unknown/missing setting")
    if any(not isinstance(v, str) or not v for v in config.values()):
        raise TransportError("RESPONSES_CONFIG_INVALID: empty setting")
    url = urlsplit(config["base_url"])
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or config["wire_api"] != "responses"
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config["env_key"])
    ):
        raise TransportError(
            "RESPONSES_CONFIG_INVALID: HTTPS Responses endpoint and env_key required"
        )


def require_key(config):
    validate_settings(config)
    key = os.environ.get(config["env_key"])
    if not key:
        raise TransportError(f"RESPONSES_API_KEY_NOT_CONFIGURED: {config['env_key']}")
    return key


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_answer(answer):
    """Exactly one JSON object, no fences, trailing objects, duplicate keys or NaN."""

    def invalid(_):
        raise ValueError("nonfinite JSON")

    try:
        result = json.loads(answer, object_pairs_hook=_pairs, parse_constant=invalid)
        if not isinstance(result, dict):
            raise ValueError("JSON object required")
        return result
    except (ValueError, TypeError):
        raise TransportError("RESPONSES_INVALID_JSON_ANSWER") from None


def payload(messages, config, max_output_tokens):
    validate_settings(config)
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("positive output token budget required")
    if not messages or any(
        set(m) != {"role", "content"}
        or m["role"] not in ("system", "user", "assistant")
        or not isinstance(m["content"], str)
        for m in messages
    ):
        raise ValueError("text-only host-controlled messages required")
    return {
        "model": config["model"],
        "reasoning": {"effort": config["reasoning_effort"]},
        "input": messages,
        "tools": [],
        "store": False,
        "max_output_tokens": max_output_tokens,
    }


def encode(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransportError("RESPONSES_REDIRECT_FORBIDDEN")


def call(messages, config, max_output_tokens=MAX_OUTPUT_TOKENS, *, opener=None):
    key = require_key(config)
    body = encode(payload(messages, config, max_output_tokens))
    request_hash = hashlib.sha256(body).hexdigest()
    request = urllib.request.Request(
        config["base_url"] + "/responses",
        data=body,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        open_request = opener or urllib.request.build_opener(_NoRedirect()).open
        with open_request(request, timeout=600) as response:
            if not 200 <= response.status < 300:
                raise TransportError("RESPONSES_HTTP_FAILURE")
            result = json.loads(response.read())
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
    except (urllib.error.URLError, OSError, ValueError):
        raise TransportError("RESPONSES_NETWORK_OR_SCHEMA_FAILURE") from None
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
    messages_out = [i for i in outputs if i["type"] == "message"]
    if len(messages_out) != 1 or messages_out[0].get("role") != "assistant":
        raise TransportError("RESPONSES_UNIQUE_MESSAGE_REQUIRED")
    parts = messages_out[0].get("content", [])
    if (
        len(parts) != 1
        or parts[0].get("type") != "output_text"
        or not isinstance(parts[0].get("text"), str)
    ):
        raise TransportError("RESPONSES_UNIQUE_TEXT_ANSWER_REQUIRED")
    answer = parts[0]["text"]
    # Discard any accidental credential echo BEFORE parsing, persistence, or hashing.
    if key in json.dumps(result):
        raise TransportError("RESPONSES_SENSITIVE_ECHO_REJECTED")
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
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "provider_id": config["provider_id"],
        "base_url_hostname": urlsplit(config["base_url"]).hostname,
        "client_user_agent": USER_AGENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "native_tool_calls": 0,
    }


def preflight(path, config=None, *, transport=call):
    """Content-free check only. Failure never changes provider/model or retries."""
    config = settings() if config is None else config
    require_key(config)
    messages = [{"role": "user", "content": 'Return exactly {"status":"ok"}'}]
    try:
        answer, receipt = transport(messages, config, 2048)
        if parse_answer(answer) != {"status": "ok"}:
            raise TransportError("RESPONSES_PREFLIGHT_ANSWER_MISMATCH")
    except TransportError as error:
        atomic_json(
            path,
            {
                "status": "FAILED",
                "error": str(error),
                "config": config,
                "scientific_content": False,
            },
        )
        raise
    record = {
        "status": "PASS",
        "config": config,
        "config_sha256": stable_hash(config),
        "scientific_content": False,
        "tools": [],
        "store": False,
        "receipt": receipt,
        "limitations": "Request accepted; cannot independently verify provider retention policy or context limit.",
    }
    atomic_json(path, record)
    return record
