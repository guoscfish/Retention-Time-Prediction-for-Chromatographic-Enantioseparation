"""Text-only Responses transport using the explicitly frozen local provider."""

import json
import os
from pathlib import Path
import tomllib
import urllib.error
import urllib.request


def settings():
    config = tomllib.loads((Path.home() / ".codex/config.toml").read_text())
    provider = config.get("model_providers", {}).get(config.get("model_provider", "openai"), {})
    return {
        "model": "gpt-6-sol", "reasoning_effort": "high",
        "base_url": provider.get("base_url", "https://api.openai.com/v1").rstrip("/"),
        "wire_api": "responses", "store": False, "native_tools": [],
        "max_output_tokens": 16000, "automatic_fallback": False,
    }


def call(messages, config):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        key = json.loads((Path.home() / ".codex/auth.json").read_text()).get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No API key configured for the frozen Responses provider")
    if config != settings():
        raise RuntimeError("Provider or model settings differ from the frozen transport")
    payload = {"model": config["model"], "reasoning": {"effort": config["reasoning_effort"]},
               "input": messages, "tools": [], "store": False,
               "max_output_tokens": config["max_output_tokens"]}
    request = urllib.request.Request(config["base_url"] + "/responses",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "User-Agent": "hplc-al-study/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Responses HTTP {error.code}; no selection accepted") from None
    if result.get("status") != "completed":
        raise RuntimeError(f"Responses incomplete: {result.get('status')}")
    model = result.get("model", "")
    if model != config["model"] and not model.startswith(config["model"] + "-"):
        raise RuntimeError("Served model differs from the frozen model")
    if any(item["type"] not in ("message", "reasoning") for item in result["output"]):
        raise RuntimeError("Native tool call is forbidden")
    answer = "".join(part["text"] for item in result["output"] if item["type"] == "message"
                     for part in item["content"] if part["type"] == "output_text")
    return answer, {"response_id": result["id"], "served_model": model,
                    "usage": result.get("usage"), "native_tool_calls": 0}
