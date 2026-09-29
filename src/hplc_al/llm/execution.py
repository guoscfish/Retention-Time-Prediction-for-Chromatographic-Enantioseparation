"""User-authorized operational retries and terminal progress; no scientific edits."""

import csv
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from ..common import read_json, write_once
from .responses_transport import TransportError

MAX_ATTEMPTS = 5
RETRY_DELAYS = (15, 30, 60, 60)
MAX_RETRY_DELAY = 15 * 60
HEARTBEAT_SECONDS = 15


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{utc()}] {message}", flush=True)


def retryable(error):
    return str(error) in {
        "RESPONSES_HTTP_408",
        "RESPONSES_HTTP_429",
        "RESPONSES_HTTP_500",
        "RESPONSES_HTTP_502",
        "RESPONSES_HTTP_503",
        "RESPONSES_HTTP_504",
        "RESPONSES_HTTP_520",
        "RESPONSES_HTTP_522",
        "RESPONSES_HTTP_524",
        "RESPONSES_NETWORK_FAILURE",
    }


def safe_error(error):
    # Never persist arbitrary exception text, provider bodies or local secrets.
    value = str(error)
    if isinstance(error, TransportError) and re.fullmatch(
        r"RESPONSES_[A-Z0-9_]+", value
    ):
        return value
    return "LOCAL_OR_VALIDATION_FAILURE"


@contextmanager
def heartbeat(label, detail=None):
    started = time.monotonic()
    stop = threading.Event()

    def report():
        while not stop.wait(HEARTBEAT_SECONDS):
            extra = ""
            if detail is not None:
                try:
                    extra = detail()
                except (OSError, ValueError, KeyError, csv.Error):
                    extra = "progress file not ready"
            log(
                f"{label}: still running, elapsed {time.monotonic() - started:.0f}s {extra}"
            )

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def training_detail(directory):
    curves = sorted(directory.glob("fit/attempt_*/training_curve.csv"))
    if not curves:
        return "preparing graphs/model"
    with curves[-1].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # A writer may currently be halfway through the last row.
    rows = [r for r in rows if r.get("observed_batch_sizes") is not None]
    if not rows:
        return "first epoch in progress"
    last = rows[-1]
    return f"epoch={last['epoch']} train_loss={last['train_loss']} validation_mse={last['validation_mse']}"


def retry_delay(attempt):
    """Return a bounded delay before a transient retry."""
    if attempt <= len(RETRY_DELAYS):
        return RETRY_DELAYS[attempt - 1]
    return min(
        MAX_RETRY_DELAY,
        RETRY_DELAYS[-1] * 2 ** (attempt - len(RETRY_DELAYS)),
    )


def request_with_retries(
    transport,
    messages,
    config,
    budget,
    directory,
    name,
    digest,
    *,
    legacy=False,
    retry_forever=False,
):
    """Retry transient failures and reuse the first valid receipt.

    A started attempt without an outcome is ambiguous, including Ctrl-C or a
    process crash. The explicitly authorized policy permits identical replay,
    recording possible duplicate provider charges. The default remains the
    bounded five-attempt behavior used by callers and tests. The continuous
    runner opts into persistent retries so a temporary gateway outage cannot
    terminate a multi-hour study; Ctrl-C and non-retryable failures still stop.
    """

    def path(number, kind):
        return directory / f"{name}.attempt_{number:02d}.{kind}.json"

    if legacy and not path(1, "started").exists():
        write_once(
            path(1, "started"),
            {
                "request_sha256": digest,
                "attempt": 1,
                "origin": "pre-amendment request without receipt",
                "timestamp": utc(),
            },
        )
    count = 0
    while path(count + 1, "started").exists():
        count += 1
        started = read_json(path(count, "started"))
        if started["request_sha256"] != digest or started["attempt"] != count:
            raise RuntimeError("retry request binding mismatch")
        failed = path(count, "failed")
        if failed.exists():
            outcome = read_json(failed)
            if outcome["request_sha256"] != digest or not outcome["retryable"]:
                raise RuntimeError("previous terminal failure; no resampling permitted")
    if count >= MAX_ATTEMPTS and not retry_forever:
        raise TransportError("RESPONSES_RETRY_LIMIT_EXHAUSTED")
    attempt = count + 1
    while True:
        if attempt > 1:
            delay = retry_delay(attempt - 1)
            log(
                f"{name}: retry {attempt}"
                + (f"/{MAX_ATTEMPTS}" if not retry_forever else "")
                + f" in {delay}s; identical request, possible duplicate charge"
            )
            time.sleep(delay)
        write_once(
            path(attempt, "started"),
            {
                "request_sha256": digest,
                "attempt": attempt,
                "timestamp": utc(),
                "origin": "authorized identical-request retry"
                if attempt > 1
                else "initial request",
            },
        )
        started = time.monotonic()
        log(
            f"{name}: sending attempt {attempt}"
            + (f"/{MAX_ATTEMPTS}" if not retry_forever else "")
        )
        try:
            with heartbeat(
                f"{name} attempt {attempt}"
                + (f"/{MAX_ATTEMPTS}" if not retry_forever else "")
            ):
                result = transport(messages, config, budget)
        except Exception as error:
            transient = isinstance(error, TransportError) and retryable(error)
            code = safe_error(error)
            write_once(
                path(attempt, "failed"),
                {
                    "request_sha256": digest,
                    "attempt": attempt,
                    "timestamp": utc(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": code,
                    "retryable": transient,
                },
            )
            log(
                f"{name}: attempt {attempt}"
                + (f"/{MAX_ATTEMPTS}" if not retry_forever else "")
                + f" failed: {code}"
            )
            if not transient or (not retry_forever and attempt == MAX_ATTEMPTS):
                raise
        else:
            log(f"{name}: response received in {time.monotonic() - started:.1f}s")
            return result
        attempt += 1


def preflight_with_retries(operation):
    """Content-free attempts have a fresh, bounded budget per invocation."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"Content-free preflight: attempt {attempt}/{MAX_ATTEMPTS}")
        try:
            with heartbeat("Content-free preflight"):
                return operation()
        except TransportError as error:
            log(f"Preflight failed: {safe_error(error)}")
            if not retryable(error) or attempt == MAX_ATTEMPTS:
                raise
            delay = retry_delay(attempt)
            log(f"Preflight retry in {delay}s")
            time.sleep(delay)


def attempt_summary(directory):
    """Keep failed/unknown dispatch cost visible without inventing provider usage."""
    intents = sorted(directory.glob("round_*/llm/*.request.json"))
    total = receipts = unknown = 0
    for intent in intents:
        name = intent.name.removesuffix(".request.json")
        attempts = list(intent.parent.glob(f"{name}.attempt_*.started.json"))
        count = max(1, len(attempts))
        completed = (intent.parent / f"{name}.receipt.json").exists()
        total += count
        receipts += int(completed)
        unknown += count - int(completed)
    return {
        "scientific_dispatch_intents": total,
        "validated_response_receipts": receipts,
        "attempts_without_usage_receipts": unknown,
        "billing_for_attempts_without_receipts": "unknown; may include duplicate charges",
        "note": "Write-ahead intents can include a crash before network dispatch. Receipt usage excludes unreturned attempts.",
    }
