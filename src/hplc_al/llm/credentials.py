"""Launcher-only key injection from the user's explicitly designated local file."""

import getpass
import os
import sys
from pathlib import Path

from .responses_transport import TransportError

ENV_KEY = "TOKEN4RESEARCH_API_KEY"
KEY_FILE = Path.home() / ".config/qgeognn-scientist/token4research.api-key"


def inject_key(*, interactive=False, path=None):
    """No key return value, logging, copying, hashing or credential-file writes.

    Existing environment wins unless an explicit file was requested. The HTTP
    transport continues to read only os.environ[env_key].
    """
    if path is None and os.environ.get(ENV_KEY):
        return True
    source = Path(path).expanduser() if path is not None else KEY_FILE
    try:
        key = source.read_text().strip()
    except FileNotFoundError:
        if path is not None:
            raise TransportError("RESPONSES_KEY_FILE_NOT_FOUND") from None
        if not interactive or not sys.stdin.isatty():
            return False
        key = getpass.getpass(
            "token4research API key (hidden, this process only): "
        ).strip()
    except (OSError, UnicodeError):
        raise TransportError("RESPONSES_KEY_FILE_UNREADABLE") from None
    if (
        not key
        or not key.isascii()
        or any(c.isspace() or ord(c) < 33 or ord(c) == 127 for c in key)
    ):
        raise TransportError(
            "RESPONSES_KEY_FORMAT_INVALID: expected one nonempty key; value suppressed"
        )
    os.environ[ENV_KEY] = key
    return True
