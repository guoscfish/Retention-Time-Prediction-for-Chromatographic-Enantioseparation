import os

import pytest

from hplc_al.llm import credentials
from hplc_al.llm.responses_transport import TransportError


@pytest.fixture
def key_file(monkeypatch, tmp_path):
    monkeypatch.delenv(credentials.ENV_KEY, raising=False)
    path = tmp_path / "token4research.api-key"
    monkeypatch.setattr(credentials, "KEY_FILE", path)
    return path


def test_file_injects_environment_without_output_or_modifying_file(key_file, capsys):
    content = "test-fixture-credential\n"
    key_file.write_text(content)
    assert credentials.inject_key()
    assert os.environ[credentials.ENV_KEY] == content.strip()
    assert key_file.read_text() == content
    assert capsys.readouterr() == ("", "")


def test_existing_environment_wins_unless_file_explicit(monkeypatch, key_file):
    monkeypatch.setenv(credentials.ENV_KEY, "existing-test-credential")
    key_file.write_text("file-test-credential")
    assert credentials.inject_key()
    assert os.environ[credentials.ENV_KEY] == "existing-test-credential"
    assert credentials.inject_key(path=key_file)
    assert os.environ[credentials.ENV_KEY] == "file-test-credential"


def test_absent_file_does_not_prompt_without_interactive_mode(monkeypatch, key_file):
    monkeypatch.setattr(
        credentials.getpass, "getpass", lambda *a: pytest.fail("must not prompt")
    )
    assert credentials.inject_key() is False
    with pytest.raises(TransportError, match="FILE_NOT_FOUND"):
        credentials.inject_key(path=key_file)


@pytest.mark.parametrize(
    "value", ["", "secret first\nsecond", "secret\x00value", "nonascii秘密"]
)
def test_bad_file_reports_no_value(key_file, value):
    key_file.write_text(value)
    with pytest.raises(TransportError, match="FORMAT_INVALID") as error:
        credentials.inject_key()
    assert "secret" not in str(error.value) and "秘密" not in str(error.value)
    assert credentials.ENV_KEY not in os.environ
