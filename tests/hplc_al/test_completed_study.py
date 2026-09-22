"""Refactoring must preserve historical evidence and keep execution gates strict."""

import hashlib
from zipfile import ZipFile

import pytest

from hplc_al import common, runner
from hplc_al.common import atomic_json, verify_frozen_source


def make_snapshot(directory, content=b"original source\n"):
    name = "src/example.py"
    with ZipFile(directory / "frozen_source.zip", "w") as archive:
        archive.writestr(name, content)
    return {name: hashlib.sha256(content).hexdigest()}


def test_snapshot_verifies_original_code_after_refactor(tmp_path, monkeypatch):
    expected = make_snapshot(tmp_path)
    monkeypatch.setattr(common, "code_hashes", lambda: {"src/example.py": "new version"})
    assert verify_frozen_source(tmp_path, expected) == "frozen_source.zip"


def test_snapshot_tampering_rejected(tmp_path, monkeypatch):
    expected = make_snapshot(tmp_path)
    monkeypatch.setattr(common, "code_hashes", lambda: {})
    make_snapshot(tmp_path, b"altered historical source\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_frozen_source(tmp_path, expected)


def test_missing_snapshot_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "code_hashes", lambda: {})
    with pytest.raises(RuntimeError, match="missing"):
        verify_frozen_source(tmp_path, {"source.py": "old hash"})


def test_snapshot_cannot_override_live_training_gate(tmp_path, monkeypatch):
    expected = make_snapshot(tmp_path)
    atomic_json(tmp_path / "protocol.json", {"code": expected})
    monkeypatch.setattr(runner, "code_hashes", lambda: {})
    with pytest.raises(RuntimeError, match="frozen code/environment drift"):
        runner.assert_frozen_protocol(tmp_path)


@pytest.mark.parametrize("action", ["prepare", "tests_gate", "duration_audit", "freeze_protocol"])
def test_completed_study_rejects_mutating_stages(tmp_path, action):
    marker = tmp_path / "completion_manifest.json"
    atomic_json(marker, {"status": "complete"})
    before = marker.read_bytes()
    with pytest.raises(RuntimeError, match="study is complete"):
        getattr(runner, action)(tmp_path)
    assert marker.read_bytes() == before
    assert list(tmp_path.iterdir()) == [marker]


@pytest.mark.parametrize("action", ["run", "report"])
def test_completed_entrypoints_only_verify(tmp_path, monkeypatch, action):
    from hplc_al import reporting

    atomic_json(tmp_path / "completion_manifest.json", {"status": "complete"})
    verified = {"status": "VERIFIED"}
    module = runner if action == "run" else reporting
    monkeypatch.setattr(module, "verify_completed_study", lambda study: verified)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed study must not enter execution or label access")

    monkeypatch.setattr(module, "assert_frozen_protocol", forbidden)
    result = runner.run_trajectories(tmp_path) if action == "run" else reporting.report(tmp_path)
    assert result == verified
