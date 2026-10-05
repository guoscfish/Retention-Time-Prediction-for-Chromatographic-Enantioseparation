"""Read-only historical artifact contracts after working-tree relocation."""

import hashlib
import subprocess

from hplc_al.common import ROOT, read_json, sha

TAG = "archive/pre-cleanup-free-llm32-2026-09-29"
V1 = ROOT / "studies/active_learning/odh_free_llm32_scientist_v1"


def test_retained_v3_dependencies_match_original_hashes():
    mapping = read_json(V1 / "provenance/relocation_manifest.json")
    inventory = read_json(ROOT / "docs/repository/cleanup_20261005.json")
    retained = set(inventory["retained_frozen_v3_dependency_paths"])
    for old, item in mapping.items():
        if item["path"] not in retained:
            continue
        destination = ROOT / item["path"]
        if item["tracked"]:
            original = subprocess.check_output(["git", "show", f"{TAG}:{old}"], cwd=ROOT)
            assert hashlib.sha256(original).hexdigest() == item["sha256"] == sha(destination)
        elif destination.exists():
            assert sha(destination) == item["sha256"]


def test_retired_experiments_are_absent():
    assert not (ROOT / "studies/archive/llm_hybrid").exists()
    assert not (V1 / "provenance/successful_execution/final_audit.json").exists()
    assert not (V1 / "results/partial_aulc.json").exists()
    assert not (ROOT / "studies/active_learning/odh_free_llm32_fullpool_v2/runtime/seed_1525/free_llm32_fullpool/round_0/llm").exists()


def test_no_legacy_scientific_runtime():
    retired = [
        "llm_cli_transport",
        "llm_transport",
        "llm_hybrid",
        "free_llm_runner",
        "free_llm_scientist",
        "free_llm_report",
        "llm_catalog",
    ]
    for name in retired:
        assert not (ROOT / "src/hplc_al" / (name + ".py")).exists()
    for path in (ROOT / "src/hplc_al/llm").glob("*.py"):
        text = path.read_text()
        assert "auth.json" not in text and "OPENAI_API_KEY" not in text
        assert "codex exec" not in text
