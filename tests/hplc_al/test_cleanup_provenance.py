"""Read-only historical artifact contracts after working-tree relocation."""

import hashlib
import subprocess

from hplc_al.common import ROOT, read_json, sha

TAG = "archive/pre-cleanup-free-llm32-2026-09-29"
V1 = ROOT / "studies/active_learning/odh_free_llm32_scientist_v1"


def test_relocated_success_and_legacy_bytes_unchanged():
    for path in [
        V1 / "provenance/relocation_manifest.json",
        ROOT / "studies/archive/llm_hybrid/relocation_manifest.json",
    ]:
        mapping = read_json(path)
        for old, item in mapping.items():
            destination = ROOT / item["path"]
            if item["tracked"]:
                original = subprocess.check_output(
                    ["git", "show", f"{TAG}:{old}"], cwd=ROOT
                )
                assert (
                    hashlib.sha256(original).hexdigest()
                    == item["sha256"]
                    == sha(destination)
                )
            elif destination.exists():
                assert sha(destination) == item["sha256"]


def test_frozen_v1_results():
    metrics = read_json(V1 / "results/validation_metrics.json")
    rows = [r for r in metrics if r["method"] == "free_llm32_scientist"]
    assert [r["budget"] for r in rows] == [333, 365, 397]
    assert [round(r["rmse"], 6) for r in rows] == [8.043997, 8.052217, 8.022899]
    aulc = read_json(V1 / "results/partial_aulc.json")
    assert [round(r["mean_nrmse"], 6) for r in aulc] == [0.905802, 0.906640, 0.883564]
    audit = read_json(V1 / "provenance/successful_execution/final_audit.json")
    assert audit["actual_llm_calls"] == 10 and audit["test_truth_access_count"] == 0
    assert not (V1 / "EXECUTION_STATUS.md").exists()


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
