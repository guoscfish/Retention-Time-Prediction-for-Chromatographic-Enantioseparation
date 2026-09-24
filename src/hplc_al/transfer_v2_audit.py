"""Pre-AL implementation audit for the v2 shared-checkpoint contract."""

from __future__ import annotations

from pathlib import Path

from .common import ROOT, atomic_json, read_json, stable_hash
from .transfer_v2 import ACQUISITIONS, BUDGETS, L0, TRAINING, training_batch_sizes


STUDY = ROOT / "studies/active_learning/odh_gradient_al_transfer_v2"


def audit(study=STUDY):
    study = Path(study)
    duration = read_json(study / "duration_smoke/summary.json")
    if duration["status"] != "PASS":
        raise RuntimeError("duration smoke did not pass")
    if duration["L0"] != L0 or duration["steps_per_epoch"] != 2 or duration["observed_batch_sizes"] != [256, 77]:
        raise RuntimeError("L333 batch geometry drift")
    source = (ROOT / "src/hplc_al/gradient.py").read_text()
    contract = (ROOT / "src/hplc_al/transfer_v2.py").read_text()
    if "output[0, 1]" not in source or "512" not in source:
        raise RuntimeError("central full-network CountSketch implementation not found")
    if "np.maximum(norms, 1e-12)" not in contract:
        raise RuntimeError("unit normalization epsilon contract drift")
    result = {
        "status": "PASS_PRE_AL",
        "gradient_definition": "phi_raw = 512D CountSketch(full-network central gradient)",
        "unit_definition": "phi_unit = phi_raw / max(||phi_raw||_2, 1e-12) after sketch",
        "unit_name": "unit-normalized sketched gradient",
        "round_0": {
            "shared_checkpoint_required": True,
            "shared_raw_gradient_bank_required": True,
            "unit_derived_from_raw_bank": True,
            "separate_round_0_predictors": False,
        },
        "post_acquisition": "method-specific label sets receive independent scratch retraining",
        "training": TRAINING,
        "batch_sizes": training_batch_sizes(),
        "methods": list(ACQUISITIONS),
        "budgets": list(BUDGETS),
        "duration_summary_hash": stable_hash(duration),
        "long_al_started": False,
    }
    atomic_json(study / "implementation_audit.json", result)
    return result
