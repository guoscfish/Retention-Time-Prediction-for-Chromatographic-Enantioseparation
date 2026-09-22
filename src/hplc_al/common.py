"""Small, explicit artifact contracts shared by the AL stages."""

import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/active_learning/odh_gradient_al_smoke"
BASELINE = ROOT / "artifacts/reproduction/odh_baseline_20260922"
SOURCE = ROOT / "dataset/ODH_charity_0616.csv"
CACHE = ROOT / "artifacts/reproduction/odh_cache"
METHODS = ("random", "lcmd", "maxdet")
BUDGETS = (356, 388, 420, 452)
SEED = 73
INIT_SEED = 525
SKETCH_SEED = 4_000_037 + SEED
RANDOM_SEED = SEED * 1_000_003 + 900_001


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def ids_hash(ids):
    return stable_hash(sorted(int(i) for i in ids))


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_once(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise RuntimeError(f"immutable artifact differs: {path}")
    else:
        atomic_json(path, value)


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return stable_hash([str(a.dtype), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()])


def code_hashes():
    paths = sorted((ROOT / "src/hplc_al").glob("*.py"))
    paths += [
        ROOT / "src/reproduction/official.py",
        ROOT / "code/Single_column_prediction.py",
        ROOT / "code/compound_tools.py",
        ROOT / "scripts/run_hplc_al_smoke.py",
    ]
    paths += sorted((ROOT / "tests/hplc_al").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_files(root, files):
    for name, expected in files.items():
        path = Path(root) / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"artifact hash mismatch: {path}")


def verify_frozen_source(study, expected_hashes):
    """Verify historical code without claiming that refactored code ran the study.

    This fallback is only for read-only verification of completed experiments.
    Training must still match the working tree through assert_frozen_protocol.
    Snapshot members are read as bytes, never extracted or executed.
    """
    if code_hashes() == expected_hashes:
        return "working_tree"

    archive = Path(study) / "frozen_source.zip"
    if not archive.is_file():
        raise RuntimeError("historical source differs and frozen_source.zip is missing")
    with ZipFile(archive) as snapshot:
        names = snapshot.namelist()
        if len(names) != len(expected_hashes) or set(names) != set(expected_hashes):
            raise RuntimeError("frozen source snapshot membership mismatch")
        for name, expected in expected_hashes.items():
            if hashlib.sha256(snapshot.read(name)).hexdigest() != expected:
                raise RuntimeError(f"frozen source snapshot hash mismatch: {name}")
    return archive.name


def metrics(pred, truth, scale):
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("invalid metric inputs")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid frozen L0 scale")
    error = pred - truth
    mse = np.mean(error**2)
    total = np.sum((truth - truth.mean()) ** 2)
    if total <= 0:
        raise ValueError("R2 undefined for constant targets")
    return dict(
        rmse=float(np.sqrt(mse)),
        mae=float(np.abs(error).mean()),
        r2=float(1 - np.sum(error**2) / total),
        nrmse=float(np.sqrt(mse) / scale),
    )


def partial_aulc(budgets, errors):
    if list(budgets) != list(BUDGETS) or len(errors) != 4 or not np.isfinite(errors).all():
        raise ValueError("partial AULC requires all four finite registered points")
    values = np.asarray(errors, dtype=np.float64)
    raw = float(np.sum(np.diff(budgets) * (values[1:] + values[:-1]) / 2))
    return {"raw": raw, "mean": raw / 96.0}
