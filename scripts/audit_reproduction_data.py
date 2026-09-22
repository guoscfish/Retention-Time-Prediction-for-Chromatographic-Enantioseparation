"""Audit source rows and chemistry; no training or active-learning decisions."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(smiles, stereo=True):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=stereo) if mol else None


def metrics(true, pred):
    error = np.asarray(pred) - np.asarray(true)
    return {
        "n": len(error), "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error**2) / np.sum((true - np.mean(true))**2)),
        "median_relative_error": float(np.median(np.abs(error) / true)),
        "mean_relative_error": float(np.mean(np.abs(error) / true)),
        "relative_l2_error": float(np.sqrt(np.sum(error**2) / np.sum(true**2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path)
    args = parser.parse_args()
    out = ROOT / "artifacts/reproduction/audit"
    out.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.warning")
    records, tables = {}, []
    for path in sorted((ROOT / "dataset").glob("*.csv")):
        df = pd.read_csv(path)
        smiles = df.SMILES.dropna().unique()
        canonical = {s: identity(s) for s in smiles}
        nonstereo = {s: identity(s, False) for s in smiles}
        col = df.get("Column", pd.Series("ADH", index=df.index))
        ids = df.SMILES.map(canonical)
        chemistry_columns = [x for x in df.columns if not x.startswith("Unnamed") and x != "index"]
        numeric = df[["RT", "Speed", "i-PrOH_proportion"]].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric).all(axis=1) & (numeric.RT > 0) & (numeric.Speed > 0)
        valid &= numeric["i-PrOH_proportion"].between(0, 1) & ids.notna()
        rtv = (numeric.RT * numeric.Speed).astype(np.float32)
        bad_file = ROOT / "dataset" / ("bad_all_column.npy" if path.stem.startswith("All") else f"bad_{col.iloc[0]}.npy")
        bad = np.load(bad_file).astype(int).tolist() if bad_file.exists() else ([4231] if col.iloc[0] == "ODH" else [])
        bad_mask = df.index.isin(bad)
        record = {
            "sha256": sha256(path), "bytes": path.stat().st_size,
            "rows": len(df), "columns": df.columns.tolist(),
            "unique_raw_smiles": len(smiles), "unique_canonical_isomeric": ids.nunique(),
            "unique_canonical_nonisomeric": df.SMILES.map(nonstereo).nunique(),
            "invalid_smiles_rows": df.index[ids.isna()].tolist(),
            "numeric_invalid_rows": df.index[~valid].tolist(),
            "duplicate_full_rows": int(df.duplicated().sum()),
            "duplicate_content_rows_ignoring_export_ids": int(df.duplicated(chemistry_columns).sum()),
            "repeated_canonical_rows": int(ids.duplicated().sum()),
            "unique_literature_values": int(df.Literature.nunique()),
            "literature_missing": int(df.Literature.isna().sum()),
            "known_conformer_exclusion_rows": bad,
            "rtv_gt_60_count": int((rtv > 60).sum()),
            "eligible_known_rules_count": int((valid & ~bad_mask & (rtv <= 60)).sum()),
            "ranges": {c: [float(numeric[c].min()), float(numeric[c].max())] for c in numeric},
        }
        records[path.name] = record
        for column in sorted(col.unique()):
            mask = col == column
            tables.append({"file": path.name, "column": column, "raw_rows": int(mask.sum()),
                           "unique_smiles": int(df.loc[mask, "SMILES"].nunique()),
                           "known_bad_rows": int((mask & bad_mask).sum()),
                           "rtv_gt_60": int((mask & (rtv > 60)).sum()),
                           "eligible_known_rules": int((mask & valid & ~bad_mask & (rtv <= 60)).sum())})
        if col.iloc[0] == "ODH" and not path.stem.startswith("All"):
            eligible = df.index[valid & ~bad_mask & (rtv <= 60)].to_numpy()
            order = np.random.RandomState(388).permutation(len(eligible))
            n = len(order)
            splits = {"train": eligible[order[:int(n*.9)]],
                      "valid": eligible[order[int(n*.9):int(n*.9)+int(n*.05)]],
                      "test": eligible[order[n-int(n*.05):]]}
            for name in ("valid", "test"):
                record[f"{name}_canonical_overlap_train_rows"] = int(ids.loc[splits[name]].isin(ids.loc[splits["train"]]).sum())
                nc = df.SMILES.map(nonstereo)
                record[f"{name}_nonisomeric_overlap_train_rows"] = int(nc.loc[splits[name]].isin(nc.loc[splits["train"]]).sum())
    if args.source_zip:
        with zipfile.ZipFile(args.source_zip) as archive:
            source = pd.read_csv(archive.open("Source data/source_data_ODH.csv"))
            source.to_csv(out / "paper_source_data_ODH.csv", index=False)
            records["paper_source_data"] = {
                "zip_sha256": sha256(args.source_zip),
                "odh_csv_byte_identical": archive.read("Source data/ODH_charity.csv") == (ROOT / "dataset/ODH_charity_0616.csv").read_bytes(),
                "computed_metrics": metrics(source.true.to_numpy(), source.pred.to_numpy()),
                "figure_3c_reported": {"mae": 2.74, "median_relative_error": .158, "r2": .778},
            }
    (out / "data_audit.json").write_text(json.dumps(records, indent=2) + "\n")
    pd.DataFrame(tables).to_csv(out / "column_counts.csv", index=False)
    for filename, record in records.items():
        print(filename, {k: v for k, v in record.items() if k in ["rows", "unique_raw_smiles", "unique_canonical_isomeric", "eligible_known_rules_count", "computed_metrics", "test_canonical_overlap_train_rows"]})


if __name__ == "__main__":
    main()
