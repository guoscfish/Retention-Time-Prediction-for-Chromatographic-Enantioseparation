"""Deterministic, label-free chemical and learned representation coverage."""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .common import SOURCE


def nested_random_order(unlabeled, seed):
    ids = sorted(int(i) for i in unlabeled)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate IDs")
    return np.random.default_rng(seed).permutation(ids).tolist()


def kcenter(distance, ids, labeled, unlabeled, batch=32):
    """Farthest-first from current L; exact ties use smallest sample ID."""
    ids = list(ids)
    d = np.asarray(distance, dtype=np.float64)
    if (len(set(ids)) != len(ids) or d.shape != (len(ids), len(ids))
            or not np.isfinite(d).all() or np.any(d < -1e-6)):
        raise ValueError("invalid distance bank")
    if (not labeled or len(set(labeled)) != len(labeled)
            or len(set(unlabeled)) != len(unlabeled) or set(labeled) & set(unlabeled)
            or not set(labeled + unlabeled) <= set(ids) or not 0 < batch <= len(unlabeled)):
        raise ValueError("invalid coreset state")
    index = {key: i for i, key in enumerate(ids)}
    pool = sorted(unlabeled)
    up = np.asarray([index[i] for i in pool])
    lp = [index[i] for i in labeled]
    nearest = d[np.ix_(up, lp)].min(axis=1).copy()
    selected, trace = [], []
    for _ in range(batch):
        pos = int(np.argmax(nearest))
        selected.append(pool[pos])
        trace.append({"id": pool[pos], "nearest_center_distance": float(nearest[pos])})
        nearest = np.minimum(nearest, d[up, up[pos]])
        nearest[[pool.index(i) for i in selected]] = -np.inf
    return selected, trace


def cosine_distance(unit):
    unit = np.asarray(unit, dtype=np.float64)
    if unit.ndim != 2 or not np.isfinite(unit).all():
        raise ValueError("finite feature matrix required")
    if not np.allclose(np.linalg.norm(unit, axis=1), 1, atol=1e-6):
        raise ValueError("cosine coverage requires nonzero unit rows")
    result = np.clip(1 - unit @ unit.T, 0, 2)
    np.fill_diagonal(result, 0)
    return result.astype(np.float32)


def morgan_bank(ids, source=SOURCE):
    """Read only molecular identities; preserve explicit stereochemistry."""
    frame = pd.read_csv(source, usecols=["SMILES"])
    generators = {chiral: rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=chiral) for chiral in (False, True)}
    fps, molecules, records, unique = [], [], [], {}
    for key in ids:
        mol = Chem.MolFromSmiles(str(frame.loc[key, "SMILES"]))
        if mol is None:
            raise ValueError(f"invalid SMILES at {key}")
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        fp = generators[True].GetFingerprint(mol)
        fps.append(fp)
        molecules.append(mol)
        unique[canonical] = mol
        records.append(dict(
            id=int(key), smiles=canonical, atom_count=mol.GetNumAtoms(),
            heavy_atom_count=mol.GetNumHeavyAtoms(), molecular_weight=Descriptors.MolWt(mol),
            scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
            specified_tetrahedral_centers=sum(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                                              for a in mol.GetAtoms()),
        ))
    bits = np.asarray([list(fp) for fp in fps], dtype=np.uint8)
    distance = np.empty((len(ids), len(ids)), dtype=np.float32)
    for i, fp in enumerate(fps):
        distance[i] = 1 - np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps))
    pairs, seen = [], set()
    for canonical, mol in unique.items():
        mirror = Chem.Mol(mol)
        for atom in mirror.GetAtoms():
            if atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                      Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                atom.InvertChirality()
        # Inverting a tag leaves cached CIP labels behind; refresh them before
        # both canonicalization and fingerprint generation.
        Chem.AssignStereochemistry(mirror, cleanIt=True, force=True)
        mirror_smiles = Chem.MolToSmiles(mirror, isomericSmiles=True)
        pair = tuple(sorted((canonical, mirror_smiles)))
        if canonical == mirror_smiles or pair in seen:
            continue
        seen.add(pair)
        similarities = {str(chiral): float(DataStructs.TanimotoSimilarity(
            gen.GetFingerprint(mol), gen.GetFingerprint(mirror)))
            for chiral, gen in generators.items()}
        pairs.append(dict(smiles=canonical, mirror_smiles=mirror_smiles,
                          mirror_observed=mirror_smiles in unique, similarities=similarities))
    audit = dict(
        api="rdFingerprintGenerator.GetMorganGenerator", radius=2, nBits=2048,
        useChirality=True, rdkit_atom_count="implicit H excluded",
        scaffold_chirality=False, empty_scaffold_policy="single acyclic scaffold class",
        duplicate_fingerprint_rows=len(ids) - len(np.unique(bits, axis=0)),
        enantiomer_definition="invert all specified tetrahedral centers; exclude identical canonical mirror",
        mirror_pairs=pairs,
        observed_mirror_pairs=sum(p["mirror_observed"] for p in pairs),
        observed_distinguished_chiral=sum(p["mirror_observed"] and p["similarities"]["True"] < 1 for p in pairs),
        observed_distinguished_achiral=sum(p["mirror_observed"] and p["similarities"]["False"] < 1 for p in pairs),
        limitation="Unspecified, axial and non-tetrahedral stereochemistry is not inferred.",
    )
    return bits, distance, records, audit
