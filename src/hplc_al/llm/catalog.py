"""Label-free chemical metadata and gradient-distance percentiles."""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..common import SOURCE, stable_hash

GROUPS = (
    "fr_Al_OH",
    "fr_Ar_OH",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_amide",
    "fr_ester",
    "fr_ether",
    "fr_ketone",
    "fr_COO",
    "fr_halogen",
    "fr_nitrile",
)


def metadata(ids):
    frame = pd.read_csv(SOURCE, usecols=["SMILES", "Speed", "i-PrOH_proportion"])
    result, fingerprints = {}, {}
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    for i in ids:
        molecule = Chem.MolFromSmiles(frame.loc[i, "SMILES"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES in legal outer pool row {i}")
        smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        centers = Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False
        )
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=False
        )
        row = {
            "smiles": smiles,
            "scaffold": scaffold,
            "identity": stable_hash(smiles)[:16],
            "scaffold_group": stable_hash(scaffold)[:16],
            "functional_groups": [
                g[3:] for g in GROUPS if getattr(Fragments, g)(molecule)
            ],
            "stereocenters": [{"atom": int(a), "CIP": c} for a, c in centers],
            "MW": Descriptors.MolWt(molecule),
            "LogP": Descriptors.MolLogP(molecule),
            "TPSA": Descriptors.TPSA(molecule),
            "HBD": Descriptors.NumHDonors(molecule),
            "HBA": Descriptors.NumHAcceptors(molecule),
            "rotatable_bonds": Descriptors.NumRotatableBonds(molecule),
            "aromatic_rings": Descriptors.NumAromaticRings(molecule),
            "chiral_centers": len(centers),
            "unassigned_centers": sum(c == "?" for _, c in centers),
            "ipa_fraction": float(frame.loc[i, "i-PrOH_proportion"]),
            "flow": float(frame.loc[i, "Speed"]),
        }
        result[i] = row
        fingerprints[i] = generator.GetFingerprint(molecule)
    return result, fingerprints


def coverage_percentile(phi, labeled_positions):
    from scipy.spatial.distance import cdist
    from scipy.stats import rankdata

    distance = cdist(phi, phi[labeled_positions], metric="sqeuclidean").min(axis=1)
    return rankdata(distance, method="average") / len(distance)
