"""Allowlisted ODH full-pool queries with a matched chemical-information ablation."""

import copy
import math

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Fragments, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .common import SOURCE, stable_hash

QUERY_BUDGET = 8
VIEW_BUDGET = 480
CALL_BUDGET = 28
PAGE_SIZE = 24
NUMERIC = ("MW", "LogP", "TPSA", "HBD", "HBA", "rotatable_bonds", "aromatic_rings",
           "chiral_centers", "unassigned_centers", "ipa_fraction", "flow")
ANONYMOUS = {name: f"x{i:02d}" for i, name in enumerate(NUMERIC)}
GROUPS = ("fr_Al_OH", "fr_Ar_OH", "fr_NH0", "fr_NH1", "fr_NH2", "fr_amide",
          "fr_ester", "fr_ether", "fr_ketone", "fr_COO", "fr_halogen", "fr_nitrile")
COMMON_PROMPT = """You select experiments to improve a retrained regression model on a fixed
Row distribution. Select exactly 16 additional rows, complementing 16 pending numerical
LCMD rows. All 32 responses are revealed together AFTER the batch and hypotheses freeze.
Use only the supplied trajectory. No native tools, files, web, literature lookups, other
trajectories, validation or test outcomes. Search the full legal pool before selecting.
The objective is overall predictive accuracy, not maximizing the selected response.
pred_center is MSE-trained, not a median. pred_q10 and pred_q90 are quantile outputs;
their difference is NOT calibrated uncertainty. coverage is distance to current labeled
rows in raw full-network gradient geometry, expressed as a percentile.
Initial measured rows have no premeasurement errors. Later signed_error is the frozen
premeasurement center prediction minus the measured response. Use feedback to revise
hypotheses; distinguish observations, model behavior, priors and untested explanations.

Return only a JSON object. Query form: {"type":"query","queries":[{"pool":"candidates",
"offset":0,"limit":24,"sort_by":"coverage","order":"desc"}]}.
Pools: candidates, observed, pending. Optional filters: ids (list), ranges (object of
field:[min,max], either bound may be null), same_identity_as (id), similar_to (id),
min_similarity (0 to 1), same_scaffold_as (id). Similarity uses radius-2 2048-bit Morgan
Tanimoto with includeChirality=False; exact identity separately preserves stereochemistry.
sort_by accepts any numeric field visible on cards or similarity (requires similar_to).
offset supports paging; total_matches covers the whole filtered pool. Limit <=24.
Default order is a round-salted hash of opaque IDs. No shortlist restricts eligibility.
At most 24 queries, 480 distinct candidate views (including 20 initial), 28 replies.
Query errors count against this budget. No per-family quotas apply.

Final form: {"type":"selection","packet_hash":"copy exactly","choices":[{"id":"...",
"evidence_ids":[],"reason":"expected learning value; which competing explanations
this measurement distinguishes, or explicitly exploratory"}],"hypotheses":[{"id":"h1",
"claim":"...","evidence_ids":[],"alternative":"...","expected_sign":0,
"candidate_ids":["..."],"learning_value":"..."}],"feedback_interpretation":"...",
"batch_rationale":"..."}. Choose exactly 16 distinct VIEWED eligible candidates.
Evidence IDs must be measured observed records you viewed. Zero to eight hypotheses;
do not invent strong mechanisms when evidence is absent. expected_sign is -1, 0 or 1,
predicting signed_error (prediction minus measurement), where 0 means no directional
claim. candidate_ids must be among your choices. Explicitly explain how prior outcomes
support, weaken or contradict previous hypotheses; round zero may use an empty string.
"""
CHEMISTRY_PROMPT = """
CHEMICAL SETTING: HPLC on CHIRALCEL OD-H, cellulose tris(3,5-dimethylphenylcarbamate)
coated on silica. Target response is retention volume RTv=retention time*flow, in mL.
ipa_fraction is the recorded isopropanol proportion; flow is the recorded flow rate.
Do not invent unrecorded solvent identities, temperature, pH or additives. Use chiral
structure, H-bonding, steric environment and structure-condition interactions as priors,
not universal rules. Do not impose monotonic retention with solvent composition.
Canonical isomeric SMILES preserve exact stereochemical identity; Murcko scaffolds and
nonchiral fingerprints support analogy, not proof of identical chiral recognition.
Additional filters: smiles_contains (substring), functional_group (a visible group name).
"""
MASKED_PROMPT = """
The task has anonymous numerical inputs x00-x10, opaque exact-identity and scaffold
groups, numerical similarity, predictions and observed responses. No domain semantics
or descriptor-name mapping is supplied. Make judgments from these numerical data.
"""


def metadata(ids):
    frame = pd.read_csv(SOURCE, usecols=["SMILES", "Speed", "i-PrOH_proportion"])
    result, fingerprints = {}, {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                           includeChirality=False)
    for i in ids:
        molecule = Chem.MolFromSmiles(frame.loc[i, "SMILES"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES in legal outer pool row {i}")
        smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True,
                                            useLegacyImplementation=False)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        row = {"smiles": smiles, "scaffold": scaffold,
               "identity": stable_hash(smiles)[:16], "scaffold_group": stable_hash(scaffold)[:16],
               "functional_groups": [g[3:] for g in GROUPS if getattr(Fragments, g)(molecule)],
               "stereocenters": [{"atom": int(a), "CIP": c} for a, c in centers],
               "MW": Descriptors.MolWt(molecule), "LogP": Descriptors.MolLogP(molecule),
               "TPSA": Descriptors.TPSA(molecule), "HBD": Descriptors.NumHDonors(molecule),
               "HBA": Descriptors.NumHAcceptors(molecule),
               "rotatable_bonds": Descriptors.NumRotatableBonds(molecule),
               "aromatic_rings": Descriptors.NumAromaticRings(molecule),
               "chiral_centers": len(centers), "unassigned_centers": sum(c == "?" for _, c in centers),
               "ipa_fraction": float(frame.loc[i, "i-PrOH_proportion"]),
               "flow": float(frame.loc[i, "Speed"])}
        result[i] = row
        fingerprints[i] = generator.GetFingerprint(molecule)
    return result, fingerprints


class Catalog:
    def __init__(self, features, fingerprints, labeled, unlabeled, pending, predictions,
                 coverage, observations, *, salt, chemical):
        self.chemical = chemical
        self.salt = salt
        self.query_budget = QUERY_BUDGET
        self.queries = 0
        self.viewed = set()
        self.observed_viewed = set()
        # Opaque IDs prevent source-row / literature identification in either arm.
        self.to_id = {i: "r" + stable_hash(["ODH-LLM-v1", i])[:12] for i in features}
        self.from_id = {v: k for k, v in self.to_id.items()}
        if len(self.from_id) != len(features):
            raise RuntimeError("Opaque identity collision")
        self.fingerprints = {self.to_id[i]: fp for i, fp in fingerprints.items()}
        self.pools = {"candidates": {}, "observed": {}, "pending": {}}
        pending_set, labeled_set = set(pending), set(labeled)
        if not pending_set <= set(unlabeled) or labeled_set & set(unlabeled):
            raise ValueError("Invalid catalog membership")
        for i in sorted(labeled + unlabeled):
            original = features[i]
            row = {"id": self.to_id[i], "identity": original["identity"],
                   "scaffold_group": original["scaffold_group"]}
            row.update({(k if chemical else ANONYMOUS[k]): original[k] for k in NUMERIC})
            if chemical:
                row.update({k: original[k] for k in ("smiles", "scaffold", "functional_groups", "stereocenters")})
            if i in labeled_set:
                row.update(observations[i])
                pool = "observed"
            else:
                row.update(dict(zip(("pred_q10", "pred_center", "pred_q90"),
                                    map(float, predictions[i]))))
                row["coverage"] = float(coverage[i])
                pool = "pending" if i in pending_set else "candidates"
            self.pools[pool][row["id"]] = row
        self.references = {k: v for pool in self.pools.values() for k, v in pool.items()}

    def key(self, row):
        return stable_hash([self.salt, row["id"]])

    def initial(self):
        cards = sorted(self.pools["candidates"].values(), key=self.key)[:20]
        self.viewed.update(r["id"] for r in cards)
        return copy.deepcopy(cards)

    def query(self, query):
        self.queries += 1
        if self.queries > QUERY_BUDGET:
            raise ValueError("query budget exceeded")
        allowed = {"pool", "ids", "offset", "limit", "ranges", "sort_by", "order",
                   "same_identity_as", "same_scaffold_as", "similar_to", "min_similarity"}
        if self.chemical:
            allowed |= {"smiles_contains", "functional_group"}
        if set(query) - allowed or query.get("pool") not in self.pools:
            raise ValueError("Unknown query field or pool")
        pool = query["pool"]
        rows = list(self.pools[pool].values())
        if "ids" in query:
            if not set(query["ids"]) <= set(self.pools[pool]):
                raise ValueError("ID outside requested pool")
            rows = [r for r in rows if r["id"] in query["ids"]]
        for key, field in (("same_identity_as", "identity"), ("same_scaffold_as", "scaffold_group")):
            if key in query:
                ref = self.references[query[key]]
                rows = [r for r in rows if r[field] == ref[field]]
        if "similar_to" in query:
            fp = self.fingerprints[query["similar_to"]]
            rows = [{**r, "similarity": DataStructs.TanimotoSimilarity(fp, self.fingerprints[r["id"]])} for r in rows]
            rows = [r for r in rows if r["similarity"] >= float(query.get("min_similarity", 0))]
        if "smiles_contains" in query:
            rows = [r for r in rows if query["smiles_contains"] in r["smiles"]]
        if "functional_group" in query:
            rows = [r for r in rows if query["functional_group"] in r["functional_groups"]]
        numeric = set(NUMERIC if self.chemical else ANONYMOUS.values())
        numeric |= ({"response", "premeasurement_center", "signed_error", "abs_error"}
                    if pool == "observed" else {"pred_q10", "pred_center", "pred_q90", "coverage"})
        if "similar_to" in query:
            numeric.add("similarity")
        for name, bounds in query.get("ranges", {}).items():
            if name not in numeric or len(bounds) != 2:
                raise ValueError("Invalid numeric range field")
            low, high = bounds
            low, high = -math.inf if low is None else float(low), math.inf if high is None else float(high)
            rows = [r for r in rows if name in r and low <= r[name] <= high]
        rows.sort(key=self.key)
        field = query.get("sort_by", "similarity" if "similar_to" in query else None)
        if field:
            if field not in numeric or query.get("order", "desc") not in ("asc", "desc"):
                raise ValueError("Invalid numeric sort")
            known, missing = [r for r in rows if field in r], [r for r in rows if field not in r]
            rows = sorted(known, key=lambda r: r[field], reverse=query.get("order", "desc") == "desc") + missing
        offset, limit = query.get("offset", 0), query.get("limit", PAGE_SIZE)
        if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= PAGE_SIZE:
            raise ValueError("Invalid page bounds")
        page = rows[offset:offset + limit]
        ids = {r["id"] for r in page}
        if pool == "candidates":
            if len(self.viewed | ids) > VIEW_BUDGET:
                raise ValueError("Distinct candidate-view budget exceeded")
            self.viewed |= ids
        if pool == "observed":
            self.observed_viewed |= ids
        return {"pool": pool, "total_matches": len(rows), "offset": offset,
                "next_offset": offset + len(page) if offset + len(page) < len(rows) else None,
                "records": copy.deepcopy(page)}

    def validate(self, value, packet_hash):
        if value.get("type") != "selection" or value.get("packet_hash") != packet_hash:
            raise ValueError("Invalid selection packet binding")
        choices = value.get("choices", [])
        ids = [c["id"] for c in choices]
        if len(ids) != 16 or len(set(ids)) != 16 or not set(ids) <= self.viewed:
            raise ValueError("Exactly 16 unique viewed candidate IDs required")
        if not set(ids) <= set(self.pools["candidates"]) or not self.queries:
            raise ValueError("Select only legal candidates after querying")
        hypotheses = value.get("hypotheses", [])
        if len(hypotheses) > 8 or len({h["id"] for h in hypotheses}) != len(hypotheses):
            raise ValueError("Invalid hypothesis count/identity")
        for item in choices + hypotheses:
            if not set(item.get("evidence_ids", [])) <= self.observed_viewed:
                raise ValueError("Evidence must be viewed observed IDs")
        for item in choices:
            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                raise ValueError("Each choice requires a learning reason")
        for h in hypotheses:
            if not set(h["candidate_ids"]) <= set(ids) or h["expected_sign"] not in (-1, 0, 1):
                raise ValueError("Invalid hypothesis prediction")
            if any(not isinstance(h.get(k), str) or not h[k].strip()
                   for k in ("id", "claim", "alternative", "learning_value")):
                raise ValueError("Hypothesis lacks scientific content")
        if not value.get("batch_rationale") or not isinstance(value.get("feedback_interpretation"), str):
            raise ValueError("Missing batch rationale/feedback interpretation")
        return [self.from_id[i] for i in ids]


def coverage_percentile(phi, labeled_positions):
    from scipy.spatial.distance import cdist
    from scipy.stats import rankdata
    distance = cdist(phi, phi[labeled_positions], metric="sqeuclidean").min(axis=1)
    return rankdata(distance, method="average") / len(distance)
