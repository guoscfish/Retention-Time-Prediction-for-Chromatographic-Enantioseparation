"""Pure label-free full-pool serialization, chunking, coverage and summaries."""

import math
from collections import Counter
from functools import lru_cache
from itertools import combinations

import numpy as np
import tiktoken
from rdkit import Chem

from ..common import stable_hash
from .responses_transport import MAX_OUTPUT_TOKENS, encode

BATCH_SIZE = 32
SEED = 1525
METHOD = "free_llm32_fullpool"
STUDY_ID = "odh_free_llm32_fullpool_v2"
BUDGETS = [333, 365, 397]
LIMITS = {
    "context_tokens": 262144,
    "tokenizer": "o200k_base",
    "token_safety_factor": 1.2,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "target_chunk_size": 250,
    "nominees_per_chunk": 16,
    "arbitration_calls": 8,
}
FIELDS = (
    "id",
    "smiles",
    "stereocenters",
    "scaffold",
    "ipa_fraction",
    "flow",
    "pred_center",
    "q_width",
    "coverage",
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "identity",
    "scaffold_group",
    "pred_q10",
    "pred_q90",
    "functional_groups",
)


def opaque(row):
    return "r" + stable_hash([STUDY_ID, int(row)])[:12]


def validate_cards(cards, legal_ids):
    ids = [c["id"] for c in cards]
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(legal_ids)
        or len(legal_ids) != len(set(legal_ids))
    ):
        raise ValueError("candidate cards must equal legal U exactly")
    for card in cards:
        if set(card) != set(FIELDS):
            raise ValueError("candidate card violates label-free allowlist")
        for field in (
            "ipa_fraction",
            "flow",
            "pred_center",
            "q_width",
            "coverage",
            "MW",
            "LogP",
            "TPSA",
            "HBD",
            "HBA",
            "pred_q10",
            "pred_q90",
        ):
            if not isinstance(card[field], (int, float)) or not math.isfinite(
                card[field]
            ):
                raise ValueError("nonfinite candidate field")
        if (
            not 0 <= card["coverage"] <= 1
            or not card["smiles"]
            or Chem.MolFromSmiles(card["smiles"]) is None
        ):
            raise ValueError("invalid coverage or missing SMILES")
        if not math.isclose(
            card["q_width"], card["pred_q90"] - card["pred_q10"], abs_tol=1e-7
        ):
            raise ValueError("q-width/prediction mismatch")
    return {c["id"]: c for c in cards}


def table(cards):
    """Lossless columnar JSON; every required field remains explicit in columns."""
    return {"columns": list(FIELDS), "rows": [[c[k] for k in FIELDS] for c in cards]}


@lru_cache(maxsize=1)
def tokenizer():
    return tiktoken.get_encoding(LIMITS["tokenizer"])


def token_bound(value):
    # Explicit tokenizer estimate + 20% margin + framing reserve. Not a claim
    # that the third-party provider uses this tokenizer or context capacity.
    return (
        math.ceil(
            len(tokenizer().encode(encode(value).decode(), disallowed_special=()))
            * LIMITS["token_safety_factor"]
        )
        + 4096
    )


def ensure_context(messages, limits):
    if token_bound(messages) + limits["max_output_tokens"] > limits["context_tokens"]:
        raise ValueError(
            "CONTEXT_BUDGET_EXCEEDED: stop; never truncate scientific information"
        )


def chunks(cards, salt, prompt, common, limits):
    ordered = sorted(cards, key=lambda c: stable_hash([salt, c["id"]]))

    def messages(part):
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": encode({**common, "cards": table(part)}).decode(),
            },
        ]

    overhead = token_bound(messages([]))
    result, current, cost = [], [], overhead

    def append_checked(part):
        try:
            ensure_context(messages(part), limits)
        except ValueError:
            if len(part) < 2:
                raise ValueError(
                    "single card plus memory exceeds context budget"
                ) from None
            middle = len(part) // 2
            append_checked(part[:middle])
            append_checked(part[middle:])
        else:
            result.append(part)

    for card in ordered:
        # Incremental estimate avoids repeatedly tokenizing the whole L333 memory.
        size = token_bound([card[k] for k in FIELDS]) - 4096 + 16
        if current and (
            len(current) >= limits["target_chunk_size"]
            or cost + size + limits["max_output_tokens"] > limits["context_tokens"]
        ):
            append_checked(current)
            current, cost = [], overhead
        current.append(card)
        cost += size
    if current:
        append_checked(current)
    return result


def compact_records(records):
    """Lossless groups by exact field membership: initial observations stay error-free."""
    groups = {}
    for record in records:
        keys = tuple(sorted(record))
        groups.setdefault(keys, []).append([record[k] for k in keys])
    return [{"columns": list(keys), "rows": values} for keys, values in groups.items()]


def coverage_audit(legal_ids, screened_chunks, nominees):
    seen = [i for chunk in screened_chunks for i in chunk]
    counts = Counter(seen)
    legal = set(legal_ids)
    result = {
        "total_legal_candidates": len(legal_ids),
        "number_of_chunks": len(screened_chunks),
        "chunk_sizes": [len(c) for c in screened_chunks],
        "all_screened_ids": sorted(counts),
        "duplicate_screen_count": sum(n - 1 for n in counts.values()),
        "missing_ids": sorted(legal - counts.keys()),
        "illegal_ids": sorted(counts.keys() - legal),
        "nominee_count": len(set(nominees)),
        "full_pool_screen_coverage": len(legal & counts.keys()) / len(legal)
        if legal
        else 0,
    }
    if (
        len(legal) != len(legal_ids)
        or result["missing_ids"]
        or result["illegal_ids"]
        or result["duplicate_screen_count"]
        or result["full_pool_screen_coverage"] != 1.0
        or not set(nominees) <= legal
    ):
        raise ValueError("FULL_POOL_COVERAGE_FAILED")
    return result


def distribution(values):
    return dict(
        zip(
            ("min", "q10", "q25", "median", "q75", "q90", "max"),
            map(float, np.quantile(values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])),
        )
    )


def pair_counts(cards):
    nonstereo = {
        c["id"]: Chem.MolToSmiles(Chem.MolFromSmiles(c["smiles"]), isomericSmiles=False)
        for c in cards
    }
    # Group before comparing: avoids quadratic cross-scaffold work for full U.
    stereo_groups, identity_groups = {}, {}
    for card in cards:
        stereo_groups.setdefault(nonstereo[card["id"]], []).append(card)
        identity_groups.setdefault(card["identity"], []).append(card)
    stereo = sum(
        a["smiles"] != b["smiles"]
        for group in stereo_groups.values()
        for a, b in combinations(group, 2)
    )
    condition = sum(
        (a["ipa_fraction"], a["flow"]) != (b["ipa_fraction"], b["flow"])
        for group in identity_groups.values()
        for a, b in combinations(group, 2)
    )
    return {
        "stereo_contrast_pairs": stereo,
        "exact_identity_condition_contrast_pairs": condition,
    }


def summary(cards, observations):
    centers = distribution([c["pred_center"] for c in cards])
    high = sorted(
        (o for o in observations if "abs_error" in o),
        key=lambda o: (-o["abs_error"], o["id"]),
    )[:16]
    scaffolds = Counter(c["scaffold"] for c in cards)
    return {
        "candidate_count": len(cards),
        "pred_center": centers,
        "q_width": distribution([c["q_width"] for c in cards]),
        "coverage": distribution([c["coverage"] for c in cards]),
        "ipa_flow_counts": dict(
            sorted(Counter(f"{c['ipa_fraction']}|{c['flow']}" for c in cards).items())
        ),
        "scaffold_counts": dict(sorted(scaffolds.items())),
        "identity_counts": dict(sorted(Counter(c["identity"] for c in cards).items())),
        **pair_counts(cards),
        "prediction_extreme_counts": {
            "low_q10": sum(c["pred_center"] <= centers["q10"] for c in cards),
            "high_q90": sum(c["pred_center"] >= centers["q90"] for c in cards),
        },
        "highest_error_related_scaffold_coverage": [
            {
                "observed_id": o["id"],
                "scaffold": o["scaffold"],
                "observed_abs_error": o["abs_error"],
                "candidate_count": scaffolds[o["scaffold"]],
            }
            for o in high
        ],
    }
