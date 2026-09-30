"""All pages are in one request before the only global selection response."""

from hplc_al.common import stable_hash, write_once
from hplc_al.llm.execution import log
from hplc_al.llm.full_pool import LIMITS as V2_LIMITS
from hplc_al.llm.full_pool import token_bound, validate_cards
from hplc_al.llm.memory import validate_observations
from hplc_al.llm.planner import ARBITRATE_PROMPT, SCIENCE, Journal, validate_selection
from hplc_al.llm.responses_transport import call, encode
from hplc_al.llm.wire import display_numbers, replace

from .codec import pack, table, tables, unpack

LIMITS = {
    k: V2_LIMITS[k]
    for k in ("context_tokens", "tokenizer", "token_safety_factor", "max_output_tokens")
} | {"page_size": 300}
PROMPT = (
    SCIENCE.split("Host packets use")[0]
    + """
GLOBAL SELECTION: All legal candidates are supplied in the following candidate pages
in THIS request. Pages are delivery partitions, never screening or nomination stages.
Read every page before selecting exactly 32 unique candidates across the whole pool.
Compare cross-page chemical controls, condition contrasts and redundant experiments.
There are NO per-page, scaffold, uncertainty, stereochemistry or condition quotas.
Do not nominate, discard, rank or select separately by page. No expansion calls are needed.

The first message contains metadata, observed tables, memory and column dictionaries.
Each table row is zipped with its columns. For columns named in dictionaries, every
cell is an integer index into that column's dictionary (zero based). Dictionary values
are literal, never recursively decoded. Other cells are literal. Restore observed row
order using positions. Memory is ordinary JSON, not dictionary encoded.
cN and oN are reversible round-local candidate and observation IDs. gN denotes equality
groups for identity/scaffold. Use these displayed IDs in your answer. All candidate
fields and chemical structures are retained. Floats display six decimal places
(maximum absolute rounding error 0.0000005); sealed full precision is used by the host
for training, metrics and frozen errors. Historical memory is included in full.
The final message marks END_OF_POOL with the total candidate count and packet_hash.
Return exactly one selection object after considering ALL supplied candidates.
"""
    + ARBITRATE_PROMPT.split("Final answer: ", 1)[1]
)


class ContextCapacityError(ValueError):
    """No request was sent: the full pool cannot fit the configured context."""


def build_request(cards, legal_ids, observed, memory, round_index, limits):
    validate_cards(cards, legal_ids)
    validate_observations(observed)
    if set(legal_ids) & {row["id"] for row in observed}:
        raise ValueError("observed/candidate overlap")
    if len(cards) < 32:
        raise ValueError("at least 32 legal candidates required")
    for key in ("context_tokens", "max_output_tokens", "page_size"):
        if type(limits[key]) is not int or limits[key] <= 0:
            raise ValueError(f"invalid {key}")
    if any(limits[k] != LIMITS[k] for k in ("tokenizer", "token_safety_factor")):
        raise ValueError("unsupported tokenizer or estimation margin")
    # Stable input order makes replay independent of caller list order.
    cards = sorted(cards, key=lambda row: row["id"])
    binding = {
        "mode": "global_all_candidates",
        "round": round_index,
        "cards_sha256": stable_hash(cards),
        "legal_ids": sorted(legal_ids),
        "observed": observed,
        "memory": memory,
        "limits": limits,
        "prompt": PROMPT,
    }
    packet_hash = stable_hash(binding)
    packed, mapping = pack(cards, observed, memory)
    if unpack(packed, mapping) != display_numbers(
        {
            "cards": cards,
            "observed": observed,
            "memory": memory,
        }
    ):
        raise ValueError("lossless codec round-trip failed")
    metadata = {
        "kind": "global_metadata",
        "packet_hash": packet_hash,
        "round": round_index,
        "candidate_count": len(cards),
        "dictionaries": packed["dictionaries"],
        "observed": tables(packed["observed"]),
        "memory": packed["memory"],
    }
    parts = [
        packed["cards"][i : i + limits["page_size"]]
        for i in range(0, len(cards), limits["page_size"])
    ]
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": encode(metadata).decode()},
    ]
    for index, part in enumerate(parts):
        messages.append(
            {
                "role": "user",
                "content": encode(
                    {
                        "kind": "candidate_page",
                        "page_index": index,
                        "page_count": len(parts),
                        "candidates": table(part),
                    }
                ).decode(),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": encode(
                {
                    "kind": "END_OF_POOL",
                    "candidate_count": len(cards),
                    "page_count": len(parts),
                    "packet_hash": packet_hash,
                    "instruction": "All candidates are now present. Select exactly 32 globally.",
                }
            ).decode(),
        }
    )
    estimated = token_bound(messages)
    audit = {
        "mode": "global_all_candidates",
        "candidate_count": len(cards),
        "all_candidate_ids": sorted(legal_ids),
        "page_sizes": list(map(len, parts)),
        "input_coverage": 1.0,
        "round_trip_verified": True,
        "screening_calls": 0,
        "nomination_cap": None,
        "estimated_input_tokens": estimated,
        "output_reserve": limits["max_output_tokens"],
        "required_context_tokens": estimated + limits["max_output_tokens"],
        "configured_context_tokens": limits["context_tokens"],
        "provider_capacity_verified": False,
    }
    audit["status"] = (
        "PASS"
        if audit["required_context_tokens"] <= limits["context_tokens"]
        else "BLOCKED_CONTEXT_CAPACITY"
    )
    return messages, mapping, binding, audit


def require_capacity(audit):
    if audit["status"] != "PASS":
        raise ContextCapacityError(
            f"FULL_POOL_CONTEXT_TOO_LARGE: input {audit['estimated_input_tokens']} + "
            f"output {audit['output_reserve']} = {audit['required_context_tokens']} > "
            f"configured {audit['configured_context_tokens']}; no API call sent. "
            "All candidates retained; no screening or truncation fallback."
        )


def plan(
    cards,
    legal_ids,
    observed,
    memory,
    round_index,
    directory,
    config,
    limits,
    transport=call,
    *,
    retry_requests=False,
    retry_forever=False,
    request_name="global_selection",
):
    messages, mapping, binding, audit = build_request(
        cards, legal_ids, observed, memory, round_index, limits
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_once(directory / "binding.json", {**binding, "config": config})
    write_once(directory / "wire_aliases.json", mapping)
    write_once(directory / "context_admission.json", audit)
    require_capacity(audit)
    log(
        f"Global selection: all {len(cards)} candidates in {len(audit['page_sizes'])} "
        f"pages in ONE request; estimated input {audit['estimated_input_tokens']} tokens"
    )
    journal = Journal(
        directory,
        config,
        limits,
        transport,
        retry_requests=retry_requests,
        retry_forever=retry_forever,
    )
    wire_response = journal.ask(request_name, messages)
    response = replace(wire_response, {v: k for k, v in mapping.items()})
    packet_hash = stable_hash(binding)
    selected = validate_selection(
        response,
        packet_hash,
        legal_ids,
        legal_ids,
        [o["id"] for o in observed],
        [h["id"] for h in memory["previous_hypotheses"]],
    )
    saved = {
        "mode": "global_all_candidates",
        "selected_ids": selected,
        "response": response,
        "input_audit": audit,
        "receipts": journal.receipts,
        "packet_hash": packet_hash,
        "visible_candidates": sorted(legal_ids),
    }
    write_once(directory / "selection.json", saved)
    return saved
