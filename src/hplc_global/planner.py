"""All pages are in one request before the only global selection response."""

from hplc_al.common import stable_hash, write_once
from hplc_al.llm.execution import log
from hplc_al.llm.full_pool import LIMITS as V2_LIMITS
from hplc_al.llm.full_pool import BATCH_SIZE, token_bound, validate_cards
from hplc_al.llm.memory import validate_observations
from hplc_al.llm.planner import ARBITRATE_PROMPT, SCIENCE, Journal
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


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _require_text(value, *, empty=False):
    """Text validity only: length guidance is advisory in global V3."""
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError("scientific text must be a valid nonempty string")


def validate_selection(value, packet_hash, legal, visible, observed, previous):
    """Preserve V2 scientific checks; explanation length is advisory only."""
    required = {
        "type",
        "packet_hash",
        "choices",
        "hypotheses",
        "previous_hypothesis_updates",
        "feedback_interpretation",
        "batch_rationale",
        "unresolved_questions",
    }
    if (
        set(value) != required
        or value["type"] != "selection"
        or value["packet_hash"] != packet_hash
    ):
        raise ValueError("selection schema/packet binding mismatch")
    choices = value["choices"]
    ids = [c["id"] for c in choices]
    if (
        len(ids) != BATCH_SIZE
        or len(set(ids)) != BATCH_SIZE
        or not set(ids) <= set(legal) & set(visible)
    ):
        raise ValueError(
            "exactly 32 distinct legal arbitration-visible candidates required"
        )
    hypotheses = value["hypotheses"]
    hids = [h["id"] for h in hypotheses]
    if len(hids) > 8 or len(set(hids)) != len(hids) or set(hids) & set(previous):
        raise ValueError("invalid/reused hypothesis identity")
    for c in choices:
        if set(c) != {
            "id",
            "reason",
            "scientific_role",
            "hypothesis_id",
            "evidence_ids",
        }:
            raise ValueError("invalid choice schema")
        if not _text(c["reason"]) or not _text(c["scientific_role"]):
            raise ValueError("missing choice rationale")
        _require_text(c["reason"])
        _require_text(c["scientific_role"])
        if c["hypothesis_id"] is not None and c["hypothesis_id"] not in set(hids) | set(
            previous
        ):
            raise ValueError("unknown choice hypothesis")
    for h in hypotheses:
        if set(h) != {
            "id",
            "claim",
            "alternative",
            "expected_sign",
            "candidate_ids",
            "evidence_ids",
            "learning_value",
        }:
            raise ValueError("invalid hypothesis schema")
        if any(
            not _text(h[k]) for k in ("id", "claim", "alternative", "learning_value")
        ):
            raise ValueError("missing scientific hypothesis")
        for key in ("id", "claim", "alternative", "learning_value"):
            _require_text(h[key])
        if (
            not h["candidate_ids"]
            or len(h["candidate_ids"]) > 8
            or len(set(h["candidate_ids"])) != len(h["candidate_ids"])
            or not set(h["candidate_ids"]) <= set(ids)
            or h["expected_sign"] not in (-1, 0, 1)
        ):
            raise ValueError("invalid hypothesis prediction")
    for item in choices + hypotheses:
        if (
            not isinstance(item["evidence_ids"], list)
            or not set(item["evidence_ids"]) <= set(observed)
            or len(item["evidence_ids"]) > 4
        ):
            raise ValueError("evidence must be observed")
    updates = value["previous_hypothesis_updates"]
    if len(updates) != len(previous) or {u["id"] for u in updates} != set(previous):
        raise ValueError("every previous hypothesis needs one update")
    for u in updates:
        if set(u) != {
            "id",
            "status",
            "reason",
            "supporting_observations",
            "contradicting_observations",
        }:
            raise ValueError("invalid hypothesis update")
        evidence = u["supporting_observations"] + u["contradicting_observations"]
        _require_text(u["reason"])
        if (
            len(u["supporting_observations"]) > 4
            or len(u["contradicting_observations"]) > 4
        ):
            raise ValueError("too many update evidence references")
        if (
            u["status"] not in ("supported", "weakened", "contradicted", "unresolved")
            or not _text(u["reason"])
            or not set(evidence) <= set(observed)
            or (u["status"] != "unresolved" and not evidence)
        ):
            raise ValueError("unsupported hypothesis update")
    if (
        not _text(value["batch_rationale"])
        or not isinstance(value["feedback_interpretation"], str)
        or (previous and not _text(value["feedback_interpretation"]))
        or not isinstance(value["unresolved_questions"], list)
        or any(not _text(q) for q in value["unresolved_questions"])
    ):
        raise ValueError("missing feedback/batch interpretation")
    _require_text(value["batch_rationale"])
    _require_text(value["feedback_interpretation"], empty=not previous)
    if len(value["unresolved_questions"]) > 8:
        raise ValueError("too many unresolved questions")
    for question in value["unresolved_questions"]:
        _require_text(question)
    return ids


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
