"""Two-stage LLM planning; no label store, CSV, trainer or evaluation access."""

import hashlib
from pathlib import Path

from ..common import read_json, stable_hash, write_once
from .execution import log, request_with_retries
from .full_pool import (
    BATCH_SIZE,
    chunks,
    compact_records,
    coverage_audit,
    ensure_context,
    summary,
    table,
    token_bound,
    tokenizer,
    validate_cards,
)
from .memory import validate_observations
from .responses_transport import call, encode, parse_answer, payload
from .wire import aliases, content, display_numbers, memory_view, replace, summary_view

SCIENCE = """You plan chromatography experiments to improve retrained QGeoGNN accuracy on the fixed
Row distribution, CHIRALCEL OD-H. Target RTv=retention time*flow, mL. Use only this frozen prompt,
host candidate data, and this study/seed/method's observed memory. No native tools, web, shell,
MCP, files, other trajectories, validation/test labels or scores, external conversation history.
Canonical isomeric SMILES/CIP encode stereochemistry. IPA proportion and flow are recorded;
do not invent solvent identity, pH, temperature or additives. Chemical mechanisms are hypotheses.
pred_center is MSE-trained, not q50; q_width is NOT calibrated epistemic uncertainty.
coverage is a nearest-labeled raw-gradient distance percentile, NOT true error.
Initial L333 records have observed RTv but NO premeasurement errors. Later signed_error means
frozen premeasurement center minus measured RTv. Learn from support, contradiction and unresolved
hypotheses. Avoid claiming a sign match proves a mechanism. Every card row follows its columns.
Host packets use {value_table,data}: replace exact @N strings in data (keys and values)
with their literal value_table definitions (strings, arrays or objects); definitions are not recursive.
Then decode any {record_groups:[{columns,positions,rows}]} as a record list: zip columns
with each row and restore the original positions. Missing fields remain absent. gN names are
round-local opaque identity/scaffold equality IDs. All chemical structures are preserved. Host display rounds floats to six decimal places
(maximum absolute display error 0.0000005); original full precision is sealed and used for
training, metrics and frozen premeasurement errors. Displayed arithmetic can differ by rounding. Summary identity_counts/scaffold_counts use [count,keys] rows: every listed key has that exact count. A memory {observed_id:ID} references the complete row in observed, not a summary.
A historical choice_ref uses observed[id]: id, selection_reason as reason, scientific_role,
hypothesis_id, plus its explicit evidence_ids. A hypothesis_ref uses the original hypothesis
fields in previous_hypotheses[id]; update_ref uses that state's id/status/reason/evidence lists.
These references preserve exact historical values; older differing updates remain explicit.
Do not use @N references in your answer: write literal text and original candidate IDs.
Return exactly one JSON object. No prose, markdown, duplicate JSON objects or native tool calls.
"""
SCREEN_PROMPT = (
    SCIENCE
    + """
STAGE 1: Screen every candidate in this chunk for learning value, hypothesis tests, model blind
spots and chemical controls. This is NOT final batch selection. Nominate zero to nomination_cap
candidates; cap is an upper bound, NEVER a quota. Do not rank nominees against unseen chunks.
Use identical criteria for every chunk. Summarize blind spots and useful contrasts, including
non-nominee IDs when relevant so global arbitration can rescue them.
Answer: {"nominees":[{"candidate_id":"...","screen_reason":"...",
"scientific_role":"...","linked_hypothesis":null,"priority":1,
"what_evidence_would_be_learned":"..."}],"chunk_summary":"...","rescue_candidate_ids":[]}.
priority is integer 1..5 (1 highest), a within-chunk annotation, not a global ranking.
linked_hypothesis is null or an ID from previous_hypotheses. rescue_candidate_ids belong to this chunk.
Be concise: screen_reason and what_evidence_would_be_learned each at most 64 o200k_base tokens,
scientific_role at most 12 tokens, chunk_summary at most 160 tokens, at most 16 rescue IDs.
Keep the entire JSON reply below 1400 tokens; host serialized input-cost cap is 2048 per chunk.
These are output bounds, not nomination or scientific quotas. Do not fill the bounds unnecessarily.
"""
)
ARBITRATE_PROMPT = (
    SCIENCE
    + """
STAGE 2: All legal U candidates were screened in Stage 1. Review ALL nominated detailed cards,
all their rationales, the full-pool summary, chunk summaries, observed feedback and hypotheses.
Choose exactly 32 unique legal experiments with no per-chunk, scaffold, coverage, uncertainty,
stereo or condition quotas. Nomination is information compression, not an eligibility gate.
To rescue any non-nominee, return {"type":"expand","expand_candidate_ids":["..."]}.
To discover all IDs in a chunk, return {"type":"directory","chunk_indices":[0]}.
Every screened legal ID can be expanded, even if not a nominee or mentioned in summaries.
Only choose cards visible in arbitration (nominees plus expanded cards). Expansion is host JSON
relay, never a native function tool. All previous replies/pages stay in this round's transcript.
Final answer: {"type":"selection","packet_hash":"copy exactly","choices":[
{"id":"...","reason":"learning value","scientific_role":"...","hypothesis_id":null,
"evidence_ids":[]}],"hypotheses":[{"id":"new unique id","claim":"...","alternative":"...",
"expected_sign":0,"candidate_ids":["chosen id"],"evidence_ids":[],"learning_value":"..."}],
"previous_hypothesis_updates":[{"id":"old id","status":"unresolved","reason":"...",
"supporting_observations":[],"contradicting_observations":[]}],
"feedback_interpretation":"...","batch_rationale":"...","unresolved_questions":[]}.
Zero to eight new hypotheses; expected_sign -1/0/1 predicts frozen signed_error.
Update EVERY old hypothesis exactly once: supported/weakened/contradicted/unresolved.
Evidence IDs must be observed records. Each supported/weakened/contradicted update needs actual
observations. Round zero has no updates. Roles are annotations, not quotas. Do not request more
than the remaining arbitration_calls; leave a final call for selection.
Request at most 32 expanded IDs or one chunk directory per reply. If a page cannot fit,
the host returns CONTEXT_PAGE_TOO_LARGE; request fewer cards or select visible cards.
Keep choice reason <=32 tokens and scientific_role <=16 tokens; hypothesis id <=16 tokens,
claim/alternative <=48 tokens each, learning_value <=32 tokens; update reason <=32 tokens.
Each evidence list <=4 distinct IDs; each hypothesis names <=8 distinct representative chosen candidate_ids. feedback_interpretation and batch_rationale <=128 tokens each.
At most eight unresolved_questions, each <=48 tokens. Bounds use o200k_base tokens.
"""
)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def bounded_text(value, maximum, *, empty=False):
    if (
        not isinstance(value, str)
        or (not empty and not value.strip())
        or len(tokenizer().encode(value, disallowed_special=())) > maximum
    ):
        raise ValueError("scientific text exceeds registered field bound")


def validate_screen(value, ids, previous, cap):
    if token_bound(encode(value).decode()) - 4096 > 2048:
        raise ValueError("screening reply exceeds registered serialized bound")
    if set(value) != {"nominees", "chunk_summary", "rescue_candidate_ids"} or not _text(
        value["chunk_summary"]
    ):
        raise ValueError("invalid screening schema")
    nominees = value["nominees"]
    bounded_text(value["chunk_summary"], 160)
    if not isinstance(nominees, list) or len(nominees) > cap:
        raise ValueError("nomination cap exceeded")
    chosen = [n["candidate_id"] for n in nominees]
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(ids):
        raise ValueError("illegal or duplicate nomination")
    if (
        not isinstance(value["rescue_candidate_ids"], list)
        or not set(value["rescue_candidate_ids"]) <= set(ids)
        or len(value["rescue_candidate_ids"]) > 16
    ):
        raise ValueError("illegal rescue reference")
    for n in nominees:
        if set(n) != {
            "candidate_id",
            "screen_reason",
            "scientific_role",
            "linked_hypothesis",
            "priority",
            "what_evidence_would_be_learned",
        }:
            raise ValueError("invalid nominee schema")
        if any(
            not _text(n[k])
            for k in (
                "screen_reason",
                "scientific_role",
                "what_evidence_would_be_learned",
            )
        ):
            raise ValueError("missing scientific rationale")
        if type(n["priority"]) is not int or not 1 <= n["priority"] <= 5:
            raise ValueError("invalid priority")
        bounded_text(n["screen_reason"], 64)
        bounded_text(n["what_evidence_would_be_learned"], 64)
        bounded_text(n["scientific_role"], 12)
        if (
            n["linked_hypothesis"] is not None
            and n["linked_hypothesis"] not in previous
        ):
            raise ValueError("unknown hypothesis")
    return nominees


def validate_selection(value, packet_hash, legal, visible, observed, previous):
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
        bounded_text(c["reason"], 32)
        bounded_text(c["scientific_role"], 16)
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
        for key, bound in (
            ("id", 16),
            ("claim", 48),
            ("alternative", 48),
            ("learning_value", 32),
        ):
            bounded_text(h[key], bound)
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
        bounded_text(u["reason"], 32)
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
    bounded_text(value["batch_rationale"], 128)
    bounded_text(value["feedback_interpretation"], 128, empty=not previous)
    if len(value["unresolved_questions"]) > 8:
        raise ValueError("too many unresolved questions")
    for question in value["unresolved_questions"]:
        bounded_text(question, 48)
    return ids


def admission(parts, initial, mapping, limits):
    """Reject impossible arbitration BEFORE spending on any screening calls.

    Bound every possible nominee by the largest cards in its chunk, plus the
    registered serialized reply caps. No candidate is removed to meet this gate.
    """
    base = [
        {"role": "system", "content": ARBITRATE_PROMPT},
        {"role": "user", "content": content(initial, mapping)},
    ]
    shared = {}

    def collect(value):
        if isinstance(value, str) and len(value) > 24:
            shared[value] = "@shared_000000"
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(key)
                collect(item)

    # Structures already present in the complete summary/observations cost only
    # a reference when repeated in a nominee. The base already pays their literals.
    collect(replace(display_numbers(initial), mapping))
    card_cost = 0
    for part in parts:
        costs = sorted(
            (
                token_bound(
                    encode(
                        replace(
                            replace(display_numbers(table([c])["rows"][0]), mapping),
                            shared,
                        )
                    ).decode()
                )
                - 4096
                + 32
                for c in part
            ),
            reverse=True,
        )
        card_cost += sum(costs[: limits["nominees_per_chunk"]])
    estimate = token_bound(base) + card_cost + len(parts) * (2048 + 128)
    reserve = 4096
    record = {
        "status": "PASS",
        "arbitration_input_bound": estimate,
        "relay_reserve": reserve,
        "output_reserve": limits["max_output_tokens"],
        "context_tokens": limits["context_tokens"],
        "screen_reply_cap": 2048,
    }
    if estimate + reserve + limits["max_output_tokens"] > limits["context_tokens"]:
        raise ValueError(
            f"CONTEXT_ADMISSION_FAILED: estimated {estimate} + relay {reserve} + output {limits['max_output_tokens']} > {limits['context_tokens']}; no screening requests sent"
        )
    return record


class Journal:
    """Write-ahead intent and exact replay; retries require explicit host policy."""

    def __init__(
        self,
        directory,
        config,
        limits,
        transport=call,
        *,
        retry_requests=False,
        retry_forever=False,
    ):
        self.directory = Path(directory)
        self.config = config
        self.limits = limits
        self.transport = transport
        self.receipts = []
        self.retry_requests = retry_requests
        self.retry_forever = retry_forever

    def ask(self, name, messages):
        ensure_context(messages, self.limits)
        log(
            f"{name}: estimated input {token_bound(messages)} tokens; output cap {self.limits['max_output_tokens']}"
        )
        request = payload(messages, self.config, self.limits["max_output_tokens"])
        digest = hashlib.sha256(encode(request)).hexdigest()
        intent = self.directory / f"{name}.request.json"
        receipt_path = self.directory / f"{name}.receipt.json"
        record = {
            "request_sha256": digest,
            "config_sha256": stable_hash(self.config),
            "request": request,
        }
        legacy = intent.exists() and not list(
            self.directory.glob(f"{name}.attempt_*.started.json")
        )
        if intent.exists():
            if read_json(intent) != record:
                raise RuntimeError("request/protocol drift; start a new study version")
            if not receipt_path.exists() and not self.retry_requests:
                raise RuntimeError(
                    "ambiguous request without receipt; never automatically retry"
                )
        else:
            if receipt_path.exists():
                raise RuntimeError("orphan receipt")
            write_once(intent, record)
        if receipt_path.exists():
            log(f"{name}: reusing saved response; no API call")
            saved = read_json(receipt_path)
        else:
            if self.retry_requests:
                # A newly written intent has no previous attempt; a pre-existing
                # intent from the original runner consumes attempt 1.
                answer, receipt = request_with_retries(
                    self.transport,
                    messages,
                    self.config,
                    self.limits["max_output_tokens"],
                    self.directory,
                    name,
                    digest,
                    legacy=legacy,
                    retry_forever=self.retry_forever,
                )
            else:
                answer, receipt = self.transport(
                    messages, self.config, self.limits["max_output_tokens"]
                )
            saved = {"answer": answer, "receipt": receipt}
            # Persist validated HTTP response even when downstream scientific schema fails.
            write_once(receipt_path, saved)
        receipt = saved["receipt"]
        answer = saved["answer"]
        if (
            receipt["request_sha256"] != digest
            or receipt["answer_sha256"] != hashlib.sha256(answer.encode()).hexdigest()
            or receipt["native_tool_calls"] != 0
        ):
            raise RuntimeError("receipt binding mismatch")
        self.receipts.append(receipt)
        return parse_answer(answer)


def plan(
    cards,
    legal_ids,
    observed,
    memory,
    round_index,
    directory,
    config,
    limits,
    *,
    transport=call,
    retry_requests=False,
    retry_forever=False,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lookup = validate_cards(cards, legal_ids)
    validate_observations(observed)
    if set(legal_ids) & {o["id"] for o in observed}:
        raise ValueError("observed/candidate overlap")
    previous = [h["id"] for h in memory["previous_hypotheses"]]
    binding = {
        "cards_sha256": stable_hash(sorted(cards, key=lambda c: c["id"])),
        "legal_ids_sha256": stable_hash(sorted(legal_ids)),
        "observed": observed,
        "memory": memory,
        "round": round_index,
        "config": config,
        "limits": limits,
        "screen_prompt": SCREEN_PROMPT,
        "arbitration_prompt": ARBITRATE_PROMPT,
    }
    write_once(directory / "binding.json", binding)
    packet_hash = stable_hash(binding)
    mapping = aliases(cards, observed)
    write_once(directory / "wire_aliases.json", mapping)
    wire_memory = memory_view(memory, observed)
    common = {
        "packet_hash": packet_hash,
        "round": round_index,
        "memory": wire_memory,
        "observed": compact_records(observed),
        "nomination_cap": limits["nominees_per_chunk"],
    }
    parts = chunks(
        cards,
        f"{memory['study']}/{memory['seed']}/{memory['method']}/{round_index}",
        SCREEN_PROMPT,
        common,
        limits,
        mapping,
    )
    full_summary = summary(cards, observed)
    initial = {
        "packet_hash": packet_hash,
        "nominees": table([]),
        "screening_results": [],
        "memory": wire_memory,
        "observed": compact_records(observed),
        "full_pool_summary": summary_view(full_summary),
        "arbitration_calls": limits["arbitration_calls"],
    }
    write_once(
        directory / "context_admission.json", admission(parts, initial, mapping, limits)
    )
    journal = Journal(
        directory,
        config,
        limits,
        transport,
        retry_requests=retry_requests,
        retry_forever=retry_forever,
    )
    log(f"Stage 1: {len(legal_ids)} candidates, {len(parts)} screening chunks")
    results, nominees, screened = [], {}, []
    for index, part in enumerate(parts):
        log(f"Stage 1 chunk {index + 1}/{len(parts)}: {len(part)} candidates")
        value = {**common, "cards": table(part)}
        messages = [
            {"role": "system", "content": SCREEN_PROMPT},
            {"role": "user", "content": content(value, mapping)},
        ]
        result = journal.ask(f"screen_{index:03d}", messages)
        ids = [c["id"] for c in part]
        for nomination in validate_screen(
            result, ids, previous, limits["nominees_per_chunk"]
        ):
            nominees[nomination["candidate_id"]] = nomination
        screened.append(ids)
        results.append({"chunk_index": index, **result})
        log(
            f"Stage 1 chunk {index + 1}/{len(parts)} validated; {len(result['nominees'])} nominees; covered {sum(map(len, screened))}/{len(legal_ids)}"
        )
    audit = coverage_audit(legal_ids, screened, nominees)
    write_once(directory / "screening_audit.json", audit)
    write_once(directory / "full_pool_summary.json", full_summary)
    initial.update(
        nominees=table([lookup[i] for i in sorted(nominees)]), screening_results=results
    )
    messages = [
        {"role": "system", "content": ARBITRATE_PROMPT},
        {"role": "user", "content": content(initial, mapping)},
    ]
    visible = set(nominees)
    log(
        f"Stage 1 complete: coverage=1.0; {len(nominees)} nominees. Starting global arbitration"
    )
    for turn in range(limits["arbitration_calls"]):
        log(
            f"Stage 2 arbitration {turn + 1}/{limits['arbitration_calls']}; {len(visible)} visible cards"
        )
        result = journal.ask(f"arbitrate_{turn:02d}", messages)
        if result.get("type") == "selection":
            selected = validate_selection(
                result,
                packet_hash,
                legal_ids,
                visible,
                [o["id"] for o in observed],
                previous,
            )
            # Recheck complete request coverage immediately before selection can freeze.
            coverage_audit(legal_ids, screened, nominees)
            saved = {
                "selected_ids": selected,
                "response": result,
                "screening_audit": audit,
                "nominee_ids": sorted(nominees),
                "receipts": journal.receipts,
                "packet_hash": packet_hash,
                "visible_in_arbitration": sorted(visible),
            }
            write_once(directory / "selection.json", saved)
            log(
                "Stage 2 complete: 32 selections validated; pending Git seal before label reveal"
            )
            return saved
        if result.get("type") == "expand" and set(result) == {
            "type",
            "expand_candidate_ids",
        }:
            ids = result["expand_candidate_ids"]
            if (
                not isinstance(ids, list)
                or not ids
                or len(set(ids)) != len(ids)
                or not set(ids) <= set(legal_ids)
            ):
                raise ValueError("expand only screened legal U IDs")
            reply = (
                {"expanded_cards": table([lookup[i] for i in ids])}
                if len(ids) <= 32
                else {"error": "REQUEST_AT_MOST_32_CARDS"}
            )
        elif result.get("type") == "directory" and set(result) == {
            "type",
            "chunk_indices",
        }:
            indices = result["chunk_indices"]
            if (
                not isinstance(indices, list)
                or not indices
                or any(type(i) is not int or not 0 <= i < len(parts) for i in indices)
            ):
                raise ValueError("invalid chunk directory request")
            reply = (
                {"chunk_ids": {str(i): screened[i] for i in indices}}
                if len(indices) == 1
                else {"error": "REQUEST_ONE_CHUNK_DIRECTORY"}
            )
        else:
            raise ValueError("invalid arbitration action")
        additions = [
            {"role": "assistant", "content": encode(result).decode()},
            {
                "role": "user",
                "content": content(
                    {
                        **reply,
                        "remaining_calls": limits["arbitration_calls"] - turn - 1,
                    },
                    mapping,
                ),
            },
        ]
        try:
            ensure_context(
                messages + additions,
                {**limits, "max_output_tokens": limits["max_output_tokens"] + 2048},
            )
        except ValueError:
            reply = {
                "error": "CONTEXT_PAGE_TOO_LARGE",
                "instruction": "Request fewer cards or select visible cards.",
                "remaining_calls": limits["arbitration_calls"] - turn - 1,
            }
            additions[-1]["content"] = content(reply)
        if "expanded_cards" in reply:
            visible.update(ids)
        messages += additions
    raise RuntimeError("arbitration budget exhausted; no selection frozen")
