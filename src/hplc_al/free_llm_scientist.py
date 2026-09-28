"""Text-only scientific planner; receives no label store, trainer or evaluation state."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .common import read_json, sha, stable_hash, write_once
from .llm_catalog import (CALL_BUDGET, CHEMISTRY_PROMPT, NUMERIC, PAGE_SIZE,
                          QUERY_BUDGET, VIEW_BUDGET, Catalog)
from .llm_cli_transport import call

METHOD = "free_llm32_scientist"
SEED = 1525
BUDGETS = [333, 365, 397]
LIMITS = dict(query_budget=QUERY_BUDGET, view_budget=VIEW_BUDGET,
              call_budget=CALL_BUDGET, page_size=PAGE_SIZE)
ROLES = ["hypothesis_test", "matched_control", "model_failure_probe", "coverage_probe",
         "condition_contrast", "exploratory"]
STATUSES = ["supported", "weakened", "contradicted", "unresolved"]
DESCRIPTOR_SCHEMA = list(NUMERIC) + ["id", "identity", "scaffold_group", "smiles", "scaffold",
    "functional_groups", "stereocenters", "pred_q10", "pred_center", "pred_q90", "coverage", "q_width"]
SELECTION_SCHEMA = {"choices": {"count": 32, "fields": ["id", "reason", "role", "evidence_ids", "hypothesis_id (optional)"]},
    "required": ["type", "packet_hash", "choices", "batch_strategy", "batch_rationale", "hypotheses",
                 "previous_hypothesis_updates", "feedback_interpretation", "unresolved_questions"],
    "roles": ROLES, "statuses": STATUSES}
SYSTEM_PROMPT = """You are a scientific experiment planner designing the next complete batch of
32 chromatography experiments to improve retrained QGeoGNN prediction on a fixed Row distribution.
You control all 32 choices from the legal U pool. There are NO pending LCMD points, quotas,
mandatory coverage maximization, uncertainty quotas, scaffold/new-molecule quotas, or fixed
exploitation/exploration ratio. Ignore high-coverage points if a stronger learning rationale exists.
Use only this packet, its trajectory-local memory, and the JSON query relay. No web, shell, MCP,
files, native tools, literature search, other trajectories or external conversation history.
No validation/test scores are available. Never request them. The target is RTv=RT*flow (mL).

Cards contain pred_q10, pred_center (MSE-trained central output, NOT a calibrated q50/median),
pred_q90 and q_width=q90-q10. Width is NOT calibrated epistemic uncertainty. coverage is the
percentile of nearest-labeled distance in raw 512D full-network central-output gradient sketch
geometry over the original outer pool; it is a geometric signal, NOT true prediction error.
Consider systematic frozen premeasurement errors, stereochemistry, enantiomer pairs, scaffold
substitution, matched molecular pairs, same/near molecule under IPA/flow changes, structure ×
condition interactions, prediction extremes, model blind spots, geometry and competing hypotheses.
Grouped experiments are encouraged when useful, but are not required. Distinguish observed
evidence, current model behavior, chemical prior and untested hypothesis. Do not assert mechanisms
as facts. Initial L333 observations have no premeasurement errors. For acquired experiments,
signed_error = frozen premeasurement central prediction - measured response, never post-fit error.
Use new feedback to explicitly reassess ALL prior hypotheses before planning the next batch.

Return exactly one JSON object per reply, either query or selection.
Query: {"type":"query","queries":[{"pool":"candidates","offset":0,"limit":24}]}.
This is get/search_candidates or get/search_observed through pool=candidates or observed.
Allowed filters: ids, same_identity_as, same_scaffold_as, similar_to, min_similarity,
smiles_contains, functional_group, ranges={numeric_field:[low,high]} (null unbounded).
Allowed sort_by: any numeric card field, q_width, coverage, pred_center, or similarity with
similar_to; observed additionally response, premeasurement_center, signed_error, abs_error.
order=asc/desc. Default ordering is a round-salted stable hash, independent of source row order.
Similarity is radius-2 2048-bit nonchiral Morgan Tanimoto; identity preserves stereochemistry.
Each response provides total_matches and next_offset: a page is NOT the whole pool.
All returned records are actually sent to you, without truncation. Initial 20 cards count as views.
Limits: QUERY_LIMIT queries (including invalid queries), VIEW_LIMIT distinct candidate views,
CALL_LIMIT replies, PAGE_LIMIT rows per page. Observed queries count toward queries, not candidate views.
Budget is shared across observed/candidate searches. At least one query required; choose only
32 distinct legal candidate IDs actually viewed. Plan searches to leave enough views for a full batch.

Final shape:
{"type":"selection","packet_hash":"copy exactly","batch_strategy":"...","batch_rationale":"...",
"choices":[{"id":"...","reason":"...","role":"hypothesis_test","hypothesis_id":"h1","evidence_ids":[]}],
"hypotheses":[{"id":"h1","claim":"...","evidence_ids":[],"alternative":"...","expected_sign":0,
"candidate_ids":["..."],"learning_value":"..."}],
"previous_hypothesis_updates":[{"id":"previous h id","status":"unresolved",
"supporting_observations":[],"contradicting_observations":[],"reason":"..."}],
"feedback_interpretation":"...","unresolved_questions":["..."]}
Zero to eight NEW hypotheses with unique IDs never reused from history. expected_sign -1/0/1
predicts frozen signed_error for linked chosen candidate_ids; zero means no directional claim.
Each hypothesis needs claim, alternative and learning_value. Each choice needs a nonempty reason;
role must be hypothesis_test, matched_control, model_failure_probe, coverage_probe,
condition_contrast or exploratory. Roles are analysis annotations, never quotas. hypothesis_id may
reference an old hypothesis or a new one. Evidence/support/contradiction IDs must be observed records
actually seen in relay or packet memory. For each old hypothesis give exactly one update:
supported / weakened / contradicted / unresolved, with reason and actual observed IDs (or empty
if unresolved). Do not confuse a signed-error direction match with proof of a chemical mechanism.
Round zero has empty updates. Explain how feedback changes the batch design; expose unresolved
questions. There is no automatic follow-up beyond this batch.
""".replace("QUERY_LIMIT", str(QUERY_BUDGET)).replace("VIEW_LIMIT", str(VIEW_BUDGET)).replace(
    "CALL_LIMIT", str(CALL_BUDGET)).replace("PAGE_LIMIT", str(PAGE_SIZE)) + CHEMISTRY_PROMPT


def opaque(i):
    return "r" + stable_hash(["ODH-LLM-v1", int(i)])[:12]


def build_memory(history, seed, method):
    states = {}
    for batch in history:
        if batch["seed"] != seed or batch["method"] != method:
            raise ValueError("cross-trajectory scientific memory forbidden")
        response = batch["response"]
        for update in response["previous_hypothesis_updates"]:
            if update["id"] not in states:
                raise ValueError("unknown historical hypothesis")
            states[update["id"]].update(copy.deepcopy(update))
        for h in response["hypotheses"]:
            if h["id"] in states:
                raise ValueError("hypothesis identity reused")
            states[h["id"]] = {**copy.deepcopy(h), "status": "unresolved",
                "supporting_observations": [], "contradicting_observations": [],
                "origin_round": batch["round"]}
    observations = [o for b in history for o in b["observations"]]
    return {"seed": seed, "method": method, "previous_hypotheses": list(states.values()),
            "recent_batches": copy.deepcopy(history[-3:]),
            "high_error_observations": sorted(copy.deepcopy(observations),
                key=lambda o: (-o["abs_error"], o["id"]))[:16],
            "unresolved_questions": history[-1]["response"]["unresolved_questions"] if history else []}


class FreeCatalog(Catalog):
    def __init__(self, features, fingerprints, labeled, unlabeled, predictions, coverage, observations,
                 *, salt, memory, pending=()):
        if pending:
            raise ValueError("Free arm cannot contain pending points")
        super().__init__(features, fingerprints, labeled, unlabeled, [], predictions, coverage,
                         observations, salt=salt, chemical=True, selection_count=32)
        self.memory = memory
        for batch in memory["recent_batches"]:
            self.observed_viewed.update(o["id"] for o in batch["observations"])
        self.observed_viewed.update(o["id"] for o in memory["high_error_observations"])
        if not self.observed_viewed <= set(self.pools["observed"]):
            raise ValueError("memory evidence outside observed pool")

    def query(self, query):
        if query.get("pool") == "pending":
            self.queries += 1
            raise ValueError("Free arm only supports candidates/observed")
        return super().query(query)

    def validate(self, value, packet_hash):
        selected = super().validate(value, packet_hash)
        for field in ("batch_strategy", "batch_rationale"):
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise ValueError("missing batch strategy/rationale")
        previous = {h["id"] for h in self.memory["previous_hypotheses"]}
        new = {h["id"] for h in value["hypotheses"]}
        if previous & new:
            raise ValueError("reuse previous hypothesis through updates, not new hypotheses")
        updates = value.get("previous_hypothesis_updates", [])
        if len(updates) != len(previous) or {u["id"] for u in updates} != previous:
            raise ValueError("every previous hypothesis needs exactly one update")
        for update in updates:
            if update.get("status") not in STATUSES or not update.get("reason"):
                raise ValueError("invalid hypothesis update")
            for field in ("supporting_observations", "contradicting_observations"):
                if field not in update or not set(update[field]) <= self.observed_viewed:
                    raise ValueError("update cites unobserved evidence")
        for choice in value["choices"]:
            if choice.get("role") not in ROLES:
                raise ValueError("invalid analysis role")
            if choice.get("hypothesis_id") and choice["hypothesis_id"] not in previous | new:
                raise ValueError("unknown choice hypothesis")
        if not isinstance(value.get("unresolved_questions"), list) or not all(
                isinstance(q, str) for q in value["unresolved_questions"]):
            raise ValueError("unresolved questions must be strings")
        return selected


def packet(catalog, memory, round_index):
    value = {"seed": memory["seed"], "method": METHOD, "round": round_index,
             "selection_count": 32, "pending_lcmd_ids": [], "pending_count": 0,
             "limits": LIMITS, "memory": memory, "initial_cards": catalog.initial(),
             "candidate_count": len(catalog.pools["candidates"]),
             "observed_count": len(catalog.pools["observed"])}
    assert_packet(value)
    value["packet_hash"] = stable_hash(value)
    return value


def assert_packet(value):
    allowed = {"seed", "method", "round", "selection_count", "pending_lcmd_ids", "pending_count",
               "limits", "memory", "initial_cards", "candidate_count", "observed_count", "packet_hash"}
    if set(value) - allowed or value["pending_count"] != 0 or value["pending_lcmd_ids"]:
        raise ValueError("packet contains forbidden state")
    for card in value["initial_cards"]:
        if set(card) - set(DESCRIPTOR_SCHEMA):
            raise ValueError("candidate card contains forbidden data")


def parse_answer(answer):
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value, end = json.JSONDecoder().raw_decode(text)
    if text[end:].strip() and json.loads(text[end:]) != value:
        raise ValueError("multiple different responses")
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def select(packet_value, catalog, directory, config, transport=call):
    """Replay all receipts exactly; never silently retry an ambiguous remote call."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(packet_value, ensure_ascii=False)}]
    for turn in range(CALL_BUDGET):
        request = {"messages": messages, "config": config}
        path = directory / f"turn_{turn:02d}.json"
        intent = directory / f"request_{turn:02d}.json"
        digest = stable_hash(request)
        if path.exists():
            receipt = read_json(path)
            saved = read_json(intent)
            if saved != {"request_sha256": digest} or receipt["request_sha256"] != digest:
                raise RuntimeError("resumed request hash mismatch")
            if receipt["answer_sha256"] != stable_hash(receipt["answer"]):
                raise RuntimeError("receipt answer changed")
        else:
            if intent.exists():
                raise RuntimeError("ambiguous in-flight provider call; stop, audit, never auto-retry")
            write_once(intent, {"request_sha256": digest})
            answer, provenance = transport(messages, config)
            receipt = {"request_sha256": digest, "answer": answer, "answer_sha256": stable_hash(answer),
                       "provenance": {**provenance, "requested_configuration": config}}
            write_once(path, receipt)
        answer = receipt["answer"]
        try:
            value = parse_answer(answer)
            if value.get("type") == "selection":
                selected = catalog.validate(value, packet_value["packet_hash"])
                saved = {"selected": selected, "response": value, "queries": catalog.queries,
                         "calls": turn + 1, "viewed": sorted(catalog.viewed),
                         "observed_viewed": sorted(catalog.observed_viewed),
                         "receipts": {p.name: sha(p) for p in directory.glob('turn_*.json')}}
                write_once(directory / "selection.json", saved)
                return saved
            if value.get("type") != "query" or not isinstance(value.get("queries"), list) or not value["queries"]:
                raise ValueError("return query or selection JSON")
            results = []
            for query in value["queries"]:
                if catalog.queries >= QUERY_BUDGET:
                    results.append({"error": "query budget exhausted; select from viewed records"})
                    break
                try:
                    results.append(catalog.query(query))
                except (ValueError, KeyError, TypeError) as error:
                    results.append({"error": str(error)})
            reply = {"query_results": results}
        except (ValueError, KeyError, TypeError) as error:
            reply = {"error": str(error), "instruction": "correct JSON within the remaining call budget"}
        reply.update(remaining_queries=QUERY_BUDGET - catalog.queries,
                     remaining_calls=CALL_BUDGET - turn - 1,
                     remaining_candidate_views=VIEW_BUDGET - len(catalog.viewed))
        write_once(directory / f"relay_{turn:02d}.json", reply)
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": json.dumps(reply, ensure_ascii=False)}]
    raise RuntimeError("selection not completed within frozen call budget")
