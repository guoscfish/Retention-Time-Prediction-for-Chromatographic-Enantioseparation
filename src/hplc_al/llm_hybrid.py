"""ODH LCMD16 + LLM16 development trajectory.

The LLM is a selector only: labels are obtained through RestrictedLabelStore, and
all numerical/chemical cards are assembled from label-free graph inputs plus the
authorized measured history for this trajectory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .acquisition import lcmd_tp_select
from .common import ROOT, atomic_json, read_json, sha, stable_hash, ids_hash, metrics
from .data import load_graphs, predict
from .gradient import extract
from .llm_catalog import CHEMISTRY_PROMPT, COMMON_PROMPT, MASKED_PROMPT, Catalog, metadata, coverage_percentile
from .llm_transport import call, settings
from .protocol import RestrictedLabelStore, role_ids, transition
from .training import fit, load_model
from .transfer_v2 import BUDGETS, L0, OUTER_TRAIN, transfer_partition

STUDY = ROOT / "studies/active_learning/odh_llm_hybrid_v8"
SOURCE_STUDY = ROOT / "studies/active_learning/odh_lcmd_confirmation_v2"
SEEDS = (1525, 2525)
METHODS = ("chemical_llm", "masked_llm")
BATCH = 16
LLM_MODEL = "gpt-6-sol"


def _training(seed):
    return {"maximum_epochs": 500, "patience": 100, "min_delta": 0.0,
            "initialization_seed": seed, "training_seed": seed, "learning_rate": 0.001,
            "weight_decay": 1e-5, "optimizer": "Adam", "batch_size": 256,
            "shuffle": "deterministic_each_epoch", "scheduler": False,
            "loss": "original_complete", "checkpoint": "best_validation_central_MSE",
            "scratch": True, "wall_time_limit_seconds": None}


def _write(path, value):
    atomic_json(path, value)


def _parse_answer(text):
    """Parse one JSON object while tolerating provider-level duplicate output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        try:
            duplicate = json.loads(remainder)
        except json.JSONDecodeError as error:
            raise RuntimeError("LLM returned JSON plus non-JSON commentary") from error
        if duplicate != value:
            raise RuntimeError("LLM returned multiple different JSON objects")
    return value


def _packet(seed, method, round_index, labeled, pending, candidates, predictions, coverage,
            features, fingerprints, observations, history):
    catalog = Catalog(features, fingerprints, labeled, candidates + pending, pending,
                      predictions, coverage, observations, salt=f"{seed}/{method}/{round_index}",
                      chemical=method == "chemical_llm")
    return catalog, {"seed": seed, "method": method, "round": round_index,
                     "selection_count": 16, "packet_version": "odh_llm_hybrid_v1",
                     "pending_lcmd_ids": [catalog.to_id[i] for i in pending],
                     "history": history, "initial_cards": catalog.initial(),
                     "overview": {"candidate_count": len(candidates), "pending_count": len(pending),
                                  "observed_count": len(labeled), "pool_query_budget": 24,
                                  "view_budget": 480}}


def _llm_select(packet, catalog, directory, method):
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "selection.json"
    if completed.exists():
        saved = read_json(completed)
        selected = [int(i) for i in saved["selected"]]
        if len(selected) != 16 or len(set(selected)) != 16:
            raise RuntimeError("completed LLM selection is malformed")
        return selected, saved["response"]
    config = settings()
    if config["model"] != LLM_MODEL or config["reasoning_effort"] != "high":
        raise RuntimeError("LLM transport differs from frozen study configuration")
    prompt = COMMON_PROMPT + (CHEMISTRY_PROMPT if method == "chemical_llm" else MASKED_PROMPT)
    packet_text = json.dumps(packet, ensure_ascii=False)
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": packet_text}]
    for turn in range(28):
        request_hash = stable_hash({"messages": messages, "config": config})
        receipt_path = directory / f"turn_{turn:02d}.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            # A resumed run may tighten the query cap without changing the
            # scientific packet; retain the original request hash in the receipt.
        else:
            answer, provenance = call(messages, config)
            receipt = {"request_sha256": request_hash, "answer": answer,
                       "provenance": provenance}
            _write(receipt_path, receipt)
        try:
            value = _parse_answer(receipt["answer"])
        except (json.JSONDecodeError, RuntimeError):
            if turn >= 27:
                raise RuntimeError("LLM did not produce JSON within the answer budget")
            messages += [{"role": "assistant", "content": receipt["answer"]},
                         {"role": "user", "content": json.dumps({
                             "error": "your previous response was not a JSON object; return only one JSON query or final selection",
                             "remaining_queries": catalog.query_budget - catalog.queries,
                             "remaining_answers": 27 - turn}, ensure_ascii=False)}]
            continue
        if value.get("type") == "selection":
            selected = catalog.validate(value, packet["packet_hash"])
            _write(directory / "selection.json", {"selected": selected, "response": value,
                                                    "queries": catalog.queries,
                                                    "viewed": sorted(catalog.viewed),
                                                    "observed_viewed": sorted(catalog.observed_viewed),
                                                    "model": config})
            return selected, value
        if value.get("type") != "query" or not value.get("queries"):
            raise RuntimeError("LLM must return query or selection JSON")
        if catalog.queries + len(value["queries"]) > catalog.query_budget:
            messages += [{"role": "assistant", "content": receipt["answer"]},
                         {"role": "user", "content": json.dumps({
                             "error": "query budget exhausted; return the final selection now using only viewed records",
                             "remaining_queries": 0, "remaining_answers": 27 - turn}, ensure_ascii=False)}]
            continue
        results = [catalog.query(query) for query in value["queries"]]
        # Keep provider context bounded: full records remain on disk, while the
        # next turn receives only a compact page and the current packet.
        compact_results = [{**result, "records": result["records"][:8]} for result in results]
        _write(directory / f"query_{turn:02d}.json", {"queries": value["queries"], "results": results})
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": packet_text},
                    {"role": "assistant", "content": receipt["answer"]},
                    {"role": "user", "content": json.dumps({"query_results": compact_results,
                         "remaining_queries": catalog.query_budget - catalog.queries,
                         "remaining_answers": 27 - turn}, ensure_ascii=False)}]
    raise RuntimeError("LLM did not return a selection")


def _lcmd16(model, graphs, labeled, unlabeled):
    outer = sorted(labeled + unlabeled)
    phi, audit = extract(model, graphs, outer)
    positions = {sample_id: index for index, sample_id in enumerate(outer)}
    result = lcmd_tp_select(phi[[positions[i] for i in unlabeled]],
                            phi[[positions[i] for i in labeled]], BATCH)
    return [unlabeled[int(i)] for i in result.selected_pool_positions], phi, audit


def _initial_fit(graphs, partition, seed):
    valid = role_ids(partition, "validation")
    l0 = role_ids(partition, "l0")
    valid_store = RestrictedLabelStore(partition, STUDY / "label_access_audit.csv", f"{seed}/valid")
    valid_truth = valid_store.reveal(valid, "validation")
    store = RestrictedLabelStore(partition, STUDY / "label_access_audit.csv", f"{seed}/initial")
    truth = store.reveal(l0, "fit")
    source = SOURCE_STUDY / f"shared/seed_{seed}/fit"
    if (source / "fit.json").exists():
        record = read_json(source / "fit.json")
        model = load_model(source / record["checkpoint_path"])
        return model, truth, valid_truth, record
    directory = STUDY / f"shared/seed_{seed}/fit"
    record = fit(graphs, l0, truth, valid, valid_truth, _training(seed), directory,
                 {"study": STUDY.name, "seed": seed, "round": 0})
    return load_model(directory / record["checkpoint_path"]), truth, valid_truth, record


def run(seed=1525, stop_round=None):
    STUDY.mkdir(parents=True, exist_ok=True)
    # Copy the frozen row protocol from the established LCMD confirmation study.
    source_partition = read_json(SOURCE_STUDY / "splits/partition.json")
    partition = source_partition
    l0, u0 = role_ids(partition, "l0"), role_ids(partition, "u0")
    valid = role_ids(partition, "validation")
    if len(l0) != L0 or len(l0) + len(u0) != OUTER_TRAIN:
        raise RuntimeError("unexpected ODH transfer partition")
    graphs = load_graphs(partition)
    features, fingerprints = metadata(sorted(l0 + u0))
    protocol = {"study": STUDY.name, "version": "odh_llm_hybrid_v1", "seed": seed,
                "method": list(METHODS), "budgets": list(BUDGETS), "batch": BATCH,
                "llm": settings(), "objective": "LCMD16 + LLM16; chemical information vs masked ablation",
                "target": "RTv=RT*flow; central predictor output[:,1]",
                "partition_sha256": stable_hash(partition), "test_truth_access_count": 0}
    _write(STUDY / "protocol.json", protocol)
    for method in METHODS:
        method_root = STUDY / f"runtime/seed_{seed}/{method}"
        store = RestrictedLabelStore(partition, STUDY / "label_access_audit.csv", f"{seed}/{method}")
        labeled, unlabeled = list(l0), list(u0)
        history = []
        model, initial_truth, valid_truth, initial_fit = _initial_fit(graphs, partition, seed)
        scale = float(np.std(initial_truth.astype(np.float64), ddof=0))
        for round_index, budget in enumerate(BUDGETS):
            if stop_round is not None and round_index >= stop_round:
                break
            directory = method_root / f"round_{round_index}"
            directory.mkdir(parents=True, exist_ok=True)
            prediction_ids = sorted(labeled + unlabeled)
            predictions = predict(model, graphs, prediction_ids)
            prediction_map = {i: predictions[j] for j, i in enumerate(prediction_ids)}
            selected_lcmd, phi, gradient_audit = _lcmd16(model, graphs, labeled, unlabeled)
            positions = {i: j for j, i in enumerate(prediction_ids)}
            coverage = dict(zip(prediction_ids, coverage_percentile(phi,
                [positions[i] for i in labeled])))
            observed_truth = store.reveal(labeled, "fit")
            observations = {}
            for i, response in zip(labeled, observed_truth):
                entry = {"response": float(response)}
                for record in history:
                    if i in record["selected"]:
                        entry["premeasurement_center"] = record["predictions"].get(str(i))
                        entry["signed_error"] = entry["premeasurement_center"] - float(response)
                        entry["abs_error"] = abs(entry["signed_error"])
                        break
                observations[i] = entry
            candidate_predictions = {i: prediction_map[i] for i in prediction_ids}
            pending = selected_lcmd
            candidates = sorted(set(unlabeled) - set(pending))
            catalog, packet = _packet(seed, method, round_index, labeled, pending, candidates,
                                      candidate_predictions, coverage, features, fingerprints,
                                      observations, history[-3:])
            packet["packet_hash"] = stable_hash({k: v for k, v in packet.items() if k != "packet_hash"})
            _write(directory / "packet.json", packet)
            if round_index < len(BUDGETS) - 1:
                llm_dir = directory / ("llm" if round_index < 2 else "llm_compact")
                selected_llm, response = _llm_select(packet, catalog, llm_dir, method)
                selected = pending + selected_llm
                selection = {"seed": seed, "method": f"{seed}/{method}", "arm": method,
                             "round": round_index,
                             "L_hash": ids_hash(labeled), "U_hash": ids_hash(unlabeled),
                             "budget": budget, "lcmd_selected": pending, "llm_selected": selected_llm,
                             "selected": selected, "selected_hash": stable_hash(selected),
                             "lcmd_gradient_audit": gradient_audit,
                             "llm_feedback": response.get("feedback_interpretation", ""),
                             "llm_hypotheses": response.get("hypotheses", []),
                             "prediction_before_measurement": {str(i): prediction_map[i].tolist() for i in selected},
                             "labeled_ids": labeled, "unlabeled_ids": unlabeled,
                             "scale": scale}
                _write(directory / "selection.json", selection)
                history.append({"round": round_index, "selected": selected,
                                "predictions": {str(i): float(prediction_map[i][1]) for i in selected},
                                "hypotheses": response.get("hypotheses", []),
                                "feedback": response.get("feedback_interpretation", "")})
                store.commit_selection(directory / "selection.json")
                labeled, unlabeled = transition(labeled, unlabeled, selected)
                truth = store.reveal(labeled, "fit")
                fit_dir = directory / "fit"
                fit_record = fit(graphs, labeled, truth, valid, valid_truth, _training(seed), fit_dir,
                                 {"study": STUDY.name, "seed": seed, "method": method, "round": round_index + 1})
                model = load_model(fit_dir / fit_record["checkpoint_path"])
            else:
                _write(directory / "terminal.json", {"budget": budget, "labeled_ids": labeled,
                                                       "unlabeled_ids": unlabeled, "scale": scale})
            print({"seed": seed, "method": method, "round": round_index,
                   "budget": budget, "labeled": len(labeled)}, flush=True)
    _write(STUDY / f"runtime/seed_{seed}/complete.json", {"status": "STOPPED_OR_COMPLETE",
                                                           "stop_round": stop_round})


if __name__ == "__main__":
    run()
