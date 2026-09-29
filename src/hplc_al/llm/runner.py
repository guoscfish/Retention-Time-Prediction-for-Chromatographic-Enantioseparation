"""V2 host orchestration: freeze -> committed seal -> labels -> scratch fit."""

import fcntl
import hashlib
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..common import (
    ROOT,
    SOURCE,
    ids_hash,
    read_json,
    sha,
    stable_hash,
    verify_files,
    write_once,
)
from ..data import load_graphs, predict
from ..gradient import extract
from ..lcmd_confirmation import _training
from ..protocol import RestrictedLabelStore, role_ids, transition
from ..runner import assert_environment
from ..training import fit, load_model
from .catalog import coverage_percentile, metadata
from .execution import attempt_summary, heartbeat, log, training_detail
from .full_pool import BUDGETS, FIELDS, LIMITS, METHOD, ROUNDS, SEED, STUDY_ID, opaque
from .memory import OBS_FIELDS, build_memory, feedback
from .planner import ARBITRATE_PROMPT, SCREEN_PROMPT, plan
from .reporting import evaluate, label_aulc, selection_diagnostics
from .responses_transport import call, require_key, settings

STUDY = ROOT / "studies/active_learning" / STUDY_ID
BASELINE = ROOT / "studies/active_learning/odh_lcmd_confirmation_v2"
V1 = ROOT / "studies/active_learning/odh_free_llm32_scientist_v1"
V1_SUCCESS = V1 / "provenance/successful_execution"
TRAJECTORY = f"{STUDY_ID}/{SEED}/{METHOD}"
# User-requested registration target, NEVER a runtime configuration fallback.
EXPECTED_CONFIG = {
    "provider_id": "token4research",
    "model": "gpt-6-astra",
    "reasoning_effort": "high",
    "base_url": "https://token4research.cn",
    "wire_api": "responses",
    "env_key": "TOKEN4RESEARCH_API_KEY",
}


def runtime():
    return STUDY / f"runtime/seed_{SEED}/{METHOD}"


@contextmanager
def exclusive():
    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another V2 host process is running") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def bind(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_record(path):
    record = read_json(path)
    verify_files(ROOT, record["files"])
    return record


def source_hashes():
    paths = list((ROOT / "src/hplc_al").rglob("*.py"))
    paths += [
        ROOT / n
        for n in (
            "src/reproduction/official.py",
            "code/Single_column_prediction.py",
            "code/compound_tools.py",
            "scripts/run_hplc_fullpool_v2.py",
            "scripts/preflight_llm_responses.py",
        )
    ]
    return bind(sorted(paths))


def verify_execution_amendment(protocol):
    """Preserve the original protocol; accept only the user's sealed ops exception."""
    current = source_hashes()
    if current == protocol["source_hashes"]:
        return False
    amendment = read_json(STUDY / "execution_amendment.json")
    allowed = {
        "src/hplc_al/llm/execution.py",
        "src/hplc_al/llm/runner.py",
        "src/hplc_al/llm/planner.py",
        "src/hplc_al/llm/continuous.py",
        "src/hplc_al/llm/responses_transport.py",
        "scripts/run_hplc_fullpool_v2.py",
        "scripts/preflight_llm_responses.py",
    }
    old = protocol["source_hashes"]
    changed = {p for p in set(old) | set(current) if old.get(p) != current.get(p)}
    if (
        amendment["protocol_sha256"] != sha(STUDY / "protocol.json")
        or amendment["original_source_hashes"] != old
        or amendment["source_hashes"] != current
        or not changed <= allowed
        or amendment["authorization"] != "允许 V2 仅修改日志和重试，复用已有结果"
        or amendment["max_attempts_per_scientific_request"] != 5
        or amendment["scientific_request_payload_changed"] is not False
    ):
        raise RuntimeError(
            "execution amendment/source mismatch; no scientific execution"
        )
    gate = read_json(STUDY / "execution_test_gate.json")
    if gate["status"] != "PASS" or gate["source_hashes"] != current:
        raise RuntimeError("passing tests for execution amendment required")
    verify_files(ROOT, gate["evidence"])
    verify_files(ROOT, gate["test_hashes"])
    return True


def prepare():
    """Offline registration against existing sealed L333; no target CSV parsing."""
    partition = read_json(BASELINE / "splits/partition.json")
    # Recheck on every entry, including resumes, before any label-store access.
    # load_graphs also checks inputs, but advance reaches it only after feedback.
    if sha(SOURCE) != partition["source_sha256"]:
        raise RuntimeError("source data drift; no label access permitted")
    if (STUDY / "protocol.json").exists():
        protocol = read_json(STUDY / "protocol.json")
        verify_execution_amendment(protocol)
        if protocol["environment"] != assert_environment():
            raise RuntimeError(
                "protocol/source/environment drift; stop and register V3"
            )
        verify_files(ROOT, protocol["reuse_hashes"])
        if (
            sha(STUDY / "protocol.json")
            != read_json(STUDY / "protocol_freeze.json")["sha256"]
        ):
            raise RuntimeError("protocol seal mismatch")
        return protocol, partition
    old = read_json(V1_SUCCESS / "protocol.json")
    base_protocol = read_json(BASELINE / "protocol.json")
    initial = BASELINE / f"shared/seed_{SEED}/fit"
    record = read_json(initial / "fit.json")
    verify_files(initial, record["files"])
    inp = read_json(initial / "input.json")
    labeled, valid = role_ids(partition, "l0"), role_ids(partition, "validation")
    if (
        len(labeled) != 333
        or inp["labeled"] != labeled
        or inp["validation"] != valid
        or inp["config"] != _training(SEED)
        or old["training"] != _training(SEED)
        or record["checkpoint_hash"] != old["l333_checkpoint_sha256"]
        or ids_hash(labeled) != old["l333_ids_sha256"]
        or stable_hash(partition) != old["partition_sha256"]
        or sha(SOURCE) != partition["source_sha256"]
    ):
        raise RuntimeError("L333 baseline contract mismatch")
    environment = assert_environment()
    if environment != old["environment"] or environment != base_protocol["environment"]:
        raise RuntimeError("baseline environment mismatch")
    for name in (
        "src/hplc_al/training.py",
        "src/hplc_al/data.py",
        "src/hplc_al/gradient.py",
        "src/hplc_al/deterministic_shuffle.py",
        "src/reproduction/official.py",
        "code/Single_column_prediction.py",
        "code/compound_tools.py",
    ):
        if sha(ROOT / name) != old["source_hashes"][name]:
            raise RuntimeError("baseline predictor/training implementation changed")
    protocol = {
        "study": STUDY_ID,
        "seed": SEED,
        "method": TRAJECTORY,
        "budgets": BUDGETS,
        "batch_size": 32,
        "target": old["target"],
        "training": _training(SEED),
        "partition_sha256": stable_hash(partition),
        "l333_ids_sha256": ids_hash(labeled),
        "l333_checkpoint_sha256": record["checkpoint_hash"],
        "fixed_denominator": old["frozen_l333_target_population_sd"],
        "limits": LIMITS,
        "transport_registration": EXPECTED_CONFIG,
        "environment": environment,
        "screen_prompt": SCREEN_PROMPT,
        "arbitration_prompt": ARBITRATE_PROMPT,
        "candidate_fields": list(FIELDS),
        "presentation": {
            "float_decimal_places": 6,
            "full_precision_host_artifacts": True,
            "codec": "value_table + record_groups; bijective opaque group aliases; exact grouped counts",
        },
        "memory_policy": {
            "recent_full_batches": 2,
            "all_observed_measurements_and_frozen_errors": True,
            "all_previous_hypothesis_states": True,
        },
        "source_hashes": source_hashes(),
        "reuse_hashes": bind(
            [
                BASELINE / "protocol.json",
                BASELINE / "splits/partition.json",
                initial / "input.json",
                initial / "fit.json",
                V1_SUCCESS / "protocol.json",
                V1 / "results/validation_metrics.json",
                V1 / "provenance/relocation_manifest.json",
            ]
            + [initial / n for n in record["files"]]
        ),
        "test_labels": 0,
        "stop": f"L{BUDGETS[-1]}; {ROUNDS} acquisitions only",
        "context_limit_note": "Conservative local serialization budget; provider context capacity not verified by small preflight.",
        "immutability": "Any scientific response locks protocol; protocol bug requires V3, never nested revision.",
    }
    write_once(STUDY / "protocol.json", protocol)
    write_once(STUDY / "protocol_freeze.json", {"sha256": sha(STUDY / "protocol.json")})
    return protocol, partition


def require_git_commit(directory):
    path = directory / "selection_seal.json"
    sealed = verify_record(path)
    if sealed["protocol_sha256"] != sha(STUDY / "protocol.json"):
        raise RuntimeError("protocol/selection mismatch")
    # Commit protocol and seal; seal hashes all transitive request, state and selection files.
    commit_paths = [path, STUDY / "protocol.json", STUDY / "protocol_freeze.json"]
    if (STUDY / "execution_amendment.json").exists():
        commit_paths += [
            STUDY / "execution_amendment.json",
            STUDY / "execution_test_gate.json",
        ]
    for p in commit_paths:
        content = subprocess.check_output(
            ["git", "show", f"HEAD:{p.relative_to(ROOT)}"], cwd=ROOT
        )
        if hashlib.sha256(content).hexdigest() != sha(p):
            raise RuntimeError("Git/protocol seal must be committed before reveal")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def state(partition, round_index):
    labeled, unlabeled = role_ids(partition, "l0"), role_ids(partition, "u0")
    history = []
    store = RestrictedLabelStore(
        partition, STUDY / "label_access_audit.csv", TRAJECTORY
    )
    for r in range(round_index):
        directory = runtime() / f"round_{r}"
        verify_record(directory / "complete.json")
        require_git_commit(directory)
        selection = read_json(directory / "selection.json")
        store.commit_selection(directory / "selection.json")
        labeled, unlabeled = transition(labeled, unlabeled, selection["selected"])
        history.append(read_json(directory / "feedback.json"))
    return labeled, unlabeled, history, store


def initial_state():
    """Read sealed label-free L333 outputs and authorized L333 observations from V1.

    This is exact shared-checkpoint computational reuse, not V1 acquisition/memory reuse.
    """
    directory = V1_SUCCESS / f"runtime/seed_{SEED}/free_llm32_scientist/round_0"
    mapping = read_json(V1 / "provenance/relocation_manifest.json")
    for name in ("prepared.json", "model_state.npz", "observed.json"):
        path = directory / name
        item = next(
            v for v in mapping.values() if v["path"] == str(path.relative_to(ROOT))
        )
        if sha(path) != item["sha256"]:
            raise RuntimeError("L333 reused state changed")
    prepared = read_json(directory / "prepared.json")
    with np.load(directory / "model_state.npz") as saved:
        ids, prediction, coverage = (
            saved["ids"].tolist(),
            saved["predictions"],
            saved["coverage"],
        )
    observed = {int(k): v for k, v in read_json(directory / "observed.json").items()}
    return prepared, ids, prediction, coverage, observed, directory


def make_round(round_index):
    if round_index not in range(ROUNDS):
        raise ValueError(f"hard stop: {ROUNDS} acquisitions, L{BUDGETS[-1]}")
    protocol, partition = prepare()
    labeled, unlabeled, history, store = state(partition, round_index)
    directory = runtime() / f"round_{round_index}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "prepared.json").exists():
        prepared = verify_record(directory / "prepared.json")
        if (
            prepared["labeled"] != labeled
            or prepared["unlabeled"] != unlabeled
            or prepared["protocol_sha256"] != sha(STUDY / "protocol.json")
        ):
            raise RuntimeError("prepared state drift")
        packet = read_json(directory / "packet.json")
        return protocol, partition, directory, packet, labeled, unlabeled
    features, _ = metadata(sorted(labeled + unlabeled))
    if round_index == 0:
        prepared, ids, prediction, coverage, observations, reused = initial_state()
        if (
            prepared["checkpoint_hash"] != protocol["l333_checkpoint_sha256"]
            or prepared["labeled_ids"] != labeled
            or prepared["unlabeled_ids"] != unlabeled
            or set(observations) != set(labeled)
            or any(set(v) != {"response"} for v in observations.values())
        ):
            raise RuntimeError("shared L333 state mismatch")
        reuse_files = bind(
            [reused / n for n in ("prepared.json", "model_state.npz", "observed.json")]
        )
    else:
        source = runtime() / f"round_{round_index - 1}/fit"
        fit_record = read_json(source / "fit.json")
        verify_files(source, fit_record["files"])
        graphs = load_graphs(partition)
        model = load_model(source / fit_record["checkpoint_path"])
        ids = sorted(labeled + unlabeled)
        prediction = predict(model, graphs, ids)
        phi, _ = extract(model, graphs, ids)
        positions = {i: j for j, i in enumerate(ids)}
        coverage = coverage_percentile(phi, [positions[i] for i in labeled])
        # Only already-authorized trajectory observations may enter the packet.
        truth = store.reveal(labeled, "fit")
        observations = {i: {"response": float(y)} for i, y in zip(labeled, truth)}
        for batch in history:
            for row in batch["observations"]:
                i = next(i for i in labeled if opaque(i) == row["id"])
                observations[i] = row
        reuse_files = bind(
            [source / "fit.json"] + [source / n for n in fit_record["files"]]
        )
    if set(ids) != set(labeled + unlabeled) or len(ids) != len(set(ids)):
        raise RuntimeError("prediction alignment mismatch")
    positions = {i: j for j, i in enumerate(ids)}
    cards = []
    for i in unlabeled:
        pred = list(map(float, prediction[positions[i]]))
        row = {k: v for k, v in features[i].items() if k in FIELDS}
        cards.append(
            {
                **row,
                "id": opaque(i),
                "pred_q10": pred[0],
                "pred_center": pred[1],
                "pred_q90": pred[2],
                "q_width": pred[2] - pred[0],
                "coverage": float(coverage[positions[i]]),
            }
        )
    observed = [
        {
            **{k: v for k, v in features[i].items() if k in OBS_FIELDS},
            **observations[i],
            "id": opaque(i),
        }
        for i in labeled
    ]
    packet = {
        "cards": cards,
        "legal_ids": [opaque(i) for i in unlabeled],
        "observed": observed,
        "memory": build_memory(history, SEED, TRAJECTORY),
        "round_index": round_index,
    }
    write_once(directory / "packet.json", packet)
    write_once(
        directory / "prepared.json",
        {
            "labeled": labeled,
            "unlabeled": unlabeled,
            "protocol_sha256": sha(STUDY / "protocol.json"),
            "files": {**reuse_files, **bind([directory / "packet.json"])},
        },
    )
    return protocol, partition, directory, packet, labeled, unlabeled


def run_selection(round_index):
    with exclusive():
        config = settings()
        require_key(config)
        if config != EXPECTED_CONFIG:
            raise RuntimeError(
                "configured provider/model differs from registered V2; no fallback"
            )
        log(
            f"Round {round_index + 1}/{ROUNDS}: preparing candidate predictions and local memory"
        )
        with heartbeat("Candidate preparation"):
            protocol, _, directory, packet, labeled, unlabeled = make_round(round_index)
        if (directory / "selection_seal.json").exists():
            verify_record(directory / "selection_seal.json")
            return read_json(directory / "selection.json")
        preflight = read_json(STUDY / "transport_preflight.json")
        if (
            preflight.get("status") != "PASS"
            or preflight["config"] != config
            or preflight["config_sha256"] != stable_hash(config)
        ):
            raise RuntimeError("matching content-free Responses preflight required")
        gate = read_json(STUDY / "test_gate.json")
        dry = read_json(STUDY / "dry_run.json")
        stress = read_json(
            ROOT / "docs/repository/verification/context_stress_review.json"
        )
        verify_files(ROOT, gate["evidence"])
        if (
            gate["status"] != "PASS"
            or gate["source_hashes"] != protocol["source_hashes"]
            or dry["status"] != "PASS"
            or stress["status"] != "PASS"
            or stress["protocol_sha256"] != sha(STUDY / "protocol.json")
        ):
            raise RuntimeError("passing tests and dry run for frozen code required")
        if dry["protocol_sha256"] != sha(STUDY / "protocol.json"):
            raise RuntimeError("dry run protocol mismatch")
        write_once(
            STUDY / "SCIENTIFIC_LOCK.json",
            {
                "protocol_sha256": sha(STUDY / "protocol.json"),
                "rule": "Never alter this protocol; protocol bug requires V3.",
            },
        )
        saved = plan(
            **packet,
            directory=directory / "llm",
            config=config,
            limits=protocol["limits"],
            transport=call,
            retry_requests=verify_execution_amendment(protocol),
        )
        row_ids = {opaque(i): i for i in unlabeled}
        chosen = [row_ids[i] for i in saved["selected_ids"]]
        lookup = {c["id"]: c for c in packet["cards"]}
        selection = {
            "study": STUDY_ID,
            "seed": SEED,
            "method": TRAJECTORY,
            "round": round_index,
            "selected": chosen,
            "selected_hash": stable_hash(chosen),
            "L_hash": ids_hash(labeled),
            "U_hash": ids_hash(unlabeled),
            "response": saved["response"],
            "prediction_before_measurement": {
                str(i): [
                    lookup[opaque(i)][k]
                    for k in ("pred_q10", "pred_center", "pred_q90")
                ]
                for i in chosen
            },
        }
        write_once(directory / "selection.json", selection)
        write_once(
            directory / "diagnostics.json",
            selection_diagnostics(packet["cards"], saved),
        )
        files = list((directory / "llm").glob("*.json")) + [
            directory / n
            for n in (
                "selection.json",
                "diagnostics.json",
                "packet.json",
                "prepared.json",
            )
        ]
        write_once(
            directory / "selection_seal.json",
            {
                "protocol_sha256": sha(STUDY / "protocol.json"),
                "files": {
                    **read_json(directory / "prepared.json")["files"],
                    **bind(files),
                },
            },
        )
        return selection


def advance(round_index):
    if round_index not in range(ROUNDS):
        raise ValueError(f"hard stop at L{BUDGETS[-1]}")
    with exclusive():
        protocol, partition = prepare()
        directory = runtime() / f"round_{round_index}"
        if (directory / "complete.json").exists():
            return verify_record(directory / "complete.json")
        commit = require_git_commit(directory)
        labeled, unlabeled, _, store = state(partition, round_index)
        selection = read_json(directory / "selection.json")
        store.commit_selection(directory / "selection.json")
        labeled, _ = transition(labeled, unlabeled, selection["selected"])
        if not (directory / "feedback_seal.json").exists():
            truth = store.reveal(selection["selected"], "fit")
            features, _ = metadata(selection["selected"])
            write_once(
                directory / "feedback.json", feedback(selection, truth, features)
            )
            write_once(
                directory / "feedback_seal.json",
                {
                    "selection_commit": commit,
                    "files": bind([directory / "feedback.json"]),
                },
            )
        verify_record(directory / "feedback_seal.json")
        truth = store.reveal(labeled, "fit")
        valid = role_ids(partition, "validation")
        valid_truth = RestrictedLabelStore(
            partition, STUDY / "label_access_audit.csv", TRAJECTORY + "/trainer"
        ).reveal(valid, "validation")
        log(
            f"Round {round_index + 1}: feedback saved; preparing scratch fit with L{len(labeled)}"
        )
        with heartbeat("Loading training graphs"):
            graphs = load_graphs(partition)
        source = directory / "fit"
        with heartbeat(
            f"Round {round_index + 1} scratch training",
            lambda: training_detail(directory),
        ):
            record = fit(
                graphs,
                labeled,
                truth,
                valid,
                valid_truth,
                protocol["training"],
                source,
                {
                    "study": STUDY_ID,
                    "seed": SEED,
                    "method": METHOD,
                    "round": round_index + 1,
                },
            )
        with np.load(
            source
            / Path(record["checkpoint_path"]).parent
            / "validation_predictions.npz"
        ) as saved:
            if saved["ids"].tolist() != valid:
                raise RuntimeError("validation alignment mismatch")
            metric = evaluate(
                saved["predictions"][:, 1],
                valid_truth,
                protocol["fixed_denominator"],
                len(labeled),
            )
        write_once(directory / "validation.json", metric)
        log(
            f"Round {round_index + 1}: scratch fit finished; validation metrics saved at {directory / 'validation.json'}"
        )
        files = bind(
            [source / n for n in record["files"]]
            + [
                source / "fit.json",
                directory / "validation.json",
                directory / "feedback.json",
                directory / "feedback_seal.json",
                directory / "selection_seal.json",
            ]
        )
        write_once(
            directory / "complete.json",
            {"budget": len(labeled), "test_labels": 0, "files": files},
        )
        return read_json(directory / "complete.json")


def report():
    protocol, partition = prepare()
    state(partition, ROUNDS)
    # Validation metrics from the frozen baseline report; no source target read.
    baseline = read_json(V1 / "results/validation_metrics.json")
    first = next(r for r in baseline if r["budget"] == 333)
    records = [
        {
            k: first[k]
            for k in ("budget", "rmse", "mae", "r2", "nrmse", "fixed_denominator")
        }
    ]
    records += [
        read_json(runtime() / f"round_{r}/validation.json") for r in range(ROUNDS)
    ]
    result = {
        "status": f"COMPLETE_PHASE1_L{BUDGETS[-1]}",
        "study": STUDY_ID,
        "validation": records,
        "label_aulc": label_aulc(records),
        "test_labels": 0,
        "diagnostics": [
            read_json(runtime() / f"round_{r}/diagnostics.json") for r in range(ROUNDS)
        ],
    }
    write_once(STUDY / "results/summary.json", result)
    write_once(STUDY / "results/execution_attempts.json", attempt_summary(runtime()))
    return result
