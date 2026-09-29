"""Trajectory-local observations and frozen premeasurement feedback."""

import copy
import math

from .full_pool import FIELDS, STUDY_ID, opaque

OBS_FIELDS = set(FIELDS) - {
    "pred_center",
    "q_width",
    "coverage",
    "pred_q10",
    "pred_q90",
}
ERROR_FIELDS = {
    "premeasurement_prediction",
    "premeasurement_center",
    "signed_error",
    "abs_error",
    "selection_reason",
    "hypothesis_id",
    "scientific_role",
}


def validate_observations(observed):
    ids = [o["id"] for o in observed]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate observed record")
    for row in observed:
        if set(row) - (OBS_FIELDS | ERROR_FIELDS | {"response"}):
            raise ValueError("observed record contains forbidden data")
        if not math.isfinite(row["response"]):
            raise ValueError("invalid observed response")
        if set(row) & ERROR_FIELDS:
            if not ERROR_FIELDS <= set(row):
                raise ValueError("incomplete frozen feedback")
            if (
                len(row["premeasurement_prediction"]) != 3
                or row["premeasurement_prediction"][1] != row["premeasurement_center"]
                or any(not math.isfinite(v) for v in row["premeasurement_prediction"])
            ):
                raise ValueError("invalid frozen prediction")
            error = row["premeasurement_center"] - row["response"]
            if not math.isclose(error, row["signed_error"]) or not math.isclose(
                abs(error), row["abs_error"]
            ):
                raise ValueError("feedback must use frozen premeasurement prediction")


def build_memory(history, seed, method, study=STUDY_ID):
    states = {}
    for round_index, batch in enumerate(history):
        if (batch["study"], batch["seed"], batch["method"], batch["round"]) != (
            study,
            seed,
            method,
            round_index,
        ):
            raise ValueError("cross-trajectory or noncontiguous memory forbidden")
        validate_observations(batch["observations"])
        response = batch["response"]
        for update in response["previous_hypothesis_updates"]:
            if update["id"] not in states:
                raise ValueError("unknown historical hypothesis")
            states[update["id"]].update(copy.deepcopy(update))
        for hypothesis in response["hypotheses"]:
            if hypothesis["id"] in states:
                raise ValueError("hypothesis identity reused")
            states[hypothesis["id"]] = {
                **copy.deepcopy(hypothesis),
                "status": "unresolved",
                "supporting_observations": [],
                "contradicting_observations": [],
                "origin_round": round_index,
            }
    observations = [o for b in history for o in b["observations"]]
    return {
        "study": study,
        "seed": seed,
        "method": method,
        "previous_hypotheses": list(states.values()),
        "recent_batches": copy.deepcopy(history[-2:]),
        "high_error_observations": sorted(
            copy.deepcopy(observations), key=lambda o: (-o["abs_error"], o["id"])
        )[:16],
        "unresolved_questions": history[-1]["response"]["unresolved_questions"]
        if history
        else [],
    }


def feedback(selection, truth, features):
    if len(truth) != len(selection["selected"]):
        raise ValueError("feedback count mismatch")
    choices = {c["id"]: c for c in selection["response"]["choices"]}
    rows = []
    for i, measured in zip(selection["selected"], truth):
        before = selection["prediction_before_measurement"][str(i)]
        choice = choices[opaque(i)]
        row = {k: v for k, v in features[i].items() if k in OBS_FIELDS}
        error = float(before[1]) - float(measured)
        rows.append(
            {
                **row,
                "id": opaque(i),
                "premeasurement_prediction": before,
                "premeasurement_center": before[1],
                "response": float(measured),
                "signed_error": error,
                "abs_error": abs(error),
                "selection_reason": choice["reason"],
                "hypothesis_id": choice.get("hypothesis_id"),
                "scientific_role": choice["scientific_role"],
            }
        )
    validate_observations(rows)
    return {
        k: selection[k] for k in ("study", "seed", "method", "round", "response")
    } | {"observations": rows}
