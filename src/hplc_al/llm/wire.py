"""Compact scientific packets: opaque group aliases and reversible value interning."""

import re
from collections import Counter

from .responses_transport import encode


def aliases(cards, observed):
    # These hashes identify equality groups, not chemical measurements. Keep the
    # bijection in the sealed journal, and use the same names throughout a round.
    values = sorted(
        {
            r[k]
            for r in cards + observed
            for k in ("identity", "scaffold_group")
            if k in r and re.fullmatch(r"[0-9a-f]{16}", r[k])
        }
    )
    return {value: f"g{i}" for i, value in enumerate(values)}


def replace(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [replace(v, mapping) for v in value]
    if isinstance(value, dict):
        return {mapping.get(k, k): replace(v, mapping) for k, v in value.items()}
    return value


def record_tables(value):
    if isinstance(value, dict):
        return {k: record_tables(v) for k, v in value.items()}
    if not isinstance(value, list):
        return value
    rows = [record_tables(v) for v in value]
    if len(rows) < 3 or not all(isinstance(v, dict) for v in rows):
        return rows
    groups = {}
    for i, row in enumerate(rows):
        columns = tuple(sorted(row))
        group = groups.setdefault(
            columns, {"columns": list(columns), "positions": [], "rows": []}
        )
        group["positions"].append(i)
        group["rows"].append([row[k] for k in columns])
    packed = {"record_groups": list(groups.values())}
    return packed if len(encode(packed)) < len(encode(rows)) else rows


def restore_records(value):
    if isinstance(value, list):
        return [restore_records(v) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"record_groups"}:
        rows = {}
        for group in value["record_groups"]:
            for i, row in zip(group["positions"], group["rows"]):
                rows[i] = {k: restore_records(v) for k, v in zip(group["columns"], row)}
        return [rows[i] for i in range(len(rows))]
    return {k: restore_records(v) for k, v in value.items()}


def pack(value, mapping=None):
    value = record_tables(replace(value, mapping or {}))
    counts = Counter()
    originals = {}

    def count(v):
        if isinstance(v, (str, list, dict)):
            key = encode(v)
            counts[key] += 1
            originals[key] = v
        if isinstance(v, list):
            for item in v:
                count(item)
        elif isinstance(v, dict):
            for key, item in v.items():
                count(key)
                count(item)

    count(value)
    # Escape every literal starting with @, so decoding is unambiguous even for
    # model-written text. Definitions are literal and never recursively resolved.
    eligible = set()
    for key, n in counts.items():
        v = originals[key]
        if isinstance(v, str):
            if v.startswith("@") or (len(v) > 24 and n > 1) or (len(v) > 8 and n > 3):
                eligible.add(key)
        elif len(key) > 80 and n > 1:
            eligible.add(key)
    refs, definitions = {}, {}

    def compact(v):
        key = encode(v) if isinstance(v, (str, list, dict)) else None
        if key in eligible:
            if key not in refs:
                ref = f"@{len(refs)}"
                refs[key] = ref
                definitions[ref] = v  # Literal; no references inside definitions.
            return refs[key]
        if isinstance(v, list):
            return [compact(item) for item in v]
        if isinstance(v, dict):
            return {compact(k): compact(item) for k, item in v.items()}
        return v

    data = compact(value)
    return {"value_table": definitions, "data": data}


def unpack(value):
    return restore_records(replace(value["data"], value["value_table"]))


def content(value, mapping=None):
    return encode(pack(display_numbers(value), mapping)).decode()


def display_numbers(value):
    """Registered display precision only; source packets and feedback stay exact."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [display_numbers(v) for v in value]
    if isinstance(value, dict):
        return {k: display_numbers(v) for k, v in value.items()}
    return value


def memory_view(memory, observed):
    lookup = {r["id"]: r for r in observed}
    hypotheses = {h["id"]: h for h in memory.get("previous_hypotheses", [])}

    def reference(row):
        if row != lookup.get(row["id"]):
            raise ValueError("memory/observed feedback mismatch")
        return {"observed_id": row["id"]}

    def response_view(response):
        result = dict(response)
        choices = []
        for choice in response["choices"]:
            row = lookup[choice["id"]]
            expected = {
                "id": row["id"],
                "reason": row["selection_reason"],
                "scientific_role": row["scientific_role"],
                "hypothesis_id": row["hypothesis_id"],
                "evidence_ids": choice["evidence_ids"],
            }
            if choice != expected:
                raise ValueError("choice/observed feedback mismatch")
            choices.append(
                {"choice_ref": row["id"], "evidence_ids": choice["evidence_ids"]}
            )
        result["choices"] = choices
        result["hypotheses"] = []
        for hypothesis in response["hypotheses"]:
            if not all(
                hypotheses[hypothesis["id"]].get(k) == v for k, v in hypothesis.items()
            ):
                raise ValueError("historical hypothesis changed")
            result["hypotheses"].append({"hypothesis_ref": hypothesis["id"]})
        result["previous_hypothesis_updates"] = [
            {"update_ref": u["id"]}
            if all(hypotheses[u["id"]].get(k) == v for k, v in u.items())
            else u
            for u in response["previous_hypothesis_updates"]
        ]
        return result

    return {
        **memory,
        "recent_batches": [
            {
                **b,
                "observations": [reference(o) for o in b["observations"]],
                **(
                    {"response": response_view(b["response"])}
                    if "response" in b
                    else {}
                ),
            }
            for b in memory["recent_batches"]
        ],
        "high_error_observations": [
            reference(o) for o in memory["high_error_observations"]
        ],
    }


def summary_view(summary):
    result = dict(summary)
    for field in ("identity_counts", "scaffold_counts"):
        groups = {}
        for key, count in summary[field].items():
            groups.setdefault(count, []).append(key)
        result[field] = {
            "columns": ["count", "keys"],
            "rows": [[count, sorted(keys)] for count, keys in sorted(groups.items())],
        }
    return result
