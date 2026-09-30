"""Column dictionaries, lossless relative to the registered six-decimal display.

Every original field is retained. Dictionary references are column-specific integer
indices; literal values inside dictionaries are never recursively decoded.
"""

from hplc_al.llm.full_pool import tokenizer
from hplc_al.llm.responses_transport import encode
from hplc_al.llm.wire import aliases, display_numbers, replace


def tokens(value):
    return len(tokenizer().encode(encode(value).decode(), disallowed_special=()))


def pack(cards, observed, memory):
    literals = set()

    def collect(value):
        if isinstance(value, str):
            literals.add(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(key)
                collect(item)

    collect([cards, observed, memory])
    mapping = {}
    for prefix, values in (
        ("g", list(aliases(cards, observed))),
        ("c", [r["id"] for r in cards]),
        ("o", [r["id"] for r in observed]),
    ):
        index = 0
        for value in values:
            while f"{prefix}{index}" in literals:
                index += 1
            mapping[value] = f"{prefix}{index}"
            index += 1
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("non-bijective wire aliases")
    # Memory retains all observations, historical responses and hypothesis states.
    rows = replace(display_numbers(cards + observed), mapping)
    dictionaries = {}
    for field in sorted({key for row in rows for key in row}):
        values = [row[field] for row in rows if field in row]
        unique, indices, references = [], {}, []
        for value in values:
            key = encode(value)
            if key not in indices:
                indices[key] = len(unique)
                unique.append(value)
            references.append(indices[key])
        if tokens({"values": unique, "refs": references}) < tokens(values):
            dictionaries[field] = unique
            for row in rows:
                if field in row:
                    row[field] = indices[encode(row[field])]
    return {
        "dictionaries": dictionaries,
        "cards": rows[: len(cards)],
        "observed": rows[len(cards) :],
        "memory": replace(display_numbers(memory), mapping),
    }, mapping


def table(rows):
    if not rows:
        return {"columns": [], "rows": []}
    columns = sorted(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError("mixed schemas in table")
    return {"columns": columns, "rows": [[row[k] for k in columns] for row in rows]}


def tables(rows):
    groups = {}
    for position, row in enumerate(rows):
        key = tuple(sorted(row))
        positions, records = groups.setdefault(key, ([], []))
        positions.append(position)
        records.append(row)
    return [
        {"positions": positions, **table(records)}
        for positions, records in groups.values()
    ]


def decode_table(value, dictionaries):
    columns = value["columns"]
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate table columns")
    rows = []
    for cells in value["rows"]:
        if len(cells) != len(columns):
            raise ValueError("table row width mismatch")
        row = dict(zip(columns, cells))
        for field in set(row) & set(dictionaries):
            index = row[field]
            if type(index) is not int or not 0 <= index < len(dictionaries[field]):
                raise ValueError("invalid column dictionary reference")
            row[field] = dictionaries[field][index]
        rows.append(row)
    return rows


def unpack(value, mapping):
    reverse = {v: k for k, v in mapping.items()}
    restored = {}
    for key in ("cards", "observed"):
        rows = [None] * len(value[key])
        for group in tables(value[key]):
            for position, row in zip(
                group["positions"], decode_table(group, value["dictionaries"])
            ):
                rows[position] = row
        restored[key] = replace(rows, reverse)
    restored["memory"] = replace(value["memory"], reverse)
    return restored
