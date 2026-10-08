"""Scenario files: inheritance (``extends``) and sweeps (one dataset variant per value).

Fifteen scenarios share one lecture hall, so a scenario may start from a base file and change
only what its question needs: ``extends: _lecture_hall.yaml`` deep-merges the scenario over the
base (mappings merge key by key; lists and values replace; ``key: null`` removes a key the base
set, e.g. the whole-screen camera's ``margin`` when switching to the zoomed preset). Bases may
extend other bases. Files whose name starts with ``_`` are bases, not scenarios.

A ``sweep`` maps dotted keys to lists of values, e.g.

    sweep:
      perturbation.b.magnitude_px: [0, 0.25, 0.5]
      perturbation.b.direction: [across, along]

and expands to the Cartesian product, in the order written: here six variants named like
``magnitude_px=0.25__direction=across``. Values may be whole mappings (a whole ``arrangement``);
a list index is a number in the dotted key (``content.items.0.density``). Variants that are the
same scenario in effect (by a caller-supplied canonical form, e.g. any zero-size shift) are kept
once. Variants inherit the scenario's seed, so they are paired: same content, same noise.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in over.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """A scenario file with its ``extends`` chain resolved (the result has no ``extends`` key)."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"scenario: extends loop through {path.name}")
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path.name}: expected a mapping of keys")
    data = dict(data)
    base = data.pop("extends", None)
    if base is None:
        return data
    merged = deep_merge(load_yaml(path.parent / base, (*_seen, path)), data)
    if "name" not in data:
        merged.pop("name", None)  # a scenario is named after its own file, never its base
    return merged


def set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    """Set data[a][b][c] for key "a.b.c", creating mappings on the way; numbers index lists."""
    parts = key.split(".")
    node: Any = data
    for i, part in enumerate(parts[:-1]):
        child_is_index = parts[i + 1].isdigit()
        if isinstance(node, list):
            node = node[int(part)]
            continue
        if part not in node or node[part] is None or isinstance(node[part], str):
            node[part] = [] if child_is_index else {}
        node = node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = copy.deepcopy(value)
    else:
        node[last] = copy.deepcopy(value)


def _label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, Mapping):  # its scalar values in the order written, e.g. "keystone-2-y"
        parts = [_label(v) for v in value.values() if not isinstance(v, (Mapping, list, tuple))]
        return "-".join(parts) or "custom"
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, Mapping) for v in value):  # e.g. a list of nuisances: their types
            firsts = [_label(next(iter(v.values()))) for v in value]
            return "+".join(dict.fromkeys(firsts))
        return "-".join(_label(v) for v in value)
    return str(value)


def variant_name(assignment: list[tuple[str, Any]]) -> str:
    raw = "__".join(f"{key.split('.')[-1]}={_label(value)}" for key, value in assignment)
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", raw)


def expand(
    data: Mapping[str, Any], canonical: Callable[[dict[str, Any]], Any] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """(variant name, scenario data without ``sweep``) for every distinct sweep combination."""
    data = dict(data)
    sweep = data.pop("sweep", None)
    if not sweep:
        return [(str(data.get("name", "scenario")), data)]
    if not isinstance(sweep, Mapping) or not all(isinstance(v, list) and v for v in sweep.values()):
        raise ValueError("sweep: expected a mapping of dotted keys to non-empty lists of values")
    keys = list(sweep)
    out, seen, names = [], set(), set()
    for values in itertools.product(*(sweep[k] for k in keys)):
        variant = copy.deepcopy(data)
        for key, value in zip(keys, values):
            set_dotted(variant, key, value)
        signature = json.dumps(canonical(variant) if canonical else variant, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        name, n = variant_name(list(zip(keys, values))), 2
        while name in names:  # distinct variants whose labels coincide
            name, n = f"{variant_name(list(zip(keys, values)))}_{n}", n + 1
        names.add(name)
        out.append((name, variant))
    return out
