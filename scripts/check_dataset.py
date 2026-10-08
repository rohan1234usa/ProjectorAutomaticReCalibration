"""Check a generated dataset against what the scenario asked for, independently of the simulator's truth code.

What was asked for comes from the dataset's scenario.yaml: each perturbation's kind, size
(``magnitude_px``/``magnitude_mm``/``deg``/``factor``), direction, pivot and schedule. The
geometry comes from setup.json: the calibrated boxes, the overlap P (box A ∩ box B ∩ content
rect), and the coarser projector's pixel pitch at P's centroid. All of it is recomputed with the
checker's own geometry code (``scripts/check_geometry.py``), independent of ``sim/``. Per frame, the recorded ground truth must then match:

  multiplier  every schedule re-evaluated here; continuous ones (drift, ramp, bump_then_hold,
              oscillate) may differ from it by half a quantum, the step of m moving the offset 0.02 px
  shift       offset = |sum of B's shift vectors - sum of A's|, each of length
              |m| x size (mm), across = pointing away from the partner, along = at right
              angles to it; the moved projector's h_actual = T(shift) h_cal, the other's unchanged
  rotation    offset = 2 R sin(|theta| / 2), R = the overlap vertex farthest from the pivot
  scale       offset = |s - 1| R
  keystone    offset = the requested size at full strength; otherwise, and for any mix of
              kinds, a brute-force maximum over a dense grid of the overlap
  offset_px   offset_mm / the pitch computed here

within 1e-6 mm (brute force: 1e-3 mm, as a grid slightly underestimates the maximum). For a
sweep directory, variants that differ only in their perturbation must have produced
bit-identical frames before the earliest onset among them: they share their seed.

Usage:
    python -m scripts.check_dataset out/shift_sweep   (one dataset, or a sweep directory)

Prints one JSON line per dataset and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.check_geometry import Geometry, brute_force_mm

TOL_MM = 1e-6
BRUTE_TOL_MM = 1e-3
QUANTUM_PX = 0.02
CONTINUOUS = ("drift", "ramp", "bump_then_hold", "oscillate")


def _specs(scenario: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    perturbation = scenario.get("perturbation")
    if isinstance(perturbation, dict):
        for name in ("a", "b"):  # the simulator's order: projectors by name, then list order
            items = perturbation.get(name)
            if items is not None:
                out += [(name, s) for s in ([items] if isinstance(items, dict) else items)]
    return out


def _size_mm(spec: dict[str, Any], pitch: float) -> float | None:
    if "magnitude_px" in spec:
        return float(spec["magnitude_px"]) * pitch
    if "magnitude_mm" in spec:
        return float(spec["magnitude_mm"])
    return None


def _exact(value: Any) -> Fraction:
    return Fraction(str(value))  # YAML's decimal text, as written


def expected_multiplier(schedule: Any, t: Fraction) -> float:
    """The schedule's multiplier m(t), before any quantization (see sim/schedule.py's table)."""
    if schedule in (None, "none") or schedule.get("type") == "none" or t < _exact(schedule.get("t0_s", 0)):
        return 0.0
    kind, dt = schedule["type"], t - _exact(schedule.get("t0_s", 0))
    if kind == "step":
        return 1.0
    if kind == "staircase":
        levels = schedule["levels"]
        return float(levels[min(int(dt // _exact(schedule["hold_s"])), len(levels) - 1)])
    if kind == "drift":
        return float(_exact(schedule["rate_per_h"]) * dt / 3600)
    if kind == "ramp":
        return float(min(Fraction(1), dt / _exact(schedule["duration_s"])))
    if kind == "bump_then_hold":
        peak, hold, tau = (float(_exact(schedule.get(k, 1))) for k in ("peak", "hold", "settle_s"))
        return hold + (peak - hold) * (math.exp(-float(dt) / tau) if tau > 0 else 0.0)
    return 0.5 * (1 - math.cos(2 * math.pi * float(dt / _exact(schedule["period_s"]))))  # oscillate


def _full_strength(name: str, spec: dict[str, Any], geo: Geometry) -> tuple[float | None, float]:
    """(rotation angle or scale - 1 at m = 1, else None; the offset it alone then causes, in mm)."""
    size = _size_mm(spec, geo.pitch)
    if spec["kind"] in ("shift", "keystone"):
        return None, abs(size)
    r = geo.reach(geo.pivot(name, spec.get("pivot")))
    if spec["kind"] == "rotation":
        full = (math.radians(float(spec["deg"])) if size is None
                else math.copysign(2 * math.asin(min(1.0, abs(size) / (2 * r))), size))
        return full, 2 * r * abs(math.sin(full / 2))
    full = float(spec["factor"]) - 1.0 if size is None else size / r
    return full, abs(full) * r


def _frame_expectation(line: dict[str, Any], specs: list, geo: Geometry, problems: list[str]) -> tuple[float | None, bool]:
    """(expected offset in mm, or None if only brute force can tell; whether brute force is needed)."""
    entries = line["perturbation"]
    if len(entries) != len(specs):
        problems.append(f"frame {line['i']}: {len(entries)} perturbation tags for {len(specs)} specs")
        return None, False
    t = Fraction(repr(line["t_s"]))
    active = []
    for (name, spec), entry in zip(specs, entries, strict=True):
        m = expected_multiplier(spec.get("schedule"), t)
        unit = _full_strength(name, spec, geo)[1]
        kind = spec["schedule"].get("type") if isinstance(spec.get("schedule"), dict) else None
        quantum = QUANTUM_PX * geo.pitch / unit if kind in CONTINUOUS and unit > 0 else 0.0
        if abs(entry["applied"] - m) > quantum / 2 + 1e-12 or entry["projector"] != name or entry["kind"] != spec["kind"]:
            problems.append(f"frame {line['i']}: {name} {spec['kind']} applied {entry['applied']}, expected {m}")
        m = entry["applied"]  # within its quantum of the schedule: what the offset follows
        if m != 0.0:
            active.append((name, spec, entry, m))
    if not active:
        return 0.0, False
    if all(spec["kind"] == "shift" for _, spec, _, _ in active):
        total = {"a": np.zeros(2), "b": np.zeros(2)}
        for name, spec, entry, m in active:
            v = np.array(entry["vector_mm"])
            size = _size_mm(spec, geo.pitch)
            if abs(np.hypot(*v) - abs(m * size)) > TOL_MM:
                problems.append(f"frame {line['i']}: shift of {np.hypot(*v):.9f} mm, asked {abs(m * size):.9f} mm")
            total[name] += v
            if size == 0:
                continue  # a zero-size shift has no direction to check
            partner = "a" if name == "b" else "b"
            away = geo.boxes[name].mean(axis=0) - geo.boxes[partner].mean(axis=0)
            cos = float(v @ away) / (np.hypot(*v) * np.hypot(*away)) * math.copysign(1.0, m * size)
            direction = spec.get("direction", "across")
            if (direction == "across" and cos < 0.9) or (direction == "along" and abs(cos) > 0.1):
                problems.append(f"frame {line['i']}: shift {direction} points {cos:+.3f} along the centre-to-centre line")
            if isinstance(direction, list):
                given = np.array(direction, dtype=float) * math.copysign(1.0, m * size)
                if float(v @ given) / (np.hypot(*v) * np.hypot(*given)) < 1 - 1e-9:
                    problems.append(f"frame {line['i']}: shift does not follow the requested vector {direction}")
        for name in ("a", "b"):
            h_act = np.array(line["truth"]["h_actual"][name])
            want = geo.h[name] if not total[name].any() else np.array([[1, 0, total[name][0]], [0, 1, total[name][1]], [0, 0, 1.0]]) @ geo.h[name]
            if not np.allclose(h_act, want, rtol=1e-12, atol=1e-9):
                problems.append(f"frame {line['i']}: h_actual of {name} is not its shift of h_cal")
        return float(np.hypot(*(total["b"] - total["a"]))), False
    if len(active) == 1:
        name, spec, entry, m = active[0]
        pivot = geo.pivot(name, spec.get("pivot"))
        if "pivot_mm" in entry and not np.allclose(entry["pivot_mm"], pivot, atol=1e-9):
            problems.append(f"frame {line['i']}: pivot {entry['pivot_mm']} is not the requested {spec.get('pivot', 'centre')}")
        full, unit = _full_strength(name, spec, geo)
        if spec["kind"] == "rotation":
            return 2 * geo.reach(pivot) * abs(math.sin(m * full / 2)), False
        if spec["kind"] == "scale":
            return abs(m * full) * geo.reach(pivot), False
        if spec["kind"] == "keystone" and m == 1.0:
            return unit, False
    return None, True


def check(path: Path) -> dict[str, Any]:
    setup = json.loads((path / "setup.json").read_text())
    scenario = yaml.safe_load((path / "scenario.yaml").read_text())
    lines = [json.loads(s) for s in (path / "metadata.jsonl").read_text().splitlines()]
    geo = Geometry(setup)
    specs = _specs(scenario)
    problems: list[str] = []
    worst, worst_px, worst_brute, n_closed, n_brute = 0.0, 0.0, 0.0, 0, 0
    for line in lines:
        expected, brute = _frame_expectation(line, specs, geo, problems)
        got = line["truth"]["offset_mm"]
        worst_px = max(worst_px, abs(line["truth"]["offset_px"] - got / geo.pitch))
        if brute:
            worst_brute = max(worst_brute, abs(got - brute_force_mm(line, geo)))
            n_brute += 1
        elif expected is not None:
            worst = max(worst, abs(got - expected))
            n_closed += 1
    ok = (worst <= TOL_MM and worst_px <= TOL_MM / geo.pitch and worst_brute <= BRUTE_TOL_MM and not problems)
    return {"dataset": str(path), "frames": len(lines), "pitch_mm": geo.pitch, "closed_form_frames": n_closed,
            "max_offset_error_mm": worst, "max_offset_px_error": worst_px, "brute_force_frames": n_brute,
            "max_brute_force_error_mm": worst_brute, "problems": problems[:5], "n_problems": len(problems), "ok": ok}


def check_paired(root: Path, variants: list[str]) -> dict[str, Any]:
    """Variants differing only in their perturbation must share every frame before the first onset."""
    groups: dict[str, list[str]] = {}
    for v in variants:
        scenario = yaml.safe_load((root / v / "scenario.yaml").read_text())
        rest = {k: val for k, val in copy.deepcopy(scenario).items() if k != "perturbation"}
        groups.setdefault(json.dumps(rest, sort_keys=True, default=str), []).append(v)
    checked, same = 0, True
    for members in groups.values():
        hashes, onsets = {}, []
        for v in members:
            lines = [json.loads(s) for s in (root / v / "metadata.jsonl").read_text().splitlines()]
            hashes[v] = [line["frame_sha256"] for line in lines]
            moved = [line["i"] for line in lines if any(e["applied"] != 0.0 for e in line["perturbation"])]
            if moved:
                onsets.append(moved[0])
        if len(members) < 2 or any(h is None for hs in hashes.values() for h in hs):
            continue
        first = min(onsets) if onsets else min(len(h) for h in hashes.values())
        reference = hashes[members[0]][:first]
        same &= all(hashes[v][:first] == reference for v in members)
        checked += first * len(members)
    return {"paired_groups": len(groups), "paired_frames_checked": checked, "paired_identical": same, "ok": same}


def main(argv: list[str] | None = None) -> bool:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a dataset directory, or a sweep directory with variants.json")
    args = parser.parse_args(argv)
    index = args.path / "variants.json"
    ok = True
    if index.exists():
        variants = [v["dir"] for v in json.loads(index.read_text())["variants"] if (args.path / v["dir"]).exists()]
        for v in variants:
            result = check(args.path / v)
            ok &= result["ok"]
            print(json.dumps(result))
        paired = check_paired(args.path, variants)
        ok &= paired["ok"]
        print(json.dumps(paired))
    else:
        result = check(args.path)
        ok &= result["ok"]
        print(json.dumps(result))
    print(json.dumps({"ok": ok}))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
